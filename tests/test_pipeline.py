import json
import math

from PIL import Image

from image_reducer import (
    ReduceConfig,
    compute_transform,
    process_image,
    process_mask,
    transform_labels,
)
from image_reducer.dataset import process_dataset, process_folder


def test_letterbox_transform_geometry():
    # 200x100 -> 100x100 letterbox: escala 0.5, contenido 100x50, pad_y=25.
    t = compute_transform(200, 100, 100, 100, keep_aspect=True)
    assert t.scale_x == t.scale_y == 0.5
    assert (t.content_w, t.content_h) == (100, 50)
    assert t.pad_x == 0 and t.pad_y == 25
    assert t.point(0, 0) == (0, 25)
    assert t.point(200, 100) == (100, 75)


def test_transform_box_and_inverse_roundtrip():
    t = compute_transform(640, 480, 320, 320, keep_aspect=True)
    box = [76.84, 308.89, 255.5, 88.67]
    tb = t.box(box)
    # inverse debe recuperar el original (dentro de la tolerancia de redondeo).
    back = t.inverse_box(tb)
    for a, b in zip(back, box):
        assert math.isclose(a, b, abs_tol=0.2)


def test_process_image_size_and_mode():
    img = Image.new("RGB", (200, 100), (10, 20, 30))
    cfg = ReduceConfig(width=64, height=64, blur_radius=1.0)
    out, t = process_image(img, cfg)
    assert out.size == (64, 64)
    assert out.mode == "L"
    assert t.target_w == 64 and t.target_h == 64


def test_stretch_mode_no_padding():
    img = Image.new("RGB", (200, 100), (0, 0, 0))
    cfg = ReduceConfig(width=64, height=64, keep_aspect=False)
    out, t = process_image(img, cfg)
    assert out.size == (64, 64)
    assert t.pad_x == 0 and t.pad_y == 0
    assert t.scale_x != t.scale_y


def test_mask_stays_binary():
    mask = Image.new("L", (200, 100), 0)
    mask.putpixel((10, 10), 255)
    t = compute_transform(200, 100, 64, 64, keep_aspect=True)
    out = process_mask(mask, t)
    assert out.mode == "L"
    assert set(out.getdata()) <= {0, 255}


def test_transform_labels_updates_geometry():
    t = compute_transform(640, 480, 320, 320, keep_aspect=True)
    labels = {
        "image_id": "x/0", "width": 640, "height": 480, "has_overlap": False,
        "blocks": [{"block_id": "b0", "angle": 0.0,
                    "box": [100, 100, 200, 50],
                    "quad": [[100, 100], [300, 100], [300, 150], [100, 150]]}],
        "lines": [], "words": [],
    }
    out = transform_labels(labels, t)
    assert out["width"] == 320 and out["height"] == 320
    assert out["image_id"] == "x/0"  # sin cambios
    # escala 320/640 = 0.5, pad_y = (320-240)//2 = 40
    assert out["blocks"][0]["box"] == [50.0, 90.0, 100.0, 25.0]
    assert labels["blocks"][0]["box"] == [100, 100, 200, 50]  # original intacto


def _make_dataset(root):
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "labels").mkdir()
    img = Image.new("RGB", (200, 100), (128, 128, 128))
    img.save(root / "images" / "000000.png")
    mask = Image.new("L", (200, 100), 0)
    mask.save(root / "masks" / "000000.png")
    labels = {"image_id": "d/0", "width": 200, "height": 100, "has_overlap": False,
              "blocks": [{"block_id": "b0", "box": [10, 10, 40, 20],
                          "quad": [[10, 10], [50, 10], [50, 30], [10, 30]]}],
              "lines": [], "words": []}
    rec = {"index": 0, "image": "images/000000.png",
           "mask": "masks/000000.png", "labels": labels}
    (root / "labels.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (root / "dataset.json").write_text(json.dumps({"id": "d", "count": 1}), encoding="utf-8")


def test_process_dataset_end_to_end(tmp_path):
    src = tmp_path / "in"
    _make_dataset(src)
    out = tmp_path / "out"
    summary = process_dataset(src, out, ReduceConfig(width=64, height=64))

    assert summary["images"] == 1 and summary["masks"] == 1
    assert (out / "images" / "000000.png").exists()
    assert (out / "masks" / "000000.png").exists()
    assert (out / "labels" / "000000.json").exists()

    proc = Image.open(out / "images" / "000000.png")
    assert proc.size == (64, 64) and proc.mode == "L"

    recs = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines()]
    assert recs[0]["labels"]["width"] == 64
    meta = json.loads((out / "dataset.json").read_text())
    assert meta["reducer"]["config"]["width"] == 64


def test_process_folder(tmp_path):
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (200, 100), (0, 0, 0)).save(src / "a.jpg")
    Image.new("RGB", (100, 100), (0, 0, 0)).save(src / "b.png")
    out = tmp_path / "out"
    summary = process_folder(src, out, ReduceConfig(width=32, height=32))
    assert summary["images"] == 2
    assert (out / "a.png").exists() and (out / "b.png").exists()
    tr = (out / "transforms.jsonl").read_text().splitlines()
    assert len(tr) == 2
