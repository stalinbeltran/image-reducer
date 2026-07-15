import json

import pytest
from PIL import Image

from image_reducer.config import ReduceConfig
from image_reducer.registry import Registry
from image_reducer.service import PathSafetyError, detect_mode, run_job, validate_paths


# ------------------------- service: seguridad de rutas -------------------------
def test_validate_paths_rejects_output_equal_origin(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    with pytest.raises(PathSafetyError):
        validate_paths(src, src)


def test_validate_paths_rejects_output_inside_origin(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    with pytest.raises(PathSafetyError):
        validate_paths(src, src / "results")


def test_validate_paths_allows_disjoint(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    validate_paths(src, tmp_path / "out")  # no lanza


def test_detect_mode(tmp_path):
    folder = tmp_path / "imgs"
    folder.mkdir()
    img = folder / "a.png"
    Image.new("RGB", (10, 10)).save(img)
    assert detect_mode(folder) == "folder"
    assert detect_mode(img) == "image"
    (folder / "labels.jsonl").write_text("{}\n")
    assert detect_mode(folder) == "dataset"


def test_run_job_image_mode(tmp_path):
    img = tmp_path / "foto.png"
    Image.new("RGB", (200, 100), (10, 20, 30)).save(img)
    out = tmp_path / "out"
    summary = run_job(img, out, ReduceConfig(width=32, height=32))
    assert summary["mode"] == "image"
    assert (out / "foto.png").exists()
    assert (out / "foto.transform.json").exists()


# ------------------------------- registry CRUD --------------------------------
def test_registry_jobs_crud(tmp_path):
    reg = Registry(tmp_path / "data")
    job = reg.create_job({"label": "a", "mode": "dataset", "input": "i",
                          "output": "o", "config": {}, "summary": {"images": 3}})
    assert reg.get_job(job["id"])["label"] == "a"
    assert reg.update_job(job["id"], {"label": "b"})["label"] == "b"
    assert len(reg.list_jobs()) == 1
    assert reg.delete_job(job["id"]) is True
    assert reg.list_jobs() == []


def test_registry_delete_files(tmp_path):
    reg = Registry(tmp_path / "data")
    out = tmp_path / "out"
    out.mkdir()
    (out / "x.png").write_bytes(b"x")
    job = reg.create_job({"output": str(out), "mode": "folder"})
    reg.delete_job(job["id"], delete_files=True)
    assert not out.exists()


def test_registry_presets_crud(tmp_path):
    reg = Registry(tmp_path / "data")
    p = reg.create_preset("gris", {"width": 64, "height": 64})
    assert reg.update_preset(p["id"], {"name": "gris2"})["name"] == "gris2"
    assert [x["name"] for x in reg.list_presets()] == ["gris2"]
    assert reg.delete_preset(p["id"]) is True
    assert reg.list_presets() == []


# ------------------------------- API TestClient -------------------------------
def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_REDUCER_DATA", str(tmp_path / "appdata"))
    import importlib
    import image_reducer.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def test_api_process_and_jobs(tmp_path, monkeypatch):
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (100, 60), (0, 0, 0)).save(src / "a.png")
    out = tmp_path / "out"

    client = _client(tmp_path, monkeypatch)

    r = client.post("/api/process", json={
        "input": str(src), "output": str(out), "label": "t",
        "config": {"width": 48, "height": 48}})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["mode"] == "folder"
    assert (out / "a.png").exists()

    assert len(client.get("/api/jobs").json()["jobs"]) == 1

    # destino dentro del origen -> 400
    bad = client.post("/api/process", json={
        "input": str(src), "output": str(src / "sub"),
        "config": {"width": 48, "height": 48}})
    assert bad.status_code == 400

    # borrar job
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get("/api/jobs").json()["jobs"] == []


def test_api_fs_inspect(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    img = tmp_path / "z.png"
    Image.new("RGB", (10, 10)).save(img)
    r = client.get("/api/fs/inspect", params={"path": str(img)})
    assert r.status_code == 200
    assert r.json()["mode"] == "image"
