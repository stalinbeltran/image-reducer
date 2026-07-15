import json
import time

from image_reducer.audit import AuditLog


def test_audit_writes_jsonl_append(tmp_path):
    p = tmp_path / "audit.log.jsonl"
    log = AuditLog(p)
    log.log("op.one", a=1)
    log.log("op.two", b=2)
    log.close()

    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    events = [l["event"] for l in lines]
    assert "app.start" in events          # se registra al iniciar
    assert "op.one" in events and "op.two" in events
    assert all("ts" in l for l in lines)  # cada línea lleva timestamp


def test_audit_status_ok_after_write(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.log("x")
    for _ in range(50):
        if log.status()["written"] >= 1:
            break
        time.sleep(0.02)
    st = log.status()
    assert st["ok"] is True
    assert st["written"] >= 1
    assert st["dropped"] == 0
    log.close()


def test_audit_never_overwrites_existing(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"event":"previo"}\n', encoding="utf-8")
    log = AuditLog(p)
    log.log("nuevo")
    log.close()
    text = p.read_text(encoding="utf-8")
    assert '"event":"previo"' in text     # no se sobrescribió lo anterior
    assert '"nuevo"' in text


def test_audit_write_error_informs(tmp_path):
    # Apuntar el log a un directorio existente hace que la escritura falle;
    # el error debe reflejarse en status() sin lanzar en el hilo principal.
    d = tmp_path / "soy_directorio"
    d.mkdir()
    log = AuditLog(d)
    log.log("op", x=1)
    for _ in range(100):
        if not log.status()["ok"]:
            break
        time.sleep(0.02)
    st = log.status()
    assert st["ok"] is False
    assert st["last_error"]
    assert st["dropped"] >= 1
    log.close()
