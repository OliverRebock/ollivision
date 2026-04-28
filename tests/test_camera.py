import subprocess

import pytest

from ollivision.camera import capture_image


def test_capture_image_calls_rpicam_still(monkeypatch):
    called = {}

    def fake_run(cmd, check, capture_output, text):
        called["cmd"] = cmd
        called["check"] = check
        called["capture_output"] = capture_output
        called["text"] = text
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = capture_image("/tmp/test.jpg")

    assert output == "/tmp/test.jpg"
    assert called["cmd"] == ["rpicam-still", "-o", "/tmp/test.jpg", "--timeout", "1000", "--nopreview"]
    assert called["check"] is True
    assert called["capture_output"] is True
    assert called["text"] is True


def test_capture_image_raises_readable_error(monkeypatch):
    def fake_run(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(1, cmd, stderr="camera not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Kameraaufnahme fehlgeschlagen"):
        capture_image("/tmp/test.jpg")
