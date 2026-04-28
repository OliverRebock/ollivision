import pytest

from ollivision.app import describe_scene, main


def test_describe_scene_calls_hermes_client(monkeypatch):
    monkeypatch.setattr("ollivision.app.capture_image", lambda _: "/tmp/ollivision_latest.jpg")
    monkeypatch.setattr("ollivision.app.describe_image", lambda image_path, prompt: f"Beschreibung für {image_path} mit {prompt}")

    result = describe_scene()

    assert "Beschreibung für /tmp/ollivision_latest.jpg" in result


def test_cli_describe_scene_outputs_response(capsys, monkeypatch):
    monkeypatch.setattr("ollivision.app.capture_image", lambda _: "/tmp/ollivision_latest.jpg")
    monkeypatch.setattr("ollivision.app.describe_image", lambda image_path, prompt: "Dummy Hermes Antwort für /tmp/ollivision_latest.jpg")

    main(["describe-scene"])
    out = capsys.readouterr().out.strip()
    assert "Dummy Hermes Antwort" in out


def test_cli_shows_readable_error_on_camera_failure(capsys, monkeypatch):
    def fail_capture(_):
        raise RuntimeError("Kameraaufnahme fehlgeschlagen: camera not found")

    monkeypatch.setattr("ollivision.app.capture_image", fail_capture)

    with pytest.raises(SystemExit) as exc:
        main(["describe-scene"])

    assert exc.value.code == 1
    err = capsys.readouterr().err.strip()
    assert "Fehler:" in err
    assert "Kameraaufnahme fehlgeschlagen" in err
