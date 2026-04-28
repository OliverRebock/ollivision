import subprocess

import pytest

from ollivision.hermes_client import describe_image


def test_describe_image_returns_dummy_response_in_dummy_mode(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "dummy",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "auto",
        },
    )

    called = {"run": False}

    def fake_run(cmd, capture_output, text, check):
        called["run"] = True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="should-not-run", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    result = describe_image("/tmp/test.jpg", "Beschreibe das Bild")

    assert "Dummy Hermes Antwort" in result
    assert "/tmp/test.jpg" in result
    assert called["run"] is False


def test_describe_image_image_mode_builds_expected_command(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "image",
        },
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Szenenbeschreibung", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    result = describe_image(str(img), "Beschreibe das Bild")

    assert result == "Szenenbeschreibung"
    assert captured["cmd"][:4] == ["hermes", "chat", "-Q", "-q"]
    assert "--image" in captured["cmd"]
    idx = captured["cmd"].index("--image")
    assert captured["cmd"][idx + 1] == str(img)


def test_describe_image_path_only_mode_uses_legacy_prompt(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "path_only",
        },
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    describe_image(str(img), "Beschreibe das Bild")

    assert captured["cmd"][0] == "hermes"
    assert captured["cmd"][1] == "-z"
    assert f"Bildpfad: {img}" in captured["cmd"][2]


def test_describe_image_adds_model_when_set(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": "gpt-5.3-codex",
            "image_mode": "image",
        },
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    describe_image(str(img), "Beschreibe das Bild")

    assert "-m" in captured["cmd"]
    assert "gpt-5.3-codex" in captured["cmd"]


def test_describe_image_raises_when_image_missing(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "image",
        },
    )

    with pytest.raises(RuntimeError, match="Bilddatei existiert nicht"):
        describe_image("/tmp/does-not-exist.jpg", "Beschreibe das Bild")


def test_describe_image_raises_readable_error_when_cli_fails(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "auto",
        },
    )

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(args=cmd, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Hermes-CLI Anfrage fehlgeschlagen"):
        describe_image(str(img), "Beschreibe das Bild")


def test_describe_image_raises_when_cli_missing(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "auto",
        },
    )

    def fake_run(cmd, capture_output, text, check):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Hermes-CLI nicht gefunden"):
        describe_image(str(img), "Beschreibe das Bild")


def test_describe_image_raises_when_output_is_kein_bildzugriff(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "image",
        },
    )

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="KEIN_BILDZUGRIFF", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="konnte das Bild nicht visuell auswerten"):
        describe_image(str(img), "Beschreibe das Bild")


def test_describe_image_raises_when_output_is_ich_kann_das_bild_nicht_sehen(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "image",
        },
    )

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Ich kann das Bild nicht sehen.", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="konnte das Bild nicht visuell auswerten"):
        describe_image(str(img), "Beschreibe das Bild")


def test_path_only_mode_marks_debug_no_real_analysis(monkeypatch, tmp_path):
    img = tmp_path / "test.jpg"
    img.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {
            "mode": "live",
            "provider": "hermes_cli",
            "command": "hermes",
            "model": None,
            "image_mode": "path_only",
        },
    )

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Vermutung anhand Pfad", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    result = describe_image(str(img), "Beschreibe das Bild")

    assert result.startswith("[DEBUG path_only: keine echte Bildanalyse]")
