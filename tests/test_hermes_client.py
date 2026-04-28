import subprocess

import pytest

from ollivision.hermes_client import describe_image


def test_describe_image_returns_dummy_response_in_dummy_mode(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "dummy", "provider": "hermes_cli", "command": "hermes", "model": None},
    )

    result = describe_image("/tmp/test.jpg", "Beschreibe das Bild")

    assert "Dummy Hermes Antwort" in result
    assert "/tmp/test.jpg" in result


def test_describe_image_calls_hermes_cli(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "live", "provider": "hermes_cli", "command": "hermes", "model": None},
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["check"] = check
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Szenenbeschreibung", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    result = describe_image("/tmp/test.jpg", "Beschreibe das Bild")

    assert result == "Szenenbeschreibung"
    assert captured["cmd"][0] == "hermes"
    assert captured["cmd"][1] == "-z"
    assert "Bildpfad: /tmp/test.jpg" in captured["cmd"][2]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["check"] is False


def test_describe_image_calls_hermes_cli_with_optional_model(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "live", "provider": "hermes_cli", "command": "hermes", "model": "gpt-5.3-codex"},
    )

    captured = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    describe_image("/tmp/test.jpg", "Beschreibe das Bild")

    assert "-m" in captured["cmd"]
    assert "gpt-5.3-codex" in captured["cmd"]


def test_describe_image_raises_readable_error_when_cli_fails(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "live", "provider": "hermes_cli", "command": "hermes", "model": None},
    )

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(args=cmd, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Hermes-CLI Anfrage fehlgeschlagen"):
        describe_image("/tmp/test.jpg", "Beschreibe das Bild")


def test_describe_image_raises_when_cli_missing(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "live", "provider": "hermes_cli", "command": "hermes", "model": None},
    )

    def fake_run(cmd, capture_output, text, check):
        raise FileNotFoundError("not found")

    monkeypatch.setattr("ollivision.hermes_client.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Hermes-CLI nicht gefunden"):
        describe_image("/tmp/test.jpg", "Beschreibe das Bild")
