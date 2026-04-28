import json
from urllib.error import URLError

import pytest

from ollivision.hermes_client import describe_image


def test_describe_image_returns_dummy_response_in_dummy_mode(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "dummy", "base_url": "http://localhost:8000", "describe_endpoint": "/api/vision/describe"},
    )

    result = describe_image("/tmp/test.jpg", "Beschreibe das Bild")

    assert "Dummy Hermes Antwort" in result
    assert "/tmp/test.jpg" in result


def test_describe_image_calls_http_endpoint(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "http", "base_url": "http://localhost:8000", "describe_endpoint": "/api/vision/describe"},
    )

    captured = {}

    class FakeResponse:
        def __init__(self, payload: dict):
            self.status = 200
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({"description": "Ein Testbild"})

    monkeypatch.setattr("ollivision.hermes_client.urlopen", fake_urlopen)

    result = describe_image("/tmp/test.jpg", "Beschreibe das Bild")

    assert result == "Ein Testbild"
    assert captured["url"] == "http://localhost:8000/api/vision/describe"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 10
    assert captured["body"]["image_path"] == "/tmp/test.jpg"
    assert captured["body"]["prompt"] == "Beschreibe das Bild"


def test_describe_image_raises_readable_error_when_unreachable(monkeypatch):
    monkeypatch.setattr(
        "ollivision.hermes_client._load_hermes_config",
        lambda: {"mode": "http", "base_url": "http://localhost:8000", "describe_endpoint": "/api/vision/describe"},
    )

    def fake_urlopen(req, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("ollivision.hermes_client.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Hermes nicht erreichbar"):
        describe_image("/tmp/test.jpg", "Beschreibe das Bild")
