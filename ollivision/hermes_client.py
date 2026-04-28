import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def _load_hermes_config() -> dict[str, str]:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    text = config_path.read_text(encoding="utf-8")

    in_hermes = False
    config: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" ") and line.endswith(":"):
            in_hermes = line[:-1].strip() == "hermes"
            continue

        if in_hermes and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            config[key.strip()] = value.strip().strip('"').strip("'")

    return {
        "mode": config.get("mode", "dummy"),
        "base_url": config.get("base_url", "http://localhost:8000"),
        "describe_endpoint": config.get("describe_endpoint", "/api/vision/describe"),
    }


def describe_image(image_path: str, prompt: str) -> str:
    cfg = _load_hermes_config()

    if cfg["mode"] == "dummy":
        return f"Dummy Hermes Antwort für {image_path}"

    url = urljoin(cfg["base_url"].rstrip("/") + "/", cfg["describe_endpoint"].lstrip("/"))
    payload = {"image_path": image_path, "prompt": prompt}
    data = json.dumps(payload).encode("utf-8")

    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(f"Hermes nicht erreichbar: {exc.reason}") from exc
    except HTTPError as exc:
        raise RuntimeError(f"Hermes Anfrage fehlgeschlagen: HTTP {exc.code}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Hermes Antwort ist kein gültiges JSON") from exc

    description = parsed.get("description")
    if not description:
        raise RuntimeError("Hermes Antwort enthält kein Feld 'description'")

    return description
