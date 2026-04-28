from pathlib import Path
import subprocess


def _load_hermes_config() -> dict[str, str | None]:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    text = config_path.read_text(encoding="utf-8")

    in_hermes = False
    config: dict[str, str | None] = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" ") and line.endswith(":"):
            in_hermes = line[:-1].strip() == "hermes"
            continue

        if in_hermes and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            cleaned = value.strip().strip('"').strip("'")
            if cleaned.lower() in {"null", "none", "~", ""}:
                config[key.strip()] = None
            else:
                config[key.strip()] = cleaned

    return {
        "mode": config.get("mode") or "dummy",
        "provider": config.get("provider") or "hermes_cli",
        "command": config.get("command") or "hermes",
        "model": config.get("model"),
    }


def _build_hermes_cli_command(command: str, prompt: str, model: str | None = None) -> list[str]:
    cmd = [command, "-z", prompt]
    if model:
        cmd.extend(["-m", model])

    # Für später: echte Bildübergabe kann ergänzt werden (z.B. `hermes chat -q ... --image <path>`)
    return cmd


def _describe_with_hermes_cli(image_path: str, prompt: str, command: str, model: str | None = None) -> str:
    full_prompt = (
        f"{prompt}\n"
        f"Bildpfad: {image_path}\n"
        "Beschreibe die Szene kurz und präzise auf Deutsch."
    )
    cmd = _build_hermes_cli_command(command=command, prompt=full_prompt, model=model)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Hermes-CLI nicht gefunden: {command}") from exc
    except OSError as exc:
        raise RuntimeError(f"Hermes-CLI konnte nicht gestartet werden: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        detail = stderr[:300] if stderr else "Unbekannter Fehler"
        raise RuntimeError(f"Hermes-CLI Anfrage fehlgeschlagen (exit {result.returncode}): {detail}")

    output = (result.stdout or "").strip()
    if not output:
        raise RuntimeError("Hermes-CLI lieferte keine Ausgabe")

    return output


def describe_image(image_path: str, prompt: str) -> str:
    cfg = _load_hermes_config()

    if cfg["mode"] == "dummy":
        return f"Dummy Hermes Antwort für {image_path}"

    provider = cfg["provider"]
    if provider == "hermes_cli":
        return _describe_with_hermes_cli(
            image_path=image_path,
            prompt=prompt,
            command=str(cfg["command"]),
            model=cfg["model"],
        )

    raise RuntimeError(f"Unbekannter Hermes-Provider: {provider}")
