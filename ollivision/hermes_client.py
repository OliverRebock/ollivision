from pathlib import Path
import re
import subprocess


ANSWER_BEGIN_MARKER = "OLLIVISION_ANSWER_BEGIN"
ANSWER_END_MARKER = "OLLIVISION_ANSWER_END"


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
        "cli_provider": config.get("cli_provider"),
        "image_mode": config.get("image_mode") or "auto",
    }


def _build_marked_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Du erhältst ein echtes Bild als Anhang.\n"
        "Beschreibe ausschließlich den sichtbaren Bildinhalt konkret auf Deutsch.\n"
        "Antworte nicht über deine Fähigkeiten.\n"
        "Gib keine Platzhalter, keine spitzen Klammern und keine Formatbeschreibung aus.\n"
        "Beginne deine Ausgabe exakt mit der Zeile:\n"
        f"{ANSWER_BEGIN_MARKER}\n"
        "Schreibe danach direkt die echte Bildbeschreibung.\n"
        "Beende deine Ausgabe exakt mit der Zeile:\n"
        f"{ANSWER_END_MARKER}"
    )


def _extract_marked_answer(output: str) -> str:
    text = (output or "").strip()
    if not text:
        return ""

    begin = text.find(ANSWER_BEGIN_MARKER)
    if begin == -1:
        return text

    begin += len(ANSWER_BEGIN_MARKER)
    end = text.find(ANSWER_END_MARKER, begin)
    if end == -1:
        return text

    return text[begin:end].strip()


def _build_hermes_cli_command(
    command: str,
    prompt: str,
    image_path: str,
    image_mode: str,
    model: str | None = None,
    cli_provider: str | None = None,
) -> list[str]:
    marked_prompt = _build_marked_prompt(prompt)

    if image_mode in {"auto", "image"}:
        cmd = [command, "chat", "-q", marked_prompt, "--image", image_path]
        if cli_provider:
            cmd.extend(["--provider", cli_provider])
        if model:
            cmd.extend(["-m", model])
        return cmd

    if image_mode == "path_only":
        path_prompt = (
            f"{prompt}\n"
            f"Bildpfad: {image_path}\n"
            "WICHTIG: Du hast hier keinen direkten Bildzugriff. "
            "Nutze nur den Pfad-Kontext und markiere die Antwort als Debug-Hinweis.\n\n"
            "Antworte ausschließlich zwischen diesen Markern:\n"
            f"{ANSWER_BEGIN_MARKER}\n"
            "[DEBUG path_only: <dein Hinweis>]\n"
            f"{ANSWER_END_MARKER}\n"
            "Kein Text vor oder nach den Markern."
        )
        cmd = [command, "-z", path_prompt]
        if model:
            cmd.extend(["-m", model])
        return cmd

    raise RuntimeError(f"Ungültiger image_mode: {image_mode}")


_NO_IMAGE_PATTERNS = [
    r"\bkein[_\s-]?bildzugriff\b",
    r"\bcannot access image\b",
    r"\bunable to view image\b",
    r"\bi can\s*'?t see the image\b",
    r"\bich kann das bild nicht sehen\b",
]


def _looks_like_no_image_access(output: str) -> bool:
    text = output.strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _NO_IMAGE_PATTERNS)


_INVALID_PLACEHOLDER_PATTERNS = [
    r"^<\s*deine\s+kurze\s+antwort\s*>$",
    r"^deine\s+kurze\s+antwort$",
    r"^<\s*antwort\s*>$",
    r"^antwort\s+hier$",
]

_FORMAT_TEXT_PATTERN = r"ein\s+bis\s+zwei\s+deutsche\s+sätze\s+mit\s+konkreter\s+bildbeschreibung"


def _is_invalid_placeholder_answer(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    lowered = text.lower()
    return any(re.fullmatch(pattern, lowered, flags=re.IGNORECASE) for pattern in _INVALID_PLACEHOLDER_PATTERNS)


def _is_invalid_formattext_answer(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    normalized = re.sub(r"\s+", " ", text)
    if re.fullmatch(rf"<?\s*{_FORMAT_TEXT_PATTERN}\s*>?", normalized, flags=re.IGNORECASE):
        return True
    return False


def _describe_with_hermes_cli(
    image_path: str,
    prompt: str,
    command: str,
    image_mode: str,
    model: str | None = None,
    cli_provider: str | None = None,
) -> str:
    if not Path(image_path).exists():
        raise RuntimeError(f"Bilddatei existiert nicht: {image_path}")

    cmd = _build_hermes_cli_command(
        command=command,
        prompt=prompt,
        image_path=image_path,
        image_mode=image_mode,
        model=model,
        cli_provider=cli_provider,
    )

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

    answer = _extract_marked_answer(output)
    if _is_invalid_placeholder_answer(answer):
        raise RuntimeError("Hermes lieferte nur einen Platzhalter statt einer Bildbeschreibung.")
    if _is_invalid_formattext_answer(answer):
        raise RuntimeError("Hermes lieferte nur Formattext statt einer Bildbeschreibung.")

    if _looks_like_no_image_access(answer) or _looks_like_no_image_access(output):
        raise RuntimeError(
            "Hermes konnte das Bild nicht visuell auswerten. Prüfe, ob ein visionfähiges Modell aktiv ist."
        )

    if image_mode == "path_only":
        return (
            "[DEBUG path_only: keine echte Bildanalyse] "
            + answer
        )

    return answer


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
            image_mode=str(cfg["image_mode"]),
            model=cfg["model"],
            cli_provider=cfg["cli_provider"],
        )

    raise RuntimeError(f"Unbekannter Hermes-Provider: {provider}")
