import subprocess


def capture_image(output_path: str) -> str:
    cmd = ["rpicam-still", "-o", output_path, "--timeout", "1000", "--nopreview"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Kameraaufnahme fehlgeschlagen: {detail}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("Kameraaufnahme fehlgeschlagen: rpicam-still nicht gefunden") from exc

    return output_path
