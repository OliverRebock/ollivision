import argparse
import sys

from ollivision.camera import capture_image


def get_dummy_hermes_response(capture: str) -> str:
    return f"Dummy Hermes Antwort für {capture}"


def describe_scene() -> str:
    image_path = capture_image("/tmp/ollivision_latest.jpg")
    return get_dummy_hermes_response(image_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ollivision")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe-scene")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "describe-scene":
        try:
            print(describe_scene())
        except RuntimeError as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
