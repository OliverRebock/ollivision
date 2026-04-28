import argparse
import sys

from ollivision.camera import capture_image
from ollivision.hermes_client import describe_image


def describe_scene() -> str:
    image_path = capture_image("/tmp/ollivision_latest.jpg")
    prompt = "Beschreibe die Szene kurz auf Deutsch."
    return describe_image(image_path, prompt)


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
