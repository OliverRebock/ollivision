import argparse


def simulate_camera_capture() -> str:
    return "simulierte Kameraaufnahme"


def get_dummy_hermes_response(capture: str) -> str:
    return f"Dummy Hermes Antwort für {capture}"


def describe_scene() -> str:
    capture = simulate_camera_capture()
    return get_dummy_hermes_response(capture)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ollivision")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("describe-scene")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "describe-scene":
        print(describe_scene())


if __name__ == "__main__":
    main()
