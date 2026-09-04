"""Single command-line entry point for the repository."""

from __future__ import annotations

import argparse
import importlib
import sys

from deepdocforgery import __version__

COMMANDS = {
    "prepare": "deepdocforgery.prepare",
    "doctor": "deepdocforgery.doctor",
    "smoke": "deepdocforgery.smoke",
    "train": "deepdocforgery.train",
    "evaluate": "deepdocforgery.evaluate",
    "infer": "deepdocforgery.infer",
    "hpo": "deepdocforgery.hpo",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepdocforgery",
        description="Combined DocTamper + MIDV-DM forgery research pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    args, remainder = parser.parse_known_args()
    if args.command is None:
        parser.print_help()
        return
    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [f"deepdocforgery {args.command}", *remainder]
    module.main()


if __name__ == "__main__":
    main()
