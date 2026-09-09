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
    "test": "deepdocforgery.evaluate",
    "status": "deepdocforgery.status",
}


def main() -> None:
    # Dispatch a known command before constructing the top-level parser. Otherwise
    # argparse consumes a subcommand's ``--help`` itself and users see the root help
    # instead of (for example) the prepare/train options.
    raw_arguments = sys.argv[1:]
    if raw_arguments and raw_arguments[0] in COMMANDS:
        command = raw_arguments[0]
        module = importlib.import_module(COMMANDS[command])
        sys.argv = [f"deepdocforgery {command}", *raw_arguments[1:]]
        module.main()
        return

    parser = argparse.ArgumentParser(
        prog="deepdocforgery",
        description="Combined DocTamper + MIDV-DM forgery research pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    args = parser.parse_args(raw_arguments)
    if args.command is None:
        parser.print_help()
        return
    # This path is retained for completeness if a future global option allows a
    # command after it; current normal command dispatch happens above.
    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [f"deepdocforgery {args.command}"]
    module.main()


if __name__ == "__main__":
    main()
