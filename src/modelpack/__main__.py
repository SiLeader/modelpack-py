"""Command-line entry point for ModelPack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import ModelPackClient
from .models import ModelLayer


def _parse_layer(value: str) -> ModelLayer:
    """Parse ``PATH[:MEDIA_TYPE]``, where PATH may itself contain colons."""
    path, separator, media_type = value.rpartition(":")
    if separator and path and media_type.startswith("application/"):
        return ModelLayer(path=path, media_type=media_type)
    return ModelLayer(path=value)


def main() -> None:
    parser = argparse.ArgumentParser(prog="modelpack")
    parser.add_argument("--insecure", action="store_true", help="use HTTP for registry")
    subcommands = parser.add_subparsers(dest="command", required=True)

    pull_parser = subcommands.add_parser("pull", help="pull a ModelPack artifact")
    pull_parser.add_argument("reference")
    pull_parser.add_argument("destination", type=Path)
    pull_parser.add_argument("--no-unpack", action="store_true")
    pull_parser.add_argument("--keep", action="store_true")

    push_parser = subcommands.add_parser("push", help="push a ModelPack artifact")
    push_parser.add_argument("reference")
    push_parser.add_argument("--config", required=True, type=Path)
    push_parser.add_argument(
        "--layer",
        action="append",
        required=True,
        metavar="PATH[:MEDIA_TYPE]",
        help="layer path and optional ModelPack media type",
    )

    args = parser.parse_args()
    client = ModelPackClient(insecure=args.insecure)
    if args.command == "pull":
        result = client.pull(
            args.reference,
            args.destination,
            overwrite=not args.keep,
            unpack=not args.no_unpack,
        )
        print("\n".join(str(path) for path in result.files))
        return

    layers = [_parse_layer(value) for value in args.layer]
    result = client.push(args.reference, layers, args.config)
    print(json.dumps({"reference": result.reference, "digest": result.digest}))


if __name__ == "__main__":
    main()
