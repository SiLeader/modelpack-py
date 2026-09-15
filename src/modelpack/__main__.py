"""Command-line entry point for ModelPack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import ModelPackClient
from .models import ModelLayer


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

    layers = []
    for value in args.layer:
        path, separator, media_type = value.rpartition(":")
        if not separator or not Path(path).exists():
            path, media_type = value, ""
        layers.append(
            ModelLayer(path=path, media_type=media_type)
            if media_type
            else ModelLayer(path=path)
        )
    result = client.push(args.reference, layers, args.config)
    print(json.dumps({"reference": result.reference, "digest": result.digest}))
