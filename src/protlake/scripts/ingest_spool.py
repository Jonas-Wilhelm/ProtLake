#!/usr/bin/env python3
import os
import sys

_DEFAULT_RUST_LOG = "warn,deltalake_core::writer::stats=error" # to silence warnings about binrary fields

# deltalake's Rust-side logger only picks this up reliably when the process
# starts with RUST_LOG already in the environment, so re-exec once if needed.
if "RUST_LOG" not in os.environ:
    os.environ["RUST_LOG"] = _DEFAULT_RUST_LOG
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

import argparse
import logging

from protlake.write.schema_config import load_schema_config
from protlake.write.staging import SpoolIngester
from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig


def main():
    parser = argparse.ArgumentParser(
        description="Ingest staged spool batches into a Protlake",
        epilog=(
            "Schema config files must define a top-level 'fields' list. "
            "Each field needs at least 'name' and 'type', with optional 'nullable'. "
            "Example JSON/YAML: "
            "{fields: [{name: name, type: string}, {name: sample_idx, type: int32, nullable: false}]}"
        ),
    )
    parser.add_argument("--protlake-path", type=str, required=True,
                        help="Path to the Protlake directory")
    parser.add_argument("--schema-config", type=str, required=True,
                        help="Path to the JSON or YAML schema config for light metadata; expected format is a top-level 'fields' list with entries like {name: ..., type: ..., nullable: ...}")
    parser.add_argument("--interval", type=int, default=600,
                        help="Polling interval in seconds for continuous ingest mode. (default: 600)")
    parser.add_argument("--run-once", action="store_true",
                        help="Process one sweep and exit instead of polling forever")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for Protlake writer. (default: 100)")
    parser.add_argument("--shard-size", type=int, default=1 << 30,
                        help="Target maximum shard size in bytes. (default: 1 << 30)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Python logging level. (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    user_schema = load_schema_config(args.schema_config)
    writer = ProtlakeWriter(
        ProtlakeWriterConfig(
            out_path=args.protlake_path,
            user_schema=user_schema,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
        )
    )
    ingester = SpoolIngester(
        protlake_dir=args.protlake_path,
        writer=writer
    )

    if args.run_once:
        result = ingester.run_once(run_maintenance=True)
        logging.info("Processed %d batches and %d entries", result["processed_batches"], result["processed_entries"])
        writer.finalize()
        return

    try:
        ingester.run_loop(interval_seconds=args.interval, run_maintenance=True)
    finally:
        writer.finalize()


if __name__ == "__main__":
    main()
