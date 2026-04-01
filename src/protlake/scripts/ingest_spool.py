#!/usr/bin/env python3

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
    parser.add_argument("--min-entries", type=int, default=1,
                        help="Minimum number of ready staged entries before ingesting. (default: 1)")
    parser.add_argument("--run-once", action="store_true",
                        help="Process one sweep and exit instead of polling forever")
    parser.add_argument("--batch-size-metadata", type=int, default=1000,
                        help="Metadata row flush batch size for the Protlake writer. (default: 1000)")
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
            batch_size_metadata=args.batch_size_metadata,
            shard_size=args.shard_size,
        )
    )
    ingester = SpoolIngester(
        protlake_dir=args.protlake_path,
        writer=writer,
        min_entries=args.min_entries,
    )

    if args.run_once:
        result = ingester.run_once()
        logging.info("Processed %d batches and %d entries", result["processed_batches"], result["processed_entries"])
        writer.finalize()
        return

    try:
        ingester.run_loop(interval_seconds=args.interval)
    finally:
        writer.finalize()


if __name__ == "__main__":
    main()
