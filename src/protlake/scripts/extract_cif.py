#!/usr/bin/env python3

import os
from deltalake import DeltaTable
from protlake.utils import dump_cif_from_deltalake_row
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract a CIF file from a Protlake Delta table entry")
    parser.add_argument("--protlake_delta", type=str, required=False, default=None,
                        help="Path to the Protlake Delta directory. If not given, will use the PROTLAKE_DPATH environment variable.")
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of the entry to dump")
    parser.add_argument("-s", "--sample", type=int, required=False, default=0,
                        help="Sample number of the entry to dump. Default is 0")
    parser.add_argument("-z", "--seed", type=int, required=False, default=42,
                        help="Seed number of the entry to dump. Default is 42")
    parser.add_argument("-o", "--out-path", type=str, required=False, default="dump.cif",
                        help="Path to write the output CIF file to. Default is dump.cif")
    
    args = parser.parse_args()

    # if no Protlake path is given, assume environment variable "PROTLAKE_DPATH"
    if args.protlake_delta is None:
        args.protlake_delta = os.getenv("PROTLAKE_DPATH", None)
        if args.protlake_delta is None:
            raise ValueError("No Protlake path given and environment variable PROTLAKE_DPATH not set")
        
    print(f"Using Protlake Delta path: {args.protlake_delta}")
    dt = DeltaTable(f"file://{os.path.abspath(args.protlake_delta)}")
    row_dict = {"name": args.name, "sample": args.sample, "seed": args.seed}
    dump_cif_from_deltalake_row(dt, row_dict, out_path=args.out_path)

if __name__ == "__main__":
    main()
