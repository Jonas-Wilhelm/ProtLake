#!/usr/bin/env -S PYTHONUNBUFFERED=1 python

import os, time, sys
from datetime import timedelta
sys.path.append("/home/jonaswil/Software/ProtLake")
from write import IngestConfig, AF3IngestPipeline
import argparse

parser = argparse.ArgumentParser(description="Process AF3 output directories and ingest into ProtLake.")
parser.add_argument("--af3_out_path", required=True, help="Path to AF3 output directory")
parser.add_argument("--out_path", required=True, help="Path to ProtLake output directory")
args = parser.parse_args()

af3_out_path = args.af3_out_path
out_path = args.out_path

af3_dirs = [entry.path for entry in os.scandir(af3_out_path) if entry.is_dir()]

# --- SLURM array job setup ---
num_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
num_files = len(af3_dirs)

q, r = divmod(num_files, num_tasks)
task_id_0 = task_id - 1

start = task_id_0 * q + min(task_id_0, r)
stop  = start + q + (task_id_0 < r)
af3_dirs = af3_dirs[start:stop]

print(f"Task {task_id} of {num_tasks} processing files {start} to {stop} (total {len(af3_dirs)} files)")

# --- perform the processing ---
start_time = time.time()
cfg = IngestConfig(
    out_path=out_path, 
    batch_size_metadata=2000, 
    shard_size=1 << 30, 
    verbose=False,
    claim_mode=True)
AF3IngestPipeline(cfg).run(af3_dirs)
end_time = time.time()
elapsed = round(time.time() - start_time)
print(f"Processing {len(af3_dirs)} files took: {timedelta(seconds=elapsed)}")
