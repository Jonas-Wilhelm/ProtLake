#!/usr/bin/env python

import os, time, sys
from datetime import timedelta
sys.path.append("/home/jonaswil/Software/ProtLake")
from write import IngestConfig, AF3IngestPipeline

af3_out_path = "/net/scratch/jonaswil/click/design/O-prp-tyr_Pi-az_atm14/03_diffusion_v3/14_MPNN_AF_MPNN_AF_MPNN_AF_MPNN_AF"
out_path = "/net/scratch/jonaswil/click/design/O-prp-tyr_Pi-az_atm14/03_diffusion_v3/14_protlake"

af3_dirs = [entry.path for entry in os.scandir(af3_out_path) if entry.is_dir()]
# af3_dirs = af3_dirs[:3000]

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
cfg = IngestConfig(out_path=out_path, batch_size_metadata=2000, shard_size=1 << 30, verbose=False)
AF3IngestPipeline(cfg).run(af3_dirs)
end_time = time.time()
elapsed = round(time.time() - start_time)
print(f"Processing {len(af3_dirs)} files took: {timedelta(seconds=elapsed)}")
