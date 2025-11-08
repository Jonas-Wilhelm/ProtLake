#!/usr/bin/env python

import os
from protlake import ProtLake
import subprocess
from pathlib import Path

AF3_APPTAINER =  "/home/jonaswil/containers/af3_protlake/af3_protlake.sif"
AF3_MODEL_DIR =  "/home/jonaswil/weights/af3/model"

# Get the top-level directory of the git repo
repo_root = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()
os.chdir(repo_root)

# Define paths
script_dir = Path(__file__).resolve().parent.relative_to(Path.cwd())
out_dir = Path(str(script_dir).replace("tests/", "tests/output/"))
out_dir.mkdir(parents=True, exist_ok=True)
pl_dir = out_dir / "pl"

pl = ProtLake(
    path=pl_dir,
    create=True
)

# Build command
cmd = [
    AF3_APPTAINER,
    f"--run_data_pipeline=False",
    f"--output_protlake={pl_dir}",
    f"--model_dir={AF3_MODEL_DIR}",
    f"--input_dir={script_dir}/input",
    f"--buckets=192,256,320,384,448,512,768,1024,1280,1536,2048,2560,3072,3584,4096,4608,5120"
]

# Run and redirect output
log_file = out_dir / "af3.log"
print(f"Running AF3...")
with open(log_file, "w") as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

print(f"AF3 run complete.")

# Perform ProtLake maintenance
print(f"Running ProtLake maintenance...")
pl.maintenance()
print(f"ProtLake maintenance complete.")
