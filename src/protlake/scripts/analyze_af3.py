#!/usr/bin/env python3

import os
import shlex
import subprocess
from deltalake import DeltaTable
from ..utils import get_protlake_dirs
import argparse

def main():
    parser = argparse.ArgumentParser()
    # Args only for the launcher
    parser.add_argument("--num-tasks", type=int, default=1,
                        help="Number of workers in the array")
    parser.add_argument("--protlake-path", type=str, required=True, 
                        help="Path to the Protlake directory to analyze")
    parser.add_argument("--staging-path", type=str, required=False,
                        help="Path to the staging deltatable, default: <protlake-path>/delta_staging_table")
    parser.add_argument("--local", action="store_true", 
                        help="Run in local mode (no SLURM, single task)")
    parser.add_argument("--dry-run", action="store_true",
                        help="If set, the commands will be printed but not executed")
    parser.add_argument("--log-dir", type=str, default="log_slurm",
                        help="Directory to write SLURM log files to")
    parser.add_argument("--dont-delete-staging-table", action="store_true", 
                        help="if set, the staging-path directory will not be deleted after merging")
    parser.add_argument("--dont-merge", action="store_true",
                        help="if set, the merge operation will be skipped")
    parser.add_argument("--merge-only", action="store_true",
                        help="if set, only the merge operation will be performed. \
                        Expects a complete staging table at --staging-path")

    # `parse_known_args` lets us grab the rest (worker args) without erroring
    launcher_args, worker_args = parser.parse_known_args()

    if launcher_args.staging_path:
        staging_path = launcher_args.staging_path
    else:
        staging_path = os.path.join(launcher_args.protlake_path, "delta_staging_table")

    worker_script = "protlake.af3.analysis_worker"
    merge_script = "protlake.af3.analysis_merge"

    # Get current version of the delta table
    _, delta_path = get_protlake_dirs(launcher_args.protlake_path)
    dt = DeltaTable(f"file://{os.path.abspath(delta_path)}")
    snapshot_ver = dt.version()

    # Submit array job
    # Flatten the tuple for --wrap into a single string
    if not launcher_args.merge_only:
        wrap_cmd = (
            f"python -m {worker_script} {' '.join(shlex.quote(a) for a in worker_args)} "
            f"--protlake-path {launcher_args.protlake_path} "
            f"--staging-path {staging_path} "
            f"--snapshot-ver {snapshot_ver} "
        )

        sbatch_cmd = [
            f"sbatch",
            f"--cpus-per-task=1",
            f"--array=0-{launcher_args.num_tasks-1}",
            f"--time=04:00:00",
            f"--mem=8G",
            f"--partition=cpu",
            f"--job-name=protlake_af3_analysis_worker",
            f"--output={launcher_args.log_dir}/worker_%A_%a.out",
            f"--error={launcher_args.log_dir}/worker_%A_%a.out",
            f"--parsable",
            f"--wrap", wrap_cmd
        ]

        if launcher_args.local:
            print("Running in local mode, executing worker script directly")
            print("Executing command:\n", wrap_cmd)
            if launcher_args.dry_run:
                print("Dry run mode, not executing command")
            else:
                subprocess.run(shlex.split(wrap_cmd), check=True)
        else:
            print("Submitting array job with command:\n", " ".join(sbatch_cmd))
            if launcher_args.dry_run:
                print("Dry run mode, not submitting job")
            else:
                job_id = subprocess.check_output(sbatch_cmd, text=True).strip()
                print(f"Submitted array job {job_id}")

    # Submit follow-up job to merge results
    if launcher_args.dont_merge:
        print("Skipping merge step as per --dont-merge flag")
        return
    
    merge_wrap_cmd = (
        f"python -m {merge_script} "
        f"--protlake-path {launcher_args.protlake_path} "
        f"--staging-path {staging_path} "
    )

    if launcher_args.dont_delete_staging_table:
        merge_wrap_cmd += "--dont-delete-staging-table "

    merge_cmd = [
        f"sbatch",
        f"--cpus-per-task=1",
        f"--time=04:00:00",
        f"--mem=8G",
        f"--partition=cpu",
        f"--job-name=protlake_af3_analysis_merge",
        f"--output={launcher_args.log_dir}/merge_%A.out",
        f"--error={launcher_args.log_dir}/merge_%A.out",
        f"--parsable",
        "--wrap", merge_wrap_cmd
    ]

    if not launcher_args.merge_only:
        merge_cmd.append(f"--dependency=afterok:{job_id}")

    print("Submitting follow-up merge job with command:\n", " ".join(merge_cmd))
    if launcher_args.dry_run:
        print("Dry run mode, not submitting merge job")
    else:
        merge_job_id = subprocess.check_output(merge_cmd, text=True).strip()
        print(f"Submitted merge job {merge_job_id} (depends on {job_id})")


if __name__ == "__main__":
    main()

