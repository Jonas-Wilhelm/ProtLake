#!/usr/bin/env python3

import os
import shlex
import subprocess
from deltalake import DeltaTable
from protlake.utils import get_protlake_dirs
import argparse
import tempfile

def main():
    parser = argparse.ArgumentParser()
    # Args only for the launcher
    parser.add_argument("--num-tasks", type=int, default=1,
                        help="Number of workers in the array")
    parser.add_argument("--protlake-path", type=str, required=True, 
                        help="Path to the Protlake directory to analyze")
    parser.add_argument("--design-dir", type=str, required=False, 
                        help="Path to the design directory")
    parser.add_argument("--staging-path", type=str, required=False,
                        help="Path to the staging deltatable, default: <protlake-path>/delta_staging_table")
    parser.add_argument("--local", action="store_true", 
                        help="Run in local mode (no SLURM, single task)")
    parser.add_argument("--dry-run", action="store_true",
                        help="If set, the commands will be printed but not executed")
    parser.add_argument("--log-dir", type=str, default="log_slurm", # TODO: consider removing default and send output to /dev/null
                        help="Directory to write SLURM log files to")
    parser.add_argument("--dont-delete-staging-table", action="store_true", 
                        help="if set, the staging-path directory will not be deleted after merging")
    parser.add_argument("--dont-merge", action="store_true",
                        help="if set, the merge operation will be skipped")
    parser.add_argument("--merge-only", action="store_true",
                        help="if set, only the merge operation will be performed. \
                        Expects a complete staging table at --staging-path")
    parser.add_argument("--slurm-time", type=str, default="04:00:00",
                        help="Time limit for SLURM jobs (format HH:MM:SS)")
    parser.add_argument("--python-bin", type=str, default="python",
                        help="Python binary to use in SLURM jobs")
    parser.add_argument("--qos-interactive", action="store_true",
                        help="If set, adds --qos=interactive to SLURM jobs")
    parser.add_argument("--ncaa", type=str, nargs='+', required=False,
                        help="List of NCAA 3-letter CCD codes.")

    # `parse_known_args` lets us grab the rest (worker args) without erroring
    launcher_args, worker_args = parser.parse_known_args()

    if launcher_args.staging_path:
        staging_path = launcher_args.staging_path
    else:
        staging_path = os.path.join(launcher_args.protlake_path, "delta_staging_table")
    
    # Create log and staging directories if they don't exist
    if not os.path.exists(launcher_args.log_dir):
        os.makedirs(launcher_args.log_dir)
    
    if not os.path.exists(staging_path):
        os.makedirs(staging_path)

    worker_script = "protlake.af3.analysis_worker"
    merge_script = "protlake.af3.analysis_merge"

    # Submit array job
    # Flatten the tuple for --wrap into a single string
    if not launcher_args.merge_only:
        if launcher_args.design_dir is None:
            raise ValueError("Error: --design-dir argument is required when not using --merge-only mode")

        # Get current version of the delta table (only needed for worker jobs)
        _, delta_path = get_protlake_dirs(launcher_args.protlake_path)
        dt = DeltaTable(f"file://{os.path.abspath(delta_path)}")
        snapshot_ver = dt.version()
        del dt  # Explicitly release to avoid hanging on cleanup

        worker_cmd = (
            f"{launcher_args.python_bin} -m {worker_script} {' '.join(shlex.quote(a) for a in worker_args)} "
            f"--protlake-path {launcher_args.protlake_path} "
            f"--design-dir {launcher_args.design_dir} "
            f"--staging-path {staging_path} "
            f"--snapshot-ver {snapshot_ver} "
        )

        if launcher_args.ncaa:
            ncaa_args = " ".join(f"--ncaa {code}" for code in launcher_args.ncaa)
            worker_cmd += f" {ncaa_args}"

        sbatch_file = [
            f"#!/bin/bash",
            f"#SBATCH --job-name=protlake_af3_analysis_worker",
            f"#SBATCH --cpus-per-task=1",
            f"#SBATCH --array=0-{launcher_args.num_tasks-1}",
            f"#SBATCH --time={launcher_args.slurm_time}",
            f"#SBATCH --mem=8G",
            f"#SBATCH --partition=cpu",
            f"#SBATCH --output={launcher_args.log_dir}/worker_%A_%a.log",
            f"#SBATCH --error={launcher_args.log_dir}/worker_%A_%a.log",
            f"#SBATCH --parsable",
        ]
        if launcher_args.qos_interactive:
            sbatch_file.append(f"#SBATCH --qos=interactive")
        sbatch_file.extend([
            f"",
            f"{worker_cmd}"
        ])

        if launcher_args.local:
            print("Running in local mode, executing worker script directly")
            print("Executing command:\n", worker_cmd)
            if launcher_args.dry_run:
                print("Dry run mode, not executing command")
            else:
                subprocess.run(shlex.split(worker_cmd), check=True)
        else:
            print(
                "Submitting array job with sbatch script:\n",
                "--------------------------------\n",
                "\n".join([f"    {line}" for line in sbatch_file]),
                "\n--------------------------------\n"
                )
            if launcher_args.dry_run:
                print("Dry run mode, not submitting job")
            else:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
                    tmpfile.write("\n".join(sbatch_file))
                    tmpfile_path = tmpfile.name
                sbatch_cmd = ["sbatch", tmpfile_path]
                job_id = subprocess.check_output(sbatch_cmd, text=True).strip()
                print(f"Submitted array job {job_id}")
                os.remove(tmpfile_path)

    # Submit follow-up job to merge results
    if launcher_args.dont_merge:
        print("Skipping merge step as per --dont-merge flag")
        return
    
    merge_cmd_args = [
        f'{launcher_args.python_bin}', '-m', merge_script,
        '--protlake-path', launcher_args.protlake_path,
        '--staging-path', staging_path,
    ]

    if launcher_args.dont_delete_staging_table:
        merge_cmd_args.append('--dont-delete-staging-table')

    if launcher_args.local:
        # Run merge script directly in local mode
        print("Running merge script in local mode")
        print("Executing command:\n", " ".join(merge_cmd_args))
        if launcher_args.dry_run:
            print("Dry run mode, not executing merge command")
        else:
            subprocess.run(merge_cmd_args, check=True)
    else:
        # Submit merge job via SLURM
        merge_wrap_cmd = ' '.join(shlex.quote(a) for a in merge_cmd_args)
        merge_cmd = [
            "sbatch",
            "--cpus-per-task=1",
            f"--time={launcher_args.slurm_time}",
            "--mem=8G",
            "--partition=cpu",
            "--job-name=protlake_af3_analysis_merge",
            f"--output={launcher_args.log_dir}/merge_%A.log",
            f"--error={launcher_args.log_dir}/merge_%A.log",
            "--parsable",
        ]
        if launcher_args.qos_interactive:
            merge_cmd.append("--qos=interactive")
        merge_cmd.extend(["--wrap", merge_wrap_cmd])

        if not launcher_args.merge_only:
            merge_cmd.append(f"--dependency=afterok:{job_id}")

        print("Submitting follow-up merge job with command:\n", " ".join(merge_cmd))
        if launcher_args.dry_run:
            print("Dry run mode, not submitting merge job")
        else:
            merge_job_id = subprocess.check_output(merge_cmd, text=True).strip()
            if launcher_args.merge_only:
                print(f"Submitted merge job {merge_job_id}")
            else:
                print(f"Submitted merge job {merge_job_id} (depends on {job_id})")


if __name__ == "__main__":
    main()

