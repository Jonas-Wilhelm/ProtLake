#!/bin/bash
#SBATCH --job-name=ProtLake_pack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH -a 1-50 # adjust to have multiple concurrent writers.
#SBATCH --constraint='Turin'
# #SBATCH --constraint='L40|A4000|A6000|A100|A5000|H200'
#SBATCH -vvv
#SBATCH -o tests_out/logs/%A_%a.joblog
#SBATCH -e tests_out/logs/%A_%a.joblog

# Echo job info on joblog:
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID started on:   " `hostname -s`
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID started on:   " `date `
echo " "

af3_out_path="/net/scratch/jonaswil/MMO/design/03_8hlx_scaffolding/10_recycle_02_AF/"
out_path="/net/scratch/jonaswil/MMO/design/03_8hlx_scaffolding/10_recycle_02_AF_protlake/"

cd /home/jonaswil/Software/ProtLake
./tests/af3/af3_to_protlake_slurm.py --af3_out_path "$af3_out_path" --out_path "$out_path"

# Echo job info on joblog:
echo " "
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID ended on:   " `hostname -s`
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID ended on:   " `date `
