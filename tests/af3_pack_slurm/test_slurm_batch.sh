#!/bin/bash
#SBATCH --job-name=ProtLake
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH -a 1-2
#SBATCH -vvv
#SBATCH -o tests_out/af3_pack_slurm/logs/%A_%a.joblog
#SBATCH -e tests_out/af3_pack_slurm/logs/%A_%a.joblog

# Echo job info on joblog:
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID started on:   " `hostname -s`
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID started on:   " `date `
echo " "
echo "current working directory: " `pwd`
echo " "

cd /home/jonaswil/Software/ProtLake
./tests/af3_pack_slurm/test_slurm.py

# Echo job info on joblog:
echo " "
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID ended on:   " `hostname -s`
echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID ended on:   " `date `
