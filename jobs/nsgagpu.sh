#!/bin/bash --login

#SBATCH --job-name=nsgagpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=0-01:30:00
#SBATCH --mem=32G

#SBATCH --partition=gpu
#SBATCH --qos=high

#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job started at $(date)"
echo "Node: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOBID}"

cd "$HOME" || exit 1
mkdir -p logs

eval "$(micromamba shell hook --shell bash)"
micromamba activate ocstrain312

# 1-hour standalone NSGA run — baseline for the VS comparison
python -u nsga_gpu.py 51 \
    --time-limit 3600 \
    --population-size 300 \
    --db-path nsga_1h.db \
    --run-id nsga_1h

echo "Job finished at $(date)"
