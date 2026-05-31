#!/bin/bash --login

#SBATCH --job-name=nsgagpucolm
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --mem=32G

#SBATCH --partition=gpu
#SBATCH --qos=normal

#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job started at $(date)"
echo "Node: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOBID}"

cd "$HOME" || exit 1
mkdir -p logs

eval "$(micromamba shell hook --shell bash)"
micromamba activate ocstrain312

# Full NSGA + COLM pipeline (60 min total: 30 min NSGA + 10 min train + 20 min infer)
python -u pipeline.py 51 \
    --nsga-time   1800 \
    --train-time   600 \
    --infer-time  1200 \
    --db-path solutions.db \
    --nsga-run-id nsga_pipeline \
    --colm-run-id colm_pipeline

echo "Job finished at $(date)"
