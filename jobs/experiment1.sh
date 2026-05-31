#!/bin/bash --login

#SBATCH --job-name=colmexperiment1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=0-03:00:00
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

# COLM pipeline (10 min train + 20 min infer)

python pipeline.py 51 \
  --skip-nsga \
  --model-path colm_model_perm_pipeline1.pth \
  --nsga-run-id nsga_pipeline \
  --db-path solutions.db \
  --top-k 200 \
  --infer-temperature 0.1 \
  --colm-run-id colm_t01_k200

echo "Job 1 finished at $(date)"

python pipeline.py 51 \
  --skip-nsga --skip-train \
  --model-path colm_model_perm_pipeline1.pth \
  --nsga-run-id nsga_pipeline \
  --db-path solutions.db \
  --top-k 200 \
  --infer-temperature 0.2 \
  --colm-run-id colm_t02_k200

echo "Job 2 finished at $(date)"

python pipeline.py 51 \
  --skip-nsga --skip-train \
  --model-path colm_model_perm_pipeline1.pth \
  --nsga-run-id nsga_pipeline \
  --db-path solutions.db \
  --top-k 200 \
  --infer-temperature 0.5 \
  --colm-run-id colm_t05_k200

echo "Job 3 finished at $(date)"

python pipeline.py 51 \
  --skip-nsga --skip-train \
  --model-path colm_model_perm_pipeline1.pth \
  --nsga-run-id nsga_pipeline \
  --db-path solutions.db \
  --top-k 200 \
  --infer-temperature 1.0 \
  --colm-run-id colm_t10_k200

echo "Job 4 finished at $(date)"

python pipeline.py 51 \
  --skip-nsga --skip-train \
  --model-path colm_model_perm_pipeline1.pth \
  --nsga-run-id nsga_pipeline \
  --db-path solutions.db \
  --top-k 200 \
  --infer-temperature 1.2 \
  --colm-run-id colm_t12_k200

echo "Job 5 finished at $(date)"
