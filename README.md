# Beyond NSGA-II: Enhancing Multi-Objective Scheduling with COLM-Generated Solutions -- Optimisation for Computer Science course


Hybrid pipeline combining GPU-accelerated NSGA-II with COLM, a compact GPT-style transformer, for multi-objective permutation flow-shop scheduling (makespan + total tardiness) on Taillard benchmark instances.

## Pipeline

1. **NSGA-II** explores the solution space and builds a Pareto front
2. **Corpus extraction** selects the top-K non-dominated solutions as training data
3. **COLM training** learns the distribution of high-quality permutations
4. **COLM inference** generates new candidate solutions via autoregressive sampling

## Repository structure

| File/Folder | Description |
|---|---|
| `pipeline.py` | Main orchestrator for the full hybrid pipeline |
| `nsga_gpu.py` | GPU-accelerated NSGA-II via EvoX |
| `colm/train_perm.py` | COLM training on permutation corpus |
| `colm/inference_perm.py` | Constrained autoregressive inference |
| `db.py` | SQLite solution storage and Pareto ranking |
| `helper_fn/` | Taillard instance generator and due date computation |
| `analyze.py` | Pareto front comparison plots |
| `hypervolume.py` | Hypervolume indicator computation and plot |
| `heatmap.py` | Parameter sweep heatmap (temperature × top-k) |
| `compare.py` | Run statistics table |
| `jobs/` | SLURM batch scripts for cluster execution |

## Authors

David Pereira de Magalhaes, Bacem Etteib, Sadin Avdusinovic — University of Luxembourg
