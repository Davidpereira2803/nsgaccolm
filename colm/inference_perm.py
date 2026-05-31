import sys
import time
from pathlib import Path
import argparse
import numpy as np
import torch
from colm import COLM
from helper_fn.due_dates_computation import compute_due_dates, estimate_average_completion_time
from helper_fn.flowshop import generate_taillard_processing_times

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def evaluate_sequence( schedule: list[int], processing_times: np.ndarray, due_dates: np.ndarray,) -> tuple[int, float]:
    # Compute flow-shop completion times C and derive makespan and total tardiness for the given job sequence
    n_jobs = len(schedule)
    n_machines = processing_times.shape[0]
    C = np.zeros((n_jobs, n_machines))
    for i, job in enumerate(schedule):
        for m in range(n_machines):
            prev_job  = C[i - 1, m] if i > 0 else 0.0
            prev_mach = C[i, m - 1] if m > 0 else 0.0
            C[i, m]   = max(prev_job, prev_mach) + processing_times[m, job]
    makespan  = int(round(C[-1, -1]))
    tardiness = float(sum(max(0.0, C[i, -1] - due_dates[job]) for i, job in enumerate(schedule)))
    return makespan, tardiness

def is_valid(seq: list[int], n_jobs: int) -> bool:
    # Check permutation validity: contains each job index exactly once
    return len(seq) == n_jobs and set(seq) == set(range(n_jobs))

def main():
    parser = argparse.ArgumentParser(description="Generate permutations via constrained COLM decoding")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--instance-id", type=int, default=51)
    parser.add_argument("--R", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=0.2)
    parser.add_argument("--time-limit", type=float, default=1200.0, help="Wall-clock inference budget (seconds)")
    parser.add_argument("--num-samples", type=int, default=0, help="Fixed sample count (0 = use --time-limit)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prompt-file", type=str, default="", help="Seed permutations (one per line)")
    parser.add_argument("--prompt-prefix-len", type=int, default=0, help="Tokens from each seed used as context (0 = half)")
    parser.add_argument("--db-path", type=str, default="")
    parser.add_argument("--run-id", type=str, default="colm")
    parser.add_argument("--output", type=str, default="", help="Optional output file for generated permutations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    # Load COLM model and move to device for generation
    print(f"Loading model from {model_path} via COLM...")
    colm = COLM.from_pretrained(str(model_path))
    colm.model.to(device)

    n_jobs = colm.meta["sol_size"]
    print(f"Model loaded — n_jobs={n_jobs}")

    # Build processing times and due dates for the specified Taillard instance
    processing_times = np.array(generate_taillard_processing_times(args.instance_id))
    from helper_fn.due_dates_computation import estimate_average_completion_time
    P         = estimate_average_completion_time(processing_times)
    due_dates = np.array(compute_due_dates(processing_times, P, R=args.R, T=args.T), dtype=float)

    # Load optional prompt seeds
    prompts_pool: list[list[int]] = []
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                tokens = [int(x) for x in line.split()]
                if len(tokens) == n_jobs:
                    if min(tokens) >= 1:
                        tokens = [x - 1 for x in tokens]
                    prompts_pool.append(tokens)
        print(f"Loaded {len(prompts_pool)} seed permutations")

    # Determine prefix length used as prompt context for generation
    prefix_len = args.prompt_prefix_len if args.prompt_prefix_len > 0 else (n_jobs // 2 if prompts_pool else 0)

    all_results: dict[tuple[int, ...], tuple[int, float]] = {}
    n_generated = 0
    n_duplicates = 0
    use_time_limit = args.num_samples == 0

    print(f"Generating (batch={args.batch_size}, prefix_len={prefix_len}, is_unique=True)...")
    start_time = time.time()
    iteration  = 0

    while True:
        # Stop conditio:either time budget or fixed sample count
        if use_time_limit:
            if time.time() - start_time >= args.time_limit:
                break
        else:
            if n_generated >= args.num_samples:
                break

        # Prepare batch of prompts
        if prompts_pool and prefix_len > 0:
            idx          = np.random.randint(0, len(prompts_pool), size=args.batch_size)
            batch_prompts = [prompts_pool[i][:prefix_len] for i in idx]
        else:
            batch_prompts = [[] for _ in range(args.batch_size)]

        # Generate unique sequences from COLM
        sequences = colm.generate(
            prompts=batch_prompts,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            is_unique=True,
        )

        for seq in sequences:
            n_generated += 1
            if is_valid(seq, n_jobs):
                key = tuple(seq)
                if key in all_results:
                    # Track duplicates
                    n_duplicates += 1
                else:
                    # Evaluate and store makespan and total tardiness for the new unique sequence
                    ms, tt = evaluate_sequence(seq, processing_times, due_dates)
                    all_results[key] = (ms, tt)

        iteration += 1
        if iteration % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  iter={iteration}  generated={n_generated}  unique={len(all_results)}  dupes={n_duplicates}  {elapsed:.0f}s elapsed")

    elapsed = time.time() - start_time
    n_unique = len(all_results)
    print(f"\nDone: {n_generated} generated, {n_unique} unique, {n_duplicates} duplicates skipped in {elapsed:.1f}s")

    solutions_list = [(perm, ms, tt) for perm, (ms, tt) in all_results.items()]

    # Optionally save results to a database and/or output file
    if args.db_path and solutions_list:
        from db import init_db, save_solutions, compute_and_update_ranks
        init_db(args.db_path)
        n_saved = save_solutions(args.db_path, solutions_list, args.run_id, args.instance_id)
        print(f"Saved {n_saved} solutions to {args.db_path} (run_id={args.run_id})")
        n_fronts = compute_and_update_ranks(args.db_path, args.run_id, args.instance_id)
        print(f"Computed {n_fronts} Pareto fronts")

    # Optionally write generated permutations to an output file
    if args.output and solutions_list:
        with open(args.output, "w", encoding="utf-8") as f:
            for perm, _, _ in solutions_list:
                f.write(" ".join(map(str, perm)) + "\n")
        print(f"Wrote {len(solutions_list)} permutations to {args.output}")

    if solutions_list:
        print(f"Best makespan  : {min(ms for _, (ms, _) in all_results.items())}")
        print(f"Best tardiness : {min(tt for _, (_, tt) in all_results.items()):.2f}")

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    main()
