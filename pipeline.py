import argparse
import subprocess
import sys
import time
from pathlib import Path
from analysis.compare import compare

PROJECT_ROOT = Path(__file__).resolve().parent

def _python() -> str:
    # Use the same Python interpreter for subprocesses to ensure consistent environment and dependencies
    return sys.executable

def _run(script_rel: str, extra_args: list[str]) -> None:
    # Spawn a subprocess to run a script relative to the project root with additional command-line arguments
    cmd = [_python(), str(PROJECT_ROOT / script_rel)] + extra_args
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

def _get_n_jobs(inst_id: int) -> int:
    # Helper to query Taillard instance job count from helper module
    from helper_fn.flowshop import taillard_get_nb_jobs
    return taillard_get_nb_jobs(inst_id)

def run_pipeline(args: argparse.Namespace) -> None:
    db = args.db_path
    inst = args.inst_id

    print(f"PIPELINE  instance={inst}  db={db}")
    # Print high-level phase plan and durations or skip flags
    if args.skip_nsga:
        print(f"  Phase 1  NSGA    SKIPPED (using run_id={args.nsga_run_id})")
    else:
        print(f"  Phase 1  NSGA    {args.nsga_time:.0f}s")
    print(f"  Phase 2  top {args.top_k} -> corpus")
    if args.skip_train:
        print(f"  Phase 3  train   SKIPPED (using model={args.model_path})")
    else:
        print(f"  Phase 3  train   {args.train_time:.0f}s")
    print(f"  Phase 4  infer   {args.infer_time:.0f}s")

    # Phase 1: optionally run NSGA to populate DB 
    if args.skip_nsga:
        print(f"\n[1] Skipping NSGA — using existing run_id={args.nsga_run_id}")
    else:
        print(f"\n[1] GPU NSGA for {args.nsga_time:.0f}s...")
        t0 = time.time()
        _run("nsga_gpu.py", [
            str(inst),
            "--time-limit", str(args.nsga_time),
            "--population-size", str(args.nsga_pop_size),
            "--db-path", db,
            "--run-id", args.nsga_run_id,
            "--max-solutions", str(args.max_solutions),
        ])
        print(f"[1] Done in {time.time()-t0:.0f}s")

    # Phase 2: extract top-k solutions from NSGA run and write to corpus file for COLM training
    print(f"\n[2] Extracting top {args.top_k} solutions from DB...")
    from db import get_top_k
    top_perms = get_top_k(db, args.nsga_run_id, inst, args.top_k)
    if not top_perms:
        print("[2] ERROR: No solutions in DB — aborting.")
        sys.exit(1)
    print(f"[2] Got {len(top_perms)} permutations")

    corpus = Path(args.corpus_path)
    # Write one permutation per line for COLM training
    with open(corpus, "w", encoding="utf-8") as f:
        for perm in top_perms:
            f.write(" ".join(map(str, perm)) + "\n")
    print(f"[2] Corpus written to {corpus}")

    # Phase 3: optionally train COLM model on the corpus
    if args.skip_train:
        print(f"\n[3] Skipping training — using existing model={args.model_path}")
        if not Path(args.model_path).exists():
            print(f"[3] ERROR: Model not found at {args.model_path} — aborting.")
            sys.exit(1)
    else:
        print(f"\n[3] Training COLM perm model for {args.train_time:.0f}s...")
        t0 = time.time()
        _run("colm/train_perm.py", [
            "--corpus", str(corpus),
            "--n-jobs", str(_get_n_jobs(inst)),
            "--output", args.model_path,
            "--time-limit", str(args.train_time),
        ])
        print(f"[3] Done in {time.time()-t0:.0f}s")

    # Phase 4: run COLM inference using the trained model
    prefix_len = max(1, int(_get_n_jobs(inst) * args.infer_prefix_fraction))
    print(f"\n[4] Running inference for {args.infer_time:.0f}s  (temperature={args.infer_temperature}, prefix_len={prefix_len})...")
    t0 = time.time()
    _run("colm/inference_perm.py", [
        "--model-path", str(PROJECT_ROOT / args.model_path),
        "--instance-id", str(inst),
        "--R", str(args.R),
        "--T", str(args.T),
        "--time-limit", str(args.infer_time),
        "--batch-size", str(args.infer_batch_size),
        "--db-path", db,
        "--run-id", args.colm_run_id,
        "--prompt-file", str(corpus),
        "--temperature", str(args.infer_temperature),
        "--prompt-prefix-len", str(prefix_len),
    ])
    print(f"[4] Done in {time.time()-t0:.0f}s")

    # Final comparison: call analysis.compare to summarise results
    runs = [(args.nsga_run_id, "NSGA-30min"), (args.colm_run_id, "COLM")]
    if args.baseline_run_id:
        runs.append((args.baseline_run_id, "NSGA-1h"))
    compare(db, inst, runs)

def main():
    parser = argparse.ArgumentParser(description="GPU NSGA + COLM pipeline")
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--db-path", default="solutions.db")
    parser.add_argument("--nsga-time", type=float, default=1800.0, help="NSGA phase duration (s)")
    parser.add_argument("--train-time", type=float, default=600.0, help="COLM training duration (s)")
    parser.add_argument("--infer-time", type=float, default=1200.0, help="Inference duration (s)")
    parser.add_argument("--top-k", type=int, default=1000, help="Solutions passed to COLM training")
    parser.add_argument("--nsga-pop-size", type=int, default=300)
    parser.add_argument("--nsga-run-id", default="nsga_pipeline")
    parser.add_argument("--colm-run-id", default="colm_pipeline")
    parser.add_argument("--model-path", default="colm_model_perm_pipeline.pth")
    parser.add_argument("--corpus-path", default="pipeline_corpus.txt")
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--infer-temperature", type=float, default=0.5,
                        help="Sampling temperature for COLM inference (default 0.5)")
    parser.add_argument("--infer-prefix-fraction", type=float, default=0.25,
                        help="Fraction of each seed permutation used as prompt prefix (default 0.25)")
    parser.add_argument("--R", type=float, default=0.4)
    parser.add_argument("--T", type=float, default=0.2)
    parser.add_argument("--max-solutions", type=int, default=100_000, help="Max unique solutions accumulated by NSGA (default 100 000)")
    parser.add_argument("--skip-nsga", action="store_true", help="Skip Phase 1 and use an existing NSGA run_id already in the DB")
    parser.add_argument("--skip-train", action="store_true", help="Skip Phase 3 and use an existing model file")
    parser.add_argument("--baseline-run-id", default="",
                        help="Optional 1h NSGA run_id to include in the Phase 5 comparison")
    parser.add_argument("--only-compare", action="store_true", help="Skip pipeline, just compare two existing run_ids in the DB")
    args = parser.parse_args()

    # Allow running only the comparion step without executing the pipeline
    if args.only_compare:
        runs = [(args.nsga_run_id, "NSGA-30min"), (args.colm_run_id, "COLM")]
        if args.baseline_run_id:
            runs.append((args.baseline_run_id, "NSGA-1h"))
        compare(args.db_path, args.inst_id, runs)
    else:
        run_pipeline(args)

if __name__ == "__main__":
    main()
