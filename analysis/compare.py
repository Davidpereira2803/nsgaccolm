import argparse
import sqlite3
import numpy as np
from db import _find_pareto_front_2d

def compare(db_path: str, instance_id: int, runs: list[tuple[str, str]]) -> None:
    
    conn = sqlite3.connect(db_path)

    print("\nResults Comparison:")

    run_rows: list[tuple[str, str, list[tuple[int, float]]]] = []

    for run_id, label in runs:
        # Basic summary stats for each run
        row = conn.execute(
            """
            SELECT COUNT(*),
                   MIN(makespan),
                   MIN(tardiness),
                   SUM(CASE WHEN pareto_front_rank = 0 THEN 1 ELSE 0 END)
            FROM solutions
            WHERE run_id=? AND instance_id=?
            """,
            (run_id, instance_id),
        ).fetchone()
        total, min_ms, min_tt, rank0 = row
        print(f"\n  {label}  (run_id={run_id})")
        print(f"    Total solutions : {total}")
        print(f"    Rank-0 front    : {rank0}")
        print(f"    Best makespan   : {min_ms}")
        if min_tt is not None:
            print(f"    Best tardiness  : {min_tt:.2f}")
        else:
            print( "    Best tardiness  : n/a")

        rows = conn.execute(
            "SELECT makespan, tardiness FROM solutions WHERE run_id=? AND instance_id=?",
            (run_id, instance_id),
        ).fetchall()
        run_rows.append((run_id, label, rows))

    conn.close()

    if any(not r[2] for r in run_rows):
        print("\n  (one or more runs have no data — skipping combined front)")
        return

    all_ms: list[float] = []
    all_tt: list[float] = []
    boundaries: list[int] = [0]

    for _, _, rows in run_rows:
        all_ms.extend(r[0] for r in rows)
        all_tt.extend(r[1] for r in rows)
        boundaries.append(len(all_ms))

    ms_arr = np.array(all_ms, dtype=np.float64)
    tt_arr = np.array(all_tt, dtype=np.float64)

    # Build the combined Pareto front accross all runs
    combined_front = _find_pareto_front_2d(ms_arr, tt_arr)
    total_front = int(combined_front.sum())

    print(f"\n  Combined Pareto front  ({total_front} solutions total)")
    for i, (_, label, _) in enumerate(run_rows):
        lo, hi = boundaries[i], boundaries[i + 1]
        count = int(combined_front[lo:hi].sum())
        print(f"    From {label:<14}: {count}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Pareto fronts from the solutions DB")
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--db-path", default="solutions.db")
    parser.add_argument("--nsga-run-id", default="nsga_pipeline", help="NSGA run_id (e.g. nsga_30min)")
    parser.add_argument("--colm-run-id", default="colm_pipeline", help="COLM run_id")
    parser.add_argument("--baseline-run-id", default="",
                        help="Optional extra run_id to use as baseline (e.g. nsga_1h)")
    args = parser.parse_args()

    runs = [
        (args.nsga_run_id, "NSGA-30min"),
        (args.colm_run_id, "COLM"),
    ]
    if args.baseline_run_id:
        runs.append((args.baseline_run_id, "NSGA-1h"))

    compare(args.db_path, args.inst_id, runs)

if __name__ == "__main__":
    main()
