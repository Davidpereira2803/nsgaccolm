import argparse
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from db import _find_pareto_front_2d

def _query(db_path, sql, params=()):
    # Run one SQL query and return all rows as a list of tuples
    conn = sqlite3.connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows

def get_all_run_ids(db_path, instance_id):
    # List all runs that have data for this instance
    rows = _query(
        db_path,
        "SELECT DISTINCT run_id FROM solutions WHERE instance_id=? ORDER BY run_id",
        (instance_id,),
    )
    return [r[0] for r in rows]

def get_stats(db_path, run_id, instance_id):
    # Compute basic summary stats for one run
    row = _query(db_path, """
        SELECT COUNT(*),
               MIN(makespan),
               MIN(tardiness),
               SUM(CASE WHEN pareto_front_rank = 0 THEN 1 ELSE 0 END)
        FROM solutions
        WHERE run_id=? AND instance_id=?
    """, (run_id, instance_id))[0]
    return {
        "total":   row[0],
        "best_ms": row[1],
        "best_tt": row[2],
        "rank0":   row[3] or 0,
    }

def get_rank0_objectives(db_path, run_id, instance_id):
    # Return each run's rank-0 solutions as a 2D array of (makespan, tardiness)
    rows = _query(db_path,
        "SELECT makespan, tardiness FROM solutions "
        "WHERE run_id=? AND instance_id=? AND pareto_front_rank=0",
        (run_id, instance_id),
    )
    if not rows:
        return np.empty((0, 2))
    return np.array(rows, dtype=np.float64)

def compute_combined_front(run_ids, rank0_objs):
    # Concatenate all rank-0 solutions and find the combined Pareto front across all runs
    all_ms, all_tt = [], []
    boundaries = [0]
    for r in run_ids:
        obj = rank0_objs[r]
        if len(obj):
            all_ms.extend(obj[:, 0].tolist())
            all_tt.extend(obj[:, 1].tolist())
        boundaries.append(len(all_ms))

    if not all_ms:
        return np.array([], dtype=bool), [0] * len(run_ids), np.array([]), np.array([])

    ms_arr = np.array(all_ms, dtype=np.float64)
    tt_arr = np.array(all_tt, dtype=np.float64)
    mask = _find_pareto_front_2d(ms_arr, tt_arr)

    counts = []
    for i in range(len(run_ids)):
        lo, hi = boundaries[i], boundaries[i + 1]
        counts.append(int(mask[lo:hi].sum()))

    return mask, counts, ms_arr, tt_arr

def print_table(run_ids, stats, combined_counts):
    # Print a summary table comparing each run's stats and how many of its rank-0 solutions are on the combined front
    rows = list(zip(run_ids, [stats[r] for r in run_ids], combined_counts))
    rows.sort(key=lambda x: x[2], reverse=True)

    header = f"{'run_id':<28} {'total':>10} {'own rank-0':>10} {'combined':>10} {'best_ms':>9} {'best_tt':>14}"
    print("\n" + header)
    print("-" * len(header))
    for run_id, s, combined in rows:
        tt_str = f"{s['best_tt']:.2f}" if s["best_tt"] is not None else "n/a"
        print(
            f"{run_id:<28} {s['total']:>10} {s['rank0']:>10} {combined:>10} "
            f"{s['best_ms']:>9} {tt_str:>14}"
        )

def plot_results(run_ids, rank0_objs, combined_mask, ms_arr, tt_arr,
                 nsga_run_ids, output_path):
    # Plot the rank-0 solutions for each run, highlighting NSGA baselines and the combined Pareto front
    colm_runs = [r for r in run_ids if r not in nsga_run_ids]
    nsga_runs  = [r for r in run_ids if r in nsga_run_ids]

    colm_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(colm_runs), 1)))
    colm_color_map = dict(zip(colm_runs, colm_colors))
    nsga_palette  = ["#ff0004", "#ff7f00", "#dd00ff", "#09ff00"]
    nsga_color_map = {r: nsga_palette[i % len(nsga_palette)] for i, r in enumerate(nsga_runs)}

    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot each run's rank-0 solutions with different colors/markers for NSGA vs. others
    for run_id in run_ids:
        obj = rank0_objs[run_id]
        if len(obj) == 0:
            continue
        is_nsga = run_id in nsga_run_ids
        color   = nsga_color_map.get(run_id) or colm_color_map.get(run_id, "gray")
        marker  = "^" if is_nsga else "o"
        size    = 40  if is_nsga else 15
        alpha   = 0.9 if is_nsga else 0.6
        zorder  = 4   if is_nsga else 2
        ax.scatter(
            obj[:, 0], obj[:, 1],
            s=size, alpha=alpha, color=color,
            marker=marker, label=run_id, zorder=zorder,
        )

    if len(ms_arr):
        # Draw the combined Pareto front as a step curve
        front_ms = ms_arr[combined_mask]
        front_tt = tt_arr[combined_mask]
        sort_idx = np.argsort(front_ms)
        ax.step(
            front_ms[sort_idx], front_tt[sort_idx],
            where="post", color="black", linewidth=1.5,
            linestyle="--", alpha=0.6, zorder=5, label="Combined front",
        )
        ax.scatter(
            front_ms, front_tt,
            s=60, color="black", marker="x",
            linewidths=1.5, zorder=6,
        )

    ax.set_xlabel("Makespan", fontsize=12)
    ax.set_ylabel("Tardiness", fontsize=12)
    ax.set_title("Pareto Front Comparison (rank-0 solutions per run)", fontsize=13)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

def export_front(run_ids, rank0_objs, combined_mask, csv_path):
    # Export the combined Pareto front to CSV for reuse
    import csv
    boundaries = [0]
    for r in run_ids:
        boundaries.append(boundaries[-1] + len(rank0_objs[r]))

    rows = []
    for i, run_id in enumerate(run_ids):
        obj = rank0_objs[run_id]
        lo, hi = boundaries[i], boundaries[i + 1]
        front_indices = combined_mask[lo:hi]
        for j, on_front in enumerate(front_indices):
            if on_front:
                rows.append((int(obj[j, 0]), float(obj[j, 1]), run_id))

    rows.sort(key=lambda x: x[0])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["makespan", "tardiness", "run_id"])
        writer.writerows(rows)

    print(f"Combined front exported to {csv_path} ({len(rows)} solutions)")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize experiment results")
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--db-path", default="solutions.db")
    parser.add_argument("--runs", nargs="+", default=[],
                        help="Run IDs to include (default: all runs in DB)")
    parser.add_argument("--nsga-runs", nargs="+", default=[],
                        help="Run IDs to treat as NSGA baselines (plotted as triangles)")
    parser.add_argument("--output", default="pareto_comparison.png",
                        help="Output PNG path (default: pareto_comparison.png)")
    parser.add_argument("--export-front", default="",
                        help="Export combined Pareto front solutions with run_id to this CSV path")
    args = parser.parse_args()

    run_ids = args.runs if args.runs else get_all_run_ids(args.db_path, args.inst_id)
    nsga_run_ids = set(args.nsga_runs)

    if not run_ids:
        print("No runs found in DB.")
        return

    print(f"Analyzing {len(run_ids)} runs for instance {args.inst_id}...")

    stats       = {r: get_stats(args.db_path, r, args.inst_id) for r in run_ids}
    rank0_objs  = {r: get_rank0_objectives(args.db_path, r, args.inst_id) for r in run_ids}

    combined_mask, combined_counts, ms_arr, tt_arr = compute_combined_front(run_ids, rank0_objs)

    print_table(run_ids, stats, combined_counts)

    plot_results(
        run_ids, rank0_objs, combined_mask, ms_arr, tt_arr,
        nsga_run_ids, args.output,
    )

    if args.export_front:
        export_front(run_ids, rank0_objs, combined_mask, args.export_front)

if __name__ == "__main__":
    main()
