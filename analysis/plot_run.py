import argparse
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def query(db_path, sql, params=()):
    # small helper to run one read-only query and return all rows as a list of tuples
    conn = sqlite3.connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows

def pareto_staircase(ms, tt):
    # Sort points by makespan so the Pareto front can be drawn as a step curve
    idx = np.argsort(ms)
    return ms[idx], tt[idx]

def main():
    parser = argparse.ArgumentParser(description="Plot full solution space of a single run")
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--run-id", required=True, help="run_id to plot")
    parser.add_argument("--db-path", default="solutions.db")
    parser.add_argument("--output", default="", help="Output PNG path (default: <run_id>.png)")
    parser.add_argument("--max-points", type=int, default=100_000,
                        help="Max background points to plot (random sample if exceeded, default 100000)")
    args = parser.parse_args()

    output = args.output or f"{args.run_id}.png"

    all_rows = query(args.db_path,
        "SELECT makespan, tardiness, pareto_front_rank FROM solutions "
        "WHERE run_id=? AND instance_id=? AND pareto_front_rank IS NOT NULL",
        (args.run_id, args.inst_id),
    )

    if not all_rows:
        print(f"No ranked solutions found for run_id='{args.run_id}' instance={args.inst_id}.")
        print("Run compute_and_update_ranks first if ranks are missing.")
        return

    ms_all   = np.array([r[0] for r in all_rows], dtype=np.float64)
    tt_all   = np.array([r[1] for r in all_rows], dtype=np.float64)
    rank_all = np.array([r[2] for r in all_rows], dtype=np.int32)

    max_rank = int(rank_all.max())
    print(f"Run '{args.run_id}': {len(ms_all)} solutions, {max_rank + 1} Pareto fronts")

    mask_r0   = rank_all == 0
    mask_last = rank_all == max_rank
    mask_mid  = ~mask_r0 & ~mask_last

    print(f"  Rank-0 (best front) : {mask_r0.sum()} solutions")
    print(f"  Rank-{max_rank} (last front): {mask_last.sum()} solutions")

    fig, ax = plt.subplots(figsize=(13, 8))

    # Sample the middle ranks if the cloud is too dense
    mid_idx = np.where(mask_mid)[0]
    if len(mid_idx) > args.max_points:
        mid_idx = np.random.choice(mid_idx, size=args.max_points, replace=False)
        print(f"  Background points   : sampled {args.max_points} of {mask_mid.sum()}")
    ax.scatter(ms_all[mid_idx], tt_all[mid_idx],
               s=4, alpha=0.2, color="steelblue", label="All solutions", zorder=1)

    # Highlight the last Pareto front
    ms_last, tt_last = pareto_staircase(ms_all[mask_last], tt_all[mask_last])
    ax.scatter(ms_last, tt_last,
               s=20, alpha=0.8, color="tomato", label=f"Last front (rank {max_rank})", zorder=3)
    ax.step(ms_last, tt_last, where="post",
            color="tomato", linewidth=1.2, alpha=0.7, zorder=3)

    # Highlight the best Pareto front
    ms_r0, tt_r0 = pareto_staircase(ms_all[mask_r0], tt_all[mask_r0])
    ax.scatter(ms_r0, tt_r0,
               s=30, alpha=1.0, color="black", label="Rank-0 (best front)", zorder=5)
    ax.step(ms_r0, tt_r0, where="post",
            color="black", linewidth=1.8, alpha=0.9, zorder=5)

    ax.set_xlabel("Makespan", fontsize=12)
    ax.set_ylabel("Tardiness", fontsize=12)
    ax.set_title(f"Solution space — run_id='{args.run_id}'  (instance {args.inst_id})", fontsize=13)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"\nPlot saved to {output}")

if __name__ == "__main__":
    main()

