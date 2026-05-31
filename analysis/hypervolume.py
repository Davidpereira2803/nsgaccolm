import argparse
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def _query(db_path, sql, params=()):
    # Run a single SQL query and return all rows as a list of tuples
    conn = sqlite3.connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows

def get_pareto_front(db_path, run_id, instance_id):
    # Load the rank-0 Pareto front solutions for one run and instance
    rows = _query(
        db_path,
        "SELECT makespan, tardiness FROM solutions "
        "WHERE run_id=? AND instance_id=? AND pareto_front_rank=0",
        (run_id, instance_id),
    )
    if not rows:
        return np.empty((0, 2))
    return np.array(rows, dtype=np.float64)

def _sort_front(front):
    # Sort the Pareto front points by makespan so the staircase can be drawn correctly.
    if len(front) == 0:
        return front
    return front[np.argsort(front[:, 0])]

def compute_hv(front, ref_point):
    # Compute teh 2-D hypervolume under the Pareto front
    front = _sort_front(front)
    ref_ms, ref_tt = ref_point

    mask = (front[:, 0] < ref_ms) & (front[:, 1] < ref_tt)
    front = front[mask]
    if len(front) == 0:
        return 0.0

    hv = 0.0
    for i in range(len(front)):
        ms_next = front[i + 1, 0] if i + 1 < len(front) else ref_ms
        hv += (ms_next - front[i, 0]) * (ref_tt - front[i, 1])
    return hv

def _staircase(front, ref_point):
    # Convert the front into step-like curve for plotting the hypervolume boundary. Returns two arrays of x and y coordinates.
    front = _sort_front(front)
    if len(front) == 0:
        return np.array([]), np.array([])

    xs, ys = [front[0, 0]], [front[0, 1]]
    for i in range(1, len(front)):
        xs += [front[i, 0], front[i, 0]]
        ys += [front[i - 1, 1], front[i, 1]]
    xs.append(ref_point[0])
    ys.append(front[-1, 1])
    return np.array(xs), np.array(ys)

def _fill_hv(ax, front, ref_point, color, alpha=0.18):
    # Shade each rectangle contribution to the hypervolume
    front = _sort_front(front)
    ref_ms, ref_tt = ref_point
    mask = (front[:, 0] < ref_ms) & (front[:, 1] < ref_tt)
    front = front[mask]

    for i in range(len(front)):
        ms_next = front[i + 1, 0] if i + 1 < len(front) else ref_ms
        rect = patches.Rectangle(
            xy=(front[i, 0], front[i, 1]),
            width=ms_next - front[i, 0],
            height=ref_tt - front[i, 1],
            linewidth=0,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot the 2-D hypervolume indicator for two Pareto fronts"
    )
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--db-path", default="solutions.db")
    parser.add_argument("--run1", default="nsga1h",
                        help="First run ID (default: nsga1h)")
    parser.add_argument("--run2", default="colmt02k200",
                        help="Second run ID (default: colmt02k200)")
    parser.add_argument("--ref-margin", type=float, default=0.05,
                        help="Fractional margin beyond worst point for the reference point "
                             "(default: 0.05 = 5%%)")
    parser.add_argument("--ref-ms", type=float, default=None,
                        help="Fix the reference point makespan (overrides --ref-margin)")
    parser.add_argument("--ref-tt", type=float, default=None,
                        help="Fix the reference point tardiness (overrides --ref-margin)")
    parser.add_argument("--output", default="hypervolume_comparison.png",
                        help="Output PNG path (default: hypervolume_comparison.png)")
    args = parser.parse_args()

    front1 = get_pareto_front(args.db_path, args.run1, args.inst_id)
    front2 = get_pareto_front(args.db_path, args.run2, args.inst_id)

    if len(front1) == 0:
        print(f"WARNING: no rank-0 solutions for run '{args.run1}' instance {args.inst_id}")
    if len(front2) == 0:
        print(f"WARNING: no rank-0 solutions for run '{args.run2}' instance {args.inst_id}")

    print(f"  {args.run1}: {len(front1)} Pareto-front points")
    print(f"  {args.run2}: {len(front2)} Pareto-front points")

    non_empty = [f for f in [front1, front2] if len(f)]
    if not non_empty:
        print("No solutions found — nothing to plot.")
        return
    all_pts = np.vstack(non_empty)

    if args.ref_ms is not None and args.ref_tt is not None:
        ref_ms, ref_tt = args.ref_ms, args.ref_tt
        print(f"\nReference point (fixed): makespan={ref_ms:.2f}, tardiness={ref_tt:.4f}")
    else:
        # Auto-place the reference point just beyond the worst observed point
        m = args.ref_margin
        ref_ms = float(all_pts[:, 0].max()) * (1.0 + m)
        ref_tt = float(all_pts[:, 1].max()) * (1.0 + m)
        print(f"\nReference point (auto): makespan={ref_ms:.2f}, tardiness={ref_tt:.4f}  ({m*100:.0f}% margin)")
    ref_point = (ref_ms, ref_tt)

    hv1 = compute_hv(front1, ref_point) if len(front1) else 0.0
    hv2 = compute_hv(front2, ref_point) if len(front2) else 0.0
    ratio = hv2 / hv1 if hv1 > 0 else float("nan")

    print(f"\nHypervolume  {args.run1:>20}: {hv1:>16.4f}")
    print(f"Hypervolume  {args.run2:>20}: {hv2:>16.4f}")
    print(f"Ratio (run2 / run1)           : {ratio:>16.4f}")

    COLOR1 = "#c0392b"
    COLOR2 = "#2471a3"

    fig, ax = plt.subplots(figsize=(13, 8))

    if len(front1):
        _fill_hv(ax, front1, ref_point, COLOR1, alpha=0.18)
    if len(front2):
        _fill_hv(ax, front2, ref_point, COLOR2, alpha=0.18)

    if len(front1):
        xs, ys = _staircase(front1, ref_point)
        ax.plot(xs, ys, color=COLOR1, linewidth=1.8, zorder=3)
    if len(front2):
        xs, ys = _staircase(front2, ref_point)
        ax.plot(xs, ys, color=COLOR2, linewidth=1.8, zorder=3)

    if len(front1):
        f1 = _sort_front(front1)
        ax.scatter(f1[:, 0], f1[:, 1],
                   s=55, color=COLOR1, marker="^", zorder=5,
                   label=f"{args.run1}  (HV = {hv1:,.2f},  n = {len(front1)})")
    if len(front2):
        f2 = _sort_front(front2)
        ax.scatter(f2[:, 0], f2[:, 1],
                   s=30, color=COLOR2, marker="o", zorder=5,
                   label=f"{args.run2}  (HV = {hv2:,.2f},  n = {len(front2)})")

    ax.scatter([ref_ms], [ref_tt],
               s=150, color="black", marker="*", zorder=7,
               label=f"Reference point  ({ref_ms:.0f}, {ref_tt:.2f})")
    ax.annotate(
        f"  ref ({ref_ms:.0f}, {ref_tt:.2f})",
        xy=(ref_ms, ref_tt), fontsize=9, va="center",
    )

    ax.axvline(ref_ms, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.axhline(ref_tt, color="black", linewidth=0.7, linestyle=":", alpha=0.5)

    ratio_label = f"HV ratio (run2/run1) = {ratio:.4f}" if not np.isnan(ratio) else "HV ratio: n/a"
    ax.text(
        0.02, 0.97, ratio_label,
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    ax.set_xlabel("Makespan", fontsize=13)
    ax.set_ylabel("Tardiness", fontsize=13)
    ax.set_title(
        f"Hypervolume Indicator — Instance {args.inst_id}  |  "
        f"{args.run1}: HV={hv1:,.2f}   vs   {args.run2}: HV={hv2:,.2f}",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")

if __name__ == "__main__":
    main()
