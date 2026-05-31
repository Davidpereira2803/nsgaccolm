import argparse
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEMPERATURES = [0.1, 0.2, 0.5, 1.0, 1.2]
TOPK_VALUES  = [100, 200, 300, 500, 1000]

def _t_code(t):
    return f"{int(round(t * 10)):02d}"

def _run_id(t, k):
    return f"colm_t{_t_code(t)}_k{k}"

def _query(db_path, sql, params=()):
    # Run one SQL query and return all rows
    conn = sqlite3.connect(db_path)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows

def get_pareto_front(db_path, run_id, instance_id):
    # Load the rank-0 solutions for one run and instance
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
    return front[np.argsort(front[:, 0])] if len(front) else front

def compute_hv(front, ref_point):
    # Compute the 2-D hypervolume relative to the reference point
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

def main():
    parser = argparse.ArgumentParser(
        description="Heatmap of HV and rank-0 count across 25 COLM parameter experiments"
    )
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--db-path", default="solutions.db")
    parser.add_argument("--nsga-run-id", default="nsga_1h",
                        help="NSGA baseline run ID used to anchor the reference point")
    parser.add_argument("--ref-margin", type=float, default=0.05)
    parser.add_argument("--output-hv",    default="heatmap_hv.png")
    parser.add_argument("--output-rank0", default="heatmap_rank0.png")
    args = parser.parse_args()

    all_fronts = {}
    for t in TEMPERATURES:
        for k in TOPK_VALUES:
            rid = _run_id(t, k)
            all_fronts[(t, k)] = get_pareto_front(args.db_path, rid, args.inst_id)

    nsga_front = get_pareto_front(args.db_path, args.nsga_run_id, args.inst_id)

    non_empty = [f for f in list(all_fronts.values()) + [nsga_front] if len(f)]
    if not non_empty:
        print("No rank-0 solutions found for any run. Check run IDs and instance id.")
        return

    all_pts = np.vstack(non_empty)
    m = args.ref_margin
    ref_point = (float(all_pts[:, 0].max()) * (1 + m),
                 float(all_pts[:, 1].max()) * (1 + m))
    print(f"Reference point: ms={ref_point[0]:.2f}, tt={ref_point[1]:.4f}  ({m*100:.0f}% margin)")

    nsga_hv = compute_hv(nsga_front, ref_point)
    print(f"NSGA baseline ({args.nsga_run_id}) HV: {nsga_hv:.2f}\n")

    hv_mat    = np.zeros((len(TEMPERATURES), len(TOPK_VALUES)))
    rank0_mat = np.zeros((len(TEMPERATURES), len(TOPK_VALUES)), dtype=int)

    print(f"{'run_id':<22} {'rank-0':>7} {'HV':>16}  {'HV/nsga':>8}")
    print("-" * 58)
    for i, t in enumerate(TEMPERATURES):
        for j, k in enumerate(TOPK_VALUES):
            front = all_fronts[(t, k)]
            hv = compute_hv(front, ref_point)
            hv_mat[i, j]    = hv
            rank0_mat[i, j] = len(front)
            ratio = hv / nsga_hv if nsga_hv > 0 else float("nan")
            print(f"{_run_id(t, k):<22} {len(front):>7} {hv:>16.2f}  {ratio:>8.4f}")

    t_labels = [str(t) for t in TEMPERATURES]
    k_labels = [str(k) for k in TOPK_VALUES]

    def _plot_heatmap(mat, title, cmap, fmt, out_path, baseline_val=None, baseline_label=""):
        # Render one heatmap and annotate each cell
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(mat, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(TOPK_VALUES)))
        ax.set_yticks(range(len(TEMPERATURES)))
        ax.set_xticklabels(k_labels, fontsize=11)
        ax.set_yticklabels(t_labels, fontsize=11)
        ax.set_xlabel("top-k", fontsize=13)
        ax.set_ylabel("temperature", fontsize=13)
        ax.set_title(title, fontsize=13, pad=12)

        vmin, vmax = mat.min(), mat.max()
        for i in range(len(TEMPERATURES)):
            for j in range(len(TOPK_VALUES)):
                val = mat[i, j]
                text_color = "white" if (val - vmin) / (vmax - vmin + 1e-9) > 0.6 else "black"
                ax.text(j, i, fmt.format(val),
                        ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        if baseline_val is not None and baseline_val > 0:
            ax.text(1.18, 0.5,
                    f"NSGA baseline\n{baseline_label}\n{fmt.format(baseline_val)}",
                    transform=ax.transAxes, fontsize=8, va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        best_idx = np.unravel_index(np.argmax(mat), mat.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="lime", linewidth=3, zorder=5,
        ))
        ax.text(best_idx[1], best_idx[0] - 0.42, "best",
                ha="center", va="top", fontsize=7, color="lime", fontweight="bold")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out_path}")
        plt.close()

    _plot_heatmap(
        hv_mat,
        title=f"Hypervolume — Instance {args.inst_id}  (NSGA baseline = {nsga_hv:,.0f})",
        cmap="YlOrRd",
        fmt="{:,.0f}",
        out_path=args.output_hv,
        baseline_val=nsga_hv,
        baseline_label=args.nsga_run_id,
    )

    _plot_heatmap(
        rank0_mat.astype(float),
        title=f"Rank-0 solution count — Instance {args.inst_id}  (NSGA baseline = {len(nsga_front)})",
        cmap="YlGn",
        fmt="{:.0f}",
        out_path=args.output_rank0,
        baseline_val=float(len(nsga_front)),
        baseline_label=args.nsga_run_id,
    )

if __name__ == "__main__":
    main()
