import argparse
import csv
import sqlite3
import sys

def get_rank_counts(db_path: str, run_id: str, instance_id: int) -> dict[int, int]:
    # Count how many solutions fall into each Pareto rank for one run
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT pareto_front_rank, COUNT(*)
        FROM solutions
        WHERE run_id=? AND instance_id=? AND pareto_front_rank IS NOT NULL
        GROUP BY pareto_front_rank
        ORDER BY pareto_front_rank
        """,
        (run_id, instance_id),
    ).fetchall()
    conn.close()
    return {rank: count for rank, count in rows}

def main():
    parser = argparse.ArgumentParser(description="Pareto front rank distribution table")
    parser.add_argument("inst_id", type=int, help="Taillard instance id")
    parser.add_argument("--db-path",    default="solutions.db")
    parser.add_argument("--nsga-1h",    required=True, help="run_id for the 1h NSGA baseline")
    parser.add_argument("--nsga-30min", required=True, help="run_id for the 30min NSGA / learning set")
    parser.add_argument("--colm-run",   required=True, help="run_id for the COLM inference to compare")
    parser.add_argument("--csv",        default="", help="Optional path to save table as CSV")
    args = parser.parse_args()

    runs = {
        "NSGA-1h":             args.nsga_1h,
        "NSGA-30min (corpus)": args.nsga_30min,
        args.colm_run:         args.colm_run,
    }

    data: dict[str, dict[int, int]] = {}
    for label, run_id in runs.items():
        counts = get_rank_counts(args.db_path, run_id, args.inst_id)
        if not counts:
            print(f"WARNING: no ranked solutions found for run_id='{run_id}'", file=sys.stderr)
        data[label] = counts

    # Collect every rank that appears in any run
    all_ranks = sorted(set(r for counts in data.values() for r in counts))

    if not all_ranks:
        print("No ranked solutions found for any run.")
        return

    labels = list(runs.keys())
    col_w  = max(20, *(len(l) for l in labels))

    header = f"{'Rank':>6} | " + " | ".join(f"{l:>{col_w}}" for l in labels)
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    rows_out = []
    for rank in all_ranks:
        counts = [data[l].get(rank, 0) for l in labels]
        print(f"{rank:>6} | " + " | ".join(f"{c:>{col_w}}" for c in counts))
        rows_out.append([rank] + counts)

    print(sep)
    totals = [sum(data[l].values()) for l in labels]
    print(f"{'TOTAL':>6} | " + " | ".join(f"{t:>{col_w}}" for t in totals))
    print(sep)

    if args.csv:
        # Export the table to CSV
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rank"] + labels)
            writer.writerows(rows_out)
            writer.writerow(["TOTAL"] + totals)
        print(f"\nCSV saved to {args.csv}")

if __name__ == "__main__":
    main()
