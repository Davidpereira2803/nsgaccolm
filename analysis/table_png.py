import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    # Reads a CSV file and returns the header and rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Render CSV as a trimmed PNG table")
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--head", type=int, default=5, help="Number of rows from the top (default 5)")
    parser.add_argument("--tail", type=int, default=5, help="Number of rows from the bottom (default 5)")
    parser.add_argument("--output", default="", help="Output PNG path (default: <csv>.png)")
    args = parser.parse_args()

    output = args.output or args.csv.replace(".csv", ".png")

    header, rows = read_csv(args.csv)

    total_row = None
    data_rows = rows
    if rows and rows[-1][0].strip().upper() == "TOTAL":
        # Keep summary row seperate so it stays at the bottom of the table
        total_row = rows[-1]
        data_rows = rows[:-1]

    # Show only top and bottom rows, with "..." in the middle if needed
    head = data_rows[:args.head]
    tail = data_rows[-args.tail:] if len(data_rows) > args.head + args.tail else []
    dots = [["..."] * len(header)] if tail else []

    display_rows = head + dots + tail
    if total_row:
        display_rows.append([""] * len(header))
        display_rows.append(total_row)

    table_data = [header] + display_rows

    n_rows = len(table_data)
    n_cols = len(header)

    # Scale the figure to fit the table cleanly
    fig_w = max(8, n_cols * 2.2)
    fig_h = max(2, n_rows * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(n_cols)))

    # Style the header row
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(color="black", fontweight="bold")

    # Style seperators, totals and alternating body rows
    for row_idx, row in enumerate(display_rows, start=1):
        is_dots    = row[0].strip() == "..."
        is_total   = row[0].strip().upper() == "TOTAL"
        is_spacer  = all(v == "" for v in row)

        for col in range(n_cols):
            cell = tbl[row_idx, col]
            if is_dots or is_spacer:
                cell.set_facecolor("#eeeeee")
                cell.set_text_props(color="black", style="italic")
            elif is_total:
                cell.set_facecolor("#eeeeee")
                cell.set_text_props(color="black", fontweight="bold")
            elif row_idx % 2 == 0:
                cell.set_facecolor("#eeeeee")
            else:
                cell.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Table saved to {output}")

if __name__ == "__main__":
    main()
