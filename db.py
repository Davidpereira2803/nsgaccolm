import json
import sqlite3
from pathlib import Path
import numpy as np

def init_db(db_path: str | Path) -> None:
    # Create solutions table one row per permutation and index for fast lookups
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS solutions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            TEXT    NOT NULL,
            instance_id       INTEGER NOT NULL,
            permutation       TEXT    NOT NULL,
            makespan          INTEGER NOT NULL,
            tardiness         REAL    NOT NULL,
            pareto_front_rank INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run ON solutions(run_id, instance_id)")
    conn.commit()
    conn.close()

def save_solutions(db_path: str | Path, solutions: list[tuple[tuple[int, ...], int, float]], run_id: str, instance_id: int,) -> int:
    # Insert a batch of (permutation, makespan, tardiness) tuples into the database with the associated run_id and instance_id. Returns the number of rows inserted
    if not solutions:
        return 0
    conn = sqlite3.connect(str(db_path))
    rows = [
        (run_id, instance_id, json.dumps(list(perm)), int(ms), float(tt))
        for perm, ms, tt in solutions
    ]
    conn.executemany(
        "INSERT INTO solutions (run_id, instance_id, permutation, makespan, tardiness) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)

def compute_and_update_ranks(db_path: str | Path, run_id: str, instance_id: int) -> int:
    # Load solutions for a run, compute Pareto front ranks based on makespan and tardiness, and update the database with the computed ranks. Returns the number of distinct Pareto fronts found
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, makespan, tardiness FROM solutions WHERE run_id=? AND instance_id=?",
        (run_id, instance_id),
    ).fetchall()
    conn.close()

    if not rows:
        return 0

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    makespan = np.array([r[1] for r in rows], dtype=np.float64)
    tardiness = np.array([r[2] for r in rows], dtype=np.float64)

    ranks = _nondominated_sort(makespan, tardiness)
    n_fronts = int(ranks.max()) + 1

    # Persist computed ranks back to the database
    conn = sqlite3.connect(str(db_path))
    conn.executemany(
        "UPDATE solutions SET pareto_front_rank=? WHERE id=?",
        [(int(r), int(i)) for r, i in zip(ranks, ids)],
    )
    conn.commit()
    conn.close()
    return n_fronts

def get_top_k( db_path: str | Path, run_id: str, instance_id: int, k: int) -> list[list[int]]:
    # Return the top-k permutations ordered by front rank then makespan for a given run_id and instance_id
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT permutation FROM solutions
        WHERE run_id=? AND instance_id=? AND pareto_front_rank IS NOT NULL
        ORDER BY pareto_front_rank ASC, makespan ASC
        LIMIT ?
        """,
        (run_id, instance_id, k),
    ).fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]

def _find_pareto_front_2d(ms: np.ndarray, tt: np.ndarray) -> np.ndarray:
    # Return boolean mask of points on the Pareto-optimal front rank 0
    ranks = _nondominated_sort(ms, tt)
    return ranks == 0

def _nondominated_sort(makespan: np.ndarray, tardiness: np.ndarray) -> np.ndarray:
    # Lightweight 2-D non-dominated sorting that assigns integer ranks to each point based on Pareto dominance
    N = len(makespan)
    if N == 0:
        return np.array([], dtype=np.int32)
    if N == 1:
        return np.zeros(1, dtype=np.int32)

    ranks = np.empty(N, dtype=np.int32)

    # Process points in order of incresing makespan
    order = np.lexsort((tardiness, makespan))

    # best_tt[k] holds the smallest tardiness observed so far for front k
    best_tt: list[float] = []

    prev_ms = prev_tt = None
    prev_rank = -1

    for idx in order:
        ms_i = float(makespan[idx])
        tt_i = float(tardiness[idx])

        # Identical point to previous -> inherit previous rank
        if ms_i == prev_ms and tt_i == prev_tt:
            ranks[idx] = prev_rank
            continue

        # Binary search to find smallest k with best_tt[k] > tt_i
        lo, hi = 0, len(best_tt)
        while lo < hi:
            mid = (lo + hi) >> 1
            if best_tt[mid] <= tt_i:
                lo = mid + 1
            else:
                hi = mid
        k = lo

        ranks[idx] = k
        prev_ms, prev_tt, prev_rank = ms_i, tt_i, k

        # Update best_tt: append new front or replace existing tardiness if better
        if k == len(best_tt):
            best_tt.append(tt_i)
        else:
            if tt_i < best_tt[k]:
                best_tt[k] = tt_i

    return ranks
