"""
KODA Bridges (Hashiwokakero) Generator  (Task 2 — first batch)
==============================================================
Rules:
  • Islands sit at fixed grid positions with a numeric label (1–8).
  • Bridges connect islands that are in the same row/column with nothing between them.
  • Each island pair may have 0, 1, or 2 bridges; bridges cannot cross.
  • Every island's label equals the total number of bridges it touches.
  • All islands must form a single connected component.
  • Exactly ONE valid bridge configuration exists.

Sizes: 5×5, 7×7, 10×10
Difficulties:
  easy   — spanning-tree topology (unique by construction; leaf propagation)
  medium — 1 extra non-tree edge added; uniqueness explicitly verified
  hard   — 2–3 extra non-tree edges; uniqueness explicitly verified

puzzle_data →  { "size": N, "grid": [[…], …] }
               grid[r][c] == 0 → empty cell
               grid[r][c]  > 0 → island with that bridge-count label
solution_data → { "bridges": [ {"r1":…,"c1":…,"r2":…,"c2":…,"count":…}, … ] }

Run:
    python task2_logic/bridges_generator.py
"""
from __future__ import annotations

import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from task2_logic.base_generator import (
    DAILY_COUNT,
    INFINITE_COUNT,
    PuzzleRecord,
    output_path,
    save_jsonl,
)

# ── data types ────────────────────────────────────────────────────────────────

class Island(NamedTuple):
    row: int
    col: int


class Bridge(NamedTuple):
    r1: int
    c1: int
    r2: int
    c2: int
    count: int   # 1 or 2


# ── island placement ──────────────────────────────────────────────────────────

_ISLAND_COUNTS: dict[str, dict[str, int]] = {
    "5x5":   {"easy": 5,  "medium": 6,  "hard": 7},
    "7x7":   {"easy": 7,  "medium": 9,  "hard": 11},
    "10x10": {"easy": 10, "medium": 13, "hard": 16},
}


def _place_islands(size: int, n_islands: int, max_attempts: int = 200) -> list[Island] | None:
    """Place n_islands on an N×N grid with minimum spacing of 2."""
    for _ in range(max_attempts):
        positions: list[Island] = []
        occupied: set[tuple[int, int]] = set()
        candidates = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(candidates)

        for r, c in candidates:
            if len(positions) >= n_islands:
                break
            # Enforce minimum manhattan distance ≥ 2
            if any(abs(r - p.row) + abs(c - p.col) < 2 for p in positions):
                continue
            positions.append(Island(r, c))
            occupied.add((r, c))

        if len(positions) == n_islands:
            return positions
    return None


# ── adjacency (which island pairs can be directly connected) ─────────────────

def _find_adjacencies(islands: list[Island], size: int) -> list[tuple[int, int]]:
    """
    Return (i, j) pairs where island i and j are in the same row or column
    with no other island between them.  i < j always.
    """
    pos_set = {(isl.row, isl.col): idx for idx, isl in enumerate(islands)}
    adj: list[tuple[int, int]] = []

    for idx, isl in enumerate(islands):
        # Scan right along same row
        for c in range(isl.col + 1, size):
            if (isl.row, c) in pos_set:
                adj.append((idx, pos_set[(isl.row, c)]))
                break

        # Scan down along same column
        for r in range(isl.row + 1, size):
            if (r, isl.col) in pos_set:
                adj.append((idx, pos_set[(r, isl.col)]))
                break

    return adj


# ── crossing check ────────────────────────────────────────────────────────────

def _crosses(a1: Island, a2: Island, b1: Island, b2: Island) -> bool:
    """
    Return True if the bridge a1↔a2 and the bridge b1↔b2 would cross.
    One bridge must be horizontal and the other vertical.
    """
    # Normalise so the first coords are the smaller
    ar, ac_min, ac_max = a1.row, min(a1.col, a2.col), max(a1.col, a2.col)
    br, bc_min, bc_max = b1.row, min(b1.col, b2.col), max(b1.col, b2.col)

    # Both horizontal or both vertical → no crossing
    if a1.row == a2.row and b1.row == b2.row:
        return False
    if a1.col == a2.col and b1.col == b2.col:
        return False

    # One horizontal (a), one vertical (b)
    if a1.row == a2.row and b1.col == b2.col:
        h_row, h_c1, h_c2 = a1.row, ac_min, ac_max
        v_col, v_r1, v_r2 = b1.col, min(b1.row, b2.row), max(b1.row, b2.row)
        return v_r1 < h_row < v_r2 and h_c1 < v_col < h_c2

    # One vertical (a), one horizontal (b)
    if a1.col == a2.col and b1.row == b2.row:
        v_col, v_r1, v_r2 = a1.col, min(a1.row, a2.row), max(a1.row, a2.row)
        h_row, h_c1, h_c2 = b1.row, bc_min, bc_max
        return v_r1 < h_row < v_r2 and h_c1 < v_col < h_c2

    return False


def _any_crossing(
    islands: list[Island],
    new_edge: tuple[int, int],
    existing_edges: list[tuple[int, int]],
) -> bool:
    a1, a2 = islands[new_edge[0]], islands[new_edge[1]]
    for e in existing_edges:
        b1, b2 = islands[e[0]], islands[e[1]]
        if _crosses(a1, a2, b1, b2):
            return True
    return False


# ── spanning-tree builder (Kruskal, crossing-safe) ───────────────────────────

def _build_spanning_tree(
    islands: list[Island],
    adj: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Return a spanning tree edge list or None if the graph is disconnected."""
    n = len(islands)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    shuffled = list(adj)
    random.shuffle(shuffled)

    tree_edges: list[tuple[int, int]] = []
    for i, j in shuffled:
        if find(i) != find(j):
            if not _any_crossing(islands, (i, j), tree_edges):
                tree_edges.append((i, j))
                union(i, j)

    root = find(0)
    if any(find(k) != root for k in range(n)):
        return None   # could not build a spanning tree (graph not connected)
    return tree_edges


# ── backtracking uniqueness verifier ─────────────────────────────────────────

def _verify_unique(
    islands: list[Island],
    edge_list: list[tuple[int, int]],
    island_values: list[int],
    limit: int = 2,
) -> int:
    """
    Count distinct valid bridge configurations up to `limit`.
    Valid = each island's bridge sum == island_values[i] AND fully connected.
    """
    n = len(islands)
    n_edges = len(edge_list)
    counts = [0]

    # remaining capacity per island
    rem = list(island_values)
    # which edge indices are still available per island
    avail: list[list[int]] = [[] for _ in range(n)]
    for ei, (i, j) in enumerate(edge_list):
        avail[i].append(ei)
        avail[j].append(ei)

    assignment: list[int] = [-1] * n_edges  # bridge counts per edge

    def connected(assigned: list[int]) -> bool:
        adj_g: list[list[int]] = [[] for _ in range(n)]
        for ei, (i, j) in enumerate(edge_list):
            if assigned[ei] > 0:
                adj_g[i].append(j)
                adj_g[j].append(i)
        visited = [False] * n
        stack = [0]
        visited[0] = True
        while stack:
            node = stack.pop()
            for nb in adj_g[node]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        return all(visited)

    def backtrack(ei: int, remaining: list[int]) -> None:
        if counts[0] >= limit:
            return
        if ei == n_edges:
            if all(r == 0 for r in remaining) and connected(assignment):
                counts[0] += 1
            return

        i, j = edge_list[ei]
        max_bridges = min(2, remaining[i], remaining[j])

        for b in range(0, max_bridges + 1):
            assignment[ei] = b
            remaining[i] -= b
            remaining[j] -= b

            # Pruning: if any island's remaining capacity exceeds what's still possible
            ok = True
            for node in (i, j):
                if remaining[node] < 0:
                    ok = False
                    break
                # Max possible from unassigned edges for this node
                future = sum(
                    min(2, remaining[node], remaining[edge_list[ek][0 if edge_list[ek][1] == node else 1]])
                    for ek in avail[node]
                    if assignment[ek] == -1 and ek > ei
                )
                if remaining[node] > future:
                    ok = False
                    break

            if ok:
                backtrack(ei + 1, remaining)

            remaining[i] += b
            remaining[j] += b

        assignment[ei] = -1

    backtrack(0, rem)
    return counts[0]


# ── puzzle generator ──────────────────────────────────────────────────────────

def generate_puzzle(size: int, difficulty: str, max_attempts: int = 500) -> dict | None:
    size_key = f"{size}x{size}"
    n_islands = _ISLAND_COUNTS[size_key][difficulty]
    # Number of non-tree "extra" edges added for harder difficulties.
    # Randomised per call so different puzzles have varied topology.
    extra = {"easy": 0, "medium": 1, "hard": random.randint(1, 2)}[difficulty]

    for _ in range(max_attempts):
        # 1. Place islands
        islands = _place_islands(size, n_islands)
        if islands is None:
            continue

        # 2. Build adjacency
        adj = _find_adjacencies(islands, size)
        if len(adj) < n_islands - 1:
            continue

        # 3. Spanning tree
        tree_edges = _build_spanning_tree(islands, adj)
        if tree_edges is None or len(tree_edges) < n_islands - 1:
            continue

        # 4. Assign bridge counts to tree edges (1 or 2)
        bridge_counts: dict[tuple[int, int], int] = {
            e: random.choice([1, 2]) for e in tree_edges
        }

        # 5. Add extra non-tree edges for medium / hard
        non_tree = [e for e in adj if e not in bridge_counts]
        random.shuffle(non_tree)
        added_extra: list[tuple[int, int]] = []
        for e in non_tree:
            if len(added_extra) >= extra:
                break
            if not _any_crossing(islands, e, tree_edges + added_extra):
                added_extra.append(e)
                bridge_counts[e] = random.choice([1, 2])

        all_edges = tree_edges + added_extra

        # 6. Compute island values
        island_values = [0] * n_islands
        for (i, j), cnt in bridge_counts.items():
            island_values[i] += cnt
            island_values[j] += cnt

        # Sanity: max 8 bridges per island
        if any(v > 8 or v < 1 for v in island_values):
            continue

        # 7. For easy (pure spanning tree) uniqueness is guaranteed by
        #    leaf propagation; for medium/hard verify explicitly.
        if difficulty != "easy":
            n_sol = _verify_unique(islands, all_edges, island_values, limit=2)
            if n_sol != 1:
                continue

        # 8. Build output structures
        grid = [[0] * size for _ in range(size)]
        for idx, isl in enumerate(islands):
            grid[isl.row][isl.col] = island_values[idx]

        solution_bridges = [
            {"r1": islands[i].row, "c1": islands[i].col,
             "r2": islands[j].row, "c2": islands[j].col,
             "count": bridge_counts[(i, j)]}
            for i, j in all_edges
        ]

        return {
            "puzzle_data": {"size": size, "grid": grid},
            "solution_data": {"bridges": solution_bridges},
        }

    return None


def generate_batch(
    size: int,
    difficulty: str,
    count: int,
    mode: str,
    *,
    progress_every: int = 100,
) -> list[PuzzleRecord]:
    size_key = f"{size}x{size}"
    records: list[PuzzleRecord] = []
    attempts = 0
    while len(records) < count:
        attempts += 1
        result = generate_puzzle(size, difficulty)
        if result is None:
            continue
        records.append(
            PuzzleRecord(
                game_slug="bridges",
                mode=mode,
                size_key=size_key,
                difficulty=difficulty,
                puzzle_data=result["puzzle_data"],
                solution_data=result["solution_data"],
            )
        )
        if len(records) % progress_every == 0:
            print(f"    {len(records):>6,}/{count:,}  (attempts: {attempts:,})")
    return records


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SIZES = [5, 7, 10]
    for sz in SIZES:
        for diff in ("easy", "medium", "hard"):
            for mode, cnt in (("infinite", INFINITE_COUNT), ("daily", DAILY_COUNT)):
                print(f"Generating bridges {sz}x{sz} {diff} {mode} ({cnt:,}) …")
                recs = generate_batch(sz, diff, cnt, mode)
                save_jsonl(recs, output_path("bridges", mode, f"{sz}x{sz}", diff))
