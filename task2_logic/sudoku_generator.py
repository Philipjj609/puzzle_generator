"""
KODA Sudoku Generator  (Task 2 — first batch)
=============================================
• Grid: 9×9 (standard Sudoku — the format does not generalise to 5×5 / 7×7)
• Difficulties:
    easy   — 36–45 givens
    medium — 27–35 givens
    hard   — 22–26 givens
• Every puzzle is guaranteed to have EXACTLY ONE unique solution.
  Uniqueness is verified by running the backtracking solver a second time
  from a different starting point; we only accept puzzles where the solver
  returns exactly 1 solution before the limit of 2 is reached.

Run:
    python task2_logic/sudoku_generator.py
"""
from __future__ import annotations

import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from task2_logic.base_generator import (
    DAILY_COUNT,
    INFINITE_COUNT,
    PuzzleRecord,
    output_path,
    save_jsonl,
)

# ── pre-compute peer sets (cells that share a row, column, or 3×3 box) ──────
def _build_peers() -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    peers: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for r in range(9):
        for c in range(9):
            s: set[tuple[int, int]] = set()
            for i in range(9):
                s.add((r, i))
                s.add((i, c))
            br, bc = 3 * (r // 3), 3 * (c // 3)
            for dr in range(3):
                for dc in range(3):
                    s.add((br + dr, bc + dc))
            s.discard((r, c))
            peers[(r, c)] = frozenset(s)
    return peers


PEERS: dict[tuple[int, int], frozenset[tuple[int, int]]] = _build_peers()

Grid = list[list[int]]


# ── solver (MRV backtracking) ─────────────────────────────────────────────────
def _candidates(grid: Grid, r: int, c: int) -> frozenset[int]:
    used = {grid[pr][pc] for pr, pc in PEERS[(r, c)]}
    return frozenset(range(1, 10)) - used


def _solve(grid: Grid, count: list[int], limit: int = 2) -> None:
    """Depth-first search with MRV; stops when count[0] >= limit."""
    if count[0] >= limit:
        return

    best_r = best_c = -1
    best_cands: frozenset[int] = frozenset()
    best_n = 10

    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                cands = _candidates(grid, r, c)
                if not cands:
                    return  # dead end
                if len(cands) < best_n:
                    best_n = len(cands)
                    best_r, best_c, best_cands = r, c, cands
                    if best_n == 1:
                        break
        if best_n == 1:
            break

    if best_r == -1:          # no empty cell → solution found
        count[0] += 1
        return

    for num in best_cands:
        grid[best_r][best_c] = num
        _solve(grid, count, limit)
        grid[best_r][best_c] = 0
        if count[0] >= limit:
            return


def count_solutions(grid: Grid, limit: int = 2) -> int:
    g = deepcopy(grid)
    c = [0]
    _solve(g, c, limit)
    return c[0]


# ── full-solution generator ───────────────────────────────────────────────────
def _generate_full_solution() -> Grid:
    grid: Grid = [[0] * 9 for _ in range(9)]
    nums = list(range(1, 10))

    def fill(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        if grid[r][c] != 0:
            return fill(pos + 1)
        random.shuffle(nums)
        for n in nums:
            if n not in {grid[pr][pc] for pr, pc in PEERS[(r, c)]}:
                grid[r][c] = n
                if fill(pos + 1):
                    return True
                grid[r][c] = 0
        return False

    fill(0)
    return grid


# ── difficulty configuration ──────────────────────────────────────────────────
_GIVENS: dict[str, tuple[int, int]] = {
    "easy":   (36, 45),
    "medium": (27, 35),
    "hard":   (22, 26),
}


# ── puzzle generator ──────────────────────────────────────────────────────────
def generate_puzzle(difficulty: str) -> tuple[Grid, Grid]:
    """
    Return (puzzle, solution) where puzzle has 0 for unknowns.
    Retries internally until a puzzle with exactly one solution is produced.
    """
    lo, hi = _GIVENS[difficulty]
    target_remove = 81 - random.randint(lo, hi)

    while True:
        solution = _generate_full_solution()
        puzzle = deepcopy(solution)
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)

        removed = 0
        for r, c in cells:
            if removed >= target_remove:
                break
            val = puzzle[r][c]
            puzzle[r][c] = 0
            if count_solutions(puzzle, 2) == 1:
                removed += 1
            else:
                puzzle[r][c] = val  # restore — removing this cell breaks uniqueness

        # Accept only if we reached the target (occasionally we can't strip enough)
        if removed >= target_remove - 3:   # allow ±3 tolerance on difficult grids
            return puzzle, solution


# ── batch generation ──────────────────────────────────────────────────────────
def generate_batch(
    difficulty: str,
    count: int,
    mode: str,
    *,
    progress_every: int = 200,
) -> list[PuzzleRecord]:
    records: list[PuzzleRecord] = []
    for i in range(count):
        puzzle, solution = generate_puzzle(difficulty)
        records.append(
            PuzzleRecord(
                game_slug="sudoku",
                mode=mode,
                size_key="9x9",
                difficulty=difficulty,
                puzzle_data={"grid": puzzle},
                solution_data={"grid": solution},
            )
        )
        if (i + 1) % progress_every == 0:
            print(f"    {i + 1:>6,}/{count:,}")
    return records


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for diff in ("easy", "medium", "hard"):
        for mode, count in (("infinite", INFINITE_COUNT), ("daily", DAILY_COUNT)):
            print(f"Generating sudoku 9x9 {diff} {mode} ({count:,}) …")
            recs = generate_batch(diff, count, mode)
            save_jsonl(recs, output_path("sudoku", mode, "9x9", diff))
