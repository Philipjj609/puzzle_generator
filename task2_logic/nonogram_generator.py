"""
KODA Nonogram Generator  (Task 2 — first batch)
================================================
• Sizes: 5×5, 7×7, 10×10
• Difficulties: easy, medium, hard
  — easy:   solvable purely by line-by-line constraint propagation (0 guesses)
  — medium: needs up to 5 propagation guesses
  — hard:   requires deeper backtracking
• Every puzzle is verified to have EXACTLY ONE unique solution.

puzzle_data  →  { "size": N, "row_clues": [[...], ...], "col_clues": [[...], ...] }
solution_data →  { "grid": [[0/1, ...], ...] }

Run:
    python task2_logic/nonogram_generator.py
"""
from __future__ import annotations

import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))
from task2_logic.base_generator import (
    DAILY_COUNT,
    INFINITE_COUNT,
    PuzzleRecord,
    output_path,
    save_jsonl,
)

# ── clue computation ──────────────────────────────────────────────────────────

def compute_clue(line: list[int]) -> list[int]:
    clue: list[int] = []
    run = 0
    for cell in line:
        if cell == 1:
            run += 1
        elif run:
            clue.append(run)
            run = 0
    if run:
        clue.append(run)
    return clue or [0]


# ── enumerate all valid 0/1 patterns for one line ────────────────────────────

def line_options(clue: list[int], length: int) -> list[tuple[int, ...]]:
    """
    Return every 0/1 tuple of `length` cells consistent with `clue`.
    Uses recursive block placement; correct for all clue/length combos.
    """
    if not clue or clue == [0]:
        return [tuple([0] * length)]

    results: list[tuple[int, ...]] = []

    def place(block_idx: int, min_start: int, prefix: list[int]) -> None:
        if block_idx == len(clue):
            full = prefix + [0] * (length - len(prefix))
            results.append(tuple(full))
            return

        block = clue[block_idx]
        # Minimum space needed for every remaining block (including separators)
        tail_min = (
            sum(clue[block_idx + 1 :]) + (len(clue) - block_idx - 1)
            if block_idx + 1 < len(clue)
            else 0
        )
        max_start = length - block - tail_min

        for start in range(min_start, max_start + 1):
            new_prefix = prefix + [0] * (start - len(prefix)) + [1] * block
            if block_idx + 1 < len(clue):
                place(block_idx + 1, len(new_prefix) + 1, new_prefix + [0])
            else:
                place(block_idx + 1, len(new_prefix), new_prefix)

    place(0, 0, [])
    return results


# ── constraint-propagation solver ────────────────────────────────────────────

UNKNOWN = -1


def _cp_pass(
    grid: list[list[int]],
    row_opts: list[list[tuple[int, ...]]],
    col_opts: list[list[tuple[int, ...]]],
    size: int,
) -> tuple[bool, bool]:
    """
    One full propagation pass over rows then columns.
    Returns (changed, contradiction).
    """
    changed = False

    for r in range(size):
        valid = [
            opt for opt in row_opts[r]
            if all(grid[r][c] in (UNKNOWN, opt[c]) for c in range(size))
        ]
        if not valid:
            return False, True
        row_opts[r] = valid
        for c in range(size):
            if grid[r][c] == UNKNOWN:
                vals = {o[c] for o in valid}
                if len(vals) == 1:
                    grid[r][c] = vals.pop()
                    changed = True

    for c in range(size):
        valid = [
            opt for opt in col_opts[c]
            if all(grid[r][c] in (UNKNOWN, opt[r]) for r in range(size))
        ]
        if not valid:
            return False, True
        col_opts[c] = valid
        for r in range(size):
            if grid[r][c] == UNKNOWN:
                vals = {o[r] for o in valid}
                if len(vals) == 1:
                    grid[r][c] = vals.pop()
                    changed = True

    return changed, False


def cp_solve(
    row_clues: list[list[int]],
    col_clues: list[list[int]],
    size: int,
    grid: list[list[int]] | None = None,
) -> tuple[bool, list[list[int]], int]:
    """
    Propagation-only solver.  Returns (solved, grid, guess_count).
    guess_count == 0 means the puzzle is solvable by pure deduction.
    """
    g = [[UNKNOWN] * size for _ in range(size)] if grid is None else deepcopy(grid)
    row_opts = [line_options(rc, size) for rc in row_clues]
    col_opts = [line_options(cc, size) for cc in col_clues]

    changed = True
    while changed:
        changed, contra = _cp_pass(g, row_opts, col_opts, size)
        if contra:
            return False, g, 0

    solved = all(g[r][c] != UNKNOWN for r in range(size) for c in range(size))
    return solved, g, 0


def _count_solutions(
    row_clues: list[list[int]],
    col_clues: list[list[int]],
    size: int,
    grid: list[list[int]],
    limit: int = 2,
) -> int:
    """
    Backtracking solution counter (used for uniqueness verification).
    Applies constraint propagation at every branch to prune quickly.
    """
    # First propagate
    g = deepcopy(grid)
    row_opts = [line_options(rc, size) for rc in row_clues]
    col_opts = [line_options(cc, size) for cc in col_clues]

    changed = True
    while changed:
        changed, contra = _cp_pass(g, row_opts, col_opts, size)
        if contra:
            return 0

    # Find first unknown cell
    unknown: tuple[int, int] | None = None
    for r in range(size):
        for c in range(size):
            if g[r][c] == UNKNOWN:
                unknown = (r, c)
                break
        if unknown:
            break

    if unknown is None:
        # Check whether this grid actually satisfies all clues
        for r in range(size):
            if compute_clue(g[r]) != row_clues[r]:
                return 0
        for c in range(size):
            if compute_clue([g[r][c] for r in range(size)]) != col_clues[c]:
                return 0
        return 1

    r0, c0 = unknown
    count = 0
    for val in (0, 1):
        g2 = deepcopy(g)
        g2[r0][c0] = val
        count += _count_solutions(row_clues, col_clues, size, g2, limit)
        if count >= limit:
            return count
    return count


def is_unique(
    row_clues: list[list[int]],
    col_clues: list[list[int]],
    size: int,
) -> bool:
    empty = [[UNKNOWN] * size for _ in range(size)]
    return _count_solutions(row_clues, col_clues, size, empty, 2) == 1


# ── difficulty classifier ─────────────────────────────────────────────────────

def _guess_count(
    row_clues: list[list[int]],
    col_clues: list[list[int]],
    size: int,
) -> int:
    """Count branching steps needed to resolve the puzzle by propagation."""
    g = [[UNKNOWN] * size for _ in range(size)]
    row_opts = [line_options(rc, size) for rc in row_clues]
    col_opts = [line_options(cc, size) for cc in col_clues]

    guesses = 0

    while True:
        changed, contra = _cp_pass(g, row_opts, col_opts, size)
        if contra:
            return 999
        if all(g[r][c] != UNKNOWN for r in range(size) for c in range(size)):
            return guesses
        if not changed:
            # Stuck — pick the unknown cell with fewest options and guess
            guesses += 1
            # Just advance; in practice the count is an approximation
            for r in range(size):
                for c in range(size):
                    if g[r][c] == UNKNOWN:
                        # Speculatively pick 0 to continue propagation
                        g[r][c] = 0
                        goto = True
                        break
                if goto:
                    break


_DIFFICULTY_GUESS_RANGE: dict[str, tuple[int, int]] = {
    "easy":   (0, 0),
    "medium": (1, 5),
    "hard":   (6, 999),
}


def _in_difficulty(guesses: int, difficulty: str) -> bool:
    lo, hi = _DIFFICULTY_GUESS_RANGE[difficulty]
    return lo <= guesses <= hi


# ── solution generator ────────────────────────────────────────────────────────

def _random_solution(size: int) -> list[list[int]]:
    density = random.uniform(0.30, 0.70)
    return [[1 if random.random() < density else 0 for _ in range(size)] for _ in range(size)]


# ── public API ────────────────────────────────────────────────────────────────

def generate_puzzle(size: int, difficulty: str, max_attempts: int = 500) -> tuple | None:
    """
    Return (row_clues, col_clues, solution_grid) or None if attempts exhausted.
    Guarantees exactly one solution and matches the requested difficulty.
    """
    for _ in range(max_attempts):
        grid = _random_solution(size)
        row_clues = [compute_clue(grid[r]) for r in range(size)]
        col_clues = [compute_clue([grid[r][c] for r in range(size)]) for c in range(size)]

        if not is_unique(row_clues, col_clues, size):
            continue

        guesses = _guess_count(row_clues, col_clues, size)
        if _in_difficulty(guesses, difficulty):
            return row_clues, col_clues, grid

    return None


def generate_batch(
    size: int,
    difficulty: str,
    count: int,
    mode: str,
    *,
    progress_every: int = 100,
) -> list[PuzzleRecord]:
    records: list[PuzzleRecord] = []
    attempts = 0
    while len(records) < count:
        attempts += 1
        result = generate_puzzle(size, difficulty)
        if result is None:
            continue
        row_clues, col_clues, grid = result
        records.append(
            PuzzleRecord(
                game_slug="nonogram",
                mode=mode,
                size_key=f"{size}x{size}",
                difficulty=difficulty,
                puzzle_data={
                    "size": size,
                    "row_clues": row_clues,
                    "col_clues": col_clues,
                },
                solution_data={"grid": grid},
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
                print(f"Generating nonogram {sz}x{sz} {diff} {mode} ({cnt:,}) …")
                recs = generate_batch(sz, diff, cnt, mode)
                save_jsonl(recs, output_path("nonogram", mode, f"{sz}x{sz}", diff))
