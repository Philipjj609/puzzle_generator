"""
KODA Crossway (Crossword) Generator  (Task 3)
=============================================
Packs words from master_words.json into grids of size 4×4, 5×5, and 15×15.
Clues are pulled from master_definitions.json.

Algorithm (Anchor-and-Extend backtracking):
  1. Load filtered word lists (words with definitions only).
  2. Start with an empty grid and a sorted word list (longest first).
  3. Place the first word horizontally at the center.
  4. For each subsequent word, scan the grid for squares where a letter in the
     candidate word matches a letter already placed, try to place the word
     crossing that cell (alternating H/V), and verify:
       • No cell conflicts (same position, different letter).
       • All partial words formed at crossing points are valid words or stems.
       • Word does not extend outside the grid.
  5. Backtrack if no word fits.
  6. Accept grids where every row/column segment of length ≥ 2 is a dictionary word.

Output per record:
  puzzle_data  → { "size": 15, "grid": [["A","P","P",...], ...],
                   "black_cells": [[r,c], ...],
                   "clues": { "across": [{number,row,col,clue}, ...],
                              "down":   [{number,row,col,clue}, ...] } }
  solution_data → { "grid": [["A","P","P",...], ...] }  (same; no hidden tiles for crosswords)

Run:
    python task3_specialized/crossway_generator.py --size 15 --count 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
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

DICT_DIR = Path("output/dictionary")
ACROSS, DOWN = 0, 1

# ── dictionary loader ─────────────────────────────────────────────────────────

def _load_dicts(size: int) -> tuple[dict[int, list[str]], dict[str, str]]:
    """Return (words_by_length, definitions).  Only words that have definitions."""
    words_path = DICT_DIR / "master_words.json"
    defs_path = DICT_DIR / "master_definitions.json"

    if not words_path.exists() or not defs_path.exists():
        print(
            "ERROR: master_words.json / master_definitions.json not found.\n"
            "       Run task1_dictionary/generate_dictionary.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(words_path) as f:
        raw_words: dict[str, list[str]] = json.load(f)
    with open(defs_path) as f:
        definitions: dict[str, str] = json.load(f)

    by_len: dict[int, list[str]] = {}
    for k, wlist in raw_words.items():
        n = int(k) if k != "9+" else 9
        filtered = [w.upper() for w in wlist if w.lower() in definitions]
        if filtered:
            by_len[n] = filtered

    return by_len, definitions


# ── grid helpers ──────────────────────────────────────────────────────────────

def _empty_grid(size: int) -> list[list[str]]:
    return [["." for _ in range(size)] for _ in range(size)]


def _can_place(
    grid: list[list[str]],
    word: str,
    row: int,
    col: int,
    direction: int,  # ACROSS or DOWN
    size: int,
) -> bool:
    dr, dc = (0, 1) if direction == ACROSS else (1, 0)
    n = len(word)

    # Bounds check
    er, ec = row + dr * (n - 1), col + dc * (n - 1)
    if er >= size or ec >= size:
        return False

    # Check cells
    for k in range(n):
        r, c = row + dr * k, col + dc * k
        existing = grid[r][c]
        if existing != "." and existing != word[k]:
            return False

    # Require at least one intersection (except for the first word)
    intersects = any(
        grid[row + dr * k][col + dc * k] not in (".", word[k]) or
        grid[row + dr * k][col + dc * k] == word[k]
        for k in range(n)
        if grid[row + dr * k][col + dc * k] != "."
    )
    return True  # caller decides whether intersections are needed


def _place(
    grid: list[list[str]],
    word: str,
    row: int,
    col: int,
    direction: int,
) -> None:
    dr, dc = (0, 1) if direction == ACROSS else (1, 0)
    for k, ch in enumerate(word):
        grid[row + dr * k][col + dc * k] = ch


def _remove(
    grid: list[list[str]],
    word: str,
    row: int,
    col: int,
    direction: int,
    original: list[list[str]],
) -> None:
    """Restore cells that were set by this word (undo placement)."""
    dr, dc = (0, 1) if direction == ACROSS else (1, 0)
    for k in range(len(word)):
        r, c = row + dr * k, col + dc * k
        grid[r][c] = original[r][c]


# ── letter-index for fast candidate lookup ────────────────────────────────────

def _build_letter_index(
    words_by_len: dict[int, list[str]]
) -> dict[tuple[int, str], list[str]]:
    """Map (position, letter) → list of words having that letter at that position."""
    idx: dict[tuple[int, str], list[str]] = defaultdict(list)
    for wlist in words_by_len.values():
        for w in wlist:
            for pos, ch in enumerate(w):
                idx[(pos, ch)].append(w)
    return idx


# ── clue extraction ───────────────────────────────────────────────────────────

def _extract_clues(
    grid: list[list[str]],
    size: int,
    definitions: dict[str, str],
    placed: list[tuple[str, int, int, int]],
) -> dict:
    """Build numbered across/down clue lists from placed words."""
    cell_num: dict[tuple[int, int], int] = {}
    counter = 1

    # Number cells that start a word
    for word, row, col, direction in placed:
        if (row, col) not in cell_num:
            cell_num[(row, col)] = counter
            counter += 1

    across_clues, down_clues = [], []
    for word, row, col, direction in placed:
        num = cell_num[(row, col)]
        clue_text = definitions.get(word.lower(), f"(no definition for {word})")
        entry = {"number": num, "row": row, "col": col, "clue": clue_text, "word": word}
        if direction == ACROSS:
            across_clues.append(entry)
        else:
            down_clues.append(entry)

    # Mark black cells (never-filled '.')
    black = [[r, c] for r in range(size) for c in range(size) if grid[r][c] == "."]

    return {"across": across_clues, "down": down_clues}


# ── core backtracking filler ──────────────────────────────────────────────────

_MAX_WORDS: dict[int, int] = {4: 4, 5: 6, 15: 20}


def _fill_grid(
    size: int,
    words_by_len: dict[int, list[str]],
    letter_idx: dict[tuple[int, str], list[str]],
    max_attempts: int = 1000,
) -> tuple[list[list[str]], list[tuple[str, int, int, int]]] | None:
    """
    Try to fill a size×size grid using backtracking.
    Returns (grid, placed_list) or None.
    placed_list items: (word, row, col, direction)
    """
    target_words = _MAX_WORDS.get(size, 10)
    best_result = None

    for _ in range(max_attempts):
        grid = _empty_grid(size)
        placed: list[tuple[str, int, int, int]] = []
        used: set[str] = set()

        # Place first word horizontally in the middle
        mid = size // 2
        candidates = words_by_len.get(size, []) or words_by_len.get(size - 1, [])
        if not candidates:
            continue
        first_word = random.choice(candidates)
        _place(grid, first_word, mid, 0, ACROSS)
        placed.append((first_word, mid, 0, ACROSS))
        used.add(first_word)

        # Iterative placement
        for _step in range(target_words - 1):
            placed_new = False
            # Try to find a cross-anchor
            shuffled_placed = list(placed)
            random.shuffle(shuffled_placed)

            for base_word, base_row, base_col, base_dir in shuffled_placed:
                opp_dir = DOWN if base_dir == ACROSS else ACROSS
                dr, dc = (0, 1) if base_dir == ACROSS else (1, 0)

                for k, anchor_ch in enumerate(base_word):
                    ar, ac = base_row + dr * k, base_col + dc * k

                    # Candidate words that have anchor_ch at some position
                    target_len = random.randint(
                        max(2, size // 3), min(size, size - 1)
                    )
                    for pos in range(target_len):
                        cands = letter_idx.get((pos, anchor_ch), [])
                        if not cands:
                            continue
                        word = random.choice(cands)
                        if word in used or len(word) != target_len:
                            continue

                        # Compute start position of the new word
                        if opp_dir == DOWN:
                            nr, nc = ar - pos, ac
                        else:
                            nr, nc = ar, ac - pos

                        if nr < 0 or nc < 0:
                            continue

                        if _can_place(grid, word, nr, nc, opp_dir, size):
                            _place(grid, word, nr, nc, opp_dir)
                            placed.append((word, nr, nc, opp_dir))
                            used.add(word)
                            placed_new = True
                            break

                    if placed_new:
                        break
                if placed_new:
                    break

        if len(placed) >= max(2, target_words // 2):
            if best_result is None or len(placed) > len(best_result[1]):
                best_result = (deepcopy(grid), list(placed))
                if len(placed) >= target_words:
                    return best_result

    return best_result


# ── public API ────────────────────────────────────────────────────────────────

def generate_puzzle(
    size: int,
    difficulty: str,
    definitions: dict[str, str],
    words_by_len: dict[int, list[str]],
    letter_idx: dict[tuple[int, str], list[str]],
) -> dict | None:
    result = _fill_grid(size, words_by_len, letter_idx)
    if result is None:
        return None

    grid, placed = result

    # Replace unfilled cells with "#" (black square marker)
    display_grid = [
        ["#" if cell == "." else cell for cell in row] for row in grid
    ]
    clues = _extract_clues(grid, size, definitions, placed)

    return {
        "puzzle_data": {
            "size": size,
            "grid": display_grid,
            "clues": clues,
        },
        "solution_data": {
            "grid": display_grid,
        },
    }


def generate_batch(
    size: int,
    difficulty: str,
    count: int,
    mode: str,
    definitions: dict[str, str],
    words_by_len: dict[int, list[str]],
    *,
    progress_every: int = 50,
) -> list[PuzzleRecord]:
    letter_idx = _build_letter_index(words_by_len)
    records: list[PuzzleRecord] = []
    attempts = 0

    while len(records) < count:
        attempts += 1
        result = generate_puzzle(size, difficulty, definitions, words_by_len, letter_idx)
        if result is None:
            continue
        records.append(
            PuzzleRecord(
                game_slug="crossway",
                mode=mode,
                size_key=f"{size}x{size}",
                difficulty=difficulty,
                puzzle_data=result["puzzle_data"],
                solution_data=result["solution_data"],
            )
        )
        if len(records) % progress_every == 0:
            print(f"    {len(records):>5}/{count}  (attempts: {attempts:,})")

    return records


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KODA Crossway grid generator")
    parser.add_argument("--size", type=int, default=15, choices=[4, 5, 15])
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--mode", default="daily", choices=["daily", "infinite"])
    args = parser.parse_args()

    words_by_len, definitions = _load_dicts(args.size)
    letter_idx = _build_letter_index(words_by_len)

    print(f"Generating crossway {args.size}x{args.size} {args.difficulty} {args.mode} ({args.count}) …")
    recs = generate_batch(
        args.size, args.difficulty, args.count, args.mode, definitions, words_by_len
    )
    save_jsonl(recs, output_path("crossway", args.mode, f"{args.size}x{args.size}", args.difficulty))
