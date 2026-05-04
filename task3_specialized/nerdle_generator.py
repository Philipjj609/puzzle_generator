"""
KODA Nerdle (Math Equation) Generator  (Task 3)
===============================================
Generates equations of the form:  <lhs> = <rhs>
where the full string (including '=') is at most 8 characters.

Examples:  1+2=3   12-4=8   3*4=12   9/3=3   4^2=16

Rules:
  • Only digits and operators + - * / ^ are used on the left.
  • The right-hand side is always a single integer (the evaluated result).
  • No leading zeros (e.g. "04" is forbidden).
  • Division produces an exact integer result only.
  • Exponents are positive-integer bases and exponents only.
  • The equation is unique (there is one canonical answer per puzzle, because
    the blank tiles the player is guessing are the full equation string).

Difficulty:
  easy   — single-operator, result ≤ 20
  medium — one or two operators, result ≤ 99
  hard   — two or three operators, result ≤ 999

Output per record:
  puzzle_data  → { "length": 8, "num_operators": 1 }  (the *format* hint shown to player)
  solution_data → { "equation": "12+34=46" }

Run:
    python task3_specialized/nerdle_generator.py
"""
from __future__ import annotations

import itertools
import operator
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from task2_logic.base_generator import (
    DAILY_COUNT,
    INFINITE_COUNT,
    PuzzleRecord,
    output_path,
    save_jsonl,
)

# Maximum equation string length (including '=') per difficulty.
# The 8-char display grid is the easy default; harder puzzles use more tiles.
_MAX_LEN: dict[str, int] = {"easy": 8, "medium": 10, "hard": 12}

# ── safe eval for arithmetic expressions ─────────────────────────────────────

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


def _safe_eval(expr: str) -> int | None:
    """
    Evaluate a simple integer expression using +, -, *, /, ^.
    Returns None if the expression is invalid (division non-integer, overflow, etc.)
    Only allows digit tokens separated by single operators (left-to-right precedence
    is NOT standard — we use Python's standard operator precedence via eval with
    a restricted namespace, which is safe here because we already validate the string).
    """
    # Validate character set before calling eval
    allowed = set("0123456789+-*/^")
    if not all(ch in allowed for ch in expr):
        return None
    if not expr or expr[0] in "+-*/" or expr[-1] in "+-*/^":
        return None

    # Replace ^ with ** for Python eval
    py_expr = expr.replace("^", "**")

    try:
        result = eval(py_expr, {"__builtins__": {}})  # noqa: S307 — expression is pre-validated
    except Exception:
        return None

    if not isinstance(result, (int, float)):
        return None
    if isinstance(result, float):
        if result != int(result):
            return None
        result = int(result)
    if result < 0 or result > 9999:
        return None
    return result


def _has_leading_zero(expr: str) -> bool:
    """Detect multi-digit tokens that start with 0."""
    i = 0
    while i < len(expr):
        if expr[i].isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            token = expr[i:j]
            if len(token) > 1 and token[0] == "0":
                return True
            i = j
        else:
            i += 1
    return False


def _build_equation(expr: str, max_len: int = 8) -> str | None:
    """Return 'expr=result' if valid and fits in max_len chars, else None."""
    if _has_leading_zero(expr):
        return None
    result = _safe_eval(expr)
    if result is None:
        return None
    eq = f"{expr}={result}"
    if len(eq) > max_len:
        return None
    return eq


# ── difficulty-filtered generators ───────────────────────────────────────────

def _gen_easy() -> str | None:
    """Single operator; operands up to 2 digits; ≤ 8-char budget."""
    op = random.choice(["+", "-", "*", "/"])
    a = random.randint(1, 99)
    b = random.randint(1, 99)
    if op == "/":
        b = random.randint(1, 12)
        a = b * random.randint(1, 9)  # guarantee exact integer division
    expr = f"{a}{op}{b}"
    eq = _build_equation(expr, _MAX_LEN["easy"])
    if eq is None:
        return None
    result = _safe_eval(expr)
    if result is None or result < 1:
        return None
    return eq


def _gen_medium() -> str | None:
    """Exactly two operators; operands up to 2 digits; ≤ 10-char budget."""
    a = random.randint(1, 99)
    b = random.randint(1, 99)
    c = random.randint(1, 99)
    ops = [random.choice(["+", "-", "*"]) for _ in range(2)]
    expr = f"{a}{ops[0]}{b}{ops[1]}{c}"
    eq = _build_equation(expr, _MAX_LEN["medium"])
    if eq is None:
        return None
    result = _safe_eval(expr)
    if result is None or result < 1:
        return None
    return eq


def _gen_hard() -> str | None:
    """Two or three operators; operands up to 2 digits; ≤ 12-char budget."""
    n_ops = random.choice([2, 3])
    nums = [random.randint(1, 99) for _ in range(n_ops + 1)]
    ops = [random.choice(["+", "-", "*"]) for _ in range(n_ops)]
    expr = str(nums[0])
    for i, op in enumerate(ops):
        expr += op + str(nums[i + 1])
    eq = _build_equation(expr, _MAX_LEN["hard"])
    if eq is None:
        return None
    result = _safe_eval(expr)
    if result is None or result < 1:
        return None
    return eq


_GEN_FN = {"easy": _gen_easy, "medium": _gen_medium, "hard": _gen_hard}


# ── batch generation ──────────────────────────────────────────────────────────

def generate_batch(
    difficulty: str,
    count: int,
    mode: str,
    *,
    progress_every: int = 500,
) -> list[PuzzleRecord]:
    gen_fn = _GEN_FN[difficulty]
    seen: set[str] = set()
    records: list[PuzzleRecord] = []

    while len(records) < count:
        eq = gen_fn()
        if eq is None or eq in seen:
            continue
        seen.add(eq)

        # Count operators for puzzle_data hint
        n_ops = sum(eq.count(op) for op in "+-*/^")
        max_len = _MAX_LEN[difficulty]
        records.append(
            PuzzleRecord(
                game_slug="nerdle",
                mode=mode,
                size_key=f"{max_len}char",
                difficulty=difficulty,
                puzzle_data={"length": len(eq), "num_operators": n_ops},
                solution_data={"equation": eq},
            )
        )
        if len(records) % progress_every == 0:
            print(f"    {len(records):>6,}/{count:,}")

    return records


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for diff in ("easy", "medium", "hard"):
        for mode, cnt in (("infinite", INFINITE_COUNT), ("daily", DAILY_COUNT)):
            print(f"Generating nerdle 8char {diff} {mode} ({cnt:,}) …")
            recs = generate_batch(diff, cnt, mode)
            save_jsonl(recs, output_path("nerdle", mode, "8char", diff))
