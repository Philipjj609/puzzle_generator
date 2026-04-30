"""
KODA Base Generator
Shared output schema and JSONL writer used by every game generator.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("output")

INFINITE_COUNT = 10_000
DAILY_COUNT = 365


@dataclass
class PuzzleRecord:
    game_slug: str
    mode: str          # "infinite" | "daily"
    size_key: str
    difficulty: str
    puzzle_data: dict[str, Any]
    solution_data: dict[str, Any]


def save_jsonl(records: list[PuzzleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec)) + "\n")
    print(f"[✓] {len(records):>6,} records  →  {path}")


def output_path(game_slug: str, mode: str, size_key: str, difficulty: str) -> Path:
    return OUTPUT_DIR / game_slug / f"{mode}_{size_key}_{difficulty}.jsonl"
