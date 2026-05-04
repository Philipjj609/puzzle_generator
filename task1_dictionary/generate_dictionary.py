"""
KODA Master Dictionary Generator  (Task 1)
==========================================
Outputs:
  output/dictionary/master_words.json        — words grouped by length key
  output/dictionary/master_definitions.json  — { "word": "definition" }

Length keys: "2","3","4","5","6","7","8","9+"
Words are restricted to pure A-Z, lower-cased, length 2-20.

Run:
    python task1_dictionary/generate_dictionary.py
"""
from __future__ import annotations

import json
import re
import ssl
import sys
from collections import defaultdict
from pathlib import Path

# ── bootstrap NLTK ──────────────────────────────────────────────────────────
def _ensure_nltk() -> None:
    try:
        import nltk  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])

_ensure_nltk()

import nltk  # noqa: E402

# Work around SSL errors on some macOS / CI environments
try:
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
except AttributeError:
    pass

for _corpus in ("words", "wordnet", "omw-1.4"):
    nltk.download(_corpus, quiet=True)

from nltk.corpus import wordnet as wn  # noqa: E402
from nltk.corpus import words as nltk_words  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/dictionary")
_PURE_ALPHA = re.compile(r"^[a-z]+$")


# ── helpers ──────────────────────────────────────────────────────────────────
def _length_key(n: int) -> str:
    return str(n) if n <= 8 else "9+"


def _best_definition(word: str) -> str | None:
    synsets = wn.synsets(word)
    if not synsets:
        return None
    # Prefer a synset whose lemma names include the exact word (avoids drift)
    for ss in synsets:
        if word in ss.lemma_names():
            return ss.definition()
    return synsets[0].definition()


# ── main ─────────────────────────────────────────────────────────────────────
def generate(out_dir: Path = OUTPUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading NLTK words corpus …")
    raw: set[str] = {w.lower() for w in nltk_words.words()}
    filtered: list[str] = sorted(
        w for w in raw if _PURE_ALPHA.match(w) and 2 <= len(w) <= 20
    )
    print(f"  {len(filtered):,} pure A-Z words retained")

    # ── master_words.json ────────────────────────────────────────────────────
    by_length: dict[str, list[str]] = defaultdict(list)
    for w in filtered:
        by_length[_length_key(len(w))].append(w)

    master_words = {k: sorted(v) for k, v in sorted(by_length.items())}
    words_path = out_dir / "master_words.json"
    with open(words_path, "w") as fh:
        json.dump(master_words, fh, separators=(",", ":"))
    print(f"[✓] Saved {words_path}  ({sum(len(v) for v in master_words.values()):,} words)")

    # ── master_definitions.json ──────────────────────────────────────────────
    print("Fetching WordNet definitions (this may take a minute) …")
    definitions: dict[str, str] = {}
    for i, word in enumerate(filtered):
        defn = _best_definition(word)
        if defn:
            definitions[word] = defn
        if (i + 1) % 5_000 == 0:
            pct = 100 * (i + 1) / len(filtered)
            print(f"  {i + 1:>7,}/{len(filtered):,}  ({pct:.1f}%)  —  {len(definitions):,} definitions so far")

    defs_path = out_dir / "master_definitions.json"
    with open(defs_path, "w") as fh:
        json.dump(definitions, fh, separators=(",", ":"))
    print(f"[✓] Saved {defs_path}  ({len(definitions):,} definitions)")

    # summary by bucket
    print("\nWord counts by length:")
    for k, v in master_words.items():
        print(f"  length {k:>3}: {len(v):>6,}")


if __name__ == "__main__":
    generate()
