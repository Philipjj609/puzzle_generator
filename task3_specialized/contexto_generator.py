"""
KODA Contexto Semantic-Ranking Engine  (Task 3)
================================================
For a given target word, outputs the top-500 most semantically similar words
ranked by cosine distance, using a pre-trained Word2Vec / GloVe model.

Usage (one-off lookup):
    python task3_specialized/contexto_generator.py --word apple

Bulk pre-generation mode (reads a word list and writes JSONL):
    python task3_specialized/contexto_generator.py --bulk path/to/words.txt --out output/contexto/

Output schema per record:
  game_slug    = "contexto"
  mode         = "daily" | "infinite"
  size_key     = "500"
  difficulty   = "medium"  (all Contexto puzzles have uniform difficulty)
  puzzle_data  = { "target_word": "apple" }
  solution_data = {
      "ranked_words": [
          { "rank": 1, "word": "pear",   "similarity": 0.812 },
          { "rank": 2, "word": "fruit",  "similarity": 0.798 },
          …
          { "rank": 500, "word": "…", "similarity": 0.201 }
      ]
  }

Model download note:
  This script downloads the Google News Word2Vec model on first run (~1.7 GB).
  Set KODA_W2V_PATH to point to an existing .bin/.gz file to skip the download.
  Alternatively, point KODA_GLOVE_PATH to a GloVe .txt vectors file.

Run:
    pip install gensim
    python task3_specialized/contexto_generator.py --word apple
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from task2_logic.base_generator import PuzzleRecord, save_jsonl

TOP_N = 500
_W2V_URL = (
    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
)

# ── model loader ──────────────────────────────────────────────────────────────

def _load_model():
    """Load Word2Vec model; fall back to GloVe if env var is set."""
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print("ERROR: gensim is required.  Run: pip install gensim", file=sys.stderr)
        sys.exit(1)

    glove_path = os.environ.get("KODA_GLOVE_PATH")
    w2v_path = os.environ.get("KODA_W2V_PATH")

    if glove_path and Path(glove_path).exists():
        print(f"Loading GloVe vectors from {glove_path} …")
        return KeyedVectors.load_word2vec_format(glove_path, no_header=False)

    if w2v_path and Path(w2v_path).exists():
        print(f"Loading Word2Vec model from {w2v_path} …")
        return KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # Default: download Google News vectors
    cache = Path.home() / ".koda_cache" / "GoogleNews-vectors-negative300.bin.gz"
    if not cache.exists():
        print(f"Downloading Google News Word2Vec model to {cache} …")
        print("(~1.7 GB — set KODA_W2V_PATH to skip this download in future)")
        import urllib.request
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_W2V_URL, cache)  # noqa: S310

    print(f"Loading Word2Vec model from {cache} …")
    return KeyedVectors.load_word2vec_format(str(cache), binary=True)


_MODEL = None  # lazy-loaded singleton


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


# ── ranking engine ────────────────────────────────────────────────────────────

def rank_similar_words(target: str, top_n: int = TOP_N) -> list[dict] | None:
    """
    Return a list of dicts { rank, word, similarity } for the `top_n` words
    most similar to `target`.  Returns None if the word is not in the model.
    """
    model = _get_model()
    target_lower = target.lower()

    if target_lower not in model:
        print(f"WARNING: '{target}' not in model vocabulary", file=sys.stderr)
        return None

    similar = model.most_similar(target_lower, topn=top_n)
    return [
        {"rank": rank + 1, "word": word, "similarity": round(float(sim), 6)}
        for rank, (word, sim) in enumerate(similar)
    ]


def build_record(target: str, mode: str = "daily") -> PuzzleRecord | None:
    ranked = rank_similar_words(target)
    if ranked is None:
        return None
    return PuzzleRecord(
        game_slug="contexto",
        mode=mode,
        size_key=str(TOP_N),
        difficulty="medium",
        puzzle_data={"target_word": target},
        solution_data={"ranked_words": ranked},
    )


# ── bulk generation ───────────────────────────────────────────────────────────

def generate_bulk(word_list_path: str, out_dir: str, mode: str = "daily") -> None:
    words = Path(word_list_path).read_text().splitlines()
    words = [w.strip().lower() for w in words if w.strip()]
    print(f"Processing {len(words):,} words …")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[PuzzleRecord] = []

    for i, word in enumerate(words):
        rec = build_record(word, mode)
        if rec:
            records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1:>5}/{len(words)}")

    save_jsonl(records, out / f"{mode}_500_medium.jsonl")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KODA Contexto semantic ranker")
    parser.add_argument("--word", help="Single target word to rank")
    parser.add_argument("--bulk", help="Path to newline-delimited word list for bulk mode")
    parser.add_argument("--out", default="output/contexto", help="Output directory (bulk mode)")
    parser.add_argument("--mode", default="daily", choices=["daily", "infinite"])
    args = parser.parse_args()

    if args.word:
        ranked = rank_similar_words(args.word)
        if ranked:
            print(f"\nTop {TOP_N} words closest to '{args.word}':\n")
            for entry in ranked[:20]:
                print(f"  {entry['rank']:>3}. {entry['word']:<20}  sim={entry['similarity']:.4f}")
            if len(ranked) > 20:
                print(f"  … ({len(ranked) - 20} more)")
    elif args.bulk:
        generate_bulk(args.bulk, args.out, args.mode)
    else:
        parser.print_help()
