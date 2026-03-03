"""
EcoGraphRAG — BM25 Retriever
==============================
Keyword-based retrieval baseline using BM25 (Okapi BM25).
Zero embedding cost, zero GPU needed. Classic IR baseline.

Runs on laptop CPU.

Usage:
    py -3.13 -m src.bm25_retriever
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    BM25_TOP_K,
    CHUNKS_FILE,
    DATA_DIR,
)

BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Simple tokenizer for BM25: lowercase, split on non-alphanumeric,
    remove short tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(chunks: list[dict]):
    """
    Build a BM25 index from chunks.

    Args:
        chunks: List of chunk dicts with 'text' key

    Returns:
        BM25Okapi index
    """
    from rank_bm25 import BM25Okapi

    print(f"Building BM25 index from {len(chunks)} chunks...")
    tokenized = [tokenize_for_bm25(c["text"]) for c in chunks]
    index = BM25Okapi(tokenized)
    print(f"BM25 index built ({len(tokenized)} documents)")
    return index


def bm25_retrieve(
    query: str,
    bm25_index: Any,
    chunks: list[dict],
    top_k: int = BM25_TOP_K,
) -> list[dict]:
    """
    Retrieve top-k chunks using BM25.

    Args:
        query: Query string
        bm25_index: BM25Okapi index
        chunks: List of chunk dicts (same order as index)
        top_k: Number of chunks to return

    Returns:
        List of chunk dicts with added 'bm25_score' key
    """
    query_tokens = tokenize_for_bm25(query)
    scores = bm25_index.get_scores(query_tokens)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        chunk = dict(chunks[idx])
        chunk["bm25_score"] = float(scores[idx])
        results.append(chunk)

    return results


def save_bm25_index(bm25_index: Any, filepath: Path = BM25_INDEX_FILE) -> None:
    """Save BM25 index to pickle."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved BM25 index to {filepath} ({size_mb:.2f} MB)")


def load_bm25_index(filepath: Path = BM25_INDEX_FILE) -> Any:
    """Load BM25 index from pickle."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Run `py -3.13 -m src.bm25_retriever` first."
        )
    with open(filepath, "rb") as f:
        return pickle.load(f)  # noqa: S301


if __name__ == "__main__":
    from src.chunker import load_chunks

    # Build or load
    if BM25_INDEX_FILE.exists():
        print(f"BM25 index exists at {BM25_INDEX_FILE}")
        bm25 = load_bm25_index()
        chunks = load_chunks()
    else:
        if not CHUNKS_FILE.exists():
            print("Chunks not found. Run `py -3.13 -m src.chunker` first.")
            sys.exit(1)

        chunks = load_chunks()
        bm25 = build_bm25_index(chunks)
        save_bm25_index(bm25)

    # Test queries
    test_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Which magazine was started first Arthur's Magazine or First for Women?",
        "What government position was held by the woman who portrayed Cosette?",
    ]

    print(f"\n{'='*60}")
    print("BM25 RETRIEVAL TEST")
    print(f"{'='*60}")

    for q in test_questions:
        results = bm25_retrieve(q, bm25, chunks)
        print(f"\n  Q: {q}")
        for r in results[:3]:
            print(f"    [{r['bm25_score']:.2f}] {r['chunk_id']} | {r['title']}")
            print(f"          {r['text'][:100]}...")

    print(f"\n✅ BM25 retriever working!")
