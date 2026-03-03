"""
EcoGraphRAG — Semantic Chunker
===============================
Sliding-window chunker that splits documents into overlapping
token chunks for embedding and retrieval.

Runs on laptop CPU. No GPU or LLM needed.

Usage:
    py -3.13 -m src.chunker
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKS_FILE,
    HOTPOTQA_FILE,
)


def tokenize_simple(text: str) -> list[str]:
    """
    Simple whitespace tokenizer.
    We use whitespace splitting (not a model tokenizer) because:
    1. It runs on CPU with zero dependencies
    2. Chunk boundaries don't need to align with model subwords
    3. 300 whitespace tokens ≈ 350-400 subword tokens — close enough
    """
    return text.split()


def chunk_document(
    text: str,
    title: str,
    doc_index: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a document into overlapping chunks.

    Args:
        text: Full document text
        title: Document title (for metadata)
        doc_index: Index in the documents list
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens

    Returns:
        List of chunk dicts with:
            - chunk_id: unique ID (doc_index_chunk_index)
            - text: chunk text
            - title: source document title
            - doc_index: source document index
            - chunk_index: position within document
            - start_token: start token offset
            - end_token: end token offset
    """
    tokens = tokenize_simple(text)

    if len(tokens) == 0:
        return []

    # If document is shorter than chunk_size, return as single chunk
    if len(tokens) <= chunk_size:
        return [{
            "chunk_id": f"{doc_index}_0",
            "text": text.strip(),
            "title": title,
            "doc_index": doc_index,
            "chunk_index": 0,
            "start_token": 0,
            "end_token": len(tokens),
        }]

    chunks = []
    step = chunk_size - chunk_overlap
    chunk_index = 0

    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = list(tokens[start:end])  # type: ignore[index]
        chunk_text = " ".join(chunk_tokens)

        chunks.append({
            "chunk_id": f"{doc_index}_{chunk_index}",
            "text": chunk_text,
            "title": title,
            "doc_index": doc_index,
            "chunk_index": chunk_index,
            "start_token": start,
            "end_token": end,
        })

        chunk_index += 1

        # Stop if we've reached the end
        if end >= len(tokens):
            break

    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk all documents from the dataset.

    Args:
        documents: List of document dicts with 'title' and 'text' keys

    Returns:
        List of all chunks across all documents
    """
    all_chunks = []

    for i, doc in enumerate(documents):
        doc_chunks = chunk_document(
            text=doc["text"],
            title=doc["title"],
            doc_index=i,
        )
        all_chunks.extend(doc_chunks)

    print(f"Chunked {len(documents)} documents → {len(all_chunks)} chunks")
    print(f"  Avg chunks per doc: {len(all_chunks) / max(len(documents), 1):.1f}")

    # Token statistics
    token_counts = [len(c["text"].split()) for c in all_chunks]
    print(f"  Avg tokens per chunk: {sum(token_counts) / max(len(token_counts), 1):.0f}")
    print(f"  Min tokens: {min(token_counts)}, Max tokens: {max(token_counts)}")

    return all_chunks


def save_chunks(chunks: list[dict], filepath: Path = CHUNKS_FILE) -> None:
    """Save chunks to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved {len(chunks)} chunks to {filepath} ({size_mb:.1f} MB)")


def load_chunks(filepath: Path = CHUNKS_FILE) -> list[dict]:
    """Load chunks from saved JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Run `py -3.13 -m src.chunker` first."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.data_loader import load_data

    if CHUNKS_FILE.exists():
        print(f"Chunks already exist at {CHUNKS_FILE}")
        chunks = load_chunks()
        print(f"  Total chunks: {len(chunks)}")

        # Show a sample
        c = chunks[0]
        print(f"\nSample chunk:")
        print(f"  ID: {c['chunk_id']}")
        print(f"  Title: {c['title']}")
        print(f"  Text: {c['text'][:200]}...")
    else:
        # Load dataset
        if not HOTPOTQA_FILE.exists():
            print("Dataset not found. Run `py -3.13 -m src.data_loader` first.")
            sys.exit(1)

        data = load_data()
        documents = data["documents"]

        # Chunk all documents
        chunks = chunk_all_documents(documents)
        save_chunks(chunks)
        print("\n✅ Chunking complete!")
