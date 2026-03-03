"""
EcoGraphRAG — Merged Retriever
================================
Combines FAISS vector retrieval with graph BFS retrieval.
Deduplicates and scores chunks from both sources using
configurable weights.

Used by the Graph-RAG experiment notebooks on Colab.

Usage:
    py -3.13 -m src.merged_retriever
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    MERGED_TOP_K,
    GRAPH_WEIGHT,
    FAISS_WEIGHT,
    SPACY_MODEL,
    BFS_DEPTH,
    GRAPH_TOP_K,
)


def merge_retrievals(
    faiss_results: list[dict],
    graph_chunk_ids: list[str],
    chunk_lookup: dict[str, dict],
    graph_weight: float = GRAPH_WEIGHT,
    faiss_weight: float = FAISS_WEIGHT,
    top_k: int = MERGED_TOP_K,
) -> list[dict]:
    """
    Merge FAISS and graph retrieval results with weighted scoring.

    Scoring:
        - FAISS chunks get: faiss_weight * normalized_faiss_score
        - Graph chunks get: graph_weight * (1 / rank_position)
        - Chunks appearing in both get both scores summed

    Args:
        faiss_results: List of dicts with 'chunk_id' and 'score' from FAISS
        graph_chunk_ids: List of chunk IDs from graph BFS (ordered by BFS proximity)
        chunk_lookup: Dict mapping chunk_id -> chunk dict
        graph_weight: Weight for graph-sourced chunks
        faiss_weight: Weight for FAISS-sourced chunks
        top_k: Maximum chunks to return

    Returns:
        List of merged chunk dicts with 'merged_score' and 'source' keys
    """
    chunk_scores: dict[str, float] = defaultdict(float)
    chunk_sources: dict[str, list[str]] = defaultdict(list)

    # Score FAISS results (normalize scores to 0-1 range)
    if faiss_results:
        max_score = max(r.get("score", 0) for r in faiss_results)
        min_score = min(r.get("score", 0) for r in faiss_results)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for r in faiss_results:
            cid = r["chunk_id"]
            normalized = (r.get("score", 0) - min_score) / score_range
            chunk_scores[cid] += faiss_weight * normalized
            chunk_sources[cid].append("faiss")

    # Score graph results (rank-based scoring)
    for rank, cid in enumerate(graph_chunk_ids):
        score = 1.0 / (1.0 + rank)  # Reciprocal rank
        chunk_scores[cid] += graph_weight * score
        chunk_sources[cid].append("graph")

    # Rank by merged score
    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

    # Build result list
    results = []
    for cid, score in list(ranked)[:top_k]:
        if cid not in chunk_lookup:
            continue
        chunk = dict(chunk_lookup[cid])
        chunk["merged_score"] = round(score, 4)
        chunk["source"] = "+".join(chunk_sources[cid])
        results.append(chunk)

    return results


def retrieve_graph_rag(
    question: str,
    faiss_retrieve_fn: Any,
    graph: Any,
    nlp: Any,
    chunk_lookup: dict[str, dict],
    graph_weight: float = GRAPH_WEIGHT,
    faiss_weight: float = FAISS_WEIGHT,
    top_k: int = MERGED_TOP_K,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Full Graph-RAG retrieval pipeline.

    1. FAISS vector retrieval (from query embedding)
    2. Graph BFS retrieval (from query entities)
    3. Merge + deduplicate + rank

    Args:
        question: Query string
        faiss_retrieve_fn: Function that takes (query) and returns FAISS results
        graph: NetworkX entity graph
        nlp: Loaded spaCy model
        chunk_lookup: Dict mapping chunk_id -> chunk dict
        graph_weight: Weight for graph sources
        faiss_weight: Weight for FAISS sources
        top_k: Max chunks to return

    Returns:
        (merged_chunks, retrieval_info): chunks and metadata dict
    """
    from src.graph_builder import extract_query_entities, bfs_retrieve

    # Step 1: FAISS retrieval
    faiss_results = faiss_retrieve_fn(question)

    # Step 2: Graph BFS retrieval
    query_entities = extract_query_entities(question, nlp)
    graph_chunk_ids, nodes_traversed = bfs_retrieve(
        graph, query_entities, depth=BFS_DEPTH, top_k=GRAPH_TOP_K,
    )

    # Step 3: Merge
    merged = merge_retrievals(
        faiss_results=faiss_results,
        graph_chunk_ids=graph_chunk_ids,
        chunk_lookup=chunk_lookup,
        graph_weight=graph_weight,
        faiss_weight=faiss_weight,
        top_k=top_k,
    )

    # Retrieval metadata
    info = {
        "query_entities": query_entities,
        "faiss_count": len(faiss_results),
        "graph_count": len(graph_chunk_ids),
        "merged_count": len(merged),
        "nodes_traversed": nodes_traversed,
        "sources": [c.get("source", "") for c in merged],
    }

    return merged, info


if __name__ == "__main__":
    from src.graph_builder import load_graph, extract_query_entities, bfs_retrieve
    from src.chunker import load_chunks
    from src.bm25_retriever import build_bm25_index, bm25_retrieve

    # Load data
    chunks = load_chunks()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    graph = load_graph()

    import spacy
    nlp = spacy.load(SPACY_MODEL, disable=["tagger", "parser", "lemmatizer"])

    # Use BM25 as a stand-in for FAISS (can't run FAISS without embeddings on laptop)
    bm25 = build_bm25_index(chunks)

    def mock_faiss_retrieve(question: str) -> list[dict]:
        """Stand-in for FAISS retrieval using BM25 (for laptop testing only)."""
        results = bm25_retrieve(question, bm25, chunks)
        # Rename score key to match FAISS interface
        for r in results:
            r["score"] = r.pop("bm25_score", 0)
        return results

    # Test
    test_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Which magazine was started first Arthur's Magazine or First for Women?",
        "What government position was held by the woman who portrayed Cosette?",
    ]

    print(f"\n{'='*60}")
    print("MERGED RETRIEVAL TEST (BM25 + Graph)")
    print(f"{'='*60}")

    for q in test_questions:
        merged, info = retrieve_graph_rag(
            question=q,
            faiss_retrieve_fn=mock_faiss_retrieve,
            graph=graph,
            nlp=nlp,
            chunk_lookup=chunk_lookup,
        )

        print(f"\n  Q: {q}")
        print(f"  Query entities: {info['query_entities']}")
        print(f"  FAISS: {info['faiss_count']}, Graph: {info['graph_count']}, "
              f"Merged: {info['merged_count']}, Nodes: {info['nodes_traversed']}")

        for c in merged[:4]:
            print(f"    [{c['merged_score']:.3f}] ({c['source']}) {c['chunk_id']} | {c['title']}")
            print(f"          {c['text'][:80]}...")

    print(f"\n✅ Merged retriever working!")
