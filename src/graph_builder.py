"""
EcoGraphRAG — Graph Builder
=============================
Builds an entity co-occurrence graph from the extracted entities.
Two entities sharing a chunk become connected nodes. Supports BFS
retrieval for multi-hop question answering.

Runs on laptop CPU. No GPU or LLM needed.

Usage:
    py -3.13 -m src.graph_builder
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    ENTITY_GRAPH_FILE,
    ENTITIES_FILE,
    CHUNK_ENTITY_MAP_FILE,
    BFS_DEPTH,
    GRAPH_TOP_K,
    SPACY_MODEL,
)


def build_graph(entities_data: list[dict]):
    """
    Build an entity co-occurrence graph from chunk entity data.

    Nodes: unique entities (keyed by lowercased text)
        Attributes: text, label, chunk_ids, frequency

    Edges: two entities co-occurring in the same chunk
        Attributes: weight (co-occurrence count), chunk_ids (shared chunks)

    Args:
        entities_data: Output of entity_extractor.extract_entities_from_chunks()

    Returns:
        NetworkX graph
    """
    import networkx as nx

    G = nx.Graph()

    # Track entity info for node attributes
    entity_info: dict[str, dict[str, Any]] = {}
    # Track co-occurrences for edges
    edge_chunks: dict[Any, list[str]] = defaultdict(list)

    for chunk_data in entities_data:
        chunk_id = chunk_data["chunk_id"]
        entities = chunk_data["entities"]

        if len(entities) < 1:
            continue

        # Collect unique entities in this chunk
        chunk_entities: list[dict[str, str]] = []
        seen_keys: set[str] = set()

        for ent in entities:
            key = ent["text"].lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chunk_entities.append(ent)

            # Track node info
            if key not in entity_info:
                entity_info[key] = {
                    "text": ent["text"],
                    "label": ent["label"],
                    "chunk_ids": [],
                    "frequency": 0,
                }
            cids: list[Any] = entity_info[key]["chunk_ids"]
            cids.append(chunk_id)
            entity_info[key]["frequency"] = int(entity_info[key]["frequency"]) + 1

        # Build co-occurrence edges (all pairs in this chunk)
        for i in range(len(chunk_entities)):
            for j in range(i + 1, len(chunk_entities)):
                key_a = chunk_entities[i]["text"].lower()
                key_b = chunk_entities[j]["text"].lower()

                # Consistent edge ordering
                edge = tuple(sorted([key_a, key_b]))
                edge_chunks[edge].append(chunk_id)  # type: ignore[index]

    # Add nodes
    for key, info in entity_info.items():
        G.add_node(
            key,
            text=info["text"],
            label=info["label"],
            chunk_ids=info["chunk_ids"],
            frequency=info["frequency"],
        )

    # Add edges
    for (a, b), chunks in edge_chunks.items():
        G.add_edge(a, b, weight=len(chunks), chunk_ids=chunks)

    print(f"Built entity graph:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees)
        print(f"  Avg degree: {avg_degree:.1f}")
        print(f"  Max degree: {max(degrees)}")

        # Connected components
        components = list(nx.connected_components(G))
        print(f"  Connected components: {len(components)}")
        largest = max(components, key=len)
        print(f"  Largest component: {len(largest)} nodes "
              f"({len(largest) / G.number_of_nodes() * 100:.1f}%)")

    return G


def extract_query_entities(question: str, nlp) -> list[str]:
    """
    Extract entity mentions from a question using spaCy.

    Args:
        question: The question string

    Returns:
        List of lowercased entity texts found in the question
    """
    doc = nlp(question)
    entities = []
    seen: set[str] = set()

    for ent in doc.ents:
        key = ent.text.strip().lower()
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        entities.append(key)

    return entities


def bfs_retrieve(
    graph,
    query_entities: list[str],
    depth: int = BFS_DEPTH,
    top_k: int = GRAPH_TOP_K,
) -> tuple[list[str], int]:
    """
    BFS retrieval: start from query entities, traverse graph,
    collect chunk IDs from visited nodes.

    Args:
        graph: NetworkX graph
        query_entities: List of entity keys (lowercased) from the question
        depth: Max BFS hops (default: 2)
        top_k: Max chunks to return

    Returns:
        (chunk_ids, nodes_traversed): list of chunk IDs and count of
        graph nodes visited during traversal
    """
    from collections import deque

    visited: set[str] = set()
    chunk_scores: dict[str, Any] = defaultdict(float)
    nodes_traversed: int = 0

    for entity in query_entities:
        if entity not in graph:
            continue

        # BFS from this entity
        queue: deque[tuple[str, int]] = deque([(entity, 0)])
        local_visited: set[str] = set()

        while queue:
            node, current_depth = queue.popleft()

            if node in local_visited:
                continue
            local_visited.add(node)

            if node not in visited:
                visited.add(node)
                nodes_traversed += 1

                # Score chunks from this node — closer nodes get higher score
                node_data = graph.nodes[node]
                score = 1.0 / (1.0 + current_depth)  # Decay by distance

                for chunk_id in node_data.get("chunk_ids", []):
                    chunk_scores[chunk_id] += score

            # Expand neighbors if within depth
            if current_depth < depth:
                for neighbor in graph.neighbors(node):
                    if neighbor not in local_visited:
                        queue.append((neighbor, current_depth + 1))

    # Rank chunks by total score, return top-k
    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    top_chunk_ids = [cid for cid, _ in list(ranked)[:top_k]]

    return top_chunk_ids, nodes_traversed


def save_graph(graph, filepath: Path = ENTITY_GRAPH_FILE) -> None:
    """Save graph to pickle file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved graph to {filepath} ({size_mb:.2f} MB)")


def load_graph(filepath: Path = ENTITY_GRAPH_FILE):
    """Load graph from pickle file."""
    import networkx as nx

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Run `py -3.13 -m src.graph_builder` first."
        )
    with open(filepath, "rb") as f:
        G = pickle.load(f)  # noqa: S301
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


if __name__ == "__main__":
    from src.entity_extractor import load_entities, load_entity_chunk_map

    # ── Build or load graph ───────────────────────────────────
    if ENTITY_GRAPH_FILE.exists():
        print(f"Graph already exists at {ENTITY_GRAPH_FILE}")
        G = load_graph()
    else:
        if not ENTITIES_FILE.exists():
            print("Entities not found. Run `py -3.13 -m src.entity_extractor` first.")
            sys.exit(1)

        entities_data = load_entities()
        G = build_graph(entities_data)
        save_graph(G)

    # ── Graph statistics ──────────────────────────────────────
    import networkx as nx

    print(f"\n{'='*60}")
    print("GRAPH STATISTICS")
    print(f"{'='*60}")
    print(f"Nodes (entities):  {G.number_of_nodes()}")
    print(f"Edges (co-occur):  {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        degrees = [d for _, d in G.degree()]
        print(f"Avg degree:        {sum(degrees)/len(degrees):.1f}")
        print(f"Median degree:     {sorted(degrees)[len(degrees)//2]}")
        print(f"Max degree:        {max(degrees)}")
        print(f"Density:           {nx.density(G):.6f}")

        components = list(nx.connected_components(G))
        print(f"Components:        {len(components)}")
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"Top 5 comp sizes:  {list(sizes)[:5]}")

    # ── BFS test on sample questions ──────────────────────────
    print(f"\n{'='*60}")
    print("BFS RETRIEVAL TEST")
    print(f"{'='*60}")

    test_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What government position was held by the woman who portrayed Tbuilt Cosette?",
        "Which magazine was started first Arthur's Magazine or First for Women?",
        "935 Broadway was designed by which architect who also designed a building in Manhattan?",
        "What is the name of the fight song of the university whose main campus is in Lawrence, Kansas?",
    ]

    import spacy
    nlp = spacy.load(SPACY_MODEL, disable=["tagger", "parser", "lemmatizer"])

    for q in test_questions:
        q_entities = extract_query_entities(q, nlp)
        chunk_ids, nodes_traversed = bfs_retrieve(G, q_entities)

        print(f"\n  Q: {q}")
        print(f"  Query entities: {q_entities}")
        print(f"  Nodes traversed: {nodes_traversed}")
        print(f"  Chunks retrieved: {len(chunk_ids)} → {list(chunk_ids)[:5]}")

    print(f"\n✅ Graph construction and BFS retrieval working!")
