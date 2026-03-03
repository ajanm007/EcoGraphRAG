"""
EcoGraphRAG — Entity Extractor
================================
Runs spaCy NER on all document chunks to extract named entities.
Outputs entity data and a reverse chunk-entity mapping for graph
construction.

Runs on laptop CPU. No GPU or LLM needed.

Usage:
    py -3.13 -m src.entity_extractor
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    SPACY_MODEL,
    ENTITY_TYPES,
    CHUNKS_FILE,
    ENTITIES_FILE,
    CHUNK_ENTITY_MAP_FILE,
)


def load_spacy_model():
    """
    Load the spaCy NLP model.

    Uses en_core_web_sm — small, fast, runs on CPU.
    Only NER pipeline is needed, so we disable others for speed.
    """
    import spacy

    print(f"Loading spaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL, disable=["tagger", "parser", "lemmatizer"])
    print(f"  Pipeline: {nlp.pipe_names}")
    return nlp


def extract_entities_from_chunks(
    chunks: list[dict],
    nlp,
    batch_size: int = 256,
) -> list[dict]:
    """
    Run spaCy NER on all chunks.

    Args:
        chunks: List of chunk dicts (must have 'chunk_id' and 'text')
        nlp: Loaded spaCy model
        batch_size: Batch size for nlp.pipe()

    Returns:
        List of dicts, each with:
            - chunk_id: the chunk's ID
            - entities: list of {text, label, start_char, end_char}
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    results = []
    total_entities = 0

    print(f"Extracting entities from {len(chunks)} chunks (batch_size={batch_size})...")

    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        entities = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ not in ENTITY_TYPES:
                continue

            # Normalize: strip whitespace, title-case for consistency
            text = ent.text.strip()
            if len(text) < 2:
                continue

            # Deduplicate within chunk (same text + label)
            key = (text.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)

            entities.append({
                "text": text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })

        results.append({
            "chunk_id": chunk_ids[i],
            "entities": entities,
        })
        total_entities += len(entities)

        # Progress every 500 chunks
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks ({total_entities} entities so far)")

    print(f"Extracted {total_entities} entities from {len(chunks)} chunks")
    chunks_with_entities = sum(1 for r in results if r["entities"])
    print(f"  Chunks with ≥1 entity: {chunks_with_entities}/{len(chunks)} "
          f"({chunks_with_entities / max(len(chunks), 1) * 100:.1f}%)")

    return results


def build_entity_chunk_map(entities_data: list[dict]) -> dict[str, dict]:
    """
    Build reverse mapping: entity → list of chunk_ids.

    Args:
        entities_data: Output of extract_entities_from_chunks()

    Returns:
        Dict mapping normalized entity text to:
            - chunk_ids: list of chunk IDs where this entity appears
            - label: NER label (most common if varies)
            - frequency: total occurrence count
    """
    entity_map: dict[str, dict[str, Any]] = {}
    label_counts: dict[str, Any] = defaultdict(lambda: defaultdict(int))

    for chunk_data in entities_data:
        chunk_id = chunk_data["chunk_id"]
        for ent in chunk_data["entities"]:
            key = ent["text"].lower()

            if key not in entity_map:
                entity_map[key] = {
                    "text": ent["text"],
                    "chunk_ids": [],
                    "frequency": 0,
                }

            ids: list[Any] = entity_map[key]["chunk_ids"]
            ids.append(chunk_id)
            entity_map[key]["frequency"] = int(entity_map[key]["frequency"]) + 1
            label_counts[key][ent["label"]] += 1

    # Assign most common label to each entity
    for key, counts in label_counts.items():
        entity_map[key]["label"] = max(counts, key=counts.get)  # type: ignore[arg-type]

    print(f"Built entity-chunk map: {len(entity_map)} unique entities")

    # Top entities by frequency
    top = list(sorted(entity_map.values(), key=lambda e: e["frequency"], reverse=True))[:10]
    print("  Top 10 entities by frequency:")
    for e in top:
        print(f"    {e['text']} ({e['label']}): {e['frequency']} occurrences "
              f"in {len(e['chunk_ids'])} chunks")

    return entity_map


def save_entities(
    entities_data: list[dict],
    entity_map: dict[str, dict],
    entities_path: Path = ENTITIES_FILE,
    map_path: Path = CHUNK_ENTITY_MAP_FILE,
) -> None:
    """Save entity data and chunk-entity map to JSON files."""
    entities_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities_data, f, indent=2, ensure_ascii=False)
    size_mb = entities_path.stat().st_size / (1024 * 1024)
    print(f"Saved entities to {entities_path} ({size_mb:.2f} MB)")

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(entity_map, f, indent=2, ensure_ascii=False)
    size_mb = map_path.stat().st_size / (1024 * 1024)
    print(f"Saved entity-chunk map to {map_path} ({size_mb:.2f} MB)")


def load_entities(
    entities_path: Path = ENTITIES_FILE,
) -> list[dict]:
    """Load entity data from saved JSON file."""
    if not entities_path.exists():
        raise FileNotFoundError(
            f"{entities_path} not found. Run `py -3.13 -m src.entity_extractor` first."
        )
    with open(entities_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entity_chunk_map(
    map_path: Path = CHUNK_ENTITY_MAP_FILE,
) -> dict[str, dict]:
    """Load entity-chunk map from saved JSON file."""
    if not map_path.exists():
        raise FileNotFoundError(
            f"{map_path} not found. Run `py -3.13 -m src.entity_extractor` first."
        )
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.chunker import load_chunks

    # Check if already extracted
    if ENTITIES_FILE.exists():
        print(f"Entities already exist at {ENTITIES_FILE}")
        entities_data = load_entities()
        total_ents = sum(len(e["entities"]) for e in entities_data)
        print(f"  Total chunks: {len(entities_data)}")
        print(f"  Total entities: {total_ents}")

        # Show sample
        for ed in list(entities_data)[:5]:
            if ed["entities"]:
                print(f"\n  Chunk {ed['chunk_id']}:")
                for ent in ed["entities"][:3]:
                    print(f"    {ent['text']} ({ent['label']})")
                break
    else:
        # Load chunks from Week 1
        if not CHUNKS_FILE.exists():
            print("Chunks not found. Run `py -3.13 -m src.chunker` first.")
            sys.exit(1)

        chunks = load_chunks()

        # Load spaCy and extract
        nlp = load_spacy_model()
        entities_data = extract_entities_from_chunks(chunks, nlp)
        entity_map = build_entity_chunk_map(entities_data)

        # Save
        save_entities(entities_data, entity_map)
        print("\n✅ Entity extraction complete!")
