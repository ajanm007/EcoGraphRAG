"""
EcoGraphRAG — Data Loader
==========================
Downloads HotpotQA validation split (500 questions) and saves
as a clean JSON file. Run once on laptop, never re-download.

Usage:
    py -3.13 -m src.data_loader
"""

import json
import sys
from pathlib import Path
from typing import Any, cast

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DATASET_NAME,
    DATASET_CONFIG,
    DATASET_SPLIT,
    NUM_QUESTIONS,
    HOTPOTQA_FILE,
    DATA_DIR,
)


def download_hotpotqa(num_questions: int = NUM_QUESTIONS) -> list[dict]:
    """
    Download HotpotQA distractor validation split from HuggingFace.

    Returns a list of dicts, each with:
        - id: question ID
        - question: the question string
        - answer: gold answer string
        - type: 'bridge' or 'comparison'
        - level: difficulty level ('easy', 'medium', 'hard')
        - context: list of [title, sentences] pairs
        - supporting_facts: list of [title, sentence_index] pairs
    """
    from datasets import load_dataset

    print(f"Downloading {DATASET_NAME} ({DATASET_CONFIG}) — {DATASET_SPLIT} split...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    # Take first N questions
    subset = dataset.select(range(min(num_questions, len(dataset))))

    questions = []
    for item in subset:
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "context": {
                "titles": item["context"]["title"],
                "sentences": item["context"]["sentences"],
            },
            "supporting_facts": {
                "titles": item["supporting_facts"]["title"],
                "sent_ids": item["supporting_facts"]["sent_id"],
            },
        })

    print(f"Loaded {len(questions)} questions")
    return questions


def extract_documents(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract unique documents (title + full text) from all question contexts.

    Returns a list of dicts:
        - title: document title
        - text: full concatenated text of all sentences
        - source_question_ids: list of question IDs that reference this doc
    """
    docs: dict[str, dict[str, Any]] = {}
    for q in questions:
        context: dict[str, Any] = q["context"]
        titles: list[str] = context["titles"]
        sentences_list: list[list[str]] = context["sentences"]

        for title, sentences in zip(titles, sentences_list):
            key = str(title)
            if key not in docs:
                docs[key] = {
                    "title": key,
                    "text": " ".join(sentences),
                    "sentences": sentences,
                    "source_question_ids": [],
                }
            cast(list, docs[key]["source_question_ids"]).append(q["id"])

    doc_list = list(docs.values())
    print(f"Extracted {len(doc_list)} unique documents from {len(questions)} questions")
    return doc_list


def save_data(questions: list[dict], filepath: Path = HOTPOTQA_FILE) -> None:
    """Save questions to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "num_questions": len(questions),
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "questions": questions,
        "documents": extract_documents(questions),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved to {filepath} ({size_mb:.1f} MB)")


def load_data(filepath: Path = HOTPOTQA_FILE) -> dict:
    """Load questions from saved JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Run `py -3.13 -m src.data_loader` first."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    if HOTPOTQA_FILE.exists():
        print(f"Dataset already exists at {HOTPOTQA_FILE}")
        data = load_data()
        print(f"  Questions: {data['num_questions']}")
        print(f"  Documents: {len(data['documents'])}")

        # Show a sample
        q = data["questions"][0]
        print(f"\nSample question:")
        print(f"  Q: {q['question']}")
        print(f"  A: {q['answer']}")
        print(f"  Type: {q['type']}")
    else:
        questions = download_hotpotqa()
        save_data(questions)
        print("\n✅ HotpotQA dataset ready!")
