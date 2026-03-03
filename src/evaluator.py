"""
EcoGraphRAG — Evaluator
========================
Exact Match (EM) and F1 score computation with SQuAD-style
text normalization. Runs on laptop CPU.

Usage:
    py -3.13 -m src.evaluator --results outputs/flat_rag_results.json
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def normalize_answer(s: str) -> str:
    """
    SQuAD-style answer normalization.
    Lowercase, remove articles, punctuation, and extra whitespace.
    """
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(c for c in s if c not in string.punctuation)
    # Collapse whitespace
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> int:
    """Return 1 if normalized prediction matches gold answer exactly."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """
    Token-level F1 score between prediction and gold answer.
    Returns 0.0 if no token overlap, 1.0 if perfect match.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_results(results: list[dict]) -> dict:
    """
    Compute aggregate metrics over a list of results.

    Args:
        results: List of dicts, each must have:
            - 'prediction': model's answer string
            - 'gold': ground truth answer string
            - 'type': question type ('bridge' or 'comparison')

    Returns:
        Dict with EM and F1 scores (overall + per question type)
    """
    em_scores = []
    f1_scores = []

    # Per-type tracking
    type_em: dict[str, list[int]] = {}
    type_f1: dict[str, list[float]] = {}

    for r in results:
        pred: str = r.get("prediction", "")
        gold: str = r.get("gold", "") or r.get("answer", "")
        if not isinstance(pred, str):
            pred = str(pred)
        if not isinstance(gold, str):
            gold = str(gold)

        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)

        em_scores.append(em)
        f1_scores.append(f1)

        # Track by question type
        q_type = r.get("type", "unknown")
        if q_type not in type_em:
            type_em[q_type] = []
            type_f1[q_type] = []
        type_em[q_type].append(em)
        type_f1[q_type].append(f1)

    # Compute averages
    by_type: dict[str, dict[str, float | int]] = {}
    for q_type in sorted(type_em.keys()):
        by_type[q_type] = {
            "em": sum(type_em[q_type]) / max(len(type_em[q_type]), 1),
            "f1": sum(type_f1[q_type]) / max(len(type_f1[q_type]), 1),
            "count": len(type_em[q_type]),
        }

    metrics = {
        "overall": {
            "em": sum(em_scores) / max(len(em_scores), 1),
            "f1": sum(f1_scores) / max(len(f1_scores), 1),
            "count": len(em_scores),
        },
        "by_type": by_type,
    }

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    overall = metrics["overall"]
    print(f"\nOverall ({overall['count']} questions):")
    print(f"  Exact Match:  {overall['em']:.4f}  ({overall['em']*100:.1f}%)")
    print(f"  F1 Score:     {overall['f1']:.4f}  ({overall['f1']*100:.1f}%)")

    if metrics["by_type"]:
        print(f"\nBy Question Type:")
        for q_type, scores in metrics["by_type"].items():
            print(f"  {q_type} ({scores['count']} Qs):")
            print(f"    EM:  {scores['em']:.4f}  ({scores['em']*100:.1f}%)")
            print(f"    F1:  {scores['f1']:.4f}  ({scores['f1']*100:.1f}%)")

    print("=" * 60)


def save_metrics(metrics: dict, filepath: Path) -> None:
    """Save metrics to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate EcoGraphRAG results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"File not found: {results_path}")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Handle both list and dict-with-results-key formats
    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    metrics = evaluate_results(results)
    print_metrics(metrics)

    # Save metrics alongside results
    metrics_path = results_path.parent / f"{results_path.stem}_metrics.json"
    save_metrics(metrics, metrics_path)
