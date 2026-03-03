"""
EcoGraphRAG — Checkpointer
============================
Save and resume experiment progress. Protects against Colab
disconnects by saving every N questions.

Works on both laptop (local filesystem) and Colab (Google Drive).

Usage:
    from src.checkpointer import Checkpointer
    ckpt = Checkpointer("flat_rag")
    ckpt.save(results, idx=25)
    results, start_idx = ckpt.load()
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    is_colab,
    DRIVE_CHECKPOINTS,
)


class Checkpointer:
    """
    Saves experiment results at regular intervals.

    On Colab  → saves to Google Drive (survives disconnects)
    On laptop → saves to local outputs/checkpoints/
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.interval = CHECKPOINT_INTERVAL

        # Choose storage location
        if is_colab():
            self.save_dir = Path(DRIVE_CHECKPOINTS)
        else:
            self.save_dir = CHECKPOINT_DIR

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.save_dir / f"{experiment_name}_checkpoint.json"
        print(f"Checkpointer [{experiment_name}] → {self.filepath}")

    def save(self, results: list[dict], current_index: int) -> None:
        """
        Save results and current progress index.

        Args:
            results: List of result dicts so far
            current_index: Index of last completed question
        """
        checkpoint = {
            "experiment": self.experiment_name,
            "current_index": current_index,
            "num_results": len(results),
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        print(f"  💾 Checkpoint saved: {len(results)} results (idx={current_index})")

    def load(self) -> tuple[list[dict], int]:
        """
        Load checkpoint if it exists.

        Returns:
            (results_so_far, start_index) — empty list and 0 if no checkpoint
        """
        if not self.filepath.exists():
            print(f"  No checkpoint found. Starting from scratch.")
            return [], 0

        with open(self.filepath, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        results = checkpoint["results"]
        start_index = checkpoint["current_index"] + 1
        timestamp = checkpoint.get("timestamp", "unknown")

        print(f"  📂 Checkpoint loaded: {len(results)} results")
        print(f"     Resuming from index {start_index}")
        print(f"     Last saved: {timestamp}")

        return results, start_index

    def should_save(self, index: int) -> bool:
        """Check if we should save at this index."""
        return (index + 1) % self.interval == 0

    def clear(self) -> None:
        """Delete checkpoint file (call after experiment completes)."""
        if self.filepath.exists():
            os.remove(self.filepath)
            print(f"  🗑️  Checkpoint cleared: {self.filepath}")

    def save_final(self, results: list[dict], output_path: Path) -> None:
        """
        Save final results to the output directory and clear checkpoint.

        Args:
            results: Complete list of result dicts
            output_path: Final output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Final results saved: {output_path} ({len(results)} results)")
        self.clear()


if __name__ == "__main__":
    # Quick test
    ckpt = Checkpointer("test_experiment")

    # Simulate saving
    fake_results = [{"id": i, "prediction": f"answer_{i}"} for i in range(50)]

    ckpt.save(list(fake_results[:25]), current_index=24)  # type: ignore[index]
    loaded, start = ckpt.load()
    assert len(loaded) == 25
    assert start == 25
    print(f"  Test passed: loaded {len(loaded)} results, resume at {start}")

    ckpt.clear()
    print("\n✅ Checkpointer working correctly!")
