"""
EcoGraphRAG — Central Configuration
====================================
All paths, hyperparameters, and model names in one place.
Change values here — never hard-code them in pipeline modules.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# 1. PROJECT PATHS
# ──────────────────────────────────────────────

# Root of the project (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local directories
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = PROJECT_ROOT / "indices"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Google Drive paths (used in Colab only)
DRIVE_ROOT = "/content/drive/MyDrive/graphrag_research/"
DRIVE_DATA = DRIVE_ROOT + "data/"
DRIVE_INDICES = DRIVE_ROOT + "indices/"
DRIVE_OUTPUTS = DRIVE_ROOT + "outputs/"
DRIVE_CHECKPOINTS = DRIVE_ROOT + "checkpoints/"

# Create local directories if they don't exist
for d in [DATA_DIR, INDICES_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
NUM_QUESTIONS = 500                     # First 500 from validation split

# Saved dataset file
HOTPOTQA_FILE = DATA_DIR / "hotpotqa_500.json"


# ──────────────────────────────────────────────
# 3. CHUNKING
# ──────────────────────────────────────────────

CHUNK_SIZE = 300                        # Tokens per chunk
CHUNK_OVERLAP = 50                      # Overlapping tokens between chunks
CHUNKS_FILE = DATA_DIR / "chunks.json"
CHUNK_MAPPING_FILE = INDICES_DIR / "chunk_mapping.json"


# ──────────────────────────────────────────────
# 4. EMBEDDINGS & FAISS
# ──────────────────────────────────────────────

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_DIMENSION = 768               # nomic-embed-text output dim
FAISS_INDEX_FILE = INDICES_DIR / "faiss_index.bin"
FAISS_TOP_K = 5                         # Top-k chunks for FAISS retrieval


# ──────────────────────────────────────────────
# 5. GRAPH CONSTRUCTION (Week 2)
# ──────────────────────────────────────────────

SPACY_MODEL = "en_core_web_sm"
ENTITY_GRAPH_FILE = INDICES_DIR / "entity_graph.gpickle"
ENTITIES_FILE = DATA_DIR / "entities.json"              # chunk → entities mapping
CHUNK_ENTITY_MAP_FILE = INDICES_DIR / "chunk_entity_map.json"  # entity → chunk_ids

# Entity types to extract (spaCy NER labels)
ENTITY_TYPES = [
    "PERSON", "ORG", "GPE", "LOC", "EVENT",
    "WORK_OF_ART", "FAC", "NORP", "PRODUCT", "LAW",
]

# Graph traversal
BFS_DEPTH = 1                           # Hops from query entities (depth=2 causes context flooding)
GRAPH_TOP_K = 5                         # Max chunks from graph retrieval


# ──────────────────────────────────────────────
# 6. LLM GENERATOR
# ──────────────────────────────────────────────

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_QUANTIZATION = "4bit"               # BitsAndBytes load_in_4bit
LLM_MAX_NEW_TOKENS = 64                 # Short answers (1-5 words typical)
LLM_TEMPERATURE = 0.1                   # Near-deterministic
LLM_DO_SAMPLE = False                   # Greedy decoding for reproducibility

# Prompt template for answer generation
PROMPT_TEMPLATE = """<s>[INST] Answer the following question using ONLY the provided context.
Give a short, precise answer (1-5 words). Do not explain.

Context:
{context}

Question: {question}

Answer: [/INST]"""


# ──────────────────────────────────────────────
# 7. BM25 BASELINE
# ──────────────────────────────────────────────

BM25_TOP_K = 5                          # Top-k for BM25 retrieval


# ──────────────────────────────────────────────
# 8. MERGED RETRIEVAL (Week 3)
# ──────────────────────────────────────────────

MERGED_TOP_K = 6                        # Final chunks sent to LLM
GRAPH_WEIGHT = 0.5                      # Weight for graph-retrieved chunks
FAISS_WEIGHT = 0.5                      # Weight for FAISS-retrieved chunks


# ──────────────────────────────────────────────
# 9. EVALUATION
# ──────────────────────────────────────────────

# Result files
BM25_RESULTS_FILE = OUTPUTS_DIR / "bm25_results.json"
FLAT_RAG_RESULTS_FILE = OUTPUTS_DIR / "flat_rag_results.json"
GRAPH_RAG_RESULTS_FILE = OUTPUTS_DIR / "graph_rag_results.json"
ABLATION_RESULTS_FILE = OUTPUTS_DIR / "ablation_results.json"
COST_METRICS_FILE = OUTPUTS_DIR / "cost_metrics.json"


# ──────────────────────────────────────────────
# 10. CHECKPOINTING
# ──────────────────────────────────────────────

CHECKPOINT_INTERVAL = 25                # Save every N questions
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 11. RUNTIME DETECTION
# ──────────────────────────────────────────────

def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    import torch  # type: ignore
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_storage_dir(local_path: Path, drive_path: str) -> str:
    """Return Drive path if on Colab, else local path."""
    if is_colab():
        os.makedirs(drive_path, exist_ok=True)
        return drive_path
    return str(local_path)
