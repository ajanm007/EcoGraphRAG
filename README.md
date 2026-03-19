# EcoGraphRAG

📄 **Paper:** [Coming soon — IEEE submission in progress]

**Graph-enhanced Retrieval-Augmented Generation for multi-hop question answering.**

EcoGraphRAG augments a standard flat RAG pipeline with a knowledge graph built from named-entity co-occurrence. By traversing entity relationships at query time and merging graph-retrieved context with dense (FAISS) and sparse (BM25) retrievers, EcoGraphRAG improves answer accuracy on multi-hop benchmarks — with careful calibration of graph parameters.

---

## Highlights

| Feature | Details |
|---|---|
| **Dataset** | [HotpotQA](https://hotpotqa.github.io/) — 500-question validation split |
| **Embeddings** | `nomic-ai/nomic-embed-text-v1` (768-d) via Sentence-Transformers |
| **Vector Index** | FAISS (CPU) |
| **NER** | spaCy `en_core_web_sm` |
| **Graph** | NetworkX entity co-occurrence graph |
| **Generator** | Mistral-7B-Instruct-v0.2 (4-bit quantised, greedy decoding) |
| **Evaluation** | Exact Match · F1 (standard HotpotQA metrics) |

---

## Key Results (HotpotQA, 500 questions)

| System | EM | F1 | F1 Bridge | Latency |
|---|---|---|---|---|
| BM25 + Mistral-7B | 20.6% | 37.7% | 40.3% | 3.82s |
| Flat RAG + Mistral-7B | 26.0% | 45.3% | 48.0% | 3.61s |
| Graph-RAG default (depth=2) | 21.2% | 38.8% | 40.3% | 4.15s |
| **EcoGraphRAG K3 (Ours)** | **25.0%** | **45.9%** | **49.2%** | **2.87s** |

**Key finding:** BFS depth=2 traverses avg 8,569 nodes (34% of graph) causing context flooding (p<0.001, −6.54% F1). Depth=1 reduces traversal to 663 nodes (13× reduction) achieving statistical parity with flat RAG (p=0.55) at 20% lower latency.

**Hardware:** Indexing on 16GB CPU laptop (610s, 81.8MB RAM, zero LLM calls). Inference on free-tier Colab T4 GPU.

---

## Repository Structure

```
EcoGraphRAG/
├── config/
│   └── settings.py            # Central configuration (paths, hyperparams, model names)
├── src/
│   ├── data_loader.py         # HotpotQA download & preprocessing
│   ├── chunker.py             # Token-level chunking with overlap
│   ├── entity_extractor.py    # spaCy NER extraction
│   ├── graph_builder.py       # Entity co-occurrence graph construction
│   ├── bm25_retriever.py      # BM25 sparse retrieval baseline
│   ├── merged_retriever.py    # Weighted merge of FAISS + graph retrievers
│   ├── evaluator.py           # EM / F1 / ROUGE-L / BERTScore metrics
│   └── checkpointer.py       # Incremental checkpoint save/resume
├── notebooks/
│   ├── 01_embeddings_faiss.ipynb    # Build FAISS index from chunk embeddings
│   ├── 02_flat_rag_baseline.ipynb   # Flat RAG (FAISS-only) experiment
│   ├── 03_bm25_baseline.ipynb       # BM25 baseline experiment
│   ├── 04_graph_rag.ipynb           # Graph-RAG experiment
│   └── 05_ablation.ipynb            # Ablation study (BFS depth, k, weighting)
├── scripts/
│   └── statistical_analysis.py      # t-tests & confidence intervals
├── data/                # Dataset and chunk JSON files (gitignored)
├── indices/             # FAISS index, entity graph, mappings (gitignored)
├── outputs/             # Experiment results and checkpoints (gitignored)
├── requirements.txt
└── LICENSE              # MIT
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ajanm007/EcoGraphRAG.git
cd EcoGraphRAG

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Prepare data (local)

```python
from src.data_loader import load_hotpotqa
load_hotpotqa()          # downloads & saves data/hotpotqa_500.json
```

### 3. Run the pipeline

Execute the notebooks in order:

1. **`01_embeddings_faiss.ipynb`** — generate chunk embeddings and build the FAISS index.
2. **`02_flat_rag_baseline.ipynb`** — run the Flat RAG (FAISS-only) baseline.
3. **`03_bm25_baseline.ipynb`** — run the BM25 sparse-retrieval baseline.
4. **`04_graph_rag.ipynb`** — run Graph-RAG with merged retrieval.
5. **`05_ablation.ipynb`** — ablation over BFS depth, top-k, and graph weight.

> **Note:** Notebooks 02–05 require a GPU for LLM inference and are designed to run on Google Colab (T4 or better). Notebook 01 and all `src/` modules run on CPU.

### 4. Statistical analysis

```bash
python scripts/statistical_analysis.py
```

Produces paired t-tests and 95 % confidence intervals comparing Flat RAG vs. calibrated Graph-RAG.

---

## Configuration

All hyperparameters live in [`config/settings.py`](config/settings.py). Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 300 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `FAISS_TOP_K` | 5 | Chunks retrieved by FAISS |
| `BFS_DEPTH` | 1 | Graph traversal hops (depth=2 causes context flooding — see paper findings) |
| `GRAPH_TOP_K` | 5 | Chunks retrieved via graph |
| `GRAPH_WEIGHT` | 0.5 | Merge weight for graph-retrieved chunks |
| `MERGED_TOP_K` | 6 | Final chunks sent to the LLM |

---

## License

[MIT](LICENSE) © 2026 ajanm007
