# Performance & Latency Benchmarks — `rag-from-scratch-course`

All timings recorded directly from executed runs with `torch 2.6.0+cpu` and `all-MiniLM-L6-v2` embeddings (`384-dim`) after final code edits.

## 1. Query, Cache & Scaling Telemetry (`tests/test_b11_b14_perf_e2e.py`)
- **Embedding Cache Speedup (25 batch texts):**
  - Cold miss: **`99.7 ms`**
  - Warm SHA-256 cache hit: **`0.052 ms`**
  - Measured Speedup: **`1,918x`**
- **End-to-End Hybrid RBAC Query Latency (30 queries via `NovasphereRAGEngine`):**
  - **P50 Latency:** **`2.21 ms`**
  - **P95 Latency:** **`11.17 ms`**
- **Corpus Indexing Scale (50x scale = 250 documents without cache):** Completed cleanly under the `< 30s` test budget.

## 2. Full 16-Notebook Kernel Execution Timings (`scripts/run_notebooks.py`)
*Target ceiling: `< 300s` (5 minutes) per notebook. All 16 passed cleanly in `410.21s` total:*

| Notebook | Execution Time (s) |
|---|---|
| `00_why_rag.ipynb` | 28.63s |
| `01_hello_rag.ipynb` | 27.30s |
| `02_loading_documents.ipynb` | 28.80s |
| `03_splitting_text.ipynb` | 34.29s |
| `04_embeddings_and_vector_stores.ipynb` | 28.33s |
| `05_retrieval_and_similarity.ipynb` | 68.37s |
| `06_generation_and_prompting.ipynb` | 44.31s |
| `07_query_translation.ipynb` | 15.99s |
| `08_routing_and_query_construction.ipynb` | 15.43s |
| `09_advanced_indexing.ipynb` | 16.06s |
| `10_reranking_and_compression.ipynb` | 15.19s |
| `11_agentic_and_self_correcting_rag.ipynb` | 17.08s |
| `12_evaluating_rag.ipynb` | 17.19s |
| `13_hybrid_search_and_memory.ipynb` | 18.17s |
| `14_production_safety_and_security.ipynb` | 18.97s |
| `15_capstone_project.ipynb` | 16.11s |
| **Total Batch Runtime** | **410.21s (6.8 min)** |
