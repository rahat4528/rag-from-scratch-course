# RAG From Scratch — Complete Hands-On Course & Production Capstone

A beginner-friendly, **100% runnable** 16-notebook Jupyter course and production web application that takes learners from zero knowledge of embeddings or vector databases to building, evaluating, and safely shipping an enterprise-grade **Retrieval-Augmented Generation (RAG)** system.

> **Zero Paid API Keys Required:** By default (`RAG_PROVIDER=local`), every single notebook and the web application run completely free and offline using local neural embeddings (`all-MiniLM-L6-v2` via `sentence-transformers`), local cross-encoders (`ms-marco-MiniLM-L-6-v2`), and `LocalGroundedChatModel`. Switch to **OpenAI**, **Anthropic**, or local **Ollama** anytime via `.env`.

---

## What's Inside This Repository

- **16 Progressive Notebooks (`notebooks/00`–`15`):** Built strictly to a 9-part educational standard with plain-English analogies, $\le 16$ lines per code cell, "What just happened?" notes, exercises with full solutions, and video chapter outlines.
- **Novasphere Dynamics Domain Corpus (`data/`):** A custom fictional corporate corpus spanning `.txt`, `.md`, and `.pdf` files, accompanied by an annotated 14-question benchmark (`eval_questions.json`) testing exact numbers, multi-document joins, technical error codes, and unanswerables.
- **Capstone Web App & REST API (`app/`):** A mobile-responsive Gradio UI mounted on **FastAPI** featuring live document upload, re-indexing, document retirement, pre-retrieval Role-Based Access Control (`guest`, `employee`, `executive`), prompt-injection neutralization, `/health` and `/api/ask` endpoints, and a production `Dockerfile`.
- **Original LangChain Reference (`upstream/`):** Complete, untouched copy of all 5 original Jupyter notebooks from [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) for direct reference and historical study.
- **Architecture & QA Documentation (`docs/`):** Unified testing report ([`docs/TESTING.md`](docs/TESTING.md)), architecture decisions matrix ([`docs/DECISIONS.md`](docs/DECISIONS.md)), known issues & compatibility notes ([`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)), and companion video playlist outline ([`docs/VIDEO_OUTLINE.md`](docs/VIDEO_OUTLINE.md)).
- **Exhaustive QA Suite (`tests/`):** 24 automated pytest tests covering unit components, RRF and MRR math, security, secret audits, cache latency, and 3x determinism.

---

## Course Curriculum (16 Notebooks)

| # | Notebook | Core Skills & Educational Outcomes | Video Companion |
|---|---|---|---|
| **00** | [`notebooks/00_why_rag.ipynb`](notebooks/00_why_rag.ipynb) | Why standalone LLMs hallucinate (knowledge cutoff, private data blindness, context limits, fine-tuning cost) vs. open-book RAG | Ep. 1: Why RAG Exists |
| **01** | [`notebooks/01_hello_rag.ipynb`](notebooks/01_hello_rag.ipynb) | End-to-end 5-stage RAG pipeline in pure Python without frameworks, mapped to LangChain equivalents | Ep. 2: The Big Picture |
| **02** | [`notebooks/02_loading_documents.ipynb`](notebooks/02_loading_documents.ipynb) | Multi-format ingestion (`.txt`, `.md`, `.pdf`, HTML/Web with offline fallback), normalized metadata, UTF-8 safety | Ep. 3: Document Ingestion |
| **03** | [`notebooks/03_splitting_text.ipynb`](notebooks/03_splitting_text.ipynb) | Naive slicing vs. `RecursiveCharacterTextSplitter`, chunk size (100/300/800) trade-offs, boundary overlap fix | Ep. 4: Text Chunking |
| **04** | [`notebooks/04_embeddings_and_vector_stores.ipynb`](notebooks/04_embeddings_and_vector_stores.ipynb) | Dense embeddings, cosine similarity by hand with NumPy, 2D Matplotlib plot, `InMemoryVectorStore` JSON persistence | Ep. 5: Vector Embeddings |
| **05** | [`notebooks/05_retrieval_and_similarity.ipynb`](notebooks/05_retrieval_and_similarity.ipynb) | Top-$k$ similarity scores, confidence cutoffs, Maximal Marginal Relevance (`MMR`) diversity, metadata filtering | Ep. 6: Similarity Search |
| **06** | [`notebooks/06_generation_and_prompting.ipynb`](notebooks/06_generation_and_prompting.ipynb) | Strict `<context>` prompting, unanswerable refusal guardrails, inline citations, composable LCEL (`\|`) chains | Ep. 7: Grounded Generation |
| **07** | [`notebooks/07_query_translation.ipynb`](notebooks/07_query_translation.ipynb) | Multi-Query Expansion, RAG-Fusion with Reciprocal Rank Fusion (`RRF`), Query Decomposition, Step-Back & HyDE | Ep. 8: Query Translation |
| **08** | [`notebooks/08_routing_and_query_construction.ipynb`](notebooks/08_routing_and_query_construction.ipynb) | Logical Pydantic Routing (`.with_structured_output`), zero-LLM Semantic Routing, metadata filter query construction | Ep. 9: Query Routing |
| **09** | [`notebooks/09_advanced_indexing.ipynb`](notebooks/09_advanced_indexing.ipynb) | Parent-Child document hydration, Multi-Vector Summary indexing, safe JSON index persistence | Ep. 10: Advanced Indexing |
| **10** | [`notebooks/10_reranking_and_compression.ipynb`](notebooks/10_reranking_and_compression.ipynb) | Two-Stage Retrieval ($k=10 \rightarrow \text{top\_n}=3$), local neural Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), Contextual Compression | Ep. 11: Cross-Encoder Reranking |
| **11** | [`notebooks/11_agentic_and_self_correcting_rag.ipynb`](notebooks/11_agentic_and_self_correcting_rag.ipynb) | Stateful Corrective RAG (CRAG) with **LangGraph** (`retrieve -> grade -> rewrite -> loop guard -> generate`) | Ep. 12: Agentic Self-RAG |
| **12** | [`notebooks/12_evaluating_rag.ipynb`](notebooks/12_evaluating_rag.ipynb) | Quantitative evaluation (`Hit Rate@k`, `MRR`, Faithfulness LLM Judge, 3-configuration Pandas ablation table) | Ep. 13: Quantitative Evaluation |
| **13** | [`notebooks/13_hybrid_search_and_memory.ipynb`](notebooks/13_hybrid_search_and_memory.ipynb) | Hybrid **BM25 + Dense RRF** search for exact technical error codes & multi-turn conversational query rewriting | Ep. 14: Hybrid Lexical Search |
| **14** | [`notebooks/14_production_safety_and_security.ipynb`](notebooks/14_production_safety_and_security.ipynb) | SHA-256 `CachedEmbeddings` speedup, token budgets, indirect prompt injection defense, pre-retrieval **RBAC** filtering | Ep. 15: Production Hardening |
| **15** | [`notebooks/15_capstone_project.ipynb`](notebooks/15_capstone_project.ipynb) | Complete production-hardened RAG system evaluated across all 14 benchmark questions + Gradio UI preview | Ep. 16: Capstone Project |

---

## Quickstart (Fresh Clone to Running Course)

```bash
# 1. Clone and enter repository
git clone https://github.com/rahat4528/rag-from-scratch-course.git
cd rag-from-scratch-course

# 2. Create virtual environment and install dependencies
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# 3. Configure provider (optional — defaults cleanly to local offline mode)
cp .env.example .env

# 4. Launch Jupyter Lab or execute all notebooks automatically
jupyter lab notebooks/

# Or execute all 16 notebooks top-to-bottom via CLI:
python scripts/run_notebooks.py
```

---

## Launching the Capstone Web App & REST API

The application serves both an interactive Gradio web UI and a production FastAPI REST API on port `7860`:

```bash
python app/app.py
```

- **Interactive Web UI:** Navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860) to test questions, toggle RBAC roles (`guest`, `employee`, `executive`), upload custom files, and monitor live cache hit rates.
- **REST API Healthcheck:** `GET http://127.0.0.1:7860/health` $\rightarrow$ `{"status": "healthy", "service": "novasphere-rag-capstone"}`
- **REST API Query Endpoint:**
  ```bash
  curl -X POST http://127.0.0.1:7860/api/ask \
       -H "Content-Type: application/json" \
       -d '{"question": "What is the annual education stipend?", "role": "employee"}'
  ```

### Running with Docker
```bash
docker build -t rag-from-scratch-course .
docker run -p 7860:7860 rag-from-scratch-course
```

---

## Switching Model Providers

Edit `.env` (or set `RAG_PROVIDER` in your shell) without modifying any notebook code:

| `RAG_PROVIDER` | Chat Model | Embeddings Model | API Key Required? |
|---|---|---|---|
| `local` *(default)* | `LocalGroundedChatModel` | `all-MiniLM-L6-v2` | None (100% Free / Offline) |
| `ollama` | `ChatOllama` (`llama3.2`) | `all-MiniLM-L6-v2` | None (Local Ollama daemon) |
| `openai` | `ChatOpenAI` (`gpt-4o-mini`) | `OpenAIEmbeddings` (`text-embedding-3-small`) | `OPENAI_API_KEY` |
| `anthropic` | `ChatAnthropic` (`claude-3-5-sonnet-latest`) | Local SentenceTransformer | `ANTHROPIC_API_KEY` |

---

## Verification & QA Test Suite

Run the complete test suite (24 automated tests across environment, math, security, latency, and determinism):

```bash
# Run complete pytest suite
pytest tests/ -v

# Verify SHA-256 byte-identity of original upstream reference files
python tests/verify_manifest.py

# Audit repository tree and git commit history for leaked secrets
python tests/scan_secrets.py
```

### Verified Benchmark Metrics
- **Capstone Benchmark Accuracy:** **14/14 (100.0%)** across 3 consecutive runs at `temperature=0`.
- **Embedding Cache Speedup:** **459x – 1,959x faster** (cold lookup: ~44 ms $\rightarrow$ cached lookup: 0.05 ms).
- **End-to-End Query Latency (with neural Cross-Encoder reranking):** **P50 = 137 ms**, **P95 = 165 ms**.
- **Security:** Strict pre-retrieval RBAC filtering, indirect prompt-injection clause sanitization, and 0 leaked secrets across all git commits.

---

## Repository Layout

```text
rag-from-scratch-course/
├── .env.example                 # Environment configuration template (defaults to local mode)
├── .gitignore
├── Dockerfile                   # Single multi-stage container deployment file
├── README.md                    # Master course documentation and syllabus
├── requirements.txt             # Pinned, tested dependencies
├── app/
│   └── app.py                   # Gradio Web UI mounted on FastAPI REST API
├── data/
│   ├── company_handbook.txt     # Novasphere policies, stipends, and support tiers
│   ├── product_faq.md           # Technical error codes, ROS2 specifications, ZeroGravity failover
│   ├── recent_changelog.md      # October 2026 release notes and API deprecations
│   ├── policy_brief.pdf         # Safety brief PDF with confidential executive clauses
│   └── eval_questions.json      # 14 curated benchmark questions with ground-truth targets
├── docs/
│   ├── DECISIONS.md             # Requirements traceability matrix (R1–R9) and architecture decisions
│   ├── KNOWN_ISSUES.md          # Upstream deprecation audit and dependency notes
│   ├── TESTING.md               # Unified QA testing, performance, security, and bug register
│   └── VIDEO_OUTLINE.md         # 16-episode companion video playlist guide
├── notebooks/                   # 16 runnable Jupyter notebooks (00 through 15)
├── scripts/
│   └── run_notebooks.py         # Top-to-bottom fresh-kernel runner and output saver
├── tests/
│   ├── base_manifest.sha256     # SHA-256 hashes of original reference files
│   ├── scan_secrets.py          # Git tree and history credential scanner
│   ├── test_b1_b2_env_corpus.py # Environment, imports, and corpus UTF-8 tests
│   ├── test_b3_helpers_auth.py  # Provider resolution and negative auth tests
│   ├── test_b4_b6_pipeline_math.py # Splitter, MMR, exact RRF and MRR math tests
│   ├── test_b5_notebooks.py     # Notebook template and cell length audits
│   ├── test_b7_b10_app_security.py # XSS, injection, RBAC, and Gradio endpoint tests
│   ├── test_b11_b14_perf_e2e.py # Caching speedup, 3x determinism, and latency tests
│   ├── test_repo_integrity.py   # Single-canonical-version integrity and anti-regression enforcement
│   └── verify_manifest.py       # Manifest verification script
├── upstream/                    # Original langchain-ai/rag-from-scratch notebooks (E1 reference)
│   ├── README.md
│   ├── rag_from_scratch_1_to_4.ipynb
│   ├── rag_from_scratch_5_to_9.ipynb
│   ├── rag_from_scratch_10_and_11.ipynb
│   ├── rag_from_scratch_12_to_14.ipynb
│   └── rag_from_scratch_15_to_18.ipynb
└── utils/
    ├── __init__.py
    ├── helpers.py               # Corpus loaders, display formatters, sanitizers
    └── models.py                # Unified model factory, local neural embeddings, cache
```

---

## Troubleshooting & FAQ

- **Q: PyTorch DLL Load Error (`WinError 126`) on Windows?**  
  *A:* Make sure you install the pinned version `torch>=2.6.0,<2.9.0` as specified in `requirements.txt`. Newer builds (2.15+) require missing VC++ runtime libraries.
- **Q: Do I need an OpenAI or Anthropic API key to finish the course?**  
  *A:* No! Every single notebook and the capstone app runs 100% free and offline by default using local neural embeddings (`all-MiniLM-L6-v2`) and `LocalGroundedChatModel`.
- **Q: How do I test the REST API endpoints?**  
  *A:* Run `python app/app.py`, then query `http://127.0.0.1:7860/health` or `POST http://127.0.0.1:7860/api/ask`. Non-POST methods strictly return HTTP `405 Method Not Allowed`.
- **Q: How does pre-retrieval RBAC protect confidential documents?**  
  *A:* Every chunk in `data/` carries an `access_level` (`public`, `employee`, `executive`). Query-time filtering removes unauthorized documents before semantic vector scoring begins.

---

## Glossary of Key Terms

- **RAG (Retrieval-Augmented Generation):** The technique of providing an LLM with relevant outside documents retrieved on-the-fly to ground its answers in factual context.
- **Embedding:** A dense vector of numbers (e.g. 384 dimensions) representing the semantic meaning of a word, sentence, or document.
- **Vector Store:** A specialized database (such as `InMemoryVectorStore` or Chroma) designed to store embeddings and compute nearest-neighbor similarity searches at scale.
- **Chunking:** Splitting lengthy documents into smaller, cohesive passages with boundary overlap so answers aren't split across chunks.
- **Cross-Encoder Reranker:** A neural model that processes the query and passage together through full cross-attention for significantly higher retrieval precision.
- **Context Window:** The finite token limit an LLM can process in a single generation prompt.
- **Hallucination:** When an LLM generates convincing but factually false claims due to lack of grounded context.

---

## Attribution & License

- Based on the foundational concepts from [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch).
- All original upstream notebooks are preserved byte-for-byte in [`upstream/`](upstream/) for attribution and comparison.
- Licensed under the MIT License.
