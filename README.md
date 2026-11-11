# RAG From Scratch — Complete Hands-On Course

A beginner-friendly, **100% runnable** 16-notebook Jupyter course and capstone web app that takes learners from zero knowledge of embeddings or vector databases to building, evaluating, and safely shipping a production-ready **Retrieval-Augmented Generation (RAG)** system.

> **Zero Paid API Keys Required:** By default (`RAG_PROVIDER=local`), every single notebook runs completely free and offline using local neural embeddings (`all-MiniLM-L6-v2` via `sentence-transformers` with deterministic fallback) and `LocalGroundedChatModel`. Switch to **OpenAI**, **Anthropic**, or local **Ollama** anytime via `.env`.

---

## Course Curriculum (16 Notebooks)

| # | Notebook | Core Skills & Topics |
|---|---|---|
| **00** | [`notebooks/00_why_rag.ipynb`](notebooks/00_why_rag.ipynb) | Why standalone LLMs hallucinate (knowledge cutoff, private data blindness, context limits, fine-tuning cost) vs. open-book RAG |
| **01** | [`notebooks/01_hello_rag.ipynb`](notebooks/01_hello_rag.ipynb) | Your first end-to-end 5-stage RAG pipeline (`Load -> Split -> Embed & Index -> Retrieve -> Generate`) |
| **02** | [`notebooks/02_loading_documents.ipynb`](notebooks/02_loading_documents.ipynb) | Multi-format ingestion (`.txt`, `.md`, `.pdf`, HTML/Web with offline fallback), normalized metadata, UTF-8 / Unicode safety |
| **03** | [`notebooks/03_splitting_text.ipynb`](notebooks/03_splitting_text.ipynb) | Naive slicing vs. `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, chunk size & boundary overlap math |
| **04** | [`notebooks/04_embeddings_and_vector_stores.ipynb`](notebooks/04_embeddings_and_vector_stores.ipynb) | Dense embeddings, cosine similarity, 2D PCA visualization, `InMemoryVectorStore` JSON dump/load persistence |
| **05** | [`notebooks/05_retrieval_and_similarity.ipynb`](notebooks/05_retrieval_and_similarity.ipynb) | Top-k similarity scores, confidence cutoffs, Maximal Marginal Relevance (`MMR`) diversity, metadata filtering |
| **06** | [`notebooks/06_generation_and_prompting.ipynb`](notebooks/06_generation_and_prompting.ipynb) | Strict `<context>` prompting, unanswerable refusal guardrails, inline citations, composable LCEL (`\|`) chains |
| **07** | [`notebooks/07_query_translation.ipynb`](notebooks/07_query_translation.ipynb) | Multi-Query Expansion, RAG-Fusion with Reciprocal Rank Fusion (`RRF`), Query Decomposition, Step-Back & HyDE |
| **08** | [`notebooks/08_routing_and_query_construction.ipynb`](notebooks/08_routing_and_query_construction.ipynb) | Logical Pydantic Routing (`.with_structured_output`), zero-LLM Semantic Routing, metadata filter query construction |
| **09** | [`notebooks/09_advanced_indexing.ipynb`](notebooks/09_advanced_indexing.ipynb) | Parent-Child document hydration, Multi-Vector Summary indexing, safe JSON index persistence |
| **10** | [`notebooks/10_reranking_and_compression.ipynb`](notebooks/10_reranking_and_compression.ipynb) | Two-Stage Retrieval (`k=10 -> top_n=3`), hybrid lexical + semantic reranking, sentence-level Contextual Compression |
| **11** | [`notebooks/11_agentic_and_self_correcting_rag.ipynb`](notebooks/11_agentic_and_self_correcting_rag.ipynb) | Stateful Corrective RAG (CRAG) with **LangGraph** (`retrieve -> grade -> rewrite -> loop guard -> generate`) |
| **12** | [`notebooks/12_evaluating_rag.ipynb`](notebooks/12_evaluating_rag.ipynb) | Quantitative evaluation (`Hit Rate@k`, `MRR`, Faithfulness LLM Judge, 3-configuration Pandas ablation table) |
| **13** | [`notebooks/13_hybrid_search_and_memory.ipynb`](notebooks/13_hybrid_search_and_memory.ipynb) | Hybrid **BM25 + Dense RRF** search for exact error codes & multi-turn conversational query rewriting |
| **14** | [`notebooks/14_production_safety_and_security.ipynb`](notebooks/14_production_safety_and_security.ipynb) | SHA-256 `CachedEmbeddings` speedup, token budgets, indirect prompt injection defense, pre-retrieval **RBAC** filtering |
| **15** | [`notebooks/15_capstone_project.ipynb`](notebooks/15_capstone_project.ipynb) | Complete production-hardened RAG system evaluated across all 14 benchmark questions + Gradio UI preview |

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

# 4. Launch Jupyter Lab or run all notebooks automatically
jupyter lab notebooks/
# Or execute every notebook top-to-bottom via CLI:
python scripts/run_notebooks.py
```

### Launching the Capstone Web App
```bash
python app/app.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) to ask questions, toggle RBAC roles (`guest`, `employee`, `executive`), upload your own `.txt`/`.md`/`.pdf` files, and monitor live embedding cache hits.

---

## Switching Model Providers

Edit `.env` (or set `RAG_PROVIDER` in your shell) without changing a single line of notebook code:

| `RAG_PROVIDER` | Chat Model | Embeddings | API Key Required? |
|---|---|---|---|
| `local` *(default)* | `LocalGroundedChatModel` | `all-MiniLM-L6-v2` (fallback: `LocalHashEmbeddings`) | None (Free / Offline) |
| `ollama` | `ChatOllama` (`llama3.2`) | `all-MiniLM-L6-v2` | None (Local Ollama server) |
| `openai` | `ChatOpenAI` (`gpt-4o-mini`) | `OpenAIEmbeddings` (`text-embedding-3-small`) | `OPENAI_API_KEY` |
| `anthropic` | `ChatAnthropic` (`claude-3-5-sonnet-latest`) | Local SentenceTransformer | `ANTHROPIC_API_KEY` |

---

## Repository Layout

```text
rag-from-scratch-course/
├── .env.example                 # Template configuration (defaults to local free mode)
├── .gitignore
├── README.md
├── VIDEO_OUTLINE.md             # Recording guide for a 16-episode video playlist
├── requirements.txt
├── app/
│   ├── app.py                   # Interactive Gradio RAG app + custom document uploader + RBAC
│   └── README.md
├── data/
│   ├── company_handbook.txt     # Fictional Novasphere Dynamics policies, stipends & pricing
│   ├── product_faq.md           # Technical error codes, ROS2 support & ZeroGravity specs
│   ├── recent_changelog.md      # October 2026 release notes & API deprecation notices
│   ├── policy_brief.pdf         # Safety brief PDF (confidential executive audit clauses)
│   └── eval_questions.json      # 14 benchmark questions (numbers, multi-doc, facts, unanswerable)
├── notebooks/                   # Notebooks 00 through 15 with real executed outputs
├── scripts/                     # Notebook factory, PDF generator, and batch kernel runner
├── tests/                       # Comprehensive QA plan, pytest suites, security & perf logs
└── utils/
    ├── __init__.py
    ├── helpers.py               # Corpus loaders, pretty printers, HTML/secret sanitizers
    └── models.py                # Unified provider factory (`get_llm`, `get_embeddings`, caching)
```

