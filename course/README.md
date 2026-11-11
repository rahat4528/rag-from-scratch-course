# RAG From Scratch — Beginner Hands-On Notebook Course

Welcome! This course teaches **Retrieval-Augmented Generation (RAG)** from the ground up through **16 progressive, runnable Jupyter notebooks**, a private/fictional multi-format corpus, modern LangChain `1.x` / LangGraph patterns, and a deployable Gradio + FastAPI capstone web app.

> **Upstream Attribution:** This course is an additive extension built cleanly inside `course/` on top of [`langchain-ai/rag-from-scratch`](https://github.com/langchain-ai/rag-from-scratch) (base commit `1fdb7d0e4f1d3e731c8397332452715512f18667`; note: no `LICENSE` file was included in the upstream base repository at that commit). Every original file and notebook at the repository root remains untouched.

---

## Who This Course Is For
- Motivated Python beginners who know variables, loops, and functions — **zero prior experience** with embeddings, vector stores, or LangChain is assumed.
- Engineers who want runnable step-by-step foundations, honest failure demos, rigorous retrieval evaluations, and production security patterns (caching, RBAC, prompt-injection defense).

---

## Course Map (Notebooks `00`–`15`)

| # | Notebook | Core Topics | Difficulty | Video | Estimated Time |
|---|---|---|---|---|---|
| **00** | [`notebooks/00_why_rag.ipynb`](notebooks/00_why_rag.ipynb) | Knowledge cutoffs, hallucinations, context limits, stuffing vs fine-tuning vs RAG | Beginner | Ep 00 | ~10 min |
| **01** | [`notebooks/01_rag_big_picture.ipynb`](notebooks/01_rag_big_picture.ipynb) | Index vs Query time, plain-Python framework-free RAG, LangChain mapping | Beginner | Ep 01 | ~12 min |
| **02** | [`notebooks/02_loading_documents.ipynb`](notebooks/02_loading_documents.ipynb) | `Document`, `.txt`/`.md`/`.pdf`/folder/web loading, metadata, Unicode & SSRF safety | Beginner | Ep 02 | ~12 min |
| **03** | [`notebooks/03_splitting_text.ipynb`](notebooks/03_splitting_text.ipynb) | Bad splits vs overlap fixes, `RecursiveCharacterTextSplitter`, 100/300/800 experiment | Beginner | Ep 03 | ~12 min |
| **04** | [`notebooks/04_embeddings_and_vector_stores.ipynb`](notebooks/04_embeddings_and_vector_stores.ipynb) | Hand NumPy cosine similarity, 2D PCA plot, `InMemoryVectorStore`, JSON save/load | Beginner | Ep 04 | ~15 min |
| **05** | [`notebooks/05_retrieval.ipynb`](notebooks/05_retrieval.ipynb) | Scores, `as_retriever`, `k`, MMR diversity, metadata filters & failure modes | Beginner | Ep 05 | ~12 min |
| **06** | [`notebooks/06_generation_first_rag_chain.ipynb`](notebooks/06_generation_first_rag_chain.ipynb) | Inline grounded prompts, LCEL chains & 2-node LangGraph (`retrieve -> generate`) | Beginner | Ep 06 | ~15 min |
| **07** | [`notebooks/07_rag_with_your_own_files.ipynb`](notebooks/07_rag_with_your_own_files.ipynb) | Custom folder indexing, persisted reload, citations & 14-question benchmark | Beginner | Ep 07 | ~15 min |
| **08** | [`notebooks/08_query_translation.ipynb`](notebooks/08_query_translation.ipynb) | Multi-Query, RAG-Fusion (`RRF` hand proof), Decomposition, Step-Back, HyDE | Intermediate | Ep 08 | ~15 min |
| **09** | [`notebooks/09_routing_and_query_construction.ipynb`](notebooks/09_routing_and_query_construction.ipynb) | Pydantic v2 logical routing, semantic routing, metadata filter queries & logs | Intermediate | Ep 09 | ~12 min |
| **10** | [`notebooks/10_advanced_indexing.ipynb`](notebooks/10_advanced_indexing.ipynb) | Parent-doc, Multi-Rep, Hybrid BM25+Dense, demo-scale RAPTOR & ColBERT MaxSim | Intermediate | Ep 10 | ~15 min |
| **11** | [`notebooks/11_reranking_and_compression.ipynb`](notebooks/11_reranking_and_compression.ipynb) | Wide fetch -> local `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) + compression | Intermediate | Ep 11 | ~15 min |
| **12** | [`notebooks/12_agentic_and_self_correcting_rag.ipynb`](notebooks/12_agentic_and_self_correcting_rag.ipynb) | `langchain.agents.create_agent` tool agent + capped LangGraph CRAG/Self-RAG | Intermediate | Ep 12 | ~15 min |
| **13** | [`notebooks/13_evaluating_rag.ipynb`](notebooks/13_evaluating_rag.ipynb) | Hit Rate@k, MRR hand math check, Faithfulness judge limits & 4-config ablation | Advanced | Ep 13 | ~15 min |
| **14** | [`notebooks/14_production_and_safety.ipynb`](notebooks/14_production_and_safety.ipynb) | ID-based doc retirement, caching, injection defense, RBAC bypass tests, when NOT to use | Advanced | Ep 14 | ~15 min |
| **15** | [`notebooks/15_capstone_project.ipynb`](notebooks/15_capstone_project.ipynb) | End-to-end persisted Hybrid+CrossEncoder+RBAC system + `course/app/app.py` | Capstone | Ep 15 | ~15 min |

---

## Quickstart Setup (Python 3.10+)

```bash
# 1. Clone and enter the repository
git clone https://github.com/rahat4528/rag-from-scratch-course-v2.git
cd rag-from-scratch-course-v2

# 2. Create a clean virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install verified pinned dependencies (minimum Python 3.10; tested on 3.13)
pip install -r course/requirements.txt

# 4. Configure your model provider
cp course/.env.example course/.env
```

### Provider Paths (`RAG_PROVIDER` inside `.env`)
- **Free Local Path (Recommended for learners):** Set `RAG_PROVIDER=ollama`, install [Ollama](https://ollama.com) (`ollama pull llama3.2`), and embeddings run completely free & locally via HuggingFace `all-MiniLM-L6-v2`!
- **Hosted Paths:** Set `RAG_PROVIDER=openai` (with `OPENAI_API_KEY`) or `RAG_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`; paired automatically with local sentence-transformers embeddings).
- **Offline Code-Path Demo Mode (`offline_demo`):** If neither Ollama nor paid API keys exist on your machine, `RAG_PROVIDER=offline_demo` (and notebooks' `fallback_to_demo=True`) runs every retrieval/neural-rerank cell cleanly while marking chat responses explicitly as `[OfflineDemoChatModel]`.

### Launching Jupyter & Capstone App
```bash
# Launch Jupyter notebooks:
jupyter lab course/notebooks/

# Run all 16 notebooks top-to-bottom automatically:
python course/utils/run_notebooks.py

# Launch the Capstone Gradio Web App + FastAPI server:
python course/app/app.py
```

---

## OS & Compatibility Notes
- **Windows:** `course/requirements.txt` pins `torch>=2.6.0,<2.9.0` to guarantee smooth `torch` / `sentence-transformers` DLL initialization without requiring newer VS 2022 `vcruntime140_threads.dll` builds. Persistence uses pure-Python JSON (`InMemoryVectorStore.dump/load`) instead of native Chroma binaries to avoid Windows Rust extension crashes.
- **macOS / Linux:** Standard `.venv/bin/activate` works out of the box; all paths use OS-neutral `pathlib.Path` with explicit `encoding="utf-8"`.
- **Google Colab:** Every notebook includes a top-cell git clone + `pip install` helper comment ready to uncomment.

---

## Troubleshooting
- *`No module named 'course'`*: Launch Python/Jupyter from repository root, or run the Setup cell first (it automatically resolves the repo root and appends it to `sys.path`).
- *Ollama ConnectionError*: Start Ollama (`ollama serve`) and ensure `ollama pull llama3.2` completed, OR set `RAG_PROVIDER=offline_demo` in `course/.env` to test code paths immediately.
- *Downloading HuggingFace models slowly offline*: Once `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2` cache once, subsequent runs work completely offline. Set `RAG_OFFLINE=1` to skip online checks cleanly.

---

## Beginner Glossary
- **Embedding**: A fixed-length coordinate array (`384` numbers here) representing the meaning of text — like GPS coordinates on a giant semantic map.
- **Chunk**: A focused ~300–650 character slice of a larger document small enough for accurate lookup without drowning embeddings in mixed topics.
- **Retriever**: Any component returning top-*k* relevant `Document` chunks given a user query.
- **Vector Store**: A searchable catalog storing document chunks paired with their numeric embedding coordinates.
- **Reranker**: A deeper Stage-2 reader (e.g., `CrossEncoder`) that scores query + candidate text jointly to refine initial top-k ranking.
- **Grounding**: Restricting model generation strictly to facts explicitly verified in retrieved source passages, refusing gaps honestly.
- **Hallucination**: Confident fabrication by an LLM when answering without verified supporting context.
- **Context Window**: The maximum amount of prompt + document text a model can process in one turn — like the physical size of an exam desk.
