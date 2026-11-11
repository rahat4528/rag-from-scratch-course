# Architecture, Repository & Curriculum Decisions — `course/DECISIONS.md`

## 1. Repository & Git Decisions
- **Upstream Reference Repository**: Cloned from `https://github.com/langchain-ai/rag-from-scratch` (`main` branch) into an isolated directory outside the new repository (`c:\project 03\rag-from-scratch-reference`).
- **Source Commit SHA (Reference)**: `1fdb7d0e4f1d3e731c8397332452715512f18667`
- **New Repository Base Commit SHA**: `c8cbba13cca63d581d3a2cd97c2f0480dc05b4a2` ("Initial import of base content from langchain-ai/rag-from-scratch @1fdb7d0e4f1d3e731c8397332452715512f18667")
- **Target Private Repository**: `https://github.com/rahat4528/rag-from-scratch-course-v3` (created via `gh repo create --private` with fresh history).
- **Remote Policy**: The new repository has ONLY its own `origin` remote. No other remote is configured, preventing any accidental push to the original project.
- **Working Branch**: `beginner-rag-course` branched off `main`; merged cleanly into `main` with `--no-ff` merge commit and tagged `v1.0-course` after Phase 7 verification.
- **Immutability Guarantee**: All base files recorded in `course/tests/base_manifest.sha256` remain byte-for-byte identical, with ONE link section appended to root `README.md`.

---

## 2. Requirements Traceability Matrix (R1–R9)

| Req ID | Requirement Summary | Mapped Notebooks | Verification Phase / Suite |
|---|---|---|---|
| **R1** | Explain why standalone LLMs fail on private/recent info (cutoff, hallucination, context window, fine-tuning cost/recall) | `00_why_rag` | `B5`, `B13`, `B14` |
| **R2** | Teach how to connect models to outside docs for grounded answers | `00_why_rag`, `01_rag_big_picture`, `06_generation_first_rag_chain` | `B4`, `B5`, `B13` |
| **R3** | Step-by-step build: load -> split -> index -> search -> context prompt -> generate | `01_rag_big_picture` through `07_rag_with_your_own_files` | `B4`, `B5`, `B12 (W1, W2)` |
| **R4** | Expand into advanced RAG (translation, routing, indexing, reranking, agents, eval, prod, capstone) | `08_query_translation` through `15_capstone_project` | `B5`, `B6`, `B13` |
| **R5** | Beginner-friendly explanations, analogies, short cells (<=15 lines), plain English | All 16 notebooks (`00`–`15`) | `B5` template & cell-length audit |
| **R6** | Practical, runnable Jupyter notebooks (< 5 min each, fresh clone -> venv -> `pip install`) | All 16 notebooks (`00`–`15`) | `B1`, `B5`, `B11`, `Phase 7` |
| **R7** | Current LangChain/LangGraph 1.x patterns verified against live docs (`llms.txt`) | `02`, `03`, `04`, `05`, `06`, `08`, `09`, `11`, `12`, `15` | `B1`, `B5`, `C9` zero-warning audit |
| **R8** | Video playlist feel with consistent template & chapter outlines | All 16 notebooks + `course/VIDEO_OUTLINE.md` | `B5`, `B14` |
| **R9** | Delivered in brand-new private GitHub repo seeded from upstream base clone, pushed & tagged | Git repo root + `course/` | `B12 (W8)`, `B15`, `Phase 8` |

---

## 3. Complete 16-Notebook Curriculum Plan

1. `00_why_rag.ipynb` — Closed-book vs open-book intuition, private company test, context window limits, comparison table.
2. `01_rag_big_picture.ipynb` — End-to-end architecture diagram + pure-Python framework-free mini RAG + LangChain mapping.
3. `02_loading_documents.ipynb` — `Document` abstraction, `.txt`/`.md`, PDF (`pypdf`), folder batch load, web loader + offline fallback, messy PDFs/encoding.
4. `03_splitting_text.ipynb` — Naive split failure vs overlap fix, `RecursiveCharacterTextSplitter`, 100/300/800 experiment table, metadata preservation.
5. `04_embeddings_and_vector_stores.ipynb` — Manual NumPy cosine similarity + 2D Matplotlib plot, real neural embeddings, `InMemoryVectorStore`, local JSON persistence.
6. `05_retrieval.ipynb` — `similarity_search`, scores, `as_retriever`, `k` choice, MMR, metadata filters, failure modes.
7. `06_generation_first_rag_chain.ipynb` — Inline grounded prompt, LCEL runnable chain, two-node LangGraph rebuild, with/without retrieval comparison + citations.
8. `07_rag_with_your_own_files.ipynb` — Mini capstone indexing custom folder, save/reload index, >=10/14 eval benchmark check + troubleshooting checklist.
9. `08_query_translation.ipynb` — Multi-Query, RAG-Fusion (hand-worked RRF), Decomposition, Step-Back, HyDE + baseline comparison & cost table.
10. `09_routing_and_query_construction.ipynb` — Logical routing (Pydantic v2 structured), semantic embedding routing, filter query construction, logging & failures.
11. `10_advanced_indexing.ipynb` — Parent-document, multi-representation summaries, BM25+Vector hybrid search, small RAPTOR tree demo, ColBERT MaxSim concept demo.
12. `11_reranking_and_compression.ipynb` — Wide fetch -> local neural `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) + contextual compression + latency/precision table.
13. `12_agentic_and_self_correcting_rag.ipynb` — Retrieval tool via modern `create_agent` + stateful LangGraph CRAG/Self-RAG grading, query rewrite loop with cap & logs.
14. `13_evaluating_rag.ipynb` — Separate retrieval (`Hit Rate@k`, `MRR` + hand-checked proof) & generation eval, faithfulness judge caveats, real ablation table.
15. `14_production_and_safety.ipynb` — Incremental updates, deleting docs by id, SHA-256 embedding cache, latency/cost, prompt injection mitigation, RBAC filter isolation, when NOT to use RAG.
16. `15_capstone_project.ipynb` — Unified "Chat with your docs" pipeline from saved index, full eval run, Gradio web UI (`course/app/`), next-steps docs roadmap.

---

## 4. Technical & Compatibility Decisions (with Reasons)
- **Zero Deprecated `langchain_community` Imports**: Consulted `https://docs.langchain.com/llms.txt` (`langchain/rag.md`, `langchain/knowledge-base.md`, `langchain/agents.md`, `langgraph/agentic-rag.md`). Replaced deprecated `langchain_community.document_loaders` with native `Path.read_text(encoding='utf-8')`, `pypdf.PdfReader`, and `BeautifulSoup` returning standard `langchain_core.documents.Document` objects, eliminating `langchain-community` sunset warnings cleanly.
- **Chat Provider Abstraction (`init_chat_model` / `ChatOllama` + Explicit `offline_demo` Mode)**: Supports OpenAI (`init_chat_model`), Anthropic (`init_chat_model`), and local Ollama (`ChatOllama`), plus a transparently labeled `OfflineDemoChatModel` code-path runner when no hosted key or Ollama server is available (explicitly kept separate from real neural embedding quality assertions).
- **Local Neural Embeddings & Reranker (`torch>=2.6.0,<2.9.0` + `sentence-transformers`)**: Runs genuine local `all-MiniLM-L6-v2` embeddings and `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking. Pinned `torch<2.9.0` because `torch 2.15.0` fails to import `shm.dll` / `vcruntime140_threads.dll` (`WinError 126`) on Windows machines without VS 2022 14.40+ redistributables.
- **Vector Store Persistence (`InMemoryVectorStore.dump/load` JSON wrapper)**: Pure-Python JSON persistence avoids C++/Rust native Windows crashes observed with Chromadb on Python 3.13 while supporting `.add_documents`, `.delete(ids=...)`, metadata filter functions, and full index round-trip integrity.
- **Capstone Web App (`Gradio` on `127.0.0.1:7860`)**: Chosen for accessible single-command launch, upload/re-index/delete controls, mobile responsiveness, loading/error/empty states, XSS escaping, and easy Dockerization.

