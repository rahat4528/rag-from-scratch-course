# Architecture, Repository & Curriculum Decisions — `DECISIONS.md`

## 1. Repository Decisions
- **Repository**: `https://github.com/rahat4528/rag-from-scratch-course`
- **Purpose**: Complete, standalone 16-notebook Jupyter course and Gradio/FastAPI capstone web app for learning RAG From Scratch.
- **Provider Architecture**: Zero paid API keys required. Free offline neural embeddings (`all-MiniLM-L6-v2`) and local rerankers (`ms-marco-MiniLM-L-6-v2`) with deterministic fallback, switchable to OpenAI, Anthropic, or Ollama via `.env`.

---

## 2. Requirements Traceability Matrix (R1–R9)

| Req ID | Requirement Summary | Mapped Notebooks | Verification Suite |
|---|---|---|---|
| **R1** | Explain why standalone LLMs fail on private/recent info (cutoff, hallucination, context limits, fine-tuning cost) | `notebooks/00_why_rag.ipynb` | `test_b1_b2`, `test_b11_b14` |
| **R2** | Teach how to connect models to outside docs for grounded answers | `notebooks/00_why_rag.ipynb`, `notebooks/01_hello_rag.ipynb`, `notebooks/06_generation_and_prompting.ipynb` | `test_b4_b6` |
| **R3** | Step-by-step build: load -> split -> index -> search -> context prompt -> generate | `notebooks/01_hello_rag.ipynb` through `notebooks/06_generation_and_prompting.ipynb` | `test_b4_b6` |
| **R4** | Expand into advanced RAG (translation, routing, indexing, reranking, agents, eval, prod, capstone) | `notebooks/07_query_translation.ipynb` through `notebooks/15_capstone_project.ipynb` | `test_b4_b6`, `test_b11_b14` |
| **R5** | Beginner-friendly explanations, analogies, short cells (<=16 lines), plain English | All 16 notebooks (`00`–`15`) | `test_b5_notebooks.py` template audit |
| **R6** | Practical, runnable Jupyter notebooks (< 5 min each, fresh clone -> venv -> pip install) | All 16 notebooks (`00`–`15`) | `scripts/run_notebooks.py`, `test_b5` |
| **R7** | Current LangChain/LangGraph 1.x patterns verified against live docs (`llms.txt`) | `01`, `02`, `03`, `04`, `05`, `06`, `08`, `09`, `11`, `15` | `test_b1`, `test_b4_b6` zero-warning audit |
| **R8** | Video playlist companion with consistent template & chapter outlines | All 16 notebooks + `VIDEO_OUTLINE.md` | `test_b14` |
| **R9** | Delivered into clean GitHub repository, pushed & verified | Repository root | `test_b11_b14`, GitHub remote |

---

## 3. Technical & Compatibility Decisions

1. **PyTorch Versioning on Windows**: Pinned `torch>=2.6.0,<2.9.0` (installed `2.8.0+cpu`) to avoid missing `vcruntime140_threads.dll` / `WinError 126` issues present in newer PyTorch builds.
2. **Native Document Ingestion**: Standardized on native `pypdf.PdfReader` and direct `Document` instantiation rather than legacy loaders, eliminating all `DeprecationWarning` logs.
3. **Hybrid Search Math**: Hand-implemented Reciprocal Rank Fusion (RRF with $k=60$) alongside BM25 and vector stores to teach the exact mathematical mechanics transparently.
4. **Security Hardening**: Implemented strict pre-retrieval Role-Based Access Control (`guest`, `employee`, `executive`) and multi-stage indirect prompt injection filtering.

