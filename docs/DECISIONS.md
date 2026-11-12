# Architecture, Repository & Curriculum Decisions — `docs/DECISIONS.md`

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
| **R8** | Video playlist companion with consistent template & chapter outlines | All 16 notebooks + `docs/VIDEO_OUTLINE.md` | `test_b14` |
| **R9** | Delivered into clean GitHub repository, pushed & verified | Repository root | `test_b11_b14`, GitHub remote |

---

## 3. Technical & Compatibility Decisions

1. **PyTorch Versioning on Windows**: Pinned `torch>=2.6.0,<2.9.0` (installed `2.8.0+cpu`) to avoid missing `vcruntime140_threads.dll` / `WinError 126` issues present in newer PyTorch builds.
2. **Native Document Ingestion**: Standardized on native `pypdf.PdfReader` and direct `Document` instantiation rather than legacy loaders, eliminating all `DeprecationWarning` logs.
3. **Hybrid Search Math**: Hand-implemented Reciprocal Rank Fusion (RRF with $k=60$) alongside BM25 and vector stores to teach the exact mathematical mechanics transparently.
4. **Security Hardening**: Implemented strict pre-retrieval Role-Based Access Control (`guest`, `employee`, `executive`) and multi-stage indirect prompt injection filtering.
5. **FastAPI Module-Level Introspection**: Placed Pydantic request models (`QueryRequest`) at module scope in `app/app.py` to prevent stringified type evaluation issues under `from __future__ import annotations`.
6. **Ephemeral Test Ports**: Dynamic socket binding (`socket.bind(('127.0.0.1', 0))`) in Gradio launch tests to prevent port collision during CI/CD test runs.

---

## 4. Repository Consolidation & Migration Decision Matrix

Consolidation audit that migrated all unique assets from prior development branches/repos into the single canonical target `rahat4528/rag-from-scratch-course`:

| Item / File Path | Canonical Target Path | Decision | Reason & Evidence |
|---|---|---|---|
| `rag_from_scratch_1_to_4.ipynb` (Upstream) | `upstream/rag_from_scratch_1_to_4.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `fb34d4f...`). |
| `rag_from_scratch_5_to_9.ipynb` (Upstream) | `upstream/rag_from_scratch_5_to_9.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `e319c8e...`). |
| `rag_from_scratch_10_and_11.ipynb` (Upstream) | `upstream/rag_from_scratch_10_and_11.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `1a905ac...`). |
| `rag_from_scratch_12_to_14.ipynb` (Upstream) | `upstream/rag_from_scratch_12_to_14.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `bba9296...`). |
| `rag_from_scratch_15_to_18.ipynb` (Upstream) | `upstream/rag_from_scratch_15_to_18.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `523da41...`). |
| `README.md` (Original Upstream) | `upstream/README.md` | **MIGRATE** | Preserves original LangChain repository documentation and attribution. |
| Verification Scripts | `tests/verify_manifest.py`, `tests/scan_secrets.py` | **MIGRATE** | Critical QA tools verifying SHA-256 byte integrity and secret leakage across history. |
| Upstream Checksums | `tests/base_manifest.sha256` | **MIGRATE** | Canonical checksum list verifying upstream file fidelity. |
| Extra Repositories History | Annotated Tags | **MIGRATE** | Preserved in TARGET via tags `archive/v2-final` (@`8055a6e`) and `archive/v3-final` (@`772b8bb`). |
| Redundant Generator Scripts | Retired | **RETIRE** | `notebooks/` is the single source of truth; generators retired. |
| Duplicate App Dockerfile / README | Retired | **RETIRE** | Merged into root `Dockerfile` and root `README.md`. |
| Fragmented Test Markdown Files | Merged into `docs/TESTING.md` | **MERGE** | Single source of truth for all testing, bug, security, and performance documentation. |
