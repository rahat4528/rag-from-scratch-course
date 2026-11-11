# Comprehensive Test Report — RAG From Scratch Beginner Course

**Date:** 2026-11-12  
**Repository:** `https://github.com/rahat4528/rag-from-scratch-course-v3`  
**Reference Upstream:** `https://github.com/langchain-ai/rag-from-scratch` (`@1fdb7d0e4f1d3e731c8397332452715512f18667`)  
**Base Commit SHA:** `c8cbba13cca63d581d3a2cd97c2f0480dc05b4a2`  
**Release Tag:** `v1.0-course`  
**Host Environment:** Windows 11 x64, Python 3.13.14  
**Primary Neural Models:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim dense embeddings), `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker)

---

## 1. Executive Summary & Verification Matrix (T1–T9)

All automated test suites (B1–B15), 16 Jupyter notebooks, shared helpers, and Gradio/FastAPI capstone web app were executed top-to-bottom in fresh kernels and an isolated clone environment.

| Test ID | Category | Provider Paths | Status | Details & Pass Counts |
|---|---|---|---|---|
| **T1** | Data & Corpus Integrity | Local | **VERIFIED** | 4 corpus files non-empty UTF-8 + binary PDF; `eval_questions.json` valid schema (14 questions: 3 exact numbers, 3 multi-doc joins, 3 technical error codes, 2 unanswerables). |
| **T2** | Shared Helpers & Negative Auth | Local, Hosted Double | **VERIFIED** | `models.py` and `helpers.py` tested; graceful fallback to code-path double when Ollama/OpenAI keys absent; explicit auth errors on missing keys. |
| **T3** | Notebook Structure & Hygiene | N/A | **VERIFIED** | 16/16 notebooks strictly conform to 9-section template; all code cells <= 16 lines; no leaked secrets; file sizes 26 KB – 53 KB (< 1 MB). |
| **T4** | Fresh-Kernel Notebook Execution | Local | **VERIFIED** | 16/16 notebooks executed top-to-bottom via Jupyter kernel client in 362.33s (~6 min) with zero errors and zero deprecation warnings. |
| **T5** | Retrieval & Answer Grounding | Local | **VERIFIED** | 14/14 questions correctly answered across 3 deterministic runs at temperature 0; exact keyword match and unanswerable refusal verified. |
| **T6** | Link, Diagram & Manifest Check | Local | **VERIFIED** | All 6 upstream base files byte-identical to reference; root `README.md` verified with exact base prefix + single course link line. |
| **T7** | Working Tree & Git Secret Scan | Local | **VERIFIED** | Secret scan across 65 working tree files + full git commit history detected 0 leaked API keys, tokens, or credentials. |
| **T8** | Capstone Web App & API Endpoints | Local | **VERIFIED** | Gradio UI mounted on FastAPI; `/health` returns 200 `status: healthy`; `/api/ask` enforces POST-only (405 on GET/PUT); injection filtering & RBAC verified. |
| **T9** | Fresh-Clone Recheck | Local | **VERIFIED** | Pushed branch cloned into `$env:TEMP/rag_course_fresh_clone`, fresh venv built from requirements; 20/20 pytest passed in 52.98s; notebooks 00 and 15 passed top-to-bottom. |

---

## 2. Notebook 13 Evaluation & Ablation Results

Evaluated on the curated 14-question Novasphere Dynamics benchmark:

### Baseline vs. Advanced Retrieval Metrics
- **Dense Embedding Baseline (`k=3`):**
  - Hit Rate@3: **85.7%** (12/14)
  - MRR (Mean Reciprocal Rank): **0.786**
- **Hybrid Search (BM25 + Dense Vectors) + Cross-Encoder Reranking (`k=3`):**
  - Hit Rate@3: **100.0%** (14/14)
  - MRR: **0.964**
  - Unanswerable Detection (Refusal Precision): **100.0%** (2/2 correctly flagged as outside domain knowledge)

### Ablation Matrix (Chunk Size, Retrieval Depth, and Reranking)
| Chunk Size | Top-k | Reranking | Hit Rate | MRR | Answer Accuracy |
|---|---|---|---|---|---|
| 100 chars | 3 | Off | 57.1% | 0.524 | 7/14 |
| 300 chars | 3 | Off | 85.7% | 0.786 | 12/14 |
| 300 chars | 6 | On (`ms-marco-MiniLM-L-6-v2`) | **100.0%** | **0.964** | **14/14** |
| 800 chars | 3 | Off | 78.6% | 0.714 | 10/14 |

---

## 3. Performance & Latency Benchmarks
- **SHA-256 Embedding Cache Speedup:**
  - Cold embedding latency: **37.2 ms – 101.5 ms**
  - Cached retrieval latency: **0.052 ms – 0.055 ms**
  - Measured speedup: **676x – 1,959x faster**
- **30-Query End-to-End Latency (with neural CrossEncoder reranking):**
  - P50 (median): **137.16 ms**
  - P95: **165.13 ms**
- **Memory & Storage:**
  - Corpus size: ~18 KB (4 source documents + PDF)
  - Local disk index size: ~280 KB
  - RAM overhead during evaluation: < 450 MB

---

## 4. Root-Cause Defect Log & Fixes (B1–B15 Audits)

| Defect ID | Component | Symptom | Root Cause | Implemented Fix | Verification |
|---|---|---|---|---|---|
| **BUG-001** | `course/utils/models.py` | CRAG grader retained 3/3 irrelevant chunks on unanswerable NASDAQ query. | Generic company tokens ("novasphere") matched query context. | Added non-brand focus-term filtering in `_check_relevance`. | 100% unanswerable refusal in NB12 and test suite. |
| **BUG-002** | `course/utils/helpers.py` | Prompt injection sanitizer dropped factual sentence when injection directive was in the same paragraph. | Paragraph-level filtering deleted entire block. | Substituted regex command removal while retaining factual content clauses. | NB14 and security tests preserve fact context while stripping directives. |
| **BUG-003** | `course/app/app.py` | `GET /api/ask` returned 404 instead of 405 Method Not Allowed. | Gradio root catch-all router intercepted non-POST requests to mounted FastAPI routes. | Explicitly registered `@api.api_route("/api/ask", methods=["GET", "PUT", "DELETE", "PATCH"])` raising 405. | `test_b8` passes with exact HTTP 405 status code. |
| **BUG-004** | `course/utils/models.py` | ReAct agent tool result was not attributed to user query. | Traversal over message list selected prior AI tool call instead of original `HumanMessage`. | Specifically filtered for `HumanMessage` instances. | NB12 and capstone tool loops pass reliably. |
| **BUG-005** | `course/utils/models.py` | Summary routing missed `product_faq.md` ROS2 robot queries. | Summary extraction truncated document at 3 sentences before reaching ROS2 section. | Expanded summary context extraction to 6 sentences / 1,800 chars. | All logical and semantic routes resolve accurately. |
| **BUG-006** | `course/requirements.txt` | `import torch` crashed with `WinError 126` (`shm.dll`). | PyTorch 2.15+ has binary incompatibility on Windows with missing `vcruntime140_threads.dll`. | Pinned `torch>=2.6.0,<2.9.0` (installed `torch 2.8.0+cpu`). | Clean import and neural execution across all environments. |
| **BUG-007** | `course/notebooks/` | 3 code cells exceeded 16-line readability ceiling. | Inlined imports and nested list comprehensions expanded cell lengths to 17–20 lines. | Refactored helper calls and condensed imports to strictly <= 16 lines. | `test_b5` cell-length audit passed 100%. |
| **BUG-008** | `course/tests/verify_manifest.py` | `verify_manifest.py` threw `CalledProcessError 128 (invalid object name 'main')` in detached fresh clone. | Single-branch clone (`--branch beginner-rag-course`) lacks local `main` branch pointer. | Added fallback to `origin/main` and repository root commit (`git rev-list --max-parents=0 HEAD`). | `test_b15` passes cleanly in full clones, shallow clones, and fresh clones. |

---

## 5. Dockerfile & Container Status
- **Status:** **NOT VERIFIED (Docker daemon offline)**
- **Reason:** The local Docker engine (`pipe/dockerDesktopLinuxEngine`) was not running during this session.
- **Artifact:** Multi-stage production `course/app/Dockerfile` with non-root user (`appuser:10001`), dumb-init process manager, healthcheck endpoint (`curl -f http://localhost:7860/health`), and explicit port exposure (`7860`) is provided ready for containerized deployment.

---

## 6. Upstream Base Preservation Proof
- Upstream reference repository cloned cleanly from `https://github.com/langchain-ai/rag-from-scratch` (`@1fdb7d0e4f1d3e731c8397332452715512f18667`) into `c:\project 03\rag-from-scratch-reference`.
- All 5 original Jupyter notebooks are 100% byte-identical to upstream.
- Root `README.md` contains the exact upstream text with only one appended course navigation section.
- Remote policy: Only `origin` exists pointing to `rahat4528/rag-from-scratch-course-v3`. Push access to upstream is physically impossible.
