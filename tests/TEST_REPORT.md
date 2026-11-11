# Final End-to-End QA Report — `rag-from-scratch-course`

**Repository:** `https://github.com/rahat4528/rag-from-scratch-course` (`main` branch)  
**Test Environment:** Fresh isolated `.venv`, Windows, Python `3.13.14`, PyTorch `2.6.0+cpu`, `sentence-transformers 6.3.0` (`all-MiniLM-L6-v2`)  
**Overall Verdict:** **PASS** (24/24 automated `pytest` suites passing, 16/16 notebooks executed cleanly with real outputs, `14/14` Capstone benchmark score across 3 deterministic runs).

---

## Executive Summary of Coverage
- **Unit, Pipeline, Math, Security, App & E2E Tests (`tests/`):** **24 / 24 PASSED**
- **Jupyter Course Notebooks (`notebooks/00`–`15`):** **16 / 16 EXECUTED CLEANLY** (`410.21s` total; `15.19s` min, `68.37s` max — all comfortably under the 5-minute ceiling; every code cell executed cleanly with <= 16 lines and full 9-part educational structure verified)
- **Capstone 14-Question Grounded Benchmark (`data/eval_questions.json`):** **14 / 14 (100.0%)** across exact numbers, multi-document joins, technical steps, and all 3 unanswerable traps — consistent across 3 consecutive runs at `temperature=0`.
- **Web Application (`app/app.py`):** Verified empty/whitespace/5000-char/XSS query handling, custom `.txt`/`.md`/`.pdf` document ingestion & reset, RBAC (`guest` vs `employee` vs `executive`), indirect prompt injection neutralization, and real Gradio HTTP `200` startup (`127.0.0.1:7869`).

---

## Key Defects Discovered & Fixed at Root Cause
Five real bugs (`BUG-001` through `BUG-005`, documented in `tests/BUG_REGISTER.md`) were identified and resolved cleanly during testing:
1. **`BUG-003` (High) — Capstone Retrieval & Unanswerable Grounding (`8/14` -> `14/14`)**: Fixed heading/sub-bullet grouping in `LocalGroundedChatModel`, added non-brand focus-term refusal gating, and improved section-aware chunk separators (`chunk_size=640, chunk_overlap=140`) in `app/app.py` and NB15.
2. **`BUG-004` (Critical) — Windows PyTorch DLL Load Failure (`torch 2.15.0`)**: Diagnosed `OSError: [WinError 126]` on `shm.dll` / `vcruntime140_threads.dll` that was silently falling back to `LocalHashEmbeddings`; pinned `torch>=2.6.0,<2.9.0` in `requirements.txt`, added explicit `warnings.warn` fallback notices, and re-verified all 16 notebooks running true neural `all-MiniLM-L6-v2` embeddings.
3. **`BUG-001`, `BUG-002`, `BUG-005` (Low)**: Fixed builder string escapes, tightened 3 code cells to `<= 16` lines, and fixed continuous-prose overlap unit assertion.

---

## Final Release Checklist
- Clean git working tree (`.env`, `.venv/`, `__pycache__/`, `.pytest_cache/` excluded via `.gitignore`).
- Zero leaked secrets or broken internal links (`test_b10`, `test_b14` passed).
- See detailed tables in `tests/TEST_LOG.md`, `tests/BUG_REGISTER.md`, `tests/SECURITY_FINDINGS.md`, `tests/PERFORMANCE.md`, and `tests/MISSING_OR_INCOMPLETE.md`.

