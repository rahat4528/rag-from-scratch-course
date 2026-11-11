# QA Test Execution Log — `rag-from-scratch-course`

Chronological log strictly matching real command runs and printed metrics in this session.

| Run ID | Area | Exact Command | Outcome | Observed Measurements |
|---|---|---|---|---|
| RUN-01 | Environment (B1) | `.\.venv\Scripts\python.exe -m pip check` | PASS | `No broken requirements found.` (before and after `torch==2.6.0` + `pip==26.3`) |
| RUN-02 | NB Batch Run #1 (pre-fix) | `.\.venv\Scripts\python.exe scripts/run_notebooks.py` | PARTIAL (15/16) | NB00–NB14 passed (`6.52s`–`23.14s`); NB15 failed assertion (`8/14`), logged as `BUG-003` |
| RUN-03 | `pip-audit` Scan #1 | `.\.venv\Scripts\python.exe -m pip_audit` | FINDINGS | Identified `pip 26.1.2` (`PYSEC-2026-3721`) and `pillow 12.3.0` (`gradio<13.0` upstream pin); upgraded `pip` -> `26.3` |
| RUN-04 | `pytest` Suite #1 | `.\.venv\Scripts\python.exe -m pytest tests/ -v -s` | PARTIAL (22/24) | Exposed `BUG-004` (`torch 2.15.0` `shm.dll` WinError 126) and `BUG-005` (`test_b4_splitter_boundaries_and_tiny_doc`) |
| RUN-05 | Torch 2.6.0 Verification | `.\.venv\Scripts\python.exe -c "import torch, sentence_transformers; ..."` | PASS | `No broken requirements found.` and `TORCH+ST OK: 2.6.0+cpu 6.3.0` |
| RUN-06 | `pytest` Suite #2 (Post-Fix) | `.\.venv\Scripts\python.exe -m pytest tests/ -v -s` | PASS (24/24) | `Cold=99.7ms`, `Warm=0.052ms (1918x)`, `P50=2.21ms`, `P95=11.17ms`, Capstone 3x deterministic score `14/14 (100.0%)`, Gradio HTTP `200` on `127.0.0.1:7869` |
| RUN-07 | NB Batch Run #2 (Neural `MiniLM`) | `.\.venv\Scripts\python.exe scripts/run_notebooks.py` | PASS (16/16) | All 16 notebooks passed cleanly in `410.21s` total (`15.19s`–`68.37s` per notebook) with real `all-MiniLM-L6-v2` embeddings; NB15 `14/14 (100.0%)`; NB12 Faithfulness verified |
| RUN-08 | `pip-audit` Scan #2 | `.\.venv\Scripts\python.exe -m pip_audit` | FINDINGS (Mitigated/Tracked) | `pip 26.3` clean (0 issues); tracks `pillow 12.3.0` (`gradio` upstream `<13.0` pin) and `torch 2.6.0` (details in `SECURITY_FINDINGS.md`) |
