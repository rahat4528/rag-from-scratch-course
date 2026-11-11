# Master QA Test Plan — `course/tests/TEST_PLAN.md`

## Scope & Inventory
- **Notebooks (`00`–`15`)**: All 16 notebooks implementing R1–R9, verified via `run_notebooks.py` + `pytest`.
- **Shared Modules**: `course/utils/models.py`, `course/utils/helpers.py`, `course/utils/generate_pdf.py`.
- **Corpus & Data**: `company_handbook.txt`, `product_faq.md`, `recent_changelog.md`, `policy_brief.pdf`, `eval_questions.json` (14 questions).
- **Web App & Deployment**: `course/app/app.py`, `course/app/README.md`, `course/app/Dockerfile`.
- **Docs & Repo State**: `course/README.md`, `course/VIDEO_OUTLINE.md`, `course/DECISIONS.md`, `course/KNOWN_ISSUES.md`, root `README.md`, `base_manifest.sha256`.

## Planned Test Suites & Areas (B1–B15)
| Area | Priority | Pre-Declared Expected Result | Test File |
|---|---|---|---|
| **B1** Environment & dependencies | Must-Have | `pip check` exits 0; all imports clean with 0 deprecation warnings; `.env.example` & Colab setup valid | `test_b1_b2_env_data.py` |
| **B2** Data & corpus | Must-Have | All 5 data files non-empty, UTF-8; PDF extracts text; `eval_questions.json` (14 Qs) grounded with zero leaks in unanswerables | `test_b1_b2_env_data.py` |
| **B3** Shared helpers & auth config | Must-Have | `get_llm`/`get_embeddings`/`get_cross_encoder` work cleanly; invalid/missing keys & unreachable servers raise clean secret-free errors; printers handle empty/giant strings | `test_b3_b4_helpers_pipeline.py` |
| **B4** Core pipeline + edge cases | Must-Have | Txt/MD/PDF/Folder/Web(SSRF+fallback), Unicode/spaces, tiny/bad-overlap split, VectorStore add/search/k>N/dump/load/delete/filter/MMR, LangGraph loop caps pass | `test_b3_b4_helpers_pipeline.py` |
| **B5** Notebooks `00`–`15` | Must-Have | All 16 pass fresh kernel run in `<300s`, `<1MB`, `<=16` lines/code-cell, full 9-part template, 0 errors/deprecation warnings/secrets | `test_b5_notebooks.py` |
| **B6** Advanced techniques & math | Must-Have | Hand RRF matches code (`1/61+1/62`), hand MRR (`0.500`), CrossEncoder beats naive lexical on exact errors, Hybrid BM25+Dense & LangGraph CRAG terminate cleanly | `test_b6_advanced_math.py` |
| **B7 & B8** App frontend & backend | Must-Have | Empty/whitespace/oversized/emoji/XSS/no-index/missing-model handled safely; custom upload/reindex/reset/withdraw work; live HTTP `200` on Gradio endpoints | `test_b7_b10_app_security.py` |
| **B9** Auth & RBAC isolation | Must-Have | Missing/placeholder keys blocked; `guest`/`employee` cannot see `executive` `policy_brief.pdf`; filter injection bypass in question rejected | `test_b7_b10_app_security.py` |
| **B10** Security audit | Must-Have | Git history & tree free of secrets; `.env` gitignored; SSRF blocks `localhost`/`169.254.169.254`/`file://`; HTML escaped; `pip-audit` documented cleanly | `test_b7_b10_app_security.py` |
| **B11** Performance benchmarks | Must-Have | Every NB `<300s`; 30 queries P50/P95 measured (`baseline`, `reranked`, `agentic`); cache speedup `>20x`; 1x/10x/100x indexing profiled; 10 parallel requests succeed | `test_b11_b15_perf_repo.py` |
| **B12** End-to-end workflows (W1–W8) | Must-Have | W1–W3 & W5–W8 automated checks PASS; W4 (`openai`/`anthropic` live keys) marked `NOT RUN` with environment reason | `test_b11_b15_perf_repo.py` |
| **B13** Output correctness (14 eval Qs) | Must-Have | Real retrieval hits & grounded code-path answers pass `>=12/14` reliably across 3 deterministic runs (`temp=0`) | `test_b11_b15_perf_repo.py` |
| **B14** Missing/incomplete audit | Must-Have | R1–R9 mapped; zero TODO/FIXME/NotImplementedError; all Markdown links & README commands verified | `test_b11_b15_perf_repo.py` |
| **B15** Upstream intact check | Must-Have | `verify_manifest.py` confirms 100% SHA-256 byte equality of all 6 base files (only 1-line link addition permitted in `README.md`) | `test_b11_b15_perf_repo.py` |

