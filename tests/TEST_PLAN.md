# Master QA Test Plan — `rag-from-scratch-course`

## Scope
Complete end-to-end QA, security audit, mathematical verification, performance benchmarking, and documentation audit across all course assets in the standalone `rag-from-scratch-course` repository.

## Test Areas (B1 – B14)

| Area | Scope & Focus | Test Asset |
|---|---|---|
| **B1** | Fresh `.venv` installation, `pip check`, zero missing imports | `tests/test_b1_b2_env_corpus.py` |
| **B2** | Corpus file existence, UTF-8 integrity, PDF extraction, `eval_questions.json` schema & strict ground truth coverage | `tests/test_b1_b2_env_corpus.py` |
| **B3** | Shared helpers (`get_llm`, `get_embeddings`, root resolution, auth error sanitization, HTML escaping) | `tests/test_b3_helpers_auth.py` |
| **B4** | Loaders (TXT, MD, PDF, HTML, Unicode/spaces), splitters (overlap & tiny docs), embeddings, vector store dump/load, MMR | `tests/test_b4_b6_pipeline_math.py` |
| **B5** | All 16 notebooks (`00`–`15`) kernel execution, zero cell errors, template compliance, <=16-line code cells, size <1MB | `tests/test_b5_notebooks.py` |
| **B6** | Mathematical & algorithmic checks (hand RRF = 1/61+1/62, MRR = 0.5, Faithfulness judge vs hallucination, BM25+Dense hybrid) | `tests/test_b4_b6_pipeline_math.py` |
| **B7 & B8** | Gradio `NovasphereRAGEngine` & UI: empty/whitespace/5000-char/XSS queries, custom file upload/reindex/reset, HTTP server check | `tests/test_b7_b10_app_security.py` |
| **B9 & B10** | Security audit: RBAC (`guest`/`employee`/`executive`), prompt injection defense, secret scan, `pip-audit` | `tests/test_b7_b10_app_security.py` |
| **B11** | Performance: embedding cache hit speedup, 30-query P50/P95 latency, 1x/10x/50x indexing scale | `tests/test_b11_b14_perf_e2e.py` |
| **B12** | End-to-end learner workflows (fresh clone, custom file upload, update/reindex, local & error paths) | `tests/test_b7_b10_app_security.py`, `tests/test_b11_b14_perf_e2e.py` |
| **B13** | Grounded accuracy gate over all 14 `eval_questions.json` benchmark questions (3 repeats at temp=0) | `tests/test_b11_b14_perf_e2e.py` |
| **B14** | Repository hygiene: zero TODO/FIXME/NotImplementedError, all Markdown links resolve on disk | `tests/test_b11_b14_perf_e2e.py` |

