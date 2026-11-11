# Comprehensive Repository Migration Map — `MIGRATION_MAP.md`

This document records the item-by-item migration audit consolidating all unique assets, features, notebooks, and documentation from `rahat4528/rag-from-scratch-course-v2` (`EXTRA 1`, `@8055a6e`) and `rahat4528/rag-from-scratch-course-v3` (`EXTRA 2`, `@772b8bb`) into the primary repository: **`rahat4528/rag-from-scratch-course` (`TARGET`, `@6faa6b0`)**.

---

## Migration Decision Matrix

| Item / File Path in Extra Repos | Target Path in `rag-from-scratch-course` | Decision | Reason & Evidence |
|---|---|---|---|
| `rag_from_scratch_1_to_4.ipynb` (Upstream) | `upstream/rag_from_scratch_1_to_4.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `fb34d4f...`). |
| `rag_from_scratch_5_to_9.ipynb` (Upstream) | `upstream/rag_from_scratch_5_to_9.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `e319c8e...`). |
| `rag_from_scratch_10_and_11.ipynb` (Upstream) | `upstream/rag_from_scratch_10_and_11.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `1a905ac...`). |
| `rag_from_scratch_12_to_14.ipynb` (Upstream) | `upstream/rag_from_scratch_12_to_14.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `bba9296...`). |
| `rag_from_scratch_15_to_18.ipynb` (Upstream) | `upstream/rag_from_scratch_15_to_18.ipynb` | **MIGRATE** | Preserves original LangChain course notebook byte-for-byte (SHA256: `523da41...`). |
| `README.md` (Original Upstream) | `upstream/README.md` | **MIGRATE** | Preserves original LangChain repository documentation and attribution. |
| `course/tests/verify_manifest.py` | `tests/verify_manifest.py` | **MIGRATE** | Critical QA tool that verifies SHA-256 byte integrity of all upstream reference files. |
| `course/tests/scan_secrets.py` | `tests/scan_secrets.py` | **MIGRATE** | Security auditing script scanning full git commit history and working tree for leaks. |
| `course/tests/base_manifest.sha256` | `tests/base_manifest.sha256` | **MIGRATE** | Canonical checksum list verifying upstream file fidelity. |
| `course/DECISIONS.md` | `DECISIONS.md` | **MIGRATE** | Master requirements traceability matrix (R1–R9) and technical architecture log. |
| `course/KNOWN_ISSUES.md` | `KNOWN_ISSUES.md` | **MIGRATE** | Documents upstream deprecations and PyTorch/Pillow Windows compatibility mitigations. |
| `course/TEST_REPORT.md` | `tests/TEST_REPORT.md` | **MIGRATE** | Exhaustive performance, latency, caching, and determinism benchmark report. |
| `course/app/Dockerfile` | `app/Dockerfile` & `Dockerfile` | **MIGRATE** | Multi-stage production container specification with non-root security. |
| `course/app/app.py` (FastAPI mount) | `app/app.py` | **MIGRATE** | FastAPI endpoints (`/health` and `/api/ask` with HTTP 405 error handling) merged into Gradio app. |
| `course/utils/models.py` (Relevance fix) | `utils/models.py` | **MIGRATE** | Brand-term filtering added to CRAG grader in `_check_relevance` to fix false positive NASDAQ recall. |
| `course/utils/helpers.py` (Native loaders) | `utils/helpers.py` | **MIGRATE** | Replaced deprecated `langchain_community` loaders with native `pypdf` to achieve 0 warnings. |
| `course/notebooks/00_why_rag.ipynb` | `notebooks/00_why_rag.ipynb` | **JUSTIFIED SKIP** | Target has active, verified version with full executed outputs and $\le 16$ lines/cell. |
| `course/notebooks/01_rag_big_picture.ipynb` | `notebooks/01_hello_rag.ipynb` | **JUSTIFIED SKIP** | Target's `01_hello_rag.ipynb` covers identical 5-stage pipeline with tested execution outputs. |
| `course/notebooks/02` through `06` | `notebooks/02` through `06` | **JUSTIFIED SKIP** | Target contains byte-tested equivalent notebooks with real outputs passing all tests. |
| `course/notebooks/07_rag_with_your_own_files.ipynb` | `notebooks/15_capstone_project.ipynb` | **JUSTIFIED SKIP** | Custom file loading mini-project is fully integrated in Target's Capstone NB15 and Gradio app. |
| `course/notebooks/08` through `15` | `notebooks/07` through `15` | **JUSTIFIED SKIP** | Target's notebooks 07-15 cover all advanced RAG topics (hybrid search, CRAG, eval, prod safety). |
| `course/data/*` (Corpus files) | `data/*` | **JUSTIFIED SKIP** | Target corpus files are already byte-identical and validated across all 24 tests. |
| `course/utils/build_phase1.py`–`phase4.py` | `scripts/build_part1`–`part4.py` | **JUSTIFIED SKIP** | Target already maintains verified builders organized by course parts under `scripts/`. |
| `course/conftest.py` & `course/__init__.py` | N/A | **JUSTIFIED SKIP** | Used only when course was nested in `course/`; Target uses clean root-level package layout. |
| `course/VERIFICATION.md` | `tests/TEST_REPORT.md` | **JUSTIFIED SKIP** | Information consolidated into Target's master `tests/TEST_REPORT.md` and README. |
| Tag `v1.0-course` on EXTRA 2 | Tag `archive/v3-final` | **MIGRATE** | Preserved in Target as annotated archive tag `archive/v3-final` referencing commit `@772b8bb`. |
| Branch `beginner-rag-course` on EXTRA 1 | Tag `archive/v2-final` | **MIGRATE** | Preserved in Target as annotated archive tag `archive/v2-final` referencing commit `@8055a6e`. |
| Git commit history of EXTRA 1 & 2 | Tagged in TARGET | **MIGRATE** | Both full commit histories fetched and preserved in Target git database via archive tags. |

---

## Summary of Decisions
- **Total Migrated Items:** 16 critical assets (5 upstream notebooks, upstream docs, 3 verification scripts, 3 architecture/QA docs, Dockerfiles, FastAPI endpoints, model fixes, 2 archive history tags).
- **Total Justified Skips:** 11 redundant or superseded items (duplicate generator scripts, nested package shims, duplicate corpus files).
- **Zero Undecided Items:** All assets accounted for with evidence.
