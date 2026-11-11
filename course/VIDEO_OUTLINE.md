# Video Playlist Production Outline — `course/VIDEO_OUTLINE.md`

Companion chapter structure for recording Notebooks `00`–`15` as a cohesive 16-episode video playlist.

| Ep | Title | Target Hook & Description | Key Live Demo Moment |
|---|---|---|---|
| **00** | Why Normal Language Models Fail | Compare closed-book guessing vs open-book RAG on 2026 Novasphere facts. | Prompt-stuffing 1 file works vs 5,000-file token overflow math. |
| **01** | The Big Picture: Plain-Python RAG | Demystify RAG in 20 lines of zero-framework Python before LangChain. | Overlap retriever finding pricing vs failing on synonyms. |
| **02** | Loading Real Documents | `.txt`, `.md`, `.pdf`, folder ingestion, SSRF-safe web loading, Unicode. | Rejecting `http://127.0.0.1` SSRF test cleanly + Bangla spaces file. |
| **03** | Splitting Text Into Smart Chunks | Show bad zero-overlap split tearing `ERR_QUANTUM_TIMEOUT` steps apart. | Pandas comparison table across chunk sizes `100`, `300`, `800`. |
| **04** | Embeddings & Vector Stores | Hand NumPy cosine similarity + live 2D PCA Matplotlib cluster chart. | Mismatched embedding space failure demo + JSON save/reload. |
| **05** | Retrieval & Failure Modes | Top-k, similarity scores, MMR diversity, metadata filters. | Three concrete failure demos proving retrieval limits answer quality. |
| **06** | Generation: LCEL + 2-Node LangGraph | Grounded prompt authoring + LCEL + rebuilding cleanly in LangGraph. | Side-by-side No-RAG vs With-RAG + NASDAQ gap refusal. |
| **07** | Mini Project: Your Own Files | Reusable folder loader + JSON saved index + 14-question benchmark. | Reloading from saved JSON & clearing Phase 1 milestone (`>=10/14`). |
| **08** | Query Translation Techniques | Multi-Query, RAG-Fusion (`RRF`), Decomposition, Step-Back, HyDE. | Decomposition rescuing compound multi-doc Q07 vs NB05 baseline. |
| **09** | Routing & Query Construction | Pydantic v2 logical routing + semantic routing + filter construction. | Deliberate wrong-route failure + structured routing decision log. |
| **10** | Advanced Indexing Strategies | Parent-Doc, Multi-Rep, Hybrid BM25+Dense, RAPTOR & ColBERT MaxSim. | Hybrid BM25 nailing `ERR_QUANTUM_TIMEOUT 9872` + ColBERT MaxSim scores. |
| **11** | Reranking & Compression | Local neural `CrossEncoder` + sentence compression vs baseline. | Before/after #1 rank precision + char reduction & latency timer. |
| **12** | Agentic & Self-Correcting RAG | Current `create_agent` tool loop + LangGraph CRAG/Self-RAG grading. | Agent skipping search on `2+2` + CRAG stopping cleanly at `retries=2`. |
| **13** | Evaluating RAG Properly | Hit Rate@k, MRR hand math check, Faithfulness judge, 4-config ablation. | Real Pandas ablation comparing chunk size, `k`, and CrossEncoder ON/OFF. |
| **14** | Production Safety & Lifecycle | Doc updates/retirement by ID, caching, injection defense, RBAC, anti-patterns. | Defeating `role='executive'` bypass text + deleting `promo-v1` cleanly. |
| **15** | Capstone & Gradio Web App | Saved-index Hybrid + CrossEncoder + RBAC app + 14-question benchmark. | Launching `python course/app/app.py` and uploading a live file in UI. |

