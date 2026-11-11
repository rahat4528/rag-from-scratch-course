# Novasphere RAG From Scratch — Capstone Web UI & API (`course/app/`)

Interactive **Gradio + FastAPI** assistant backed by the production Hybrid + CrossEncoder + RBAC RAG architecture built in Notebooks 00–15.

## Features
- **Ask Questions Tab**: Hybrid Dense (`all-MiniLM-L6-v2`) + BM25 RRF retrieval + `CrossEncoder` reranking, RBAC dropdown (`guest`, `employee`, `executive`), side-by-side citations, and live SHA-256 cache telemetry.
- **Manage Index & Own Files Tab**: Upload `.txt`, `.md`, or `.pdf` documents dynamically, retire outdated files by filename, persist indexes cleanly to JSON, clear index cleanly (`no-index` guardrail), or reset default corpus.
- **REST Endpoints**:
  - `GET /health` -> service & index readiness JSON (`chunks`, `files`)
  - `POST /api/ask` -> validated Pydantic endpoint (`{"question": "...", "role": "employee"}`)
- **Security Controls**: Pre-retrieval RBAC filter enforcement, prompt-injection stripping, HTML XSS escaping, secret redaction in logs, empty/oversized query handling, and `127.0.0.1` loopback binding.

## Run Locally
From repository root:
```bash
python course/app/app.py
```
Then open [http://127.0.0.1:7860](http://127.0.0.1:7860). Configure `RAG_PROVIDER` (`openai`, `anthropic`, `ollama`, or `offline_demo`) inside `.env` (or `course/.env`).

## Docker & Hosting Options
Build and launch locally using `course/app/Dockerfile`:
```bash
docker build -f course/app/Dockerfile -t rag-course-app .
docker run -p 7860:7860 -e RAG_PROVIDER=offline_demo rag-course-app
```
*(Deployment options: Hugging Face Spaces Gradio runtime, Railway/Render FastAPI container, or internal Docker VM behind HTTPS reverse proxy.)*

