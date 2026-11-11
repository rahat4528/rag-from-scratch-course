# Novasphere RAG From Scratch — Capstone Web UI

An interactive **Gradio** web interface connecting directly to the hybrid RBAC RAG architecture covered across Notebooks 00–15.

## Features
- **Ask Questions Tab**: Query the Novasphere knowledge base with live **Role-Based Access Control** (`guest`, `employee`, `executive`), cited source snippets, embedding cache telemetry, and query latency metrics.
- **Upload & Re-Index Docs Tab**: Drag and drop custom `.txt`, `.md`, or `.pdf` files, set an access tier, and re-index live in memory (or reset back to the default corpus).
- **Security Controls**: Empty/oversized input guards, HTML XSS escaping on outputs/filenames, pre-retrieval metadata RBAC isolation, and sanitized logs.

## Launching Locally
From the repository root:

```bash
python app/app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. Set `RAG_PROVIDER` in `.env` to switch seamlessly between `local` (default offline mode), `ollama`, `openai`, or `anthropic`.

