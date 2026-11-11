# Missing, Optional, or Environment-Blocked Items

| Item | Status | Explanation & Evidence |
|---|---|---|
| Live OpenAI API Calls (`RAG_PROVIDER=openai`) | N/A (Environment-blocked — no active paid key configured) | `get_llm('openai')` & `OpenAIEmbeddings` wiring, placeholder/missing key validation, and secret non-leakage tested cleanly via negative unit tests in `tests/test_b3_helpers_auth.py`. Live network generation requires user's `OPENAI_API_KEY` in `.env`. |
| Live Anthropic API Calls (`RAG_PROVIDER=anthropic`) | N/A (Environment-blocked — no active paid key configured) | `get_llm('anthropic')` & key validation verified cleanly in `tests/test_b3_helpers_auth.py`. Requires user's `ANTHROPIC_API_KEY` in `.env`. |
| Live Ollama Server (`RAG_PROVIDER=ollama`) | Optional local server runtime | Provider routing code in `utils/models.py` supports `ChatOllama` + `OLLAMA_BASE_URL` cleanly; live model invocation assumes local Ollama daemon running with pulled model (`llama3.2`). |
| Real-time Public URL Fetching (`WebBaseLoader`) | Optional network dependency | Demonstrated cleanly offline via local HTML parsing with BeautifulSoup (`02_loading_documents.ipynb`) so fresh clones pass 100% deterministically with zero external site flakiness. |
| Interactive Browser Rendering E2E (Selenium/Playwright) | Covered via live HTTP server + backend component API tests | `test_b10_git_secret_scan_and_gradio_server` launches a real Gradio uvicorn server on `127.0.0.1:7869`, verifies HTTP `200` + expected HTML body, and tests every UI callback (`handle_query`, upload, reset) directly. |
