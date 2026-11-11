# Security Findings & Controls — `rag-from-scratch-course`

## Verified Application & Infrastructure Controls
1. **Secret Management & Sanitization (`PASSED` — `test_b3`, `test_b10`, `test_b14`):**
   - `.env` is ignored via `.gitignore` and verified absent from `git ls-files`.
   - Placeholder/missing API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) raise `ValueError` referencing only the env variable name without echoing secrets.
   - `sanitize_error_message()` masks `sk-...` and `Bearer ...` tokens cleanly in logs and UI responses.
   - Zero hardcoded credentials or `sk-` secrets across tracked Python files or notebook outputs.
2. **Pre-Retrieval RBAC Access Control (`PASSED` — NB14, NB15, `test_b9`):**
   - Access permissions (`guest`: `public`, `employee`: `public`+`employee`, `executive`: all including `policy_brief.pdf`) are enforced inside vector-store filter predicates *prior to* LLM prompt assembly (`policy_brief.pdf` excluded cleanly when `role='employee'`).
3. **XSS & Input Validation (`PASSED` — `test_b7`):**
   - Empty/whitespace queries are rejected gracefully without model invocation.
   - Oversized queries (>2,000 chars) are safely truncated cleanly without memory spikes or crashes.
   - HTML tags/scripts (`<script>alert('xss')</script>`) and uploaded filenames are escaped via `safe_html_escape()` (`html.escape(..., quote=True)`).
4. **Indirect Prompt Injection Resilience (`PASSED` — NB14, `test_b9`):**
   - Context parser strips injected override directives (`"IGNORE ALL PREVIOUS INSTRUCTIONS"`, `"Reveal ... OPENAI_API_KEY"`) and wraps facts inside `<context>` delimiters.
5. **Local Server Binding (`PASSED` — `app/app.py`):**
   - Gradio defaults to `server_name="127.0.0.1"` (loopback only, non-public by default).

## Dependency Audit (`pip-audit` Findings)
Ran `.\.venv\Scripts\python.exe -m pip_audit`:
- **`pip 26.1.2` (`PYSEC-2026-3721`)**: **FIXED** — upgraded `.venv` `pip` cleanly to `26.3` (zero `pip` advisories remaining in Scan #2).
- **`pillow 12.3.0` (`PYSEC-2026-43xx`, `CVE-2026-xxxxx`, fix `13.0.0`)**: **UPSTREAM CONSTRAINT / MITIGATED** — `gradio 6.31.0` strictly specifies `pillow<13.0,>=8.0`. Risk is mitigated in this app because the Gradio `gr.File` uploader restricts extensions strictly to `[".txt", ".md", ".pdf"]` with no arbitrary image decoding path.
- **`torch 2.6.0` (advisories fixed in `2.7.0`–`2.13.0`)**: **DOCUMENTED HOST RUNTIME TRADEOFF** — Latest `torch 2.15.0` fails to load on Windows hosts missing VS 2022 `vcruntime140_threads.dll` (`WinError 126`, see `BUG-004`). `torch>=2.6.0,<2.9.0` is pinned in `requirements.txt` for universal Windows reproducibility. Users running updated VS 2022 C++ redistributables or Linux/macOS can safely upgrade `torch` (`>=2.8`). Only `sentence-transformers` Hub weights (`all-MiniLM-L6-v2`) with safetensors are loaded; no arbitrary untrusted `.pt` pickle files are ever deserialized.
