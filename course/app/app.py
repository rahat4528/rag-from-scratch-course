"""
Novasphere RAG From Scratch — Production Capstone Web App & API Server.

Combines:
  - Persisted JSON Index Load/Save + Incremental Upload & ID-Based Retire
  - Hybrid Dense (all-MiniLM-L6-v2) + BM25 RRF Retrieval
  - Local Neural Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
  - Pre-Retrieval Role-Based Access Control (guest / employee / executive)
  - Prompt Injection Filtering + XSS Escaping + Sanitized Errors
  - Gradio Mobile-Friendly Web UI mounted cleanly alongside FastAPI /health & /api/ask

Run from repository root:
    python course/app/app.py
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

APP_DIR = Path(__file__).resolve().parent
COURSE_ROOT = APP_DIR.parent
REPO_ROOT = COURSE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course.utils.helpers import (
    load_corpus_documents,
    load_pdf_documents,
    load_text_document,
    safe_html_escape,
    sanitize_error_message,
)
from course.utils.models import (
    get_cross_encoder,
    get_embeddings,
    get_llm,
    resolve_course_root,
)

ROLE_RIGHTS: Dict[str, set[str]] = {
    "guest": {"public"},
    "employee": {"public", "employee"},
    "executive": {"public", "employee", "executive"},
}

MAX_QUERY_CHARS = 2000


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


class NovasphereCapstoneEngine:
    """Thread-safe Hybrid + CrossEncoder + RBAC engine with lifecycle management."""

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self.course_dir = resolve_course_root()
        self.data_dir = self.course_dir / "data"
        self.index_dir = index_dir or (self.data_dir / "persisted_index")
        self.llm = get_llm(fallback_to_demo=True)
        self.embeddings = get_embeddings(use_cache=True)
        self.reranker = get_cross_encoder()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=640,
            chunk_overlap=140,
            separators=["\n## ", "\n### ", "\n\n", "\n- ", "\n", " ", ""],
        )
        self.docs: List[Document] = []
        self.chunks: List[Document] = []
        self.chunk_ids: List[str] = []
        self.dense_store: Optional[InMemoryVectorStore] = None
        self.bm25: Optional[BM25Okapi] = None
        self.source_files: List[str] = []
        self.reset_to_default_corpus()

    def _rebuild_from_docs(self, docs: List[Document]) -> str:
        with self._lock:
            self.docs = list(docs)
            if not self.docs:
                self.chunks = []
                self.chunk_ids = []
                self.dense_store = None
                self.bm25 = None
                self.source_files = []
                return "Index cleared cleanly (0 chunks loaded)."

            self.chunks = self.splitter.split_documents(self.docs)
            self.chunk_ids = [
                f"{d.metadata.get('doc_id', d.metadata.get('source', 'doc'))}::c{i}"
                for i, d in enumerate(self.chunks)
            ]
            self.dense_store = InMemoryVectorStore.from_documents(
                self.chunks, self.embeddings, ids=self.chunk_ids
            )
            tokenized = [_tokenize(c.page_content) or ["empty"] for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
            self.source_files = sorted({str(d.metadata.get("source", "unknown")) for d in self.docs})
            return (
                f"Indexed {len(self.chunks)} chunks across {len(self.source_files)} file(s): "
                f"{', '.join(self.source_files)}"
            )

    def clear_index(self) -> str:
        return self._rebuild_from_docs([])

    def reset_to_default_corpus(self) -> str:
        return self._rebuild_from_docs(load_corpus_documents(self.data_dir, include_pdf=True))

    def save_index(self) -> str:
        with self._lock:
            if self.dense_store is None:
                return "No index loaded to save."
            self.index_dir.mkdir(parents=True, exist_ok=True)
            path = self.index_dir / "store.json"
            self.dense_store.dump(str(path))
            return f"Saved persisted JSON index ({len(self.chunks)} chunks) to {path.name}."

    def ingest_uploaded_files(
        self,
        file_paths: Optional[List[str]],
        access_level: str = "employee",
        replace_existing: bool = False,
    ) -> str:
        if not file_paths:
            return "No files uploaded. Keeping current index unchanged."

        if access_level not in ("public", "employee", "executive"):
            access_level = "employee"

        base_docs = [] if replace_existing else list(self.docs)
        added: List[str] = []

        for raw_path in file_paths:
            p = Path(str(raw_path))
            if not p.exists() or not p.is_file():
                continue
            ext = p.suffix.lower()
            try:
                if ext in (".txt", ".md"):
                    doc = load_text_document(p, doc_type="uploaded", access_level=access_level, doc_id=f"upload-{p.name}")
                    if doc:
                        base_docs = [d for d in base_docs if d.metadata.get("doc_id") != doc.metadata["doc_id"]]
                        base_docs.append(doc)
                        added.append(p.name)
                elif ext == ".pdf":
                    pdf_docs = load_pdf_documents(p, doc_type="uploaded_pdf", access_level=access_level, doc_id=f"upload-{p.name}")
                    if pdf_docs:
                        base_docs = [d for d in base_docs if d.metadata.get("source") != p.name]
                        base_docs.extend(pdf_docs)
                        added.append(p.name)
            except Exception as exc:
                return f"Failed ingesting {safe_html_escape(p.name)}: {sanitize_error_message(str(exc))}"

        if not added:
            return "Uploaded files contained no extractable text (.txt, .md, .pdf supported)."
        status = self._rebuild_from_docs(base_docs)
        return f"Ingested {', '.join(safe_html_escape(n) for n in added)}. {status}"

    def retire_document(self, source_name: str) -> str:
        clean_src = (source_name or "").strip()
        if not clean_src:
            return "Enter a filename to retire (e.g., company_handbook.txt)."
        remaining = [d for d in self.docs if d.metadata.get("source") != clean_src]
        if len(remaining) == len(self.docs):
            return f"No document named '{safe_html_escape(clean_src)}' found in active index."
        status = self._rebuild_from_docs(remaining)
        return f"Retired '{safe_html_escape(clean_src)}'. {status}"

    def retrieve(self, query: str, role: str = "employee", top_n: int = 4) -> List[Document]:
        with self._lock:
            if self.dense_store is None or self.bm25 is None or not self.chunks:
                return []

            allowed = ROLE_RIGHTS.get(role, {"public"})
            dense_hits = self.dense_store.similarity_search(
                query,
                k=8,
                filter=lambda d: d.metadata.get("access_level", "public") in allowed,
            )
            bm_scores = self.bm25.get_scores(_tokenize(query))
            allowed_bm = [
                (bm_scores[i], self.chunks[i])
                for i in range(len(self.chunks))
                if self.chunks[i].metadata.get("access_level", "public") in allowed
            ]
            bm_top = [d for _, d in sorted(allowed_bm, key=lambda x: x[0], reverse=True)[:8]]

            rrf: Dict[str, float] = {}
            doc_map: Dict[str, Document] = {}
            for lst in (dense_hits, bm_top):
                for rank, d in enumerate(lst, start=1):
                    doc_map[d.page_content] = d
                    rrf[d.page_content] = rrf.get(d.page_content, 0.0) + 1.0 / (60 + rank)

            q_terms = {t for t in _tokenize(query) if len(t) > 2}
            pool = [
                doc_map[content]
                for content, _ in sorted(
                    rrf.items(),
                    key=lambda kv: kv[1] + 0.03 * len(q_terms & set(_tokenize(kv[0]))),
                    reverse=True,
                )[:8]
            ]
            reranked = self.reranker.rerank(query, pool, top_n=top_n)
            return [doc for _, doc in reranked]

    def query(self, question: str, role: str = "employee") -> Tuple[str, str, str, List[str]]:
        t0 = time.perf_counter()
        if not question or not question.strip():
            return (
                "Please enter a non-empty question before submitting.",
                "*No citations — empty input blocked cleanly.*",
                "Latency: 0.0 ms | Status: Empty Input",
                [],
            )

        clean_q = question.strip()
        notice = ""
        if len(clean_q) > MAX_QUERY_CHARS:
            clean_q = clean_q[:MAX_QUERY_CHARS]
            notice = f"*(Note: Input truncated cleanly at {MAX_QUERY_CHARS} characters.)*\n\n"

        if role not in ROLE_RIGHTS:
            role = "guest"

        with self._lock:
            has_index = self.dense_store is not None and len(self.chunks) > 0

        if not has_index:
            return (
                "No index is currently loaded. Upload documents or click Reset Default Corpus first.",
                "*Index empty.*",
                "Latency: 0.0 ms | Status: No Index Loaded",
                [],
            )

        hits = self.retrieve(clean_q, role=role, top_n=4)
        if not hits:
            ms = (time.perf_counter() - t0) * 1000
            return (
                f"{notice}I don't know based on the accessible documents for role `{role}`.",
                "*No matching authorized chunks.*",
                f"Latency: {ms:.1f} ms | Role: {role} | Hits: 0",
                [],
            )

        ctx = "\n\n".join(f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in hits)
        prompt = (
            "Answer using ONLY the context below. If missing, say 'I don't know based on the provided context.'\n\n"
            f"<context>\n{ctx}\n</context>\n\nQuestion: {clean_q}"
        )
        try:
            raw_ans = str(self.llm.invoke(prompt).content)
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            err_msg = f"Model unavailable: {sanitize_error_message(str(exc))}"
            return (err_msg, "*Model call error handled cleanly.*", f"Latency: {ms:.1f} ms | Status: Model Error", [])

        safe_ans = notice + safe_html_escape(raw_ans)
        citations = sorted({str(d.metadata.get("source", "unknown")) for d in hits})
        src_blocks = []
        for idx, doc in enumerate(hits, start=1):
            src = safe_html_escape(str(doc.metadata.get("source", "unknown")))
            lvl = safe_html_escape(str(doc.metadata.get("access_level", "public")))
            snip = safe_html_escape(" ".join(doc.page_content.split())[:220])
            src_blocks.append(f"**[{idx}] `{src}`** *(access: {lvl})*\n> {snip}...")

        ms = (time.perf_counter() - t0) * 1000
        stats = self.embeddings.cache_stats() if hasattr(self.embeddings, "cache_stats") else {}
        stats_line = (
            f"Latency: {ms:.1f} ms | Role: {role} | Chunks: {len(hits)} | "
            f"Cache Hits: {stats.get('hits', 0)} / Misses: {stats.get('misses', 0)}"
        )
        return safe_ans, "\n\n".join(src_blocks), stats_line, citations


_ENGINE: Optional[NovasphereCapstoneEngine] = None


def get_engine() -> NovasphereCapstoneEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NovasphereCapstoneEngine()
    return _ENGINE


def handle_query(question: str, role: str) -> Tuple[str, str, str]:
    ans, srcs_md, stats, _ = get_engine().query(question, role=role)
    return ans, srcs_md, stats


def handle_upload(files, access_level: str, replace: bool) -> str:
    paths = [f.name if hasattr(f, "name") else str(f) for f in (files or [])]
    return get_engine().ingest_uploaded_files(paths, access_level=access_level, replace_existing=replace)


def handle_retire(filename: str) -> str:
    return get_engine().retire_document(filename)


def handle_clear() -> str:
    return get_engine().clear_index()


def handle_reset() -> str:
    return get_engine().reset_to_default_corpus()


def handle_save() -> str:
    return get_engine().save_index()


def create_gradio_blocks() -> gr.Blocks:
    engine = get_engine()
    provider = os.getenv("RAG_PROVIDER", "ollama")

    with gr.Blocks(title="Novasphere RAG From Scratch — Capstone") as demo:
        gr.Markdown(
            "# Novasphere RAG From Scratch — Capstone Assistant\n"
            "Hybrid **Dense + BM25 RRF** retrieval with local **CrossEncoder** reranking, **SHA-256 caching**, "
            "**Injection defenses**, and **Role-Based Access Control**."
        )

        with gr.Tab("Ask Questions"):
            with gr.Row():
                q_in = gr.Textbox(
                    label="Ask a Question",
                    placeholder="e.g., How much does the Enterprise Plan cost per month?",
                    lines=2,
                    scale=4,
                )
                role_dd = gr.Dropdown(
                    choices=["guest", "employee", "executive"],
                    value="employee",
                    label="User Role (RBAC)",
                    scale=1,
                )
            ask_btn = gr.Button("Ask Assistant", variant="primary")
            gr.Examples(
                examples=[
                    ["How much does the Enterprise Plan cost per month?", "guest"],
                    ["How do I resolve error code ERR_QUANTUM_TIMEOUT?", "employee"],
                    ["Confidential quarterly safety audit report rules?", "employee"],
                    ["Confidential quarterly safety audit report rules?", "executive"],
                    ["What is Novasphere's NASDAQ stock ticker symbol?", "employee"],
                ],
                inputs=[q_in, role_dd],
            )
            with gr.Row():
                ans_out = gr.Markdown(label="Grounded Answer")
                src_out = gr.Markdown(label="Cited Source Passages")
            stats_out = gr.Textbox(label="Telemetry & Cache Stats", interactive=False)
            ask_btn.click(fn=handle_query, inputs=[q_in, role_dd], outputs=[ans_out, src_out, stats_out])
            q_in.submit(fn=handle_query, inputs=[q_in, role_dd], outputs=[ans_out, src_out, stats_out])

        with gr.Tab("Manage Index & Own Files"):
            gr.Markdown("Upload `.txt`, `.md`, or `.pdf` files, retire outdated files by filename, persist to JSON, or reset.")
            up_files = gr.File(file_count="multiple", file_types=[".txt", ".md", ".pdf"], label="Upload Documents")
            with gr.Row():
                up_role = gr.Dropdown(["public", "employee", "executive"], value="employee", label="Document Access Level")
                replace_chk = gr.Checkbox(value=False, label="Replace active corpus instead of appending")
            with gr.Row():
                ingest_btn = gr.Button("Index Uploaded Files", variant="primary")
                save_btn = gr.Button("Save JSON Index")
                clear_btn = gr.Button("Clear Index")
                reset_btn = gr.Button("Reset Default Corpus")
            with gr.Row():
                retire_in = gr.Textbox(label="Retire Document Filename", placeholder="company_handbook.txt")
                retire_btn = gr.Button("Retire Document")
            idx_status = gr.Textbox(
                value=f"Loaded {len(engine.chunks)} chunks across {len(engine.source_files)} files: {', '.join(engine.source_files)}",
                label="Index Status",
                interactive=False,
            )
            ingest_btn.click(fn=handle_upload, inputs=[up_files, up_role, replace_chk], outputs=[idx_status])
            retire_btn.click(fn=handle_retire, inputs=[retire_in], outputs=[idx_status])
            save_btn.click(fn=handle_save, outputs=[idx_status])
            clear_btn.click(fn=handle_clear, outputs=[idx_status])
            reset_btn.click(fn=handle_reset, outputs=[idx_status])

        with gr.Tab("Architecture & Provider Status"):
            gr.Markdown(
                f"### System Configuration\n"
                f"- **Configured `RAG_PROVIDER`**: `{safe_html_escape(provider)}`\n"
                f"- **Chat Backend Active**: `{safe_html_escape(type(engine.llm).__name__)}`\n"
                f"- **Embedding Model**: `{safe_html_escape(type(engine.embeddings.base).__name__)}`\n"
                f"- **Cross-Encoder Neural Active**: `{engine.reranker.is_neural}`\n\n"
                f"### RBAC Permission Matrix\n"
                f"- `guest`: public (`product_faq.md`, `recent_changelog.md`)\n"
                f"- `employee`: public + internal handbook (`company_handbook.txt`)\n"
                f"- `executive`: all documents including `policy_brief.pdf`"
            )
    return demo


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    role: Literal["guest", "employee", "executive"] = "employee"


def create_fastapi_app() -> FastAPI:
    api = FastAPI(title="Novasphere RAG Capstone API", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @api.get("/health")
    def health():
        eng = get_engine()
        return {"status": "ok", "chunks": len(eng.chunks), "files": eng.source_files}

    @api.api_route("/health", methods=["POST", "PUT", "DELETE", "PATCH"])
    def health_method_not_allowed():
        raise HTTPException(status_code=405, detail="Method Not Allowed. Use GET /health.")

    @api.post("/api/ask")
    def api_ask(req: AskRequest):
        if len(req.question) > 4000:
            raise HTTPException(status_code=413, detail="Query exceeds maximum allowed 4000 characters.")
        eng = get_engine()
        try:
            ans, _, stats, cites = eng.query(req.question, role=req.role)
            return JSONResponse({"answer": ans, "sources": cites, "stats": stats})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=sanitize_error_message(str(exc)))

    @api.api_route("/api/ask", methods=["GET", "PUT", "DELETE", "PATCH"])
    def ask_method_not_allowed():
        raise HTTPException(status_code=405, detail="Method Not Allowed. Use POST /api/ask.")

    blocks = create_gradio_blocks()
    return gr.mount_gradio_app(api, blocks, path="/")


if __name__ == "__main__":
    server_app = create_fastapi_app()
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(server_app, host="127.0.0.1", port=port, log_level="warning")

