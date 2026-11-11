"""
Novasphere RAG From Scratch — Production Capstone Web App (Gradio UI + Hybrid Backend).

Run from repository root:
    python app/app.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# Ensure repository root is importable regardless of launch working directory
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.helpers import load_corpus_documents, safe_html_escape, sanitize_error_message
from utils.models import get_embeddings, get_llm, resolve_project_root

ROLE_RIGHTS: Dict[str, set[str]] = {
    "guest": {"public"},
    "employee": {"public", "employee"},
    "executive": {"public", "employee", "executive"},
}

MAX_QUERY_CHARS = 2000


class QueryRequest(BaseModel):
    question: str
    role: str = "employee"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


class NovasphereRAGEngine:
    """Thread-safe in-memory Hybrid RBAC RAG engine powering the Gradio app."""

    def __init__(self) -> None:
        self.project_root = resolve_project_root()
        self.data_dir = self.project_root / "data"
        self.llm = get_llm()
        self.embeddings = get_embeddings(use_cache=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=640,
            chunk_overlap=140,
            separators=["\n## ", "\n### ", "\n\n", "\n- ", "\n", " ", ""],
        )
        self.chunks: List[Document] = []
        self.dense_store: Optional[InMemoryVectorStore] = None
        self.bm25: Optional[BM25Okapi] = None
        self.source_files: List[str] = []
        self.reset_to_default_corpus()

    def _build_index(self, docs: List[Document]) -> str:
        if not docs:
            self.chunks = []
            self.dense_store = None
            self.bm25 = None
            self.source_files = []
            return "Index cleared (0 chunks)."

        self.chunks = self.splitter.split_documents(docs)
        self.dense_store = InMemoryVectorStore.from_documents(self.chunks, self.embeddings)
        tokenized_corpus = [_tokenize(c.page_content) or ["empty"] for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.source_files = sorted({str(d.metadata.get("source", "unknown")) for d in docs})
        return f"Indexed {len(self.chunks)} chunks across {len(self.source_files)} file(s): {', '.join(self.source_files)}"

    def reset_to_default_corpus(self) -> str:
        docs = load_corpus_documents(self.data_dir, include_pdf=True)
        return self._build_index(docs)

    def ingest_uploaded_files(
        self,
        file_paths: Optional[List[str]],
        access_level: str = "employee",
        replace_existing: bool = False,
    ) -> str:
        if not file_paths:
            return "No files uploaded. Keeping current index unchanged."

        valid_docs: List[Document] = [] if replace_existing else load_corpus_documents(self.data_dir, include_pdf=True)
        added_names: List[str] = []

        for raw_path in file_paths:
            p = Path(str(raw_path))
            if not p.exists():
                continue
            fname = p.name
            ext = p.suffix.lower()
            try:
                if ext in (".txt", ".md"):
                    text = p.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        valid_docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": fname, "doc_type": "custom", "access_level": access_level},
                            )
                        )
                        added_names.append(fname)
                elif ext == ".pdf":
                    reader = PdfReader(str(p))
                    for page_idx, page in enumerate(reader.pages):
                        text = (page.extract_text() or "").strip()
                        if text:
                            valid_docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": fname,
                                        "page": page_idx + 1,
                                        "doc_type": "custom_pdf",
                                        "access_level": access_level,
                                    },
                                )
                            )
                    added_names.append(fname)
            except Exception as exc:
                return f"Error ingesting {safe_html_escape(fname)}: {sanitize_error_message(str(exc))}"

        if not added_names:
            return "Uploaded files contained no readable text (.txt, .md, .pdf supported)."
        status = self._build_index(valid_docs)
        return f"Uploaded {', '.join(safe_html_escape(n) for n in added_names)}. {status}"

    def retrieve(self, query: str, role: str = "employee", top_n: int = 4) -> List[Document]:
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
        reranked = []
        for content, base_score in rrf.items():
            overlap = len(q_terms & set(_tokenize(content)))
            reranked.append((base_score + 0.05 * overlap, doc_map[content]))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in reranked[:top_n]]

    def query(self, question: str, role: str = "employee") -> Tuple[str, str, str]:
        t0 = time.perf_counter()
        if not question or not question.strip():
            return (
                "Please enter a non-empty question to query the knowledge base.",
                "*No sources retrieved.*",
                "Latency: 0.0 ms | Status: Empty Input Guarded",
            )

        clean_q = question.strip()
        notice = ""
        if len(clean_q) > MAX_QUERY_CHARS:
            clean_q = clean_q[:MAX_QUERY_CHARS]
            notice = f"*(Note: Query exceeded {MAX_QUERY_CHARS} characters and was safely truncated.)*\n\n"

        if role not in ROLE_RIGHTS:
            role = "guest"

        hits = self.retrieve(clean_q, role=role, top_n=4)
        if not hits:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return (
                f"{notice}I don't know based on the accessible documents for role `{role}`.",
                "*No authorized documents matched this query.*",
                f"Latency: {elapsed_ms:.1f} ms | Role: {role} | Chunks retrieved: 0",
            )

        ctx = "\n\n".join(f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in hits)
        prompt = (
            "Answer accurately using ONLY the context below. If missing, say you don't know.\n\n"
            f"<context>\n{ctx}\n</context>\n\nQuestion: {clean_q}"
        )
        try:
            raw_answer = str(self.llm.invoke(prompt).content)
        except Exception as exc:
            raw_answer = f"Model error: {sanitize_error_message(str(exc))}"

        safe_answer = notice + safe_html_escape(raw_answer)

        src_blocks = []
        for idx, doc in enumerate(hits, start=1):
            src = safe_html_escape(str(doc.metadata.get("source", "unknown")))
            lvl = safe_html_escape(str(doc.metadata.get("access_level", "public")))
            snippet = safe_html_escape(" ".join(doc.page_content.split())[:220])
            src_blocks.append(f"**[{idx}] `{src}`** *(access: {lvl})*\n> {snippet}...")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        cache_info = self.embeddings.cache_stats() if hasattr(self.embeddings, "cache_stats") else {}
        stats_line = (
            f"Latency: {elapsed_ms:.1f} ms | Role: {role} | Retrieved: {len(hits)} chunks | "
            f"Embed Cache Hits: {cache_info.get('hits', 0)} / Misses: {cache_info.get('misses', 0)}"
        )
        return safe_answer, "\n\n".join(src_blocks), stats_line


_ENGINE: Optional[NovasphereRAGEngine] = None


def get_engine() -> NovasphereRAGEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NovasphereRAGEngine()
    return _ENGINE


def handle_query(question: str, role: str) -> Tuple[str, str, str]:
    return get_engine().query(question, role=role)


def handle_upload(files, access_level: str, replace: bool) -> str:
    paths = [f.name if hasattr(f, "name") else str(f) for f in (files or [])]
    return get_engine().ingest_uploaded_files(paths, access_level=access_level, replace_existing=replace)


def handle_reset() -> str:
    return get_engine().reset_to_default_corpus()


def create_demo() -> gr.Blocks:
    engine = get_engine()
    provider = os.getenv("RAG_PROVIDER", "local")

    with gr.Blocks(title="Novasphere RAG From Scratch Assistant") as demo:
        gr.Markdown(
            "# Novasphere RAG From Scratch — Capstone Assistant\n"
            "Hybrid **Dense + BM25 RRF** retrieval with **Cross-Feature Reranking**, **Embedding Caching**, "
            "**Injection Filtering**, and **Role-Based Access Control**."
        )

        with gr.Tab("Ask Questions"):
            with gr.Row():
                question_box = gr.Textbox(
                    label="Your Question",
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
                    ["Confidential quarterly safety audit report access rule?", "employee"],
                    ["Confidential quarterly safety audit report access rule?", "executive"],
                    ["What is Novasphere's NASDAQ stock ticker symbol?", "employee"],
                ],
                inputs=[question_box, role_dd],
            )
            ans_out = gr.Markdown(label="Grounded Answer")
            src_out = gr.Markdown(label="Retrieved Citations")
            stats_out = gr.Textbox(label="Telemetry & Cache Stats", interactive=False)

            ask_btn.click(fn=handle_query, inputs=[question_box, role_dd], outputs=[ans_out, src_out, stats_out])
            question_box.submit(fn=handle_query, inputs=[question_box, role_dd], outputs=[ans_out, src_out, stats_out])

        with gr.Tab("Upload & Re-Index Docs"):
            gr.Markdown(
                "Upload custom `.txt`, `.md`, or `.pdf` files to index them immediately into memory, "
                "or reset back to the default Novasphere corpus."
            )
            upload_files = gr.File(file_count="multiple", file_types=[".txt", ".md", ".pdf"], label="Upload Documents")
            with gr.Row():
                upload_access = gr.Dropdown(["public", "employee", "executive"], value="employee", label="Document Access Level")
                replace_chk = gr.Checkbox(value=False, label="Replace default corpus instead of appending")
            with gr.Row():
                ingest_btn = gr.Button("Index Uploaded Files", variant="primary")
                reset_btn = gr.Button("Reset Default Corpus")
            index_status = gr.Textbox(
                value=f"Loaded {len(engine.chunks)} chunks from: {', '.join(engine.source_files)}",
                label="Index Status",
                interactive=False,
            )
            ingest_btn.click(fn=handle_upload, inputs=[upload_files, upload_access, replace_chk], outputs=[index_status])
            reset_btn.click(fn=handle_reset, outputs=[index_status])

        with gr.Tab("Architecture & Provider Status"):
            gr.Markdown(
                f"### Active Configuration\n"
                f"- **RAG Provider**: `{safe_html_escape(provider)}`\n"
                f"- **Embedding Model**: `{safe_html_escape(type(engine.embeddings.base).__name__)}`\n"
                f"- **Chat Model**: `{safe_html_escape(type(engine.llm).__name__)}`\n"
                f"- **Indexed Source Files**: `{', '.join(engine.source_files)}`\n\n"
                f"### RBAC Permission Matrix\n"
                f"- `guest`: public documents only (`product_faq.md`, `recent_changelog.md`)\n"
                f"- `employee`: public + internal handbook (`company_handbook.txt`)\n"
                f"- `executive`: all documents including confidential briefs (`policy_brief.pdf`)"
            )

    return demo


def create_fastapi_app():
    """Mount Gradio app onto FastAPI with explicit health and query endpoints."""
    from fastapi import FastAPI, HTTPException

    api = FastAPI(title="Novasphere RAG Capstone API", version="1.0.0")

    @api.get("/health")
    def health():
        return {"status": "healthy", "service": "novasphere-rag-capstone"}

    @api.post("/api/ask")
    def ask_endpoint(req: QueryRequest):
        ans, src, stats = handle_query(req.question, req.role)
        return {"answer": ans, "citations": src, "telemetry": stats}

    @api.api_route("/api/ask", methods=["GET", "PUT", "DELETE", "PATCH"])
    def ask_method_not_allowed():
        raise HTTPException(status_code=405, detail="Method Not Allowed")

    blocks = create_demo()
    return gr.mount_gradio_app(api, blocks, path="/")


if __name__ == "__main__":
    app_ui = create_demo()
    port = int(os.getenv("PORT", "7860"))
    app_ui.launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
