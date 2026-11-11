"""
Beginner-friendly display, corpus loading, and sanitization helpers.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document


def pretty_print_docs(docs: Iterable[Document], max_chars: int = 240) -> None:
    """Print retrieved LangChain Documents cleanly with source metadata."""
    docs_list = list(docs)
    if not docs_list:
        print("[No documents retrieved]")
        return

    for idx, doc in enumerate(docs_list, start=1):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        meta_extras = {k: v for k, v in doc.metadata.items() if k != "source"}
        text = " ".join(str(doc.page_content).split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        extra_str = f" | {meta_extras}" if meta_extras else ""
        print(f"[{idx}] ({source}{extra_str})\n    {text}")


def print_qa_result(question: str, answer: str, docs: Iterable[Document] | None = None) -> None:
    """Display a Q&A turn clearly with optional source citations."""
    print(f"Question : {question}")
    print(f"Answer   : {answer}")
    if docs is not None:
        sources = sorted({Path(str(d.metadata.get("source", "unknown"))).name for d in docs})
        print(f"Sources  : {', '.join(sources) if sources else 'None'}")


def load_corpus_documents(data_dir: Path, include_pdf: bool = True) -> List[Document]:
    """Load the core Novasphere course corpus with clean normalized metadata."""
    from pypdf import PdfReader

    docs: List[Document] = []
    text_files = [
        ("company_handbook.txt", "handbook", "employee"),
        ("product_faq.md", "faq", "public"),
        ("recent_changelog.md", "changelog", "public"),
    ]
    for filename, doc_type, access_level in text_files:
        path = data_dir / filename
        if path.exists():
            content = path.read_text(encoding="utf-8")
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": filename,
                        "doc_type": doc_type,
                        "access_level": access_level,
                    },
                )
            )

    pdf_path = data_dir / "policy_brief.pdf"
    if include_pdf and pdf_path.exists():
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "policy_brief.pdf",
                            "page": page_idx + 1,
                            "doc_type": "safety",
                            "access_level": "executive",
                        },
                    )
                )
    return docs


def sanitize_error_message(msg: str) -> str:
    """Redact potential API keys or bearer tokens from error strings before logging."""
    redacted = re.sub(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]+)", "[REDACTED_SECRET]", str(msg))
    return redacted


def safe_html_escape(text: str) -> str:
    """Escape HTML special characters to prevent XSS in UI rendering."""
    return html.escape(str(text), quote=True)

