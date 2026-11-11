"""
Native, zero-deprecation document loaders, SSRF-safe web loader,
display printers, and security sanitizers for the RAG From Scratch course.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader


def sanitize_error_message(msg: str) -> str:
    """Redact potential API keys or bearer tokens before printing or returning errors."""
    redacted = re.sub(
        r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]+|AIza[0-9A-Za-z_-]{16,})",
        "[REDACTED_SECRET]",
        str(msg),
    )
    return redacted


def safe_html_escape(text: str) -> str:
    """Escape HTML special characters to neutralize XSS payloads in UI outputs."""
    return html.escape(str(text), quote=True)


def pretty_print_docs(docs: Iterable[Document], max_chars: int = 240) -> None:
    """Print retrieved LangChain Documents cleanly with source and metadata."""
    docs_list = list(docs)
    if not docs_list:
        print("[No documents retrieved]")
        return

    for idx, doc in enumerate(docs_list, start=1):
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        extras = {k: v for k, v in doc.metadata.items() if k != "source"}
        text = " ".join(str(doc.page_content).split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        extra_str = f" | {extras}" if extras else ""
        print(f"[{idx}] ({source}{extra_str})\n    {text}")


def print_qa_result(question: str, answer: str, docs: Iterable[Document] | None = None) -> None:
    """Display a Q&A turn clearly with cited filenames."""
    print(f"Question : {question}")
    print(f"Answer   : {answer}")
    if docs is not None:
        sources = sorted({Path(str(d.metadata.get("source", "unknown"))).name for d in docs})
        print(f"Sources  : {', '.join(sources) if sources else 'None'}")


def load_text_document(
    path: Path,
    doc_type: str = "text",
    access_level: str = "public",
    doc_id: Optional[str] = None,
) -> Optional[Document]:
    """Load a UTF-8 .txt or .md file safely, returning None on empty files."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.name}")
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return None
    meta = {
        "source": path.name,
        "doc_type": doc_type,
        "access_level": access_level,
        "doc_id": doc_id or path.name,
    }
    return Document(page_content=content, metadata=meta)


def load_pdf_documents(
    path: Path,
    doc_type: str = "pdf",
    access_level: str = "public",
    doc_id: Optional[str] = None,
) -> List[Document]:
    """Extract non-empty text pages from a PDF natively via pypdf (zero deprecation warnings)."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path.name}")
    reader = PdfReader(str(path))
    docs: List[Document] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page": idx,
                    "doc_type": doc_type,
                    "access_level": access_level,
                    "doc_id": doc_id or f"{path.name}#p{idx}",
                },
            )
        )
    return docs


def load_folder_documents(folder: Path, access_level: str = "public") -> List[Document]:
    """Load all .txt, .md, and .pdf files in a folder safely."""
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    docs: List[Document] = []
    for item in sorted(folder.iterdir()):
        if not item.is_file():
            continue
        ext = item.suffix.lower()
        if ext in (".txt", ".md"):
            doc = load_text_document(item, doc_type=ext.lstrip("."), access_level=access_level)
            if doc:
                docs.append(doc)
        elif ext == ".pdf":
            try:
                docs.extend(load_pdf_documents(item, doc_type="pdf", access_level=access_level))
            except Exception as exc:
                print(f"[Skipping unreadable PDF {item.name}: {sanitize_error_message(str(exc))}]")
    return docs


def validate_safe_url(url: str) -> None:
    """Block SSRF vectors: non-HTTP schemes, localhost, loopback, link-local, and private IPs."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked unsafe URL scheme '{parsed.scheme}': only http/https allowed.")
    host = (parsed.hostname or "").strip().lower()
    if not host or host in ("localhost", "ip6-localhost"):
        raise ValueError(f"Blocked internal/loopback host '{host}' (SSRF defense).")
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
            raise ValueError(f"Blocked non-public IP address '{host}' (SSRF defense).")
    except ValueError as exc:
        if "Blocked non-public IP" in str(exc):
            raise
        # Hostname rather than literal IP: attempt resolution check if online
        try:
            resolved = socket.gethostbyname(host)
            ip_obj = ipaddress.ip_address(resolved)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                raise ValueError(f"Blocked hostname resolving to internal IP '{resolved}'.")
        except socket.gaierror:
            pass


def load_web_page(
    url: str,
    fallback_html: Optional[str] = None,
    timeout_s: int = 6,
) -> Document:
    """
    Fetch an external web page safely with SSRF validation and strip HTML tags.
    Falls back cleanly to `fallback_html` if offline or the remote request fails.
    """
    validate_safe_url(url)
    html_text: Optional[str] = None
    source_note = "live_web"
    try:
        resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "RAGFromScratchCourse/1.0"})
        resp.raise_for_status()
        html_text = resp.text
    except Exception:
        if fallback_html is None:
            raise
        html_text = fallback_html
        source_note = "offline_fallback"

    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    clean_text = soup.get_text("\n", strip=True)
    return Document(
        page_content=clean_text,
        metadata={"source": url, "doc_type": "web", "fetch_mode": source_note, "access_level": "public"},
    )


def load_corpus_documents(data_dir: Path, include_pdf: bool = True) -> List[Document]:
    """Load the official Novasphere course corpus with normalized metadata and doc IDs."""
    docs: List[Document] = []
    specs = [
        ("company_handbook.txt", "handbook", "employee", "doc-handbook"),
        ("product_faq.md", "faq", "public", "doc-faq"),
        ("recent_changelog.md", "changelog", "public", "doc-changelog"),
    ]
    for fname, dtype, level, did in specs:
        p = data_dir / fname
        if p.exists():
            d = load_text_document(p, doc_type=dtype, access_level=level, doc_id=did)
            if d:
                docs.append(d)

    pdf_path = data_dir / "policy_brief.pdf"
    if include_pdf and pdf_path.exists():
        docs.extend(load_pdf_documents(pdf_path, doc_type="safety", access_level="executive", doc_id="doc-safety"))
    return docs

