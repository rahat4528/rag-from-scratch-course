"""
Shared utilities for the RAG From Scratch beginner course.
"""

from course.utils.helpers import (
    load_corpus_documents,
    load_folder_documents,
    load_pdf_documents,
    load_text_document,
    load_web_page,
    pretty_print_docs,
    print_qa_result,
    safe_html_escape,
    sanitize_error_message,
    validate_safe_url,
)
from course.utils.models import (
    CachedEmbeddings,
    LocalCrossEncoderReranker,
    LocalHashFallbackEmbeddings,
    LocalSentenceTransformerEmbeddings,
    OfflineDemoChatModel,
    get_cross_encoder,
    get_embeddings,
    get_llm,
    resolve_course_root,
)

__all__ = [
    "get_llm",
    "get_embeddings",
    "get_cross_encoder",
    "resolve_course_root",
    "OfflineDemoChatModel",
    "LocalSentenceTransformerEmbeddings",
    "LocalHashFallbackEmbeddings",
    "CachedEmbeddings",
    "LocalCrossEncoderReranker",
    "pretty_print_docs",
    "print_qa_result",
    "load_text_document",
    "load_pdf_documents",
    "load_folder_documents",
    "load_web_page",
    "load_corpus_documents",
    "sanitize_error_message",
    "safe_html_escape",
    "validate_safe_url",
]

