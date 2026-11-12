"""
Shared utilities for the RAG From Scratch beginner course.
"""

from utils.models import (
    get_llm,
    get_embeddings,
    resolve_project_root,
    cosine_similarity,
    LocalGroundedChatModel,
    LocalHashEmbeddings,
    LocalSentenceTransformerEmbeddings,
    CachedEmbeddings,
)
from utils.helpers import (
    pretty_print_docs,
    print_qa_result,
    load_corpus_documents,
    sanitize_error_message,
)

__all__ = [
    "get_llm",
    "get_embeddings",
    "resolve_project_root",
    "cosine_similarity",
    "LocalGroundedChatModel",
    "LocalHashEmbeddings",
    "LocalSentenceTransformerEmbeddings",
    "CachedEmbeddings",
    "pretty_print_docs",
    "print_qa_result",
    "load_corpus_documents",
    "sanitize_error_message",
]
