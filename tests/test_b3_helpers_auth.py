"""
Area B3 Tests: Shared Helpers, Provider Configuration, Auth Error Handling & Sanitizers.
"""

import pytest
from langchain_core.documents import Document

from utils.helpers import (
    load_corpus_documents,
    pretty_print_docs,
    print_qa_result,
    safe_html_escape,
    sanitize_error_message,
)
from utils.models import (
    CachedEmbeddings,
    LocalGroundedChatModel,
    get_embeddings,
    get_llm,
    resolve_project_root,
)


def test_b3_project_root_and_default_local_models(monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", "local")
    root = resolve_project_root()
    assert (root / "utils" / "models.py").exists()
    assert (root / "data" / "company_handbook.txt").exists()

    llm = get_llm("local")
    emb = get_embeddings("local")
    assert isinstance(llm, LocalGroundedChatModel)
    assert isinstance(emb, CachedEmbeddings)
    vec = emb.embed_query("Enterprise Plan")
    assert len(vec) == 384


def test_b3_missing_or_placeholder_keys_fail_safely(monkeypatch):
    fake_secret = "sk-placeholder-do-not-leak-me"
    monkeypatch.setenv("OPENAI_API_KEY", fake_secret)
    with pytest.raises(ValueError) as exc_openai:
        get_llm("openai")
    assert fake_secret not in str(exc_openai.value)
    assert "OPENAI_API_KEY" in str(exc_openai.value)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-replace")
    with pytest.raises(ValueError) as exc_anth:
        get_llm("anthropic")
    assert "ANTHROPIC_API_KEY" in str(exc_anth.value)

    with pytest.raises(ValueError):
        get_llm("nonexistent-provider-xyz")


def test_b3_display_and_sanitization_helpers(capsys):
    pretty_print_docs([])
    captured = capsys.readouterr()
    assert "No documents retrieved" in captured.out

    giant_doc = Document(page_content="word " * 400, metadata={"source": "big.txt", "tag": "test"})
    pretty_print_docs([giant_doc], max_chars=80)
    out = capsys.readouterr().out
    assert "..." in out and "big.txt" in out

    print_qa_result("Q?", "Ans.", [giant_doc])
    assert "big.txt" in capsys.readouterr().out

    leaked = "Error connecting with sk-1234567890abcdef1234 and Bearer eyJhbGciOi.payload.sig"
    cleaned = sanitize_error_message(leaked)
    assert "sk-1234" not in cleaned
    assert "eyJhbGci" not in cleaned
    assert "[REDACTED_SECRET]" in cleaned

    escaped = safe_html_escape("<script>alert('xss')</script>")
    assert "<script>" not in escaped and "&lt;script&gt;" in escaped


def test_b3_corpus_loader_metadata():
    root = resolve_project_root()
    docs = load_corpus_documents(root / "data", include_pdf=True)
    assert len(docs) == 4
    levels = {d.metadata["source"]: d.metadata["access_level"] for d in docs}
    assert levels["company_handbook.txt"] == "employee"
    assert levels["product_faq.md"] == "public"
    assert levels["recent_changelog.md"] == "public"
    assert levels["policy_brief.pdf"] == "executive"

