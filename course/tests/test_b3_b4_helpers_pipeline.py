"""
Area B3 & B4 Tests: Shared Helpers, Auth Error Paths, Loaders, Splitters, VectorStore & LangGraph.
"""

from pathlib import Path
from typing import TypedDict

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

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
)
from course.utils.models import (
    CachedEmbeddings,
    LocalSentenceTransformerEmbeddings,
    OfflineDemoChatModel,
    get_cross_encoder,
    get_embeddings,
    get_llm,
    resolve_course_root,
)

COURSE = resolve_course_root()
DATA_DIR = COURSE / "data"


def test_b3_providers_and_negative_auth_errors(monkeypatch):
    # Real neural embeddings & cross-encoder
    emb = get_embeddings("ollama")
    assert isinstance(emb, CachedEmbeddings)
    assert isinstance(emb.base, LocalSentenceTransformerEmbeddings)
    assert emb.base.is_neural, "SentenceTransformer neural backend must initialize cleanly without silent hash fallback!"

    ce = get_cross_encoder()
    assert ce.is_neural, "CrossEncoder neural reranker must load cleanly!"

    # Missing / invalid hosted keys must raise clean secret-free ValueError
    secret = "sk-placeholder-secret-value-99999"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    with pytest.raises(ValueError) as exc_o:
        get_llm("openai", fallback_to_demo=False)
    assert secret not in str(exc_o.value) and "OPENAI_API_KEY" in str(exc_o.value)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-replace")
    with pytest.raises(ValueError):
        get_llm("anthropic", fallback_to_demo=False)

    # Unreachable Ollama server must raise ConnectionError when strict
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59999")
    with pytest.raises(ConnectionError):
        get_llm("ollama", fallback_to_demo=False)

    # Explicit offline_demo code-path mode
    demo_llm = get_llm("offline_demo")
    assert isinstance(demo_llm, OfflineDemoChatModel)


def test_b3_display_and_sanitizers(capsys):
    pretty_print_docs([])
    assert "No documents retrieved" in capsys.readouterr().out
    big = Document(page_content="token " * 300, metadata={"source": "big.txt"})
    pretty_print_docs([big], max_chars=70)
    assert "..." in capsys.readouterr().out
    print_qa_result("Q", "A", [big])
    assert "big.txt" in capsys.readouterr().out

    cleaned = sanitize_error_message("Key sk-abcdef1234567890abcd leaked")
    assert "sk-abcdef" not in cleaned and "[REDACTED_SECRET]" in cleaned
    assert "<script>" not in safe_html_escape("<script>alert(1)</script>")


def test_b4_loaders_edge_cases(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n", encoding="utf-8")
    assert load_text_document(empty_file) is None

    with pytest.raises(FileNotFoundError):
        load_text_document(tmp_path / "missing.txt")

    uni = tmp_path / "নিরাপত্তা brief (2026).txt"
    uni.write_text("নিরাপত্তা — Zurich Tuesday", encoding="utf-8")
    assert "নিরাপত্তা" in load_text_document(uni).page_content

    corrupt_pdf = tmp_path / "broken.pdf"
    corrupt_pdf.write_bytes(b"%PDF-corrupt-binary")
    loaded_folder = load_folder_documents(tmp_path)
    assert len(loaded_folder) == 1  # Broken PDF skipped safely without crashing

    # Web offline fallback
    w = load_web_page("https://example.invalid/info", fallback_html="<html><body><p>Novasphere gRPC v2</p></body></html>")
    assert "gRPC v2" in w.page_content and w.metadata["fetch_mode"] == "offline_fallback"


def test_b4_splitting_vector_store_delete_dump_load_mmr_and_graph(tmp_path: Path):
    sp = RecursiveCharacterTextSplitter(chunk_size=140, chunk_overlap=35)
    sample = (
        "Novasphere OS handles high-frequency telemetry collection, dual safety perimeters, "
        "ZeroGravity peer failover under 40 milliseconds, and QuantumCache synchronization."
    )
    pieces = sp.split_text(sample)
    assert len(pieces) >= 2 and (set(pieces[0].split()) & set(pieces[1].split()))

    emb = get_embeddings("offline_demo")
    docs = load_corpus_documents(DATA_DIR)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)
    ids = [f"c-{i}" for i in range(len(chunks))]
    store = InMemoryVectorStore.from_documents(chunks, emb, ids=ids)

    # k > N safe, delete/withdraw doc by ID verified
    tiny = InMemoryVectorStore.from_texts(["A", "B"], emb, ids=["id-a", "id-b"])
    assert len(tiny.similarity_search("A", k=25)) == 2
    tiny.delete(ids=["id-a"])
    assert all(d.page_content != "A" for d in tiny.similarity_search("A", k=5))

    # Dump/load persistence & missing path error
    p = str(tmp_path / "idx.json")
    store.dump(p)
    reloaded = InMemoryVectorStore.load(p, emb)
    assert store.similarity_search("Enterprise Plan", k=1)[0].page_content == reloaded.similarity_search("Enterprise Plan", k=1)[0].page_content
    with pytest.raises(FileNotFoundError):
        InMemoryVectorStore.load(str(tmp_path / "missing.json"), emb)

    # MMR returns diverse hits
    assert len(store.max_marginal_relevance_search("Enterprise HyperSync", k=3, fetch_k=8)) == 3

    # LangGraph bounded termination
    class S(TypedDict):
        n: int
    g = StateGraph(S).add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc").add_conditional_edges("inc", lambda s: END if s["n"] >= 3 else "inc")
    assert g.compile().invoke({"n": 0})["n"] == 3

