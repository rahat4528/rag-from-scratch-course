"""
Area B4 & B6 Tests: Core Pipeline Components, Persistence, MMR, RRF, MRR, Faithfulness & Hybrid Determinism.
"""

import re
import tempfile
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from utils.helpers import load_corpus_documents
from utils.models import LocalHashEmbeddings, get_embeddings, get_llm, resolve_project_root

ROOT = resolve_project_root()
DATA_DIR = ROOT / "data"


def test_b4_loaders_unicode_and_spaces(tmp_path: Path):
    spaced = tmp_path / "safety notes (2026 final).txt"
    spaced.write_text("নিরাপত্তা — Zurich Pet Day Tuesday", encoding="utf-8")
    loaded = TextLoader(str(spaced), encoding="utf-8").load()
    assert "নিরাপত্তা" in loaded[0].page_content

    pdf_docs = PyPDFLoader(str(DATA_DIR / "policy_brief.pdf")).load()
    assert len(pdf_docs) == 1 and "Emergency Stop" in pdf_docs[0].page_content

    soup = BeautifulSoup("<div><script>bad()</script><p>gRPC v2</p></div>", "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()
    assert soup.get_text(" ", strip=True) == "gRPC v2"


def test_b4_splitter_boundaries_and_tiny_doc():
    splitter = RecursiveCharacterTextSplitter(chunk_size=140, chunk_overlap=35)
    sample = (
        "Novasphere OS provides automated telemetry collection, dual-channel safety watchdogs, "
        "ZeroGravity automatic peer failover within 40 milliseconds, and QuantumCache state "
        "synchronization for industrial robots and autonomous warehouse drones worldwide."
    )
    chunks = splitter.split_text(sample)
    assert len(chunks) >= 2
    assert set(chunks[0].split()) & set(chunks[1].split())
    assert splitter.split_text("Hi.") == ["Hi."]


def test_b4_embeddings_vector_store_dump_load_and_mmr(tmp_path: Path):
    emb = get_embeddings("local")
    v1 = emb.embed_query("ZeroGravity failover")
    v2 = emb.embed_query("ZeroGravity failover")
    assert len(v1) == 384
    assert v1 == v2
    assert len(emb.embed_query("")) == 384

    docs = load_corpus_documents(DATA_DIR)
    chunks = RecursiveCharacterTextSplitter(chunk_size=360, chunk_overlap=40).split_documents(docs)
    store = InMemoryVectorStore.from_documents(chunks, emb)

    # k larger than corpus subset does not crash
    tiny_store = InMemoryVectorStore.from_texts(["alpha", "beta"], emb)
    assert len(tiny_store.similarity_search("alpha", k=50)) == 2

    # Persistence parity
    dump_file = str(tmp_path / "vs.json")
    store.dump(dump_file)
    reloaded = InMemoryVectorStore.load(dump_file, emb)
    q = "Enterprise tier cost per month"
    assert store.similarity_search(q, k=2)[0].page_content == reloaded.similarity_search(q, k=2)[0].page_content

    # MMR diversity returns requested k distinct docs
    mmr = store.max_marginal_relevance_search("Enterprise failover HyperSync", k=3, fetch_k=10)
    assert len(mmr) == 3


def test_b6_rrf_math_and_mrr_math():
    def rrf(lists, k=60):
        sc, dm = {}, {}
        for l in lists:
            for r, d in enumerate(l, 1):
                dm[d.page_content] = d
                sc[d.page_content] = sc.get(d.page_content, 0.0) + 1.0 / (k + r)
        return sorted(sc.items(), key=lambda x: x[1], reverse=True)

    a, b = Document(page_content="A"), Document(page_content="B")
    res = rrf([[a, b], [b, a]], k=60)
    assert abs(res[0][1] - (1 / 61 + 1 / 62)) < 1e-12

    ranks = [1, 2, 0]
    mrr = sum(1 / r if r > 0 else 0.0 for r in ranks) / len(ranks)
    assert abs(mrr - 0.5) < 1e-12


def test_b6_faithfulness_judge_and_3x_determinism():
    llm = get_llm("local", temperature=0.0)
    faithful_prompt = (
        "Output FAITHFUL or HALLUCINATED.\n"
        "CONTEXT:\nNovasphere allows pets in Zurich on Tuesdays and Thursdays.\n"
        "ANSWER:\nPets are allowed on Tuesdays and Thursdays in Zurich.\nVERDICT:"
    )
    hallucinated_prompt = (
        "Output FAITHFUL or HALLUCINATED.\n"
        "CONTEXT:\nNovasphere allows pets in Zurich on Tuesdays and Thursdays.\n"
        "ANSWER:\nPets are allowed every day, and Novasphere offers free dog treats.\nVERDICT:"
    )
    v_faith = llm.invoke(faithful_prompt).content.upper()
    v_hall = llm.invoke(hallucinated_prompt).content.upper()
    assert "FAITHFUL" in v_faith
    assert "HALLUCINATED" in v_hall

    # Run grounded query 3 times at temp=0 verifying determinism
    qa_prompt = "<context>\nEnterprise Plan: $4,200 per month.\n</context>\nQuestion: How much does the Enterprise Plan cost per month?"
    outputs = [llm.invoke(qa_prompt).content for _ in range(3)]
    assert outputs[0] == outputs[1] == outputs[2]
    assert "4,200" in outputs[0]

