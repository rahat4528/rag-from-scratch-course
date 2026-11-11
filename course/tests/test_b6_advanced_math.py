"""
Area B6 Tests: Advanced RAG Math Verification (RRF, MRR, HitRate, CrossEncoder Reranking, Hybrid, Determinism).
"""

import re
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from course.utils.helpers import load_corpus_documents
from course.utils.models import get_cross_encoder, get_embeddings, get_llm, resolve_course_root

DATA_DIR = resolve_course_root() / "data"


def test_b6_exact_rrf_and_mrr_mathematics():
    def rrf(lists, k=60):
        sc = {}
        for lst in lists:
            for rank, d in enumerate(lst, start=1):
                sc[d.page_content] = sc.get(d.page_content, 0.0) + 1.0 / (k + rank)
        return sorted(sc.items(), key=lambda x: x[1], reverse=True)

    a, b = Document(page_content="A"), Document(page_content="B")
    res = rrf([[a, b], [b, a]], k=60)
    assert abs(res[0][1] - (1.0 / 61 + 1.0 / 62)) < 1e-12

    ranks = [1, 2, 0]
    mrr = sum(1.0 / r for r in ranks if r > 0) / len(ranks)
    assert abs(mrr - 0.5) < 1e-12


def test_b6_cross_encoder_and_hybrid_bm25_precision():
    emb = get_embeddings("offline_demo")
    ce = get_cross_encoder()
    chunks = RecursiveCharacterTextSplitter(chunk_size=640, chunk_overlap=140, separators=["\n## ", "\n### ", "\n\n", "\n- ", "\n", " ", ""]).split_documents(load_corpus_documents(DATA_DIR))
    store = InMemoryVectorStore.from_documents(chunks, emb)

    q = "What exact steps resolve ERR_QUANTUM_TIMEOUT port 9872?"
    cands = store.similarity_search(q, k=8)
    reranked = ce.rerank(q, cands, top_n=2)
    assert "9872" in reranked[0][1].page_content and "novasphere-edge" in reranked[0][1].page_content

    # Hybrid BM25 + Dense RRF
    tok = lambda s: re.findall(r"[a-zA-Z0-9_]+", s.lower())
    bm25 = BM25Okapi([tok(c.page_content) for c in chunks])
    bm_top = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: bm25.get_scores(tok(q))[i], reverse=True)[:4]]
    assert any("9872" in d.page_content for d in bm_top)


def test_b6_codepath_3x_determinism_and_faithfulness():
    llm = get_llm("offline_demo", temperature=0.0)
    prompt = "<context>\nEnterprise Plan: $4,200 per month.\n</context>\nQuestion: How much does the Enterprise Plan cost per month?"
    outs = [llm.invoke(prompt).content for _ in range(3)]
    assert outs[0] == outs[1] == outs[2] and "4,200" in outs[0]

    v_f = llm.invoke("FAITHFUL or HALLUCINATED\nCONTEXT:\nPets allowed Tuesday.\nANSWER:\nPets allowed Tuesday.\nVERDICT:").content.upper()
    v_h = llm.invoke("FAITHFUL or HALLUCINATED\nCONTEXT:\nPets allowed Tuesday.\nANSWER:\nPets allowed daily with free treats.\nVERDICT:").content.upper()
    assert "FAITHFUL" in v_f and "HALLUCINATED" in v_h

