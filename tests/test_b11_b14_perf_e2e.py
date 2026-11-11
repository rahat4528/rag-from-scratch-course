"""
Area B11, B13 & B14 Tests: Performance Profiling, 14/14 Capstone Benchmark, Scaling & Hygiene.
"""

import html
import json
import re
import statistics
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.app import NovasphereRAGEngine
from utils.models import get_embeddings

ROOT = Path(__file__).resolve().parent.parent


def test_b11_cache_speedup_and_query_latency():
    emb = get_embeddings("local", use_cache=True)
    queries = [f"Novasphere benchmark latency test sentence #{i}" for i in range(25)]

    t0 = time.perf_counter()
    emb.embed_documents(queries)
    cold_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    emb.embed_documents(queries)
    warm_ms = max(0.001, (time.perf_counter() - t1) * 1000)
    stats = emb.cache_stats()
    assert stats["hits"] >= 25
    assert cold_ms / warm_ms >= 20.0, f"Cache speedup too low: {cold_ms/warm_ms:.1f}x"

    # 30-query P50 & P95 profile
    engine = NovasphereRAGEngine()
    test_qs = [
        "Enterprise Plan monthly cost",
        "ZeroGravity timeout threshold",
        "ERR_QUANTUM_TIMEOUT resolution",
        "Zurich hub pet days",
        "ROS2 distributions supported",
        "NASDAQ stock ticker",
    ] * 5
    latencies = []
    for q in test_qs:
        ts = time.perf_counter()
        engine.query(q, role="executive")
        latencies.append((time.perf_counter() - ts) * 1000)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    assert p95 < 1500.0, f"P95 query latency too high: {p95:.1f}ms"
    print(f"\n[PERF] Cold={cold_ms:.1f}ms | Warm={warm_ms:.3f}ms ({cold_ms/warm_ms:.0f}x) | Q P50={p50:.2f}ms | Q P95={p95:.2f}ms")


def test_b11_corpus_scaling_50x():
    emb = get_embeddings("local", use_cache=False)
    base_docs = [Document(page_content=f"Novasphere sensor cluster operational doc #{i}", metadata={"source": f"doc_{i}.txt"}) for i in range(5)]
    docs_50x = [Document(page_content=f"{d.page_content} copy {c}", metadata=d.metadata) for c in range(50) for d in base_docs]
    t0 = time.perf_counter()
    store = InMemoryVectorStore.from_documents(docs_50x, emb)
    elapsed_s = time.perf_counter() - t0
    assert len(store.similarity_search("sensor cluster", k=3)) == 3
    assert elapsed_s < 30.0


def test_b13_capstone_14_questions_3x_temp0():
    engine = NovasphereRAGEngine()
    eval_qs = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    assert len(eval_qs) == 14

    scores = []
    for run in range(3):
        passed = 0
        for q in eval_qs:
            ans, _, _ = engine.query(q["question"], role="executive")
            al = html.unescape(ans).lower().replace("$", "")
            if q["type"] == "unanswerable":
                ok = ("don't know" in al or "do not contain" in al or "not" in al)
            else:
                ok = all(kf.lower().replace("$", "") in al for kf in q["key_facts"])
            passed += int(ok)
        scores.append(passed)
    assert scores[0] == scores[1] == scores[2], f"Non-deterministic runs at temp=0: {scores}"
    assert scores[0] >= 12, f"Expected >= 12/14 benchmark score, got {scores[0]}/14"
    print(f"\n[BENCHMARK] Capstone 3x deterministic score: {scores[0]}/14 ({100*scores[0]/14:.1f}%)")


def test_b14_markdown_links_and_code_hygiene():
    md_files = [ROOT / "README.md", ROOT / "VIDEO_OUTLINE.md", ROOT / "app" / "README.md"]
    link_re = re.compile(r"\[[^\]]+\]\(([^)#]+)\)")
    for md in md_files:
        text = md.read_text(encoding="utf-8")
        for match in link_re.findall(text):
            if match.startswith(("http://", "https://", "mailto:")):
                continue
            target = (md.parent / match).resolve()
            assert target.exists(), f"Broken relative link '{match}' in {md.name}"

    # Verify zero TODO / NotImplementedError in utils/ and app/
    for py in list((ROOT / "utils").glob("*.py")) + list((ROOT / "app").glob("*.py")):
        src = py.read_text(encoding="utf-8")
        assert "TODO" not in src and "FIXME" not in src and "NotImplementedError" not in src
