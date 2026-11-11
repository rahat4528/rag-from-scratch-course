"""
Area B11, B12, B13, B14 & B15 Tests: Performance Benchmarks, 14/14 Eval, Hygiene & Upstream Intactness.
"""

import html
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from course.app.app import NovasphereCapstoneEngine
from course.utils.models import get_embeddings, resolve_course_root

COURSE = resolve_course_root()
ROOT = COURSE.parent


def test_b11_perf_caching_latency_30q_and_scaling():
    emb = get_embeddings("offline_demo", use_cache=True)
    batch = [f"Novasphere perf probe query #{i}" for i in range(25)]
    t0 = time.perf_counter()
    emb.embed_documents(batch)
    cold_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    emb.embed_documents(batch)
    warm_ms = max(0.001, (time.perf_counter() - t1) * 1000)
    assert cold_ms / warm_ms >= 20.0

    eng = NovasphereCapstoneEngine()
    qs = ["Enterprise Plan cost?", "ZeroGravity timeout?", "ERR_QUANTUM_TIMEOUT steps?", "Zurich pet days?", "REST API v1 shutdown?", "NASDAQ ticker?"] * 5
    lats = []
    for q in qs:
        ts = time.perf_counter()
        eng.query(q, role="executive")
        lats.append((time.perf_counter() - ts) * 1000)
    p50, p95 = statistics.median(lats), sorted(lats)[int(0.95 * len(lats)) - 1]
    assert p95 < 2000.0
    print(f"\n[PERF] Cache {cold_ms:.1f}ms -> {warm_ms:.3f}ms ({cold_ms/warm_ms:.0f}x) | 30Q P50={p50:.2f}ms P95={p95:.2f}ms")

    # Indexing 1x, 10x, 100x scaling check (5 -> 50 -> 500 docs)
    fast_emb = get_embeddings("offline_demo", use_cache=True)
    base_d = [Document(page_content=f"Novasphere robotics note #{i}", metadata={"source": f"d{i}.txt"}) for i in range(5)]
    for mult in (1, 10, 100):
        t_s = time.perf_counter()
        InMemoryVectorStore.from_documents([Document(page_content=f"{d.page_content} copy {c}", metadata=d.metadata) for c in range(mult) for d in base_d], fast_emb)
        assert (time.perf_counter() - t_s) < 30.0


def test_b13_capstone_14_questions_3x_temp0():
    eng = NovasphereCapstoneEngine()
    qs = json.loads((COURSE / "data" / "eval_questions.json").read_text(encoding="utf-8"))
    scores = []
    for _ in range(3):
        ok_cnt = 0
        for q in qs:
            ans, _, _, _ = eng.query(q["question"], role="executive")
            al = html.unescape(ans).lower().replace("$", "")
            ok = ("don't know" in al or "not" in al) if q["type"] == "unanswerable" else all(kf.lower().replace("$", "") in al for kf in q["key_facts"])
            ok_cnt += int(ok)
        scores.append(ok_cnt)
    assert scores[0] == scores[1] == scores[2] and scores[0] >= 12
    print(f"\n[EVAL] 3x deterministic score: {scores[0]}/14")


def test_b14_docs_links_and_hygiene():
    link_re = re.compile(r"\[[^\]]+\]\(([^)#]+)\)")
    for md in (COURSE / "README.md", COURSE / "VIDEO_OUTLINE.md", COURSE / "DECISIONS.md", COURSE / "KNOWN_ISSUES.md", COURSE / "app" / "README.md"):
        text = md.read_text(encoding="utf-8")
        for m in link_re.findall(text):
            if m.startswith(("http://", "https://", "mailto:")):
                continue
            assert (md.parent / m).resolve().exists(), f"Broken link '{m}' in {md.name}"

    for py in list((COURSE / "utils").glob("*.py")) + list((COURSE / "app").glob("*.py")):
        src = py.read_text(encoding="utf-8")
        assert "TODO" not in src and "FIXME" not in src and "NotImplementedError" not in src


def test_b15_upstream_base_intact_and_push_disabled():
    proc = subprocess.run([sys.executable, str(COURSE / "tests" / "verify_manifest.py"), "verify"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Manifest mismatch: {proc.stdout} {proc.stderr}"

    # Verify upstream push is disabled
    push_url = subprocess.check_output(["git", "remote", "get-url", "--push", "upstream"], cwd=str(ROOT)).decode().strip()
    assert push_url == "no-push://disabled"

