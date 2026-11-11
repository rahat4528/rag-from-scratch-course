"""
Area B1 & B2 Tests: Environment Dependencies & Data Corpus Integrity.
"""

import json
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def test_b1_required_imports():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import langchain_community
    import langchain_core
    import langchain_text_splitters
    import langgraph
    import sentence_transformers
    import rank_bm25
    import gradio
    import pandas
    import sklearn
    import pypdf
    import nbformat
    import nbclient
    assert langchain_core is not None and gradio is not None


def test_b1_pip_check_clean():
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"pip check failed: {proc.stdout} {proc.stderr}"


def test_b2_corpus_files_exist_utf8():
    required = [
        "company_handbook.txt",
        "product_faq.md",
        "recent_changelog.md",
        "policy_brief.pdf",
        "eval_questions.json",
    ]
    for fname in required:
        p = DATA_DIR / fname
        assert p.exists(), f"Missing {fname}"
        assert p.stat().st_size > 100, f"Too small: {fname}"
        if fname.endswith((".txt", ".md", ".json")):
            txt = p.read_text(encoding="utf-8")
            assert len(txt) > 100


def test_b2_pdf_extraction():
    reader = PdfReader(str(DATA_DIR / "policy_brief.pdf"))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text() or ""
    assert "Novasphere Dynamics" in text
    assert "3-meter" in text
    assert "role='executive'" in text


def test_b2_eval_questions_schema_and_grounding():
    eval_qs = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    assert len(eval_qs) == 14
    valid_types = {"exact_number", "multi_document", "factual", "unanswerable"}

    corpus = "\n".join([
        (DATA_DIR / "company_handbook.txt").read_text(encoding="utf-8"),
        (DATA_DIR / "product_faq.md").read_text(encoding="utf-8"),
        (DATA_DIR / "recent_changelog.md").read_text(encoding="utf-8"),
        PdfReader(str(DATA_DIR / "policy_brief.pdf")).pages[0].extract_text() or "",
    ]).lower()

    for q in eval_qs:
        assert {"id", "type", "question", "expected_answer", "key_facts"}.issubset(q.keys())
        assert q["type"] in valid_types
        assert len(q["key_facts"]) >= 2
        if q["type"] != "unanswerable":
            for kf in q["key_facts"]:
                clean_kf = kf.lower().replace("$", "")
                assert clean_kf in corpus.replace("$", ""), f"Key fact '{kf}' missing from corpus for {q['id']}"
        else:
            for leak in ("nasdaq", "ticker symbol", "hardware warranty", "satellite uplink"):
                assert leak not in corpus, f"Unanswerable term '{leak}' leaked into corpus"

