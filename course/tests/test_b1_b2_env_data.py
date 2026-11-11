"""
Area B1 & B2 Tests: Environment, Zero Deprecation Warnings, Package Audit Readiness & Data Corpus.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent.parent
COURSE = ROOT / "course"
DATA_DIR = COURSE / "data"


def test_b1_modern_imports_zero_deprecation_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        import langchain
        import langchain_core
        import langchain_text_splitters
        import langgraph
        import torch
        import sentence_transformers
        import gradio
        import matplotlib
        import pandas
        import sklearn
        import rank_bm25
        import pypdf
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model

    dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning) and "langchain" in str(w.message).lower()]
    assert not dep_msgs, f"Unexpected LangChain deprecation warnings: {dep_msgs}"
    assert langchain is not None and create_agent is not None and init_chat_model is not None


def test_b1_pip_check_and_env_example():
    proc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"pip check failed: {proc.stdout} {proc.stderr}"
    env_ex = (COURSE / ".env.example").read_text(encoding="utf-8")
    for key in ("RAG_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_CHAT_MODEL"):
        assert key in env_ex


def test_b2_corpus_files_exist_utf8_pdf():
    for fname in ("company_handbook.txt", "product_faq.md", "recent_changelog.md", "policy_brief.pdf", "eval_questions.json"):
        p = DATA_DIR / fname
        assert p.exists(), f"Missing {fname}"
        assert p.stat().st_size > 100
        if fname.endswith((".txt", ".md", ".json")):
            assert len(p.read_text(encoding="utf-8")) > 100

    reader = PdfReader(str(DATA_DIR / "policy_brief.pdf"))
    assert len(reader.pages) == 1
    pdf_txt = reader.pages[0].extract_text() or ""
    assert "Novasphere Dynamics" in pdf_txt and "3-meter" in pdf_txt and "role='executive'" in pdf_txt


def test_b2_eval_questions_grounding_and_unanswerables():
    qs = json.loads((DATA_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    assert len(qs) == 14
    corpus = "\n".join([
        (DATA_DIR / "company_handbook.txt").read_text(encoding="utf-8"),
        (DATA_DIR / "product_faq.md").read_text(encoding="utf-8"),
        (DATA_DIR / "recent_changelog.md").read_text(encoding="utf-8"),
        PdfReader(str(DATA_DIR / "policy_brief.pdf")).pages[0].extract_text() or "",
    ]).lower().replace("$", "")

    for q in qs:
        assert {"id", "type", "question", "expected_answer", "key_facts"}.issubset(q.keys())
        if q["type"] != "unanswerable":
            for kf in q["key_facts"]:
                assert kf.lower().replace("$", "") in corpus, f"Missing key fact '{kf}' for {q['id']}"
        else:
            for leak in ("nasdaq", "ticker symbol", "hardware warranty", "satellite uplink"):
                assert leak not in corpus

