"""
Area B5 Tests: All 16 Notebooks Execution State, Zero Errors/Deprecations, Template & <=16-Line Cells.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NB_DIR = ROOT / "course" / "notebooks"


def test_b5_all_16_notebooks_present_under_1mb():
    nbs = sorted(NB_DIR.glob("*.ipynb"))
    assert len(nbs) == 16
    for i in range(16):
        assert any(p.name.startswith(f"{i:02d}_") for p in nbs), f"Missing {i:02d}_*"
    for p in nbs:
        assert p.stat().st_size < 1024 * 1024, f"{p.name} too large"


def test_b5_executed_zero_errors_zero_deprecation_and_template():
    required = [
        "Learning Objectives",
        "Prerequisites",
        "1. Setup",
        "2. Concept:",
        "3. Step-by-Step Build",
        "4. Try It Yourself",
        "5. Common Mistakes",
        "6. Recap",
        "7. Next Notebook",
        "8. Video Chapter Outline",
    ]
    secret_re = re.compile(r"sk-[A-Za-z0-9]{20,}")
    for p in sorted(NB_DIR.glob("*.ipynb")):
        nb = json.loads(p.read_text(encoding="utf-8"))
        md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        for req in required:
            assert req in md, f"{p.name} missing '{req}'"
        assert "What just happened?" in md, f"{p.name} missing 'What just happened?'"

        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 5
        for ci, c in enumerate(code_cells):
            src = "".join(c["source"])
            assert len(src.splitlines()) <= 16, f"{p.name} cell {ci} has >16 lines"
            assert c.get("execution_count") is not None, f"{p.name} cell {ci} not executed"
            assert "TODO" not in src and "NotImplementedError" not in src
            for out in c.get("outputs", []):
                assert out.get("output_type") != "error", f"{p.name} cell {ci} error: {out}"
                txt = "".join(out.get("text", []))
                assert not secret_re.search(txt), f"Secret leak in {p.name}"
                assert "langchain-community` is being sunset" not in txt, f"Deprecation warning in {p.name}"

