"""
Area B5 Tests: Comprehensive Notebook Execution, Structural Template & Size Verification.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def test_b5_all_16_notebooks_exist_and_under_1mb():
    nbs = sorted(NB_DIR.glob("*.ipynb"))
    assert len(nbs) == 16, f"Expected 16 notebooks, found {len(nbs)}"
    for idx in range(16):
        prefix = f"{idx:02d}_"
        assert any(p.name.startswith(prefix) for p in nbs), f"Missing notebook {prefix}*"
    for nb_path in nbs:
        size_kb = nb_path.stat().st_size / 1024
        assert size_kb < 1024, f"{nb_path.name} exceeded 1 MB ({size_kb:.1f} KB)"


def test_b5_executed_cleanly_zero_errors_and_template_compliant():
    nbs = sorted(NB_DIR.glob("*.ipynb"))
    required_sections = [
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
    secret_pat = re.compile(r"sk-[A-Za-z0-9]{20,}")

    for nb_path in nbs:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        full_md = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")

        for sec in required_sections:
            assert sec in full_md, f"{nb_path.name} missing required section '{sec}'"
        assert "What just happened?" in full_md, f"{nb_path.name} missing 'What just happened?' notes"

        code_cells = [c for c in cells if c["cell_type"] == "code"]
        assert len(code_cells) >= 5, f"{nb_path.name} has too few code cells ({len(code_cells)})"

        for ci, cell in enumerate(code_cells):
            src = "".join(cell["source"])
            lines = src.splitlines()
            assert len(lines) <= 16, f"{nb_path.name} code cell {ci} has {len(lines)} lines (>16)"
            assert cell.get("execution_count") is not None, f"{nb_path.name} code cell {ci} was NOT executed!"
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", f"{nb_path.name} cell {ci} raised error: {out}"
                out_text = "".join(out.get("text", []))
                assert not secret_pat.search(out_text), f"Potential secret leaked in {nb_path.name}"
                assert "TODO" not in src and "NotImplementedError" not in src

