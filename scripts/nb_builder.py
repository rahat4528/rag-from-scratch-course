"""
Shared Jupyter Notebook factory for the RAG From Scratch beginner course.

Enforces the unified 9-part educational template across all 16 notebooks and
validates code-cell conciseness so every cell stays beginner-friendly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


class NotebookBuilder:
    """Fluent builder producing clean, well-structured nbformat v4 notebooks."""

    def __init__(self, filename: str, max_code_lines: int = 16) -> None:
        self.filename = filename
        self.max_code_lines = max_code_lines
        self.cells: list = []

    def add_md(self, text: str) -> "NotebookBuilder":
        self.cells.append(new_markdown_cell(text.strip()))
        return self

    def add_code(self, code: str, check_len: bool = True) -> "NotebookBuilder":
        cleaned = code.strip("\n")
        line_count = len(cleaned.splitlines())
        if check_len and line_count > self.max_code_lines:
            raise ValueError(
                f"Code cell in {self.filename} has {line_count} lines "
                f"(limit {self.max_code_lines}):\n{cleaned[:120]}..."
            )
        self.cells.append(new_code_cell(cleaned))
        return self

    def header(
        self,
        title: str,
        promise: str,
        objectives: List[str],
        prerequisites: str,
    ) -> "NotebookBuilder":
        obj_md = "\n".join(f"- {o}" for o in objectives)
        md = (
            f"# {title}\n\n"
            f"**{promise}**\n\n"
            f"### Learning Objectives\n"
            f"{obj_md}\n\n"
            f"### Prerequisites\n"
            f"- {prerequisites}"
        )
        return self.add_md(md)

    def setup_section(self, extra_imports: str = "") -> "NotebookBuilder":
        self.add_md(
            "## 1. Setup\n"
            "Let's connect this notebook to our shared project helpers (`utils/models.py`), "
            "load `.env` (defaulting to free offline `local` mode), and initialize our LLM and embeddings."
        )
        code_lines = [
            "# !pip install -q -r ../requirements.txt  # Uncomment if running in a fresh remote environment",
            "import sys",
            "from pathlib import Path",
            "root = Path.cwd() if (Path.cwd() / 'utils').exists() else Path.cwd().parent",
            "if str(root) not in sys.path: sys.path.insert(0, str(root))",
            "from utils.models import get_llm, get_embeddings, resolve_project_root",
            "project_root = resolve_project_root()",
            "data_dir = project_root / 'data'",
            "llm = get_llm()",
            "embeddings = get_embeddings()",
        ]
        if extra_imports.strip():
            code_lines.append(extra_imports.strip())
        code_lines.append("print(f'Ready! Root={project_root.name} | LLM={type(llm).__name__}')")
        return self.add_code("\n".join(code_lines))

    def concept_section(self, title: str, explanation: str, diagram: str) -> "NotebookBuilder":
        md = (
            f"## 2. Concept: {title}\n\n"
            f"{explanation.strip()}\n\n"
            f"```\n{diagram.strip()}\n```"
        )
        return self.add_md(md)

    def start_build(self) -> "NotebookBuilder":
        return self.add_md("## 3. Step-by-Step Build")

    def step(
        self,
        heading: str,
        why: str,
        code: str,
        what_happened: Optional[str] = None,
    ) -> "NotebookBuilder":
        self.add_md(f"### {heading}\n{why.strip()}")
        self.add_code(code)
        if what_happened:
            self.add_md(f"> **What just happened?**\n> {what_happened.strip()}")
        return self

    def try_it_yourself(
        self,
        exercises: List[Tuple[str, str]],
        solution_code: str,
    ) -> "NotebookBuilder":
        ex_md = "\n".join(
            f"{i}. **{title}** — *Hint:* {hint}" for i, (title, hint) in enumerate(exercises, 1)
        )
        self.add_md(
            "## 4. Try It Yourself\n"
            "Experiment by editing the starter ideas below before peeking at the solution!\n\n"
            f"{ex_md}"
        )
        self.add_md("<details>\n<summary><strong>Click to reveal sample solution</strong></summary>\n\nRun the cell below to verify your solution.\n</details>")
        return self.add_code(solution_code)

    def wrap_up(
        self,
        mistakes: List[str],
        recap: List[str],
        next_teaser: str,
        video_chapters: List[Tuple[str, str]],
    ) -> "NotebookBuilder":
        m_md = "\n".join(f"- {m}" for m in mistakes)
        r_md = "\n".join(f"- {r}" for r in recap)
        v_md = "\n".join(
            f"{i}. **{title}** — {pts}" for i, (title, pts) in enumerate(video_chapters, 1)
        )
        self.add_md(f"## 5. Common Mistakes\n{m_md}\n\n## 6. Recap\n{r_md}")
        self.add_md(f"## 7. Next Notebook\n{next_teaser.strip()}")
        self.add_md(f"## 8. Video Chapter Outline\n{v_md}")
        return self

    def save(self, notebooks_dir: Path) -> Path:
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        out_path = notebooks_dir / self.filename
        nb = new_notebook(
            cells=self.cells,
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.13"},
            },
        )
        with open(out_path, "w", encoding="utf-8") as fp:
            nbformat.write(nb, fp)
        return out_path

