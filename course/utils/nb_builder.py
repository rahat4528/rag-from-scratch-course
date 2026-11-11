"""
Shared nbformat v4 Jupyter Notebook builder for course/notebooks/00..15.

Enforces the exact 9-part educational structure:
  a) Title & one-sentence promise
  b) Learning objectives (3 bullets) & Prerequisites
  c) Colab-friendly Setup cell (imports, .env load, get_llm/get_embeddings)
  d) Concept section (terms with analogies + Mermaid/ASCII diagram)
  e) Step-by-step build (<= 16 lines per code cell, 'What just happened?' notes)
  f) Try it yourself (2-3 exercises, hints, collapsed/runnable solution)
  g) Common mistakes (3-5 bullets) & Recap (3-5 bullets)
  h) Next notebook teaser
  i) Video chapter outline (4-7 chapter titles and talking points)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


class CourseNotebookBuilder:
    def __init__(self, filename: str, max_code_lines: int = 16) -> None:
        self.filename = filename
        self.max_code_lines = max_code_lines
        self.cells: list = []

    def add_md(self, text: str) -> "CourseNotebookBuilder":
        self.cells.append(new_markdown_cell(text.strip()))
        return self

    def add_code(self, code: str) -> "CourseNotebookBuilder":
        cleaned = code.strip("\n")
        n_lines = len(cleaned.splitlines())
        if n_lines > self.max_code_lines:
            raise ValueError(
                f"Code cell in {self.filename} has {n_lines} lines (max {self.max_code_lines}):\n{cleaned[:120]}"
            )
        self.cells.append(new_code_cell(cleaned))
        return self

    def header(
        self,
        title: str,
        promise: str,
        objectives: List[str],
        prerequisites: str,
    ) -> "CourseNotebookBuilder":
        obj_md = "\n".join(f"- {o}" for o in objectives)
        return self.add_md(
            f"# {title}\n\n"
            f"**{promise}**\n\n"
            f"### Learning Objectives\n{obj_md}\n\n"
            f"### Prerequisites\n- {prerequisites}"
        )

    def setup(self, extra_lines: Optional[List[str]] = None) -> "CourseNotebookBuilder":
        self.add_md(
            "## 1. Setup\n"
            "Load `.env` configuration via `course/utils/models.py` and initialize our shared LLM and embeddings helpers."
        )
        lines = [
            "# If running in Google Colab, uncomment: !git clone https://github.com/rahat4528/rag-from-scratch-course-v2.git && %cd rag-from-scratch-course-v2 && pip install -q -r course/requirements.txt",
            "import sys",
            "from pathlib import Path",
            "repo_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'course' / 'utils').exists()), Path.cwd())",
            "if str(repo_root) not in sys.path: sys.path.insert(0, str(repo_root))",
            "from course.utils.models import get_llm, get_embeddings, resolve_course_root",
            "course_dir = resolve_course_root(); data_dir = course_dir / 'data'",
            "llm = get_llm(fallback_to_demo=True); embeddings = get_embeddings()",
        ]
        if extra_lines:
            lines.extend(extra_lines)
        lines.append("print(f'Ready | LLM={type(llm).__name__} | Embeddings={type(embeddings.base).__name__}')")
        return self.add_code("\n".join(lines))

    def concept(self, title: str, explanation: str, diagram: str) -> "CourseNotebookBuilder":
        return self.add_md(f"## 2. Concept: {title}\n\n{explanation.strip()}\n\n```\n{diagram.strip()}\n```")

    def start_build(self) -> "CourseNotebookBuilder":
        return self.add_md("## 3. Step-by-Step Build")

    def step(
        self,
        heading: str,
        why: str,
        code: str,
        what_happened: Optional[str] = None,
    ) -> "CourseNotebookBuilder":
        self.add_md(f"### {heading}\n{why.strip()}")
        self.add_code(code)
        if what_happened:
            self.add_md(f"> **What just happened?**\n> {what_happened.strip()}")
        return self

    def note(self, md_text: str) -> "CourseNotebookBuilder":
        return self.add_md(md_text.strip())

    def try_it(
        self,
        exercises: List[Tuple[str, str]],
        solution_code: str,
    ) -> "CourseNotebookBuilder":
        ex_md = "\n".join(f"{i}. **{t}** — *Hint:* {h}" for i, (t, h) in enumerate(exercises, 1))
        self.add_md(f"## 4. Try It Yourself\n{ex_md}")
        self.add_md("<details>\n<summary><strong>Click to reveal sample solution</strong></summary>\nRun the cell below to check one working solution.\n</details>")
        return self.add_code(solution_code)

    def wrap_up(
        self,
        mistakes: List[str],
        recap: List[str],
        next_nb: str,
        chapters: List[Tuple[str, str]],
    ) -> "CourseNotebookBuilder":
        m_md = "\n".join(f"- {m}" for m in mistakes)
        r_md = "\n".join(f"- {r}" for r in recap)
        v_md = "\n".join(f"{i}. **{c}** — {pts}" for i, (c, pts) in enumerate(chapters, 1))
        self.add_md(f"## 5. Common Mistakes\n{m_md}\n\n## 6. Recap\n{r_md}")
        self.add_md(f"## 7. Next Notebook\n{next_nb.strip()}")
        return self.add_md(f"## 8. Video Chapter Outline\n{v_md}")

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / self.filename
        nb = new_notebook(
            cells=self.cells,
            metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.13"},
            },
        )
        with open(path, "w", encoding="utf-8") as fp:
            nbformat.write(nb, fp)
        return path

