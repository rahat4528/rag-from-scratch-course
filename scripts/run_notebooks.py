"""
Automated Jupyter notebook execution runner.

Executes notebooks top-to-bottom via nbclient with fresh kernels, writes
real executed outputs back to disk, and prints exact timing and pass/fail stats.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path
from typing import List

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run_single_notebook(nb_path: Path, timeout_s: int = 300) -> float:
    with open(nb_path, "r", encoding="utf-8") as fp:
        nb = nbformat.read(fp, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout_s,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    t0 = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - t0

    with open(nb_path, "w", encoding="utf-8") as fp:
        nbformat.write(nb, fp)
    return elapsed


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    pattern = sys.argv[1] if len(sys.argv) > 1 else str(root / "notebooks" / "*.ipynb")
    files: List[Path] = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        print(f"No notebooks matched pattern: {pattern}")
        return 1

    print(f"Executing {len(files)} notebook(s) with real Jupyter kernels...")
    print("-" * 68)
    total_time = 0.0
    failures: List[str] = []

    for nb_path in files:
        try:
            elapsed = run_single_notebook(nb_path)
            total_time += elapsed
            print(f"[PASS] {nb_path.name:<45} ({elapsed:6.2f}s)")
        except CellExecutionError as exc:
            failures.append(f"{nb_path.name}: {exc}")
            print(f"[FAIL] {nb_path.name}")
        except Exception as exc:
            failures.append(f"{nb_path.name}: {exc}")
            print(f"[ERROR] {nb_path.name}: {exc}")

    print("-" * 68)
    print(f"Finished {len(files) - len(failures)}/{len(files)} notebooks cleanly in {total_time:.2f}s")
    if failures:
        print("\nFailure details:")
        for f in failures:
            print(f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

