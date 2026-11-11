"""
Repository working-tree + git-history secret scanner for B10 security verification.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

PATTERNS = [
    ("OpenAI Key", re.compile(r"sk-(?:proj-|ant-|live-|test-)?[A-Za-z0-9]{20,}")),
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub Token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("Google API Key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

ALLOWLIST_SUBSTRINGS = (
    "sk-placeholder",
    "sk-ant-replace",
    "sk-secretmustnotleak",
    "sk-replace-with-your-key",
    "sk-1234567890",
    "sk-abcdef1234567890",
)


def scan_text(label: str, text: str) -> list[str]:
    hits = []
    for name, pat in PATTERNS:
        for m in pat.finditer(text):
            val = m.group(0)
            if any(a in val for a in ALLOWLIST_SUBSTRINGS):
                continue
            hits.append(f"{label}: matched {name} ({val[:10]}...)")
    return hits


def main() -> int:
    findings = []
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=str(ROOT)).decode().splitlines())
    course_files = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "course").rglob("*") if p.is_file() and "__pycache__" not in str(p)]
    all_targets = sorted(tracked | set(course_files))
    for rel in all_targets:
        p = ROOT / rel
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            findings.extend(scan_text(f"TREE:{rel}", content))
        except Exception:
            pass

    history = subprocess.check_output(["git", "log", "-p", "--all"], cwd=str(ROOT)).decode("utf-8", errors="ignore")
    findings.extend(scan_text("GIT_HISTORY", history))

    if findings:
        print("[FAIL] Secret scan detected potential leaks:")
        for f in findings:
            print(" ", f)
        return 1

    print(f"[PASS] Clean secret scan across {len(all_targets)} tree/course files + full git history (zero leaked credentials).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

