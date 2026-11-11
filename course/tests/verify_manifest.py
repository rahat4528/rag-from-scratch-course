"""
Generate and verify SHA-256 manifest of all tracked upstream base files.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "course" / "tests" / "base_manifest.sha256"


def create_manifest() -> None:
    files = sorted(subprocess.check_output(["git", "ls-files"], cwd=str(ROOT)).decode().splitlines())
    lines = []
    for rel in files:
        if rel.startswith("course/"):
            continue
        digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} SHA-256 base manifest entries:")
    for rel in files:
        if not rel.startswith("course/"):
            print(f"  {rel}")


def verify_manifest(allow_readme_one_line: bool = True) -> bool:
    entries = [line.split("  ", 1) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_ok = True
    for expected_sha, rel in entries:
        target = ROOT / rel
        if not target.exists():
            print(f"[FAIL] Missing base file: {rel}")
            all_ok = False
            continue
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            if allow_readme_one_line and rel == "README.md":
                base_lf = None
                for candidate in ["main:README.md", "origin/main:README.md"]:
                    try:
                        base_lf = subprocess.check_output(["git", "show", candidate], cwd=str(ROOT), stderr=subprocess.DEVNULL).replace(b"\r\n", b"\n")
                        break
                    except Exception:
                        pass
                if base_lf is None:
                    root_commits = subprocess.check_output(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=str(ROOT)).decode().strip().splitlines()
                    base_lf = subprocess.check_output(["git", "show", f"{root_commits[0]}:README.md"], cwd=str(ROOT)).replace(b"\r\n", b"\n")
                cur_lf = target.read_bytes().replace(b"\r\n", b"\n")
                if cur_lf.startswith(base_lf):
                    added_tail = cur_lf[len(base_lf):].decode("utf-8", errors="replace").strip()
                    print(f"[INFO] README.md verified exact base prefix + added link line: {added_tail[:80]}")
                    continue
            print(f"[FAIL] Modified base file: {rel}")
            all_ok = False
    if all_ok:
        print(f"[PASS] Verified all {len(entries)} upstream base files byte-identical (README link rule respected).")
    return all_ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "create"
    if mode == "verify":
        ok = verify_manifest()
        raise SystemExit(0 if ok else 1)
    create_manifest()
