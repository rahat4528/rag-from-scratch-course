"""
Generate and verify SHA-256 manifest of all tracked upstream base files.
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "base_manifest.sha256"


def create_manifest() -> None:
    upstream_dir = ROOT / "upstream"
    files = sorted(p for p in upstream_dir.glob("*") if p.is_file())
    lines = []
    for p in files:
        rel = f"upstream/{p.name}"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} SHA-256 base manifest entries:")
    for l in lines:
        print(f"  {l}")


def verify_manifest() -> bool:
    if not MANIFEST.exists():
        print(f"[FAIL] Manifest missing: {MANIFEST}")
        return False
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
            print(f"[FAIL] Modified base file: {rel} ({actual_sha} != {expected_sha})")
            all_ok = False
        else:
            print(f"[PASS] {rel} byte-identical")
    if all_ok:
        print(f"[PASS] Verified all {len(entries)} upstream base files byte-identical.")
    return all_ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "verify":
        ok = verify_manifest()
        raise SystemExit(0 if ok else 1)
    create_manifest()
