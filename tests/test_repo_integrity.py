"""
Anti-regression and single-canonical-version integrity test suite.

Ensures that no duplicates, stray packaging files, or legacy version references
can ever be reintroduced into the repository.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _get_tracked_files() -> list[str]:
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def test_no_identical_files_outside_upstream():
    """Fails if two tracked files outside upstream/ have the same SHA-256 hash."""
    tracked = _get_tracked_files()
    hash_map: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []

    for rel_path in tracked:
        if rel_path.startswith("upstream/"):
            continue
        file_path = ROOT / rel_path
        if not file_path.is_file():
            continue
        sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if sha in hash_map:
            duplicates.append((rel_path, hash_map[sha], sha))
        else:
            hash_map[sha] = rel_path

    assert not duplicates, f"Identical file duplicates found outside upstream/: {duplicates}"


def test_single_canonical_packaging_and_config():
    """Fails if more than one Dockerfile, requirements.txt, .env.example, or README.md exists outside upstream/."""
    tracked = _get_tracked_files()
    non_upstream = [f for f in tracked if not f.startswith("upstream/")]

    dockerfiles = [f for f in non_upstream if Path(f).name.lower() == "dockerfile"]
    requirements = [f for f in non_upstream if Path(f).name.lower() == "requirements.txt"]
    env_examples = [f for f in non_upstream if Path(f).name.lower() == ".env.example"]
    readmes = [f for f in non_upstream if Path(f).name.lower() == "readme.md"]

    assert len(dockerfiles) == 1, f"Expected exactly 1 Dockerfile outside upstream/, found: {dockerfiles}"
    assert len(requirements) == 1, f"Expected exactly 1 requirements.txt outside upstream/, found: {requirements}"
    assert len(env_examples) == 1, f"Expected exactly 1 .env.example outside upstream/, found: {env_examples}"
    assert len(readmes) == 1, f"Expected exactly 1 README.md outside upstream/, found: {readmes}"


def test_notebooks_canonical_numbering_and_placement():
    """Fails if notebooks/ does not hold exactly 16 notebooks numbered 00-15 once each,

    or any .ipynb exists outside notebooks/ and upstream/.
    """
    tracked = _get_tracked_files()
    all_notebooks = [f for f in tracked if f.endswith(".ipynb")]

    # Check for notebooks outside allowed directories
    stray_notebooks = [f for f in all_notebooks if not (f.startswith("notebooks/") or f.startswith("upstream/"))]
    assert not stray_notebooks, f"Notebooks found outside notebooks/ and upstream/: {stray_notebooks}"

    # Check notebooks/ specifically
    course_notebooks = sorted([Path(f).name for f in all_notebooks if f.startswith("notebooks/")])
    assert len(course_notebooks) == 16, f"Expected exactly 16 course notebooks, found {len(course_notebooks)}: {course_notebooks}"

    expected_prefixes = [f"{i:02d}_" for i in range(16)]
    actual_prefixes = [name[:3] for name in course_notebooks]
    assert actual_prefixes == expected_prefixes, (
        f"Notebook prefixes must be 00_ through 15_ once each. Got: {actual_prefixes}"
    )


def test_no_version_flavored_leftovers_or_tracked_artifacts():
    """Fails if any tracked path matches D7 version name patterns or D9 build artifacts."""
    tracked = _get_tracked_files()

    d7_patterns = [
        re.compile(r"(?:^|/)(?:v2|v3|old|copy|backup|temp|draft|fixed|_executed|\.nbconvert|checkpoint)", re.IGNORECASE),
        re.compile(r"\(\d+\)"),
    ]
    d9_patterns = [
        re.compile(r"(?:^|/)(?:__pycache__|\.pytest_cache|\.venv|\.cache|data/persisted_index)(?:/|$)", re.IGNORECASE),
        re.compile(r"\.(?:pyc|pyo|pyd|log)$", re.IGNORECASE),
    ]

    for path in tracked:
        if path.startswith("upstream/"):
            continue
        for pat in d7_patterns:
            assert not pat.search(path), f"Tracked file matches D7 version pattern ({pat.pattern}): {path}"
        for pat in d9_patterns:
            assert not pat.search(path), f"Tracked file matches D9 artifact pattern ({pat.pattern}): {path}"


def test_no_duplicate_normalized_functions_in_code():
    """Fails if the same normalized function body is defined twice across utils/, app/, and scripts/."""
    tracked = _get_tracked_files()
    py_files = [f for f in tracked if f.endswith(".py") and (f.startswith("utils/") or f.startswith("app/") or f.startswith("scripts/"))]

    def _normalize(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        clean_node = ast.FunctionDef(
            name="norm_func",
            args=node.args,
            body=body,
            decorator_list=[],
            returns=None,
        )
        return ast.dump(clean_node)

    seen: dict[str, tuple[str, str]] = {}
    for rel_path in py_files:
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(content, filename=rel_path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_dump = _normalize(node)
                if body_dump in seen:
                    prev_path, prev_name = seen[body_dump]
                    assert False, (
                        f"Duplicate normalized function body found: '{node.name}' in {rel_path} "
                        f"matches '{prev_name}' in {prev_path}"
                    )
                seen[body_dump] = (rel_path, node.name)


def test_docs_contains_only_allowed_set():
    """Fails if docs/ contains files other than the allowed set:

    DECISIONS.md, VIDEO_OUTLINE.md, KNOWN_ISSUES.md, TESTING.md
    """
    tracked = _get_tracked_files()
    docs_files = [Path(f).name for f in tracked if f.startswith("docs/")]
    allowed = {"DECISIONS.md", "VIDEO_OUTLINE.md", "KNOWN_ISSUES.md", "TESTING.md"}
    extras = set(docs_files) - allowed
    assert not extras, f"docs/ contains unauthorized files: {extras}. Allowed set is: {allowed}"
    missing = allowed - set(docs_files)
    assert not missing, f"docs/ is missing required canonical files: {missing}"


def test_upstream_matches_manifest():
    """Fails if any file in upstream/ no longer matches tests/base_manifest.sha256."""
    manifest_path = ROOT / "tests" / "base_manifest.sha256"
    assert manifest_path.is_file(), "tests/base_manifest.sha256 missing!"

    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for line in manifest_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(None, 1)
        rel_path = rel_path.strip().replace("\\", "/")
        target_file = ROOT / rel_path
        assert target_file.is_file(), f"Upstream file missing: {rel_path}"
        actual_hash = hashlib.sha256(target_file.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
        )


def test_no_tracked_file_mentions_course_v2_or_v3():
    """Fails if any tracked file mentions rag-from-scratch-course-v2 or -v3."""
    tracked = _get_tracked_files()
    pat = re.compile(r"rag-from-scratch-course-v[23]", re.IGNORECASE)
    violations: list[str] = []

    for rel_path in tracked:
        if rel_path == "tests/test_repo_integrity.py":
            continue
        file_path = ROOT / rel_path
        # Skip binary files like PDFs
        if rel_path.endswith(".pdf"):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if pat.search(content):
                violations.append(rel_path)
        except Exception:
            pass

    assert not violations, f"Files mention rag-from-scratch-course-v2 or -v3: {violations}"
