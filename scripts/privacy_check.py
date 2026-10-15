"""Fail if anything personal is about to enter the public repo.

Generic checks run everywhere, CI included: only known text file types are allowed (a PDF or DOCX resume is the
likeliest leak), data files only under examples/, no real-looking emails, phone numbers or Telegram handles.
Exact personal strings live OUTSIDE the repo (default ~/.vacancy-match-private.txt, one per line) and are checked
by the local pre-commit hook; matches are reported by index so the strings never reach a log.
The hook runs with --staged and checks what is about to be committed (the index), not the working copy.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".sh", ".lock", ".txt", ".cfg", ".ini"}
TEXT_NAMES = {".gitignore", ".gitattributes", ".python-version", "LICENSE"}
EXAMPLE_DATA_SUFFIXES = {".jsonl", ".csv"}
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org")
EMAIL = re.compile(r"[\w.+-]+@(?:[\w-]+\.)+[a-z]{2,}", re.I)
PHONE = re.compile(r"\+\d[\d ()-]{9,}\d")
TELEGRAM = re.compile(r"t\.me/\w+|telegram\W{0,3}@\w+", re.I)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def files(root: Path, staged: bool = False) -> list[str]:
    args = ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"] if staged else \
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    # -z and quotepath=off: non-ASCII names come back as real names, not "\320..." escapes
    return [p for p in _git(root, "-c", "core.quotepath=off", *args).decode("utf-8").split("\0") if p]


def _allowed_type(rel: str) -> bool:
    path = Path(rel)
    if path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return rel.startswith("examples/") and path.suffix.lower() in EXAMPLE_DATA_SUFFIXES


def check(root: Path, private_terms: list[str], staged: bool = False) -> list[str]:
    problems: list[str] = []
    for rel in files(root, staged):
        if not _allowed_type(rel):
            problems.append(f"{rel}: file type not allowed in the public repo")
            continue
        if not staged and not (root / rel).exists():  # tracked but deleted in the working tree
            continue
        raw = _git(root, "show", f":{rel}") if staged else (root / rel).read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{rel}: not valid UTF-8 text")
            continue
        for m in EMAIL.finditer(text):
            if not m.group(0).lower().endswith(ALLOWED_EMAIL_DOMAINS):
                problems.append(f"{rel}: email {m.group(0)}")
        if PHONE.search(text):
            problems.append(f"{rel}: phone number")
        if TELEGRAM.search(text):
            problems.append(f"{rel}: telegram handle")
        low = text.lower()
        problems += [f"{rel}: private term #{n}" for n, term in enumerate(private_terms, 1) if term.lower() in low]
    return problems


def main() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").decode("utf-8").strip())
    terms_file = Path(os.environ.get("VM_PRIVATE_TERMS", Path.home() / ".vacancy-match-private.txt"))
    terms = [t.strip() for t in terms_file.read_text(encoding="utf-8-sig").splitlines() if t.strip()] \
        if terms_file.exists() else []
    problems = check(root, terms, staged="--staged" in sys.argv)
    for p in problems:
        print(p)
    print(f"privacy check: {len(problems)} problem(s), {len(terms)} private term(s) checked")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
