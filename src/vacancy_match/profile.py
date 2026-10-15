"""Turn Markdown resumes into chunks: one per ## section, one per **bold** project heading."""
from __future__ import annotations

import re
from pathlib import Path

from .models import ProfileChunk, detect_lang

MIN_CHUNK = 40
_LANG_SUFFIX = re.compile(r"-(RU|EN)$", re.I)
_BOLD_LINE = re.compile(r"\*\*[^*]+\*\*")


def track_and_lang(path: Path) -> tuple[str, str]:
    stem = path.stem.removeprefix("resume-")
    m = _LANG_SUFFIX.search(stem)
    return (stem[:m.start()], m.group(1).lower()) if m else (stem, "")


def split_markdown(text: str) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    section = current = "header"
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            pieces.append((current, body))
        buf.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("## "):
            flush()
            section = current = line[3:].strip()
        elif section == "header":
            if line.startswith("### "):  # the role title is signal; name and contacts are not
                buf.append(line[4:].strip())
        elif _BOLD_LINE.fullmatch(stripped):
            flush()
            heading = stripped.strip("*").strip()
            current = f"{section}: {heading}"
            buf.append(heading)
        else:
            buf.append(line)
    flush()
    return pieces


def merge_short(pieces: list[tuple[str, str]], min_len: int = MIN_CHUNK) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    carry = ""
    for name, body in pieces:
        if carry:
            body, carry = carry + "\n" + body, ""
        if len(body) < min_len:
            carry = body
            continue
        out.append((name, body))
    if carry:
        if out:
            out[-1] = (out[-1][0], out[-1][1] + "\n" + carry)
        else:
            out.append(("header", carry))
    return out


def load_profile(paths: list[Path], exclude: set[str] = frozenset(), notes: list[str] | None = None) -> list[ProfileChunk]:
    """Chunks of every resume; `notes` collects what the person should know about how a file was read."""
    chunks: list[ProfileChunk] = []
    for path in sorted({p.resolve() for p in paths}):  # a folder plus one of its files must not double chunks
        track, lang = track_and_lang(path)
        if track in exclude:
            continue
        text = path.read_text(encoding="utf-8-sig")  # PowerShell writes a BOM
        lang = lang or detect_lang(text)
        pieces = split_markdown(text)
        if all(name == "header" for name, _ in pieces) and text.strip():
            pieces = [("resume", text.strip())]  # plain text, no sections: still the person's experience
            if notes is not None:
                notes.append(f"{path.name} has no '## ' sections; indexed as one chunk — sections and **bold** "
                             f"project lines match vacancies better")
        for n, (section, body) in enumerate(merge_short(pieces)):
            chunks.append(ProfileChunk(f"{track}-{lang}-{n:02d}", track, lang, section, body))
    return chunks
