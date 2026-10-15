from pathlib import Path

from vacancy_match.profile import load_profile, merge_short, split_markdown, track_and_lang

RESUME = """# Alex
### Backend Developer · Bots · Automation

alex@example.com · github.com/alex

## Summary

I build backend services and bots in Python and keep them running without supervision.

## Projects

**Order bot (Telegram)**
A bot that takes orders, stores them in SQLite and pings the owner.

- 300 tests, deployed with systemd.

**Landing page**
A one-page site built and deployed in an evening with plain HTML and CSS.

## Extra

Ok.
"""


def test_track_and_lang():
    assert track_and_lang(Path("resume-data-RU.md")) == ("data", "ru")
    assert track_and_lang(Path("resume-qa-auto-EN.md")) == ("qa-auto", "en")
    assert track_and_lang(Path("cv.md")) == ("cv", "")


def test_split_markdown_sections_and_projects():
    pieces = split_markdown(RESUME)
    names = [n for n, _ in pieces]
    assert names == ["header", "Summary", "Projects: Order bot (Telegram)", "Projects: Landing page", "Extra"]
    header = dict(pieces)["header"]
    assert header == "Backend Developer · Bots · Automation"  # contacts dropped, title kept
    assert "300 tests" in dict(pieces)["Projects: Order bot (Telegram)"]


def test_merge_short_glues_tiny_piece_to_neighbour():
    merged = merge_short([("a", "x" * 50), ("b", "short"), ("c", "y" * 50)])
    assert merged == [("a", "x" * 50), ("c", "short\n" + "y" * 50)]
    assert merge_short([("a", "x" * 50), ("z", "tiny")]) == [("a", "x" * 50 + "\ntiny")]


def test_load_profile_ids_and_exclude(tmp_path):
    (tmp_path / "resume-A-backend-EN.md").write_text(RESUME, encoding="utf-8")
    (tmp_path / "resume-G-coach-EN.md").write_text(RESUME, encoding="utf-8")
    chunks = load_profile(sorted(tmp_path.glob("*.md")), exclude={"G-coach"})
    assert {c.track for c in chunks} == {"A-backend"}
    assert chunks[0].id == "A-backend-en-00"
    assert all(len(c.text) >= 40 for c in chunks)


def test_same_file_twice_and_bom(tmp_path):
    f = tmp_path / "resume-A-backend-EN.md"
    f.write_text("\ufeff" + RESUME.split("\n", 1)[1], encoding="utf-8")  # starts with "### ..." after a BOM
    chunks = load_profile([f, tmp_path / "resume-A-backend-EN.md"])
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].text.startswith("Backend Developer")
