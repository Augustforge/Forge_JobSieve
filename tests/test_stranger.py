"""Second stranger pass: a Spanish developer following only the README."""
import json
from pathlib import Path

import numpy as np
import pytest

from vacancy_match import cli
from vacancy_match.cli import main
from vacancy_match.funnel import load_tracker
from vacancy_match.models import ProfileChunk, Vacancy
from vacancy_match.profile import track_and_lang
from vacancy_match.rank import rank


@pytest.mark.parametrize("name, expected", [
    ("resume-backend-ES.md", ("backend", "es")),
    ("resume-backend-EN.md", ("backend", "en")),
    ("resume-A-security-de.md", ("A-security", "de")),
    ("resume-data-AI.md", ("data-AI", "")),  # AI is not a language: it stays part of the track
    ("resume-backend.md", ("backend", "")),
])
def test_any_language_code_is_a_language_not_a_track(name, expected):
    assert track_and_lang(Path(name)) == expected


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_the_two_chunks_shown_come_from_the_winning_track():
    chunks = [ProfileChunk("a1", "A", "en", "Projects", "x"), ProfileChunk("b1", "B", "en", "Summary", "y"),
              ProfileChunk("a2", "A", "en", "Skills", "z")]
    cv = {"a1": unit(1.0, 0.0, 0.0), "b1": unit(0.9, 0.44, 0.0), "a2": unit(0.5, 0.0, 0.87)}
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "T", "text": "python " * 40})
    [r] = rank([v], {v.id: unit(1.0, 0.1, 0.0)}, chunks, cv)
    assert r.track == "A" and [m.chunk_id for m in r.top_chunks] == ["a1", "a2"]


def _base(tmp_path, runs=("2026-10-15",)):
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "resume-backend-EN.md").write_text("## Summary\nPython backend services and APIs for shops.\n",
                                                         encoding="utf-8")
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps({"source": "s", "url": f"https://x/{run}", "title": "Python dev",
                                     "text": "python backend " * 30, "run_id": run}) + "\n" for run in runs),
                 encoding="utf-8")
    db = ["--db", str(tmp_path / "vm.db"), "--model", "hash-64"]
    main(db + ["index-profile", str(tmp_path / "p")])
    main(db + ["add", str(f)])
    return db


def test_a_label_with_a_track_that_is_not_in_the_profile_is_refused(tmp_path, capsys):
    db = _base(tmp_path)
    capsys.readouterr()
    assert main(db + ["label", "https://x/2026-10-15", "fit", "--track", "backnd"]) == 1
    err = capsys.readouterr().err
    assert "backnd" in err and "backend" in err


def test_a_mistyped_database_is_not_created_silently(tmp_path, capsys):
    db = ["--db", str(tmp_path / "typo.db"), "--model", "hash-64"]
    assert main(db + ["rank", "--run", "2026-10-15"]) == 1
    assert "no database" in capsys.readouterr().err
    assert not (tmp_path / "typo.db").exists()


def test_rank_names_the_runs_that_exist(tmp_path, capsys):
    db = _base(tmp_path, runs=("2026-10-15", "2026-10-16"))
    capsys.readouterr()
    assert main(db + ["rank", "--run", "2026-10-14"]) == 1
    out = capsys.readouterr()
    assert "2026-10-16" in out.out + out.err


def test_broken_rules_are_reported_even_before_there_are_vacancies(tmp_path, capsys):
    db = _base(tmp_path)
    rules = tmp_path / "r.yaml"
    rules.write_text("wishs:\n  x: ['y']\n", encoding="utf-8")
    capsys.readouterr()
    assert main(db + ["rank", "--run", "2026-10-14", "--rules", str(rules)]) == 2
    assert "unknown section 'wishs'" in capsys.readouterr().err


def test_a_date_in_another_format_is_named_in_plain_words(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("company,sent,status\nAcme,02/10/2026,sent\n", encoding="utf-8")
    apps, skipped = load_tracker(f)
    assert not apps and "2026-10-05" in skipped[0] and "isoformat" not in skipped[0]


def test_calibrate_per_language_names_the_languages_still_on_the_shared_scale(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "MIN_FREEZE", 2)
    db = _base(tmp_path, runs=("2026-10-15", "2026-10-16"))
    capsys.readouterr()
    assert main(db + ["calibrate", "--per-language"]) == 0
    assert "en 2" in capsys.readouterr().out  # under 30: named, so it is clear nothing changed for it
