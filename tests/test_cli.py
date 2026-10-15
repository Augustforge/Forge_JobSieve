import json
import os
import subprocess
import sys

import pytest

from vacancy_match.cli import main
from vacancy_match.store import Store

RESUME = """# X
### Python developer · bots

## Projects

**Telegram bots**
Python telegram bots automation api scripts, deployed and monitored.
"""


def long(body):
    return (body + " ") * (200 // len(body) + 2)


@pytest.fixture
def env(tmp_path):
    (tmp_path / "profile").mkdir()
    (tmp_path / "profile" / "resume-A-bots-EN.md").write_text(RESUME, encoding="utf-8")
    rows = [{"source": "hh", "url": "https://hh.kz/vacancy/1", "title": "Разработчик ботов", "company": "Бот-Ко",
             "text": long("python telegram bots api")},
            {"source": "hh", "url": "https://hh.kz/vacancy/2", "title": "Продавец", "company": "Ритейл",
             "text": long("sales calls clients targets")},
            {"source": "hh", "url": "https://hh.kz/vacancy/3", "title": "Пусто", "text": ""}]
    (tmp_path / "run.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                        encoding="utf-8")
    base = ["--db", str(tmp_path / "vm.db"), "--model", "hash-256"]
    return tmp_path, base


def test_cli_end_to_end(env, capsys):
    tmp, base = env
    assert main(base + ["index-profile", str(tmp / "profile")]) == 0
    assert "1 tracks: A-bots" in capsys.readouterr().out
    assert main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"]) == 0
    assert "added 3 new, 0 already known" in capsys.readouterr().out
    out_md = tmp / "top.md"
    assert main(base + ["rank", "--run", "2026-10-14", "--out", str(out_md)]) == 0
    md = out_md.read_text(encoding="utf-8")
    assert md.index("Разработчик ботов") < md.index("Продавец") < md.index("Пусто")
    # re-index after a resume edit keeps working
    assert main(base + ["index-profile", str(tmp / "profile")]) == 0
    assert main(base + ["rank", "--run", "2026-10-14"]) == 0
    assert "Разработчик ботов" in capsys.readouterr().out


def test_label_single_and_batch(env, capsys):
    tmp, base = env
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    assert main(base + ["label", "https://krasnoyarsk.hh.ru/vacancy/2?x=1", "no_fit", "--actor", "human",
                        "--reasons", "domain"]) == 0
    batch = tmp / "labels.jsonl"
    batch.write_text('{"url": "https://hh.kz/vacancy/1", "label": "fit", "track": "A-bots"}\n', encoding="utf-8")
    assert main(base + ["label", "--file", str(batch)]) == 0
    eff = Store(tmp / "vm.db").effective_labels()
    assert sorted(l.label for l in eff.values()) == ["fit", "no_fit"]


def test_label_file_bad_line_stores_nothing(env, capsys):
    tmp, base = env
    batch = tmp / "labels.jsonl"
    batch.write_text('{"url": "x.com/1", "label": "fit"}\n{"url": "x.com/2", "label": "perhaps"}\n',
                     encoding="utf-8")
    assert main(base + ["label", "--file", str(batch)]) == 1
    assert "labels.jsonl:2" in capsys.readouterr().err
    assert Store(tmp / "vm.db").labels() == []


def test_add_without_run_is_rejected(env, capsys):
    tmp, base = env
    assert main(base + ["add", str(tmp / "run.jsonl")]) == 1  # nothing to add is a failure, said once
    assert "3 records have no dated run id" in capsys.readouterr().err


def test_eval_without_labels(env, capsys):
    tmp, base = env
    main(base + ["index-profile", str(tmp / "profile")])
    assert main(base + ["eval"]) == 1
    assert "no labeled vacancies" in capsys.readouterr().out


def test_eval_runs(env, capsys):
    tmp, base = env
    main(base + ["index-profile", str(tmp / "profile")])
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    main(base + ["label", "https://hh.kz/vacancy/1", "fit", "--track", "A-bots"])
    main(base + ["label", "https://hh.kz/vacancy/2", "no_fit"])
    out = tmp / "eval.md"
    assert main(base + ["eval", "--labels", "all", "--models", "hash-256,hash-128", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "hash-256 nw=0.0" in text and "hash-128 nw=0.0" in text


def test_cli_prints_utf8_under_cp1251(env):
    tmp, base = env
    main(base + ["index-profile", str(tmp / "profile")])
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    proc = subprocess.run([sys.executable, "-m", "vacancy_match.cli", *base, "rank", "--run", "2026-10-14"],
                          capture_output=True, env={**os.environ, "PYTHONIOENCODING": "cp1251"})
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "Разработчик ботов" in proc.stdout.decode("utf-8")


def test_add_rejects_run_without_date(env, capsys):
    tmp, base = env
    assert main(base + ["add", str(tmp / "run.jsonl"), "--run", "history"]) == 1
    assert "no dated run id" in capsys.readouterr().err
    assert Store(tmp / "vm.db").vacancies() == []


def test_add_accepts_bom_and_embeds_the_stored_text(env, capsys):
    tmp, base = env
    rec = {"source": "hh", "url": "https://hh.kz/vacancy/9", "title": "Бот", "text": long("python telegram bots api")}
    f = tmp / "bom.jsonl"
    f.write_text("\ufeff" + json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(base + ["add", str(f), "--run", "2026-10-14"]) == 0
    assert "added 1 new" in capsys.readouterr().out
    snippet = {**rec, "text": long("short teaser")[:210]}
    f.write_text(json.dumps(snippet, ensure_ascii=False) + "\n", encoding="utf-8")
    main(base + ["add", str(f), "--run", "2026-10-15"])
    from vacancy_match.embed import HashEmbedder
    s = Store(tmp / "vm.db")
    v = s.get_vacancy(next(iter(x.id for x in s.vacancies())))
    stored_vec = s.get_embeddings("vacancy", "hash-256")[v.id]
    assert (stored_vec == HashEmbedder(256).encode([v.embed_text()], "query")[0]).all()


def test_eval_with_rules_compares_both_stages(env, capsys):
    tmp, base = env
    (tmp / "rules.yaml").write_text("conditions:\n  sales: ['sales']\n", encoding="utf-8")
    main(base + ["index-profile", str(tmp / "profile")])
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    main(base + ["label", "https://hh.kz/vacancy/1", "fit"])
    main(base + ["label", "https://hh.kz/vacancy/2", "no_fit"])
    out = tmp / "eval.md"
    assert main(base + ["eval", "--labels", "all", "--models", "hash-256", "--rules", str(tmp / "rules.yaml"),
                         "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "stage=1 |" in text and "stage=1+2 |" in text and "| sales | ×0.5 | 0 | 1 | 0% | 100% |" in text


def test_neighbour_vote_needs_enough_no_fits(env, capsys):
    tmp, base = env
    main(base + ["index-profile", str(tmp / "profile")])
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    main(base + ["label", "https://hh.kz/vacancy/1", "fit"])
    main(base + ["label", "https://hh.kz/vacancy/2", "fit"])
    capsys.readouterr()
    main(base + ["rank", "--run", "2026-10-14", "--min-labels", "1"])
    assert "neighbour weight 0.0 (2 labeled, 0 no_fit)" in capsys.readouterr().out
    main(base + ["rank", "--run", "2026-10-14", "--min-labels", "1", "--min-no-fits", "0"])
    assert "neighbour weight 0.15" in capsys.readouterr().out


def test_snapshot_is_frozen_until_recalibrated(env, capsys):
    tmp, base = env
    main(base + ["index-profile", str(tmp / "profile")])
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    main(base + ["rank", "--run", "2026-10-14"])
    s = Store(tmp / "vm.db")
    assert s.get_calibration("hash-256") is None  # 2 vacancies are too few to freeze a scale on
    more = [{"source": "hh", "url": f"https://hh.kz/vacancy/{n}", "title": "Склад", "company": "W",
             "text": long("warehouse loader night shifts")} for n in range(10, 15)]
    (tmp / "more.jsonl").write_text("\n".join(json.dumps(r) for r in more) + "\n", encoding="utf-8")
    main(base + ["add", str(tmp / "more.jsonl"), "--run", "2026-10-15"])
    assert main(base + ["calibrate"]) == 0  # on purpose: stored even for a small base
    first = s.get_calibration("hash-256")
    assert first[2] == 7
    (tmp / "more2.jsonl").write_text(json.dumps({"source": "hh", "url": "https://hh.kz/vacancy/99", "title": "X",
                                                 "text": long("figma design landing")}) + "\n", encoding="utf-8")
    main(base + ["add", str(tmp / "more2.jsonl"), "--run", "2026-10-16"])
    capsys.readouterr()
    main(base + ["rank", "--run", "2026-10-16"])
    assert s.get_calibration("hash-256") == first  # the base grew, the scale did not move
    assert "fit scale from 7 vacancies" in capsys.readouterr().out
    main(base + ["index-profile", str(tmp / "profile")])
    assert s.get_calibration("hash-256") is None  # a new profile means a new scale


def test_eval_defaults_to_human_decisions(env, capsys):
    tmp, base = env
    main(base + ["index-profile", str(tmp / "profile")])
    main(base + ["add", str(tmp / "run.jsonl"), "--run", "2026-10-14"])
    main(base + ["label", "https://hh.kz/vacancy/1", "fit", "--actor", "agent"])  # agent labels only
    main(base + ["label", "https://hh.kz/vacancy/2", "no_fit", "--actor", "agent"])
    capsys.readouterr()
    assert main(base + ["eval", "--models", "hash-256"]) == 1
    assert "--labels all" in capsys.readouterr().out
