"""What a stranger hits first: inputs written by hand, typos, real-world salary strings, other languages."""
import json
from datetime import date

import pytest

from vacancy_match import fetch as fetch_mod
from vacancy_match.cli import main
from vacancy_match.fetch import collect
from vacancy_match.funnel import load_tracker
from vacancy_match.jobposting import parse_jobposting
from vacancy_match.labels import label_from_dict
from vacancy_match.models import Label, Vacancy
from vacancy_match.money import parse_salary_usd
from vacancy_match.profile import load_profile
from vacancy_match.rules import RuleSet
from vacancy_match.store import Store

RATES = {"USD": 1.0, "EUR": 1.1, "GBP": 1.25, "RUB": 0.0125, "KZT": 0.002}
T = date(2026, 10, 15)


def vac(text="", salary=""):
    return Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "Role", "text": text, "salary_text": salary})


# --- rules written by hand ---

def test_a_single_pattern_may_be_a_string():
    r = RuleSet.from_dict({"red_lines": {"upfront_fee": "pay for training"}})
    assert r.check(vac("Great team"), T).cut == []
    assert [a.rule for a in r.check(vac("You pay for training first"), T).cut] == ["upfront_fee"]


@pytest.mark.parametrize("rules, message", [
    ({"wishs": {"x": ["y"]}}, "unknown section 'wishs'"),
    ({"conditions": {"broken": ["(unclosed"]}}, "rule broken: bad regex"),
    ({"conditions": {"llm": ["\bllm\b"]}}, "single quotes"),  # "\b" in YAML double quotes is a backspace
    ({"salary": {"curve": {1000: 1.0}, "rates": {"KZT": 450}}}, "USD per one unit"),
    ({"salary": {"currency": "EUR", "curve": {4000: 1.0}}}, "rates"),
])
def test_rule_mistakes_are_named(rules, message):
    with pytest.raises(ValueError, match=message):
        RuleSet.from_dict(rules)


def test_rules_file_problems_are_one_line_errors(tmp_path):
    bad_yaml = tmp_path / "r.yaml"
    bad_yaml.write_text("conditions:\n  x: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="r.yaml"):
        RuleSet.load(bad_yaml)
    cp1251 = tmp_path / "c.yaml"
    cp1251.write_bytes("conditions:\n  ночь: ['ночн']\n".encode("cp1251"))
    with pytest.raises(ValueError, match="UTF-8"):
        RuleSet.load(cp1251)
    with pytest.raises(ValueError, match="no such file"):
        RuleSet.load(tmp_path / "missing.yaml")


def test_salary_curve_in_your_own_currency_and_period():
    r = RuleSet.from_dict({"salary": {"currency": "EUR", "period": "year", "curve": {60000: 1.0, 48000: 0.5},
                                      "rates": {"EUR": 1.1}}})
    got = {a.rule: a.factor for a in r.check(vac(salary="€54,000 per year"), T).adjustments}
    assert got["salary"] == pytest.approx(0.75)
    assert [a.rule for a in r.check(vac(salary="€40,000 per year"), T).cut] == ["salary_floor"]


# --- real salary strings ---

@pytest.mark.parametrize("text, expected", [
    ("£50,000", 5208.33),  # yearly by size
    ("GBP 110000 – 125000 per year", 12239.58),
    ("50-60k EUR per year", 5041.67),
    ("40 000 ₸ в неделю", 346.4),
    ("3 000 ₽ за смену", 787.5),
])
def test_real_salary_strings(text, expected):
    assert parse_salary_usd(text, RATES) == pytest.approx(expected, abs=0.01)


def test_hh_page_salary_and_work_format_come_from_the_page_markup():
    page = ('<script type="application/ld+json">{"@type": "JobPosting", "title": "Бухгалтер", "description": "x",'
            '"hiringOrganization": {"name": "Ко"}}</script><div data-qa="vacancy-salary"><span>от\xa0<data value="700">'
            '700</data>\xa0до\xa0<data value="1000">1\xa0000</data>\xa0<data value="USD">$</data> за\xa0месяц</span>'
            '</div><p data-qa="common-employment-text">Полная занятость</p>'
            '<p data-qa="work-formats-text">Формат работы: удалённо</p>')
    rec = parse_jobposting(page)
    assert rec["salary_text"] == "от 700 до 1 000 $ за месяц"
    assert rec["work_format"] == "Полная занятость; Формат работы: удалённо"
    assert parse_salary_usd(rec["salary_text"], RATES) == 850


# --- collecting ---

def fake(payloads):
    return lambda url, headers=None, raw=False: payloads[url]


def test_small_figures_from_boards_are_hourly():
    gh = {"jobs": [{"absolute_url": "https://g/1", "title": "Intern", "company_name": "F",
                    "content": '<div class="pay-range"><span>$46</span><span>$46 USD</span></div>'}]}
    ok = [{"legal": "x"}, {"position": "Tutor &amp; Coach", "company": "C", "url": "https://r/1", "description": "",
                           "salary_min": 40, "salary_max": 48}]
    records, errors = collect(["greenhouse:f", "remoteok"], get=fake({
        "https://boards-api.greenhouse.io/v1/boards/f/jobs?content=true": gh, "https://remoteok.com/api": ok}))
    assert errors == []
    assert [r["salary_text"] for r in records] == ["$46$46 USD per hour", "USD 40 – 48 per hour"]
    assert records[1]["title"] == "Tutor & Coach"
    assert parse_salary_usd(records[0]["salary_text"], RATES) == 46 * 160


def test_a_source_without_search_says_so_and_an_ignored_search_is_noted():
    _, errors = collect(["arbeitnow:designer"], get=fake({}))
    assert errors == ["arbeitnow:designer: takes no argument; filter with --where"]
    notes = []
    remotive = {"jobs": [{"url": "https://rm/1", "title": "Driver", "company_name": "C", "description": "Drive"}]}
    collect(["remotive:accountant"], get=fake({"https://remotive.com/api/remote-jobs?search=accountant": remotive}),
            log=notes.append)
    assert any("none of the 1 jobs mention 'accountant'" in n for n in notes)


def test_fetch_stamps_a_dated_run_so_add_needs_no_flag(tmp_path, monkeypatch):
    payload = {"jobs": [{"absolute_url": "https://g/1", "title": "Dev", "company_name": "F", "content": "Python " * 40}]}
    monkeypatch.setattr(fetch_mod, "http_get", fake({"https://boards-api.greenhouse.io/v1/boards/f/jobs?content=true":
                                                     payload}))
    out = tmp_path / "run.jsonl"
    assert main(["fetch", "greenhouse:f", "--run", "2026-10-15", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["run_id"] == "2026-10-15"
    assert main(["--db", str(tmp_path / "a.db"), "--model", "hash-64", "add", str(out)]) == 0
    assert {v.run_id for v in Store(str(tmp_path / "a.db")).vacancies()} == {"2026-10-15"}
    assert main(["--db", str(tmp_path / "b.db"), "--model", "hash-64", "add", str(out), "--run", "later-2026-10-20"]) == 0
    assert {v.run_id for v in Store(str(tmp_path / "b.db")).vacancies()} == {"later-2026-10-20"}  # --run wins


# --- adding and labelling ---

def test_add_keeps_lines_with_unicode_line_separators(tmp_path):
    rec = {"source": "s", "url": "https://x/1", "title": "T", "text": "one two\u0085three", "run_id": "2026-10-15"}
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(["--db", str(tmp_path / "vm.db"), "--model", "hash-64", "add", str(f)]) == 0
    assert len(Store(str(tmp_path / "vm.db")).vacancies()) == 1


def test_add_without_dated_runs_says_it_once(tmp_path, capsys):
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps({"source": "s", "url": f"https://x/{n}", "title": "T"}) + "\n" for n in range(30)),
                 encoding="utf-8")
    assert main(["--db", str(tmp_path / "vm.db"), "--model", "hash-64", "add", str(f)]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "30 records have no dated run id" in err


def test_labels_are_yours_by_default_and_reasons_are_your_words():
    assert label_from_dict({"vacancy_id": "a", "label": "fit"}).actor == "human"
    assert Label("a", "no_fit", reasons=("холодные_звонки", "cold-calls")).reasons == ("холодные_звонки", "cold-calls")


def test_cli_label_defaults_to_human(tmp_path):
    db = str(tmp_path / "vm.db")
    assert main(["--db", db, "label", "x.com/1", "fit"]) == 0
    assert [l.actor for l in Store(db).labels()] == ["human"]


# --- resumes ---

def test_a_resume_without_sections_is_one_chunk_with_a_note(tmp_path):
    p = tmp_path / "resume-research-EN.md"
    p.write_text("# Ann\nResearcher\n\nExperience\nRan user interviews and usability tests for five products.\n",
                 encoding="utf-8")
    notes = []
    chunks = load_profile([p], notes=notes)
    assert len(chunks) == 1 and "usability tests" in chunks[0].text
    assert notes and "no '## ' sections" in notes[0]


def test_reindexing_one_resume_warns_about_tracks_it_drops(tmp_path, capsys):
    body = "## Summary\nI keep books, reconcile accounts and close the month for small firms.\n"
    (tmp_path / "resume-buh-RU.md").write_text(body, encoding="utf-8")
    (tmp_path / "resume-ux-EN.md").write_text(body.replace("keep books", "design screens"), encoding="utf-8")
    db = ["--db", str(tmp_path / "vm.db"), "--model", "hash-64"]
    assert main(db + ["index-profile", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(db + ["index-profile", str(tmp_path / "resume-ux-EN.md")]) == 0
    assert "buh" in capsys.readouterr().err


# --- commands fail fast, before the model loads ---

def test_rank_with_broken_rules_fails_before_loading_anything(tmp_path, capsys):
    rules = tmp_path / "r.yaml"
    rules.write_text("wishs:\n  x: ['y']\n", encoding="utf-8")
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps({"source": "s", "url": "https://x/1", "title": "T", "run_id": "2026-10-15"}) + "\n",
                 encoding="utf-8")
    db = ["--db", str(tmp_path / "vm.db"), "--model", "no-such/model"]
    main(["--db", str(tmp_path / "vm.db"), "--model", "hash-64", "add", str(f)])
    assert main(db + ["rank", "--run", "2026-10-15", "--rules", str(rules)]) == 2
    assert "unknown section 'wishs'" in capsys.readouterr().err


def test_eval_uses_the_global_model_and_explains_too_few_labels(tmp_path, capsys):
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps({"source": "s", "url": "https://x/1", "title": "T", "text": "python " * 40,
                             "run_id": "2026-10-15"}) + "\n", encoding="utf-8")
    db = ["--db", str(tmp_path / "vm.db"), "--model", "hash-64"]
    main(db + ["add", str(f)])
    main(db + ["label", "https://x/1", "fit"])
    assert main(db + ["eval"]) == 1  # would try to load e5-base if --model were ignored
    assert "at least one fit and one no_fit" in capsys.readouterr().out


def test_help_shows_the_workflow(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "vm fetch" in out and "vm rank" in out and "vm funnel" in out


# --- the tracker as Excel saves it ---

def test_tracker_accepts_dotted_dates_and_russian_statuses(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Компания;Отправлено\n".replace("Компания", "company").replace("Отправлено", "sent") +
                 "Acme;05.10.2026\n", encoding="utf-8")
    p.write_text("company;sent;status\nAcme;05.10.2026;Отправлено\nBeta;06.10.2026;отказ\n", encoding="utf-8")
    apps, skipped = load_tracker(p)
    assert skipped == [] and [(a.sent, a.status) for a in apps] == [(date(2026, 10, 5), "sent"),
                                                                     (date(2026, 10, 6), "rejected")]
