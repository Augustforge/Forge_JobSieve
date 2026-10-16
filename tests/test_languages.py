"""Vacancies in different languages: one fit scale for all by default; per-language scales on request, kept in the
base until switched off."""
import json

import numpy as np
import pytest

from vacancy_match import cli
from vacancy_match.cli import main
from vacancy_match.models import ProfileChunk, Vacancy, detect_lang
from vacancy_match.rank import language_scales, rank
from vacancy_match.store import Store


@pytest.mark.parametrize("text, lang", [
    ("Ищем Python-разработчика в команду, удалённая работа", "ru"),
    ("We are looking for a Python developer to join the team and build our backend", "en"),
    ("Buscamos un desarrollador Python para el equipo, trabajo remoto con los clientes", "es"),
    ("Wir suchen einen Python-Entwickler für unser Team und die Arbeit mit den Kunden", "de"),
    ("Nous recherchons un développeur Python pour notre équipe et les clients de la société", "fr"),
    ("Procuramos um desenvolvedor Python para a nossa equipe e os clientes da empresa", "pt"),
    ("我们正在寻找一名Python开发工程师，远程工作", "zh"),
    ("Pythonエンジニアを募集しています。リモートワーク可能です", "ja"),
    ("파이썬 개발자를 모집합니다. 원격 근무 가능", "ko"),
    ("نبحث عن مطور بايثون للعمل عن بعد", "ar"),
    ("", ""),
])
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang


def test_each_language_with_enough_vacancies_gets_its_own_scale():
    raw = {"ru": [0.90, 0.91, 0.89, 0.92], "en": [0.86, 0.87, 0.85, 0.88], "es": [0.80]}
    scales = language_scales(raw, min_group=4)
    assert set(scales) == {"", "ru", "en"}  # Spanish is too thin: it uses the pooled scale
    assert scales["ru"][0] == pytest.approx(0.905) and scales["en"][0] == pytest.approx(0.865)
    assert scales["ru"][2] == 4 and scales[""][2] == 9


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_a_language_that_is_systematically_less_similar_is_not_ranked_down():
    chunk = ProfileChunk("c", "A", "en", "Projects", "python bots")
    vacs, vecs = [], {}
    for i, (lang_text, sim) in enumerate([("Ищем разработчика ботов на Python", s) for s in (0.95, 0.94, 0.93, 0.92)] +
                                         [("We need a developer for Python bots", s) for s in (0.85, 0.84, 0.83, 0.82)]):
        v = Vacancy.from_dict({"source": "s", "url": f"x.com/{i}", "title": "T", "text": (lang_text + " ") * 10,
                               "run_id": "2026-10-16"})
        vacs.append(v)
        vecs[v.id] = unit(sim, np.sqrt(1 - sim ** 2))
    out = rank(vacs, vecs, [chunk], {"c": unit(1.0, 0.0)}, min_lang_group=4)
    best = {detect_lang(r.vacancy.text): r.fit for r in sorted(out, key=lambda r: r.profile_score)}  # last wins
    assert best["en"] == pytest.approx(best["ru"], abs=0.02)  # the top English vacancy is as good as the top Russian


def test_one_scale_for_every_language_unless_asked():
    chunk = ProfileChunk("c", "A", "en", "Projects", "python bots")
    vacs, vecs = [], {}
    for i, (text, sim) in enumerate([("Ищем разработчика ботов на Python", s) for s in np.linspace(0.92, 0.95, 31)] +
                                    [("We need a developer for Python bots", s) for s in np.linspace(0.82, 0.85, 31)]):
        v = Vacancy.from_dict({"source": "s", "url": f"x.com/{i}", "title": "T", "text": (text + " ") * 10,
                               "run_id": "2026-10-16"})
        vacs.append(v)
        vecs[v.id] = unit(sim, np.sqrt(1 - sim ** 2))
    out = rank(vacs, vecs, [chunk], {"c": unit(1.0, 0.0)})
    best = {detect_lang(r.vacancy.text): r.fit for r in sorted(out, key=lambda r: r.profile_score)}
    assert best["en"] < best["ru"] - 0.3  # a less similar language stays lower: for many people it is less fit


def test_frozen_scales_are_kept_per_language(tmp_path):
    s = Store(tmp_path / "vm.db")
    s.put_calibration("m", 0.9, 0.01, 60, lang="ru")
    s.put_calibration("m", 0.86, 0.01, 40, lang="en")
    s.put_calibration("m", 0.88, 0.02, 100)
    assert s.get_calibrations("m") == {"ru": (0.9, 0.01, 60), "en": (0.86, 0.01, 40), "": (0.88, 0.02, 100)}


def test_cli_ranks_and_calibrates_per_language(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "MIN_FREEZE", 10)
    monkeypatch.setattr(cli, "MIN_LANG_GROUP", 5)
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "resume-a-EN.md").write_text("## Projects\nPython bots that take orders and notify owners.\n"
                                                   "## Боты\nБоты на Python, которые принимают заказы.\n", encoding="utf-8")
    rows = [{"source": "s", "url": f"https://x/ru{i}", "title": f"Вакансия {i}",
             "text": f"Ищем разработчика ботов на Python для заказов номер {i}. " * 6, "run_id": "2026-10-16"}
            for i in range(6)]
    rows += [{"source": "s", "url": f"https://x/en{i}", "title": f"Job {i}",
              "text": f"We need a developer of Python bots for the orders number {i}. " * 6, "run_id": "2026-10-16"}
             for i in range(6)]
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    db = ["--db", str(tmp_path / "vm.db"), "--model", "hash-64"]
    main(db + ["index-profile", str(tmp_path / "p")])
    main(db + ["add", str(f)])
    capsys.readouterr()
    assert main(db + ["rank", "--run", "2026-10-16"]) == 0  # one scale for every language unless asked
    assert "fit scale from 12 vacancies" in capsys.readouterr().out
    assert set(Store(tmp_path / "vm.db").get_calibrations("hash-64")) == {""}
    assert main(db + ["calibrate", "--per-language"]) == 0
    out = capsys.readouterr().out
    assert "ru: centre" in out and "en: centre" in out
    assert main(db + ["rank", "--run", "2026-10-16"]) == 0  # the choice is kept in the base, no flag needed
    assert "fit scale per language: en 6, ru 6" in capsys.readouterr().out
    assert main(db + ["calibrate"]) == 0  # and calibrating without the flag goes back to one scale
    assert set(Store(tmp_path / "vm.db").get_calibrations("hash-64")) == {""}


@pytest.mark.parametrize("text, lang", [
    ("We are looking for a Python developer to join the team at 株式会社メルカリ and build our backend", "en"),
    ("Ищем Python-разработчика, японский язык (にほんご) будет плюсом, работа удалённая", "ru"),
    ("Ищем backend-разработчика: Python, FastAPI, PostgreSQL, Docker, Kubernetes, Redis, Kafka, CI/CD", "ru"),
    ("Ищем backend-разработчика в e-commerce: Python, FastAPI, PostgreSQL, Docker, Kubernetes, Redis", "ru"),
    ("Ищем аналитика: SQL, Python, Tableau, Airflow, ClickHouse, A/B тесты, Looker, dbt", "ru"),
    ("招聘后端工程师 Python Django PostgreSQL Docker Kubernetes e2e", "zh"),
    ("我们在东京的客户株式会社メルカリ正在招聘后端工程师，负责支付系统的开发和维护，远程工作，薪资面议", "zh"),
    ("Backend developer, Яндекс, Python Go Kubernetes Docker Redis Kafka PostgreSQL remote", "en"),
    ("Poszukujemy doświadczonego specjalisty ds. analizy danych w Warszawie lub zdalnie. Wymagania: SQL, Python, "
     "Power BI, trzy lata doświadczenia, znajomość języka angielskiego. Oferujemy umowę B2B i elastyczne godziny.", "pl"),
    ("Dołącz do naszego zespołu analityków danych. Wymagamy znajomości SQL, Pythona oraz narzędzi BI; "
     "oferujemy pracę zdalną, szkolenia i prywatną opiekę medyczną.", "pl"),
    ("【募集職種】基盤開発技術者【勤務地】東京都港区【給与】年収800万円〜1200万円【必須経験】"
     "Python開発経験3年以上、大規模分散処理基盤の設計開発経験【歓迎経験】金融業界での開発経験", "ja"),
    ("我们的客户包括ソフトバンク株式会社与楽天グループ株式会社，现招聘后端工程师，负责支付系统开发，远程工作", "zh"),
    ("Ищем разработчика в польскую компанию (Wrocław/Łódź). Стек: Python, FastAPI, PostgreSQL, Docker, "
     "Kubernetes, Redis, Kafka, Grafana, Prometheus, GitLab CI, Terraform, Ansible, Elasticsearch, RabbitMQ, "
     "Celery, SQLAlchemy, Pydantic, Alembic, ClickHouse. Удалёнка.", "ru"),
])
def test_a_few_foreign_words_do_not_change_the_language(text, lang):
    assert detect_lang(text) == lang


def _vectors(langs):
    rng = np.random.default_rng(0)
    vecs = {vid: unit(*rng.normal(size=8)) for vid in langs}
    return vecs, {"c": unit(*rng.normal(size=8))}


def test_calibrate_drops_the_scale_of_a_language_that_shrank(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MIN_LANG_GROUP", 5)
    s = Store(tmp_path / "vm.db")
    s.put_calibration("m", 0.88, 0.02, 100)
    s.put_calibration("m", 0.80, 0.02, 31, lang="es")
    langs = {f"ru{i}": "ru" for i in range(6)} | {f"es{i}": "es" for i in range(3)}
    vecs, chunks = _vectors(langs)
    s.set_per_language(True)
    cli.frozen_scales(s, "m", vecs, chunks, langs, refresh=True)
    assert set(s.get_calibrations("m")) == {"", "ru"}  # es is down to 3: back on the pooled scale


def test_a_language_that_grew_is_frozen_and_the_others_stay_put(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MIN_LANG_GROUP", 5)
    s = Store(tmp_path / "vm.db")
    s.put_calibration("m", 0.88, 0.02, 100)
    s.put_calibration("m", 0.90, 0.01, 60, lang="ru")
    s.set_per_language(True)
    langs = {f"ru{i}": "ru" for i in range(6)} | {f"en{i}": "en" for i in range(6)}
    vecs, chunks = _vectors(langs)
    scales = cli.frozen_scales(s, "m", vecs, chunks, langs)
    assert scales[""] == (0.88, 0.02, 100) and scales["ru"] == (0.90, 0.01, 60)
    assert scales["en"][2] == 6 and s.get_calibrations("m")["en"] == scales["en"]


def test_a_model_name_with_a_bar_keeps_its_pooled_scale(tmp_path):
    s = Store(tmp_path / "vm.db")
    s.put_calibration("a|b", 0.88, 0.02, 100)
    s.put_calibration("a|b", 0.9, 0.01, 60, lang="ru")
    assert s.get_calibrations("a|b") == {"": (0.88, 0.02, 100), "ru": (0.9, 0.01, 60)}


def test_per_language_choice_is_kept_before_any_language_is_big_enough(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MIN_LANG_GROUP", 5)
    s = Store(tmp_path / "vm.db")
    langs = {f"ru{i}": "ru" for i in range(3)} | {f"en{i}": "en" for i in range(3)}
    vecs, chunks = _vectors(langs)
    s.set_per_language(True)
    assert set(cli.frozen_scales(s, "m", vecs, chunks, langs, refresh=True)) == {""}  # nobody has 5 yet
    langs |= {f"en{i}": "en" for i in range(3, 6)}
    vecs, chunks = _vectors(langs)
    assert set(cli.frozen_scales(s, "m", vecs, chunks, langs)) == {"", "en"}  # English grew: its own scale now


def test_per_language_choice_survives_a_new_profile(tmp_path):
    s = Store(tmp_path / "vm.db")
    s.set_per_language(True)
    s.replace_profile([ProfileChunk("c", "A", "en", "Projects", "python bots")])
    assert s.per_language()


def test_language_scales_from_an_older_version_are_ignored_by_default(tmp_path):
    s = Store(tmp_path / "vm.db")  # older versions froze per-language scales without being asked
    s.put_calibration("m", 0.88, 0.02, 100)
    s.put_calibration("m", 0.90, 0.01, 60, lang="ru")
    assert s.scales("m") == {"": (0.88, 0.02, 100)}
    s.set_per_language(True)
    assert s.scales("m") == {"": (0.88, 0.02, 100), "ru": (0.90, 0.01, 60)}
