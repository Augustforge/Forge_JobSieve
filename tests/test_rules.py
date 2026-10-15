from datetime import date

import pytest

from vacancy_match.models import Vacancy
from vacancy_match.rules import Adjustment, RuleSet

RULES = RuleSet.from_dict({
    "flags": {"agency": [r"outstaff|аутстаф"]},
    "wishes": {"on_call": [r"on-?call|дежурств"]},
    "conditions": {"language": [r"fluent german", r"german c1", r"немецк\w* язык"]},
    "boosts": {"open_source": [r"open[- ]source"]},
    "red_lines": {"upfront_fee": [r"оплат\w* (обучени|стартов)", r"registration fee"]},
    "stale": {"days": 30, "factor": 0.8},
    "salary": {"curve": {1500: 1.0, 1350: 0.9, 1200: 0.65, 1100: 0.5}, "rates": {"USD": 1.0, "RUB": 0.0125}},
})
T = date(2026, 10, 14)


def vac(text="", salary="", published="", work_format=""):
    return Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "Role", "text": text,
                              "salary_text": salary, "published_at": published, "work_format": work_format})


def factors(check):
    return {a.rule: a.factor for a in check.adjustments}


def test_tiers_give_their_default_levels():
    c = RULES.check(vac("German C1, дежурства по выходным, open-source проект, аутстаф"), T)
    assert c.flags == ["agency", "no_salary"]
    assert factors(c) == {"on_call": 0.9, "language": 0.5, "open_source": 1.1}
    assert c.factor == pytest.approx(0.9 * 0.5 * 1.1) and c.cut == []


def test_negated_mention_does_not_fire():
    c = RULES.check(vac("Немецкий не нужен, общение на английском", salary="$1500"), T)
    assert c.flags == [] and c.adjustments == [] and c.cut == []


def test_red_line_cuts():
    c = RULES.check(vac("Нужно оплатить обучение и стартовый набор"), T)
    assert [a.rule for a in c.cut] == ["upfront_fee"]


def test_stale_costs_a_little():
    assert factors(RULES.check(vac(published="2026-09-01T10:00:00+03:00"), T))["stale"] == 0.8
    assert "stale" not in factors(RULES.check(vac(published="2026-10-01"), T))


@pytest.mark.parametrize("salary, factor", [("$2000", None), ("$1500", None), ("$1350", 0.9), ("$1200", 0.65),
                                             ("$1150", 0.575), ("от 84 000 до 108 000 ₽", 0.65)])
def test_salary_curve(salary, factor):
    got = factors(RULES.check(vac(salary=salary), T)).get("salary")
    assert got == (pytest.approx(factor) if factor else None)


def test_salary_below_floor_is_cut_and_unknown_is_only_flagged():
    assert [(a.rule, a.detail) for a in RULES.check(vac(salary="$1000"), T).cut] == [("salary_floor", "$1000")]
    c = RULES.check(vac(salary="по договорённости"), T)
    assert "no_salary" in c.flags and c.cut == []


def test_part_time_salary_is_scaled_to_full_time_or_left_alone():
    scaled = RULES.check(vac("20 часов в неделю", salary="$900", work_format="Частичная занятость"), T)  # ≈ $1660
    assert "salary" not in factors(scaled) and scaled.cut == []
    unknown = RULES.check(vac("По вечерам", salary="$300", work_format="Подработка"), T)
    assert unknown.cut == [] and "part_time_pay" in unknown.flags
    hourly = RULES.check(vac("", salary="$4 per hour", work_format="Part-time"), T)  # $640 at full-time hours
    assert [a.rule for a in hourly.cut] == ["salary_floor"]


def test_factor_override_and_levels():
    r = RuleSet.from_dict({"levels": {"slight": 0.95},
                           "wishes": {"on_call": ["дежурств"], "office": {"patterns": ["офис"], "factor": 0.7}}})
    assert factors(r.check(vac("дежурства, офис"), T)) == {"on_call": 0.95, "office": 0.7}


def test_load_from_yaml(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("conditions:\n  clearance: ['security clearance']\n", encoding="utf-8")
    assert factors(RuleSet.load(p).check(vac("Security clearance before start"), T)) == {"clearance": 0.5}


def test_adjustment_reads_like_a_reason():
    assert str(Adjustment("salary", 0.65, "$1200")) == "salary $1200 ×0.65"
    assert str(Adjustment("upfront_fee", 0.0)) == "upfront_fee"


def test_part_time_comes_from_the_header_not_from_any_mention():
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "Менеджер", "salary_text": "50 000 ₽",
                           "text": "Это не подработка, полный день. Созвон 1 час в день."})
    assert [a.rule for a in RULES.check(v, T).cut] == ["salary_floor"]


def test_implausibly_few_hours_are_not_used_for_scaling():
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "Оператор", "work_format": "Частичная занятость",
                           "salary_text": "$300", "text": "Созвон 1 час в день."})
    c = RULES.check(v, T)
    assert c.cut == [] and "part_time_pay" in c.flags


def test_stale_defaults_to_its_own_factor():
    r = RuleSet.from_dict({"stale": {"days": 14}})
    assert factors(r.check(vac(published="2026-09-01"), T)) == {"stale": 0.8}


def test_project_pay_is_not_judged_against_a_monthly_floor():
    c = RULES.check(vac("Сделать бота", salary="30 000 ₽ за проект", work_format="Проект или разовое задание"), T)
    assert c.cut == [] and "project_pay" in c.flags
