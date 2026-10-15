import pytest

from vacancy_match.money import is_hourly, is_part_time, monthly_hours, parse_salary_usd

RATES = {"USD": 1.0, "KZT": 0.002, "RUB": 0.0125, "EUR": 1.1}


@pytest.mark.parametrize("text, expected", [
    ("от  500 000   ₸  за месяц   до вычета налогов", 1000.0),
    ("от 2 500 до 10 000 $ за месяц на руки", 6250.0),
    ("от 140 000 ₽ за месяц, на руки", 1750.0),
    ("$2000–4000", 3000.0),
    ("500.000 тенге", 1000.0),
    ("2.5k USD", 2500.0),
    ("from 120000 to 180000 RUR per month", 1875.0),
    ("€3,000 per month", 3300.0),
    ("$25 per hour", 4000.0),
    ("от 60 000 $ в год", 5000.0),
    ("USDT 1500", 1500.0),
])
def test_parse_salary_usd(text, expected):
    assert parse_salary_usd(text, RATES) == pytest.approx(expected)


@pytest.mark.parametrize("text, expected", [
    ("$120,000—$150,000 USD", 11250.0),  # US boards state a yearly salary without saying so
    ("$35,3k- $52k", 3637.5),
    ("$20,000 per month", 20000.0),  # a stated period wins
    ("от 1 500 000 ₽ в месяц", 18750.0),
    ("$9,000", 9000.0),
])
def test_a_figure_too_high_for_a_month_is_a_yearly_salary(text, expected):
    assert parse_salary_usd(text, RATES) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "по результатам собеседования", "от 100 000", "1000 BTC"])
def test_parse_salary_usd_none(text):
    assert parse_salary_usd(text, RATES) is None


def test_percentages_are_not_amounts():
    assert parse_salary_usd("$2000 + 30% от продаж", RATES) == 2000.0
    assert parse_salary_usd("120 000 ₽ + бонус 10 %", RATES) == pytest.approx(1500.0)


def test_range_counts_at_its_middle():
    assert parse_salary_usd("600–1200 $", RATES) == 900.0


def test_part_time_and_hours():
    assert is_part_time("Частичная занятость, удалённо") and is_part_time("Part-time contract")
    assert not is_part_time("Полная занятость")
    assert monthly_hours("20 часов в неделю") == pytest.approx(86.6)
    assert monthly_hours("4 h per day") == pytest.approx(84.0)
    assert monthly_hours("20 ч/нед") == pytest.approx(86.6)
    assert monthly_hours("Полный день") is None
    assert is_hourly("$25 per hour") and not is_hourly("$2500 per month")


@pytest.mark.parametrize("text, expected", [
    ("от 100 000 до 150 000 ₽ за месяц, график 5/2", 1562.5),
    ("$1000-1500 (испытательный срок 3 месяца)", 1250.0),
    ("$2000 net, 40h/week", 2000.0),
    ("1500-2000$ + бонусы, 5/2", 1750.0),
])
def test_stray_small_numbers_do_not_move_the_salary(text, expected):
    assert parse_salary_usd(text, RATES) == pytest.approx(expected)


def test_more_than_two_amounts_is_unclear_not_a_guess():
    assert parse_salary_usd("$1000, $1500 или $2500 по итогам", RATES) is None


@pytest.mark.parametrize("text, expected", [
    ("оклад 60 000 ₽ + KPI до 40 000 ₽", 750.0),
    ("от 80 000 ₽, премия до 20 000 ₽", 1000.0),
    ("$1200, bonus up to $500", 1200.0),
])
def test_bonus_is_not_the_low_end_of_a_range(text, expected):
    assert parse_salary_usd(text, RATES) == pytest.approx(expected)
