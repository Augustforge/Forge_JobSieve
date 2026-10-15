"""Salary text → monthly USD at the middle of the range, plus part-time hours for a fair comparison."""
from __future__ import annotations

import re

_NUM = re.compile(r"(\d{1,3}(?:[ \u00a0\u202f.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(k\b|к\b|тыс\.?)?", re.I)
_PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")
_GROUPED = re.compile(r"\d{1,3}(?:[ .,]\d{3})+")
_CURRENCIES = [
    ("KZT", re.compile(r"₸|тенге|\bkzt\b", re.I)),
    ("RUB", re.compile(r"₽|руб|\brub\b|\brur\b", re.I)),
    ("EUR", re.compile(r"€|\beur\b|евро", re.I)),
    ("USD", re.compile(r"\$|\busdt?\b|долл", re.I)),
]
_HOURLY = re.compile(r"в час|per hour|hourly|/\s*h(?:ou)?r\b", re.I)
_YEARLY = re.compile(r"в год|per year|annual|/\s*y(?:ea)?r\b", re.I)
_MONTHLY = re.compile(r"в месяц|за месяц|/\s*мес|per month|a month|monthly|/\s*mo\b", re.I)
YEARLY_ABOVE = 15000  # USD: with no period stated, a "monthly" figure above this is a yearly salary (US/EU boards)
_BONUS = re.compile(r"\+|преми|бонус|\bkpi\b|bonus|комисси|commission", re.I)
FULL_TIME_HOURS = 160
WEEKS_PER_MONTH = 4.33
WORKDAYS_PER_MONTH = 21
_PART_TIME = re.compile(r"частичн\w* занятост|неполн\w* (?:рабоч\w* )?(?:день|занятост)|подработк|part[- ]time", re.I)
_HOURS = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:[-–]\s*\d+\s*)?(?:ч\b|час\w*|h\b|hours?|hrs?)\s*(?:в|per|a|/)\s*"
                    r"(недел\w*|нед\b|week|wk|день|дн\w*|day)", re.I)


def is_hourly(text: str) -> bool:
    return bool(_HOURLY.search(text))


def is_part_time(text: str) -> bool:
    return bool(_PART_TIME.search(text))


def monthly_hours(text: str) -> float | None:
    """'20 часов в неделю' → 86.6, '4 h per day' → 84; None when the text names no hours."""
    m = _HOURS.search(text)
    if not m:
        return None
    per_week = m.group(2).lower().startswith(("нед", "week", "wk"))
    return float(m.group(1).replace(",", ".")) * (WEEKS_PER_MONTH if per_week else WORKDAYS_PER_MONTH)


def _to_number(raw: str, multiplier: str | None) -> float:
    s = raw.replace("\u00a0", " ").replace("\u202f", " ")
    value = float(re.sub(r"[ .,]", "", s)) if _GROUPED.fullmatch(s) else float(s.replace(",", "."))
    return value * 1000 if multiplier else value


def parse_salary_usd(text: str, rates: dict[str, float]) -> float | None:
    if not text:
        return None
    currency = next((code for code, rx in _CURRENCIES if rx.search(text)), None)
    if currency is None or currency not in rates:
        return None
    amounts_text = _PERCENT.sub(" ", text)  # "+ 30% от продаж" is not an amount
    amounts_text = _BONUS.split(amounts_text, maxsplit=1)[0]  # "+ KPI до 40 000" is not the low end of a range
    amounts = [a for a in (_to_number(m.group(1), m.group(2)) for m in _NUM.finditer(amounts_text)) if a > 0]
    if not amounts:
        return None
    amounts = [a for a in amounts if a >= max(amounts) / 10]  # "5/2", "3 месяца", "40h/week" are not pay
    if len(amounts) > 2:
        return None  # several pay figures: unclear — ask rather than guess
    monthly = (min(amounts) + max(amounts)) / 2  # a range counts at its middle: few employers start at the top
    if _HOURLY.search(text):
        monthly *= FULL_TIME_HOURS
    elif _YEARLY.search(text):
        monthly /= 12
    elif not _MONTHLY.search(text) and monthly * rates[currency] > YEARLY_ABOVE:
        monthly /= 12
    return round(monthly * rates[currency], 2)
