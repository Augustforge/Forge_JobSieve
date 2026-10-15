"""Stage 2: the candidate's own terms — what embeddings cannot see. Optional: without rules, ranking is resume fit alone.

Tiers: flags only show; wishes cost a little; conditions cost a lot; boosts add a little; red lines cut.
Every effect multiplies the calibrated fit, so a near-perfect vacancy can survive one serious miss.
The factors are starting guesses: tune them with the rule report of `vm eval`."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from .models import Vacancy
from .money import FULL_TIME_HOURS, is_hourly, is_part_time, monthly_hours, parse_salary_usd

DEFAULT_LEVELS = {"slight": 0.9, "serious": 0.5, "boost": 1.1}
TIERS = {"flags": None, "wishes": "slight", "conditions": "serious", "boosts": "boost", "red_lines": None}
STALE_FACTOR = 0.8
MIN_PART_TIME_HOURS = 40  # "созвон 1 час в день" is not a part-time schedule
_PROJECT = re.compile(r"проект\w* или разов|разов\w* (?:задани|проект)|за проект|per project|fixed[- ]price|"
                      r"project[- ]based", re.I)


@dataclass
class Adjustment:
    rule: str
    factor: float  # 0.0 = cut
    detail: str = ""

    def __str__(self) -> str:
        label = f"{self.rule} {self.detail}".strip()
        return label if self.factor == 0 else f"{label} ×{round(self.factor, 3):g}"


@dataclass
class Check:
    flags: list[str] = field(default_factory=list)
    adjustments: list[Adjustment] = field(default_factory=list)
    cut: list[Adjustment] = field(default_factory=list)

    @property
    def factor(self) -> float:
        return math.prod(a.factor for a in self.adjustments)


@dataclass
class Rule:
    name: str
    tier: str
    factor: float
    patterns: list[re.Pattern]


def _rules(tier: str, group: dict | None, levels: dict[str, float]) -> list[Rule]:
    out = []
    for name, spec in (group or {}).items():
        patterns, factor = (spec.get("patterns") or [], spec.get("factor")) if isinstance(spec, dict) else (spec, None)
        if factor is None:
            factor = {"flags": 1.0, "red_lines": 0.0}.get(tier) if TIERS[tier] is None else levels[TIERS[tier]]
        out.append(Rule(name, tier, float(factor), [re.compile(p, re.I) for p in patterns]))
    return out


def _parse_date(s: str) -> date | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None


def salary_factor(usd: float, curve: list[tuple[float, float]]) -> float | None:
    """Linear between the curve's points, the top factor above them; None below the lowest point (cut)."""
    if usd < curve[0][0]:
        return None
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if usd <= x1:
            return y0 + (y1 - y0) * (usd - x0) / (x1 - x0)
    return curve[-1][1]


@dataclass
class RuleSet:
    rules: list[Rule] = field(default_factory=list)
    stale_days: int | None = None
    stale_factor: float = STALE_FACTOR
    salary_curve: list[tuple[float, float]] = field(default_factory=list)  # (monthly usd, factor), ascending
    rates: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RuleSet":
        levels = {**DEFAULT_LEVELS, **(d.get("levels") or {})}
        stale, salary = d.get("stale") or {}, d.get("salary") or {}
        return cls([r for tier in TIERS for r in _rules(tier, d.get(tier), levels)], stale.get("days"),
                   float(stale.get("factor", STALE_FACTOR)),
                   sorted((float(k), float(v)) for k, v in (salary.get("curve") or {}).items()),
                   salary.get("rates") or {})

    @classmethod
    def load(cls, path: str | Path) -> "RuleSet":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {})

    def check(self, v: Vacancy, today: date) -> Check:
        hay = "\n".join((v.title, v.text, v.salary_text, v.work_format, v.location))
        c = Check()
        for r in self.rules:
            if any(p.search(hay) for p in r.patterns):
                if r.tier == "flags":
                    c.flags.append(r.name)
                elif r.factor == 0:
                    c.cut.append(Adjustment(r.name, 0.0))
                else:
                    c.adjustments.append(Adjustment(r.name, r.factor))
        if self.stale_days is not None and v.published_at:
            posted = _parse_date(v.published_at)
            if posted and (today - posted).days > self.stale_days:
                c.adjustments.append(Adjustment("stale", self.stale_factor, f"{(today - posted).days}d"))
        if self.salary_curve:
            self._salary(v, hay, c)
        return c

    def _salary(self, v: Vacancy, hay: str, c: Check) -> None:
        usd = parse_salary_usd(v.salary_text, self.rates)
        if usd is None:
            c.flags.append("no_salary")  # unknown is not a reason to drop: ask
            return
        # employment is read from the header fields: the body says "это не подработка" as often as "подработка"
        header = "\n".join((v.title, v.work_format, v.salary_text))
        if _PROJECT.search(header):
            c.flags.append("project_pay")  # a one-off price is not a monthly salary
            return
        if is_part_time(header) and not is_hourly(v.salary_text):
            hours = monthly_hours(hay)
            if hours is None or hours < MIN_PART_TIME_HOURS:
                c.flags.append("part_time_pay")
                return
            usd = usd * FULL_TIME_HOURS / hours
        factor = salary_factor(usd, self.salary_curve)
        if factor is None:
            c.cut.append(Adjustment("salary_floor", 0.0, f"${usd:.0f}"))
        elif factor != 1.0:
            c.adjustments.append(Adjustment("salary", factor, f"${usd:.0f}"))
