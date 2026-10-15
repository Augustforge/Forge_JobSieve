"""After the ranking: what happened to the applications you sent, and whom to remind.

Reads a tracker (CSV: company, sent, status, channel, track, source, replied, updated, note) and answers three
questions: how many answers per week, which channel / resume / source gets answered, and who has been silent long
enough for one reminder. Silence is counted in working days from the last message either way."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

STATUSES = ("draft", "sent", "replied", "interview", "offer", "rejected", "closed")
ANSWERED = {"replied", "interview", "offer", "rejected"}
INTERVIEWED = {"interview", "offer"}
OPEN = ("sent", "replied", "interview", "offer")
REQUIRED = ("company", "sent")
DEFAULT_WAIT = 5  # working days of silence before one reminder
STATUS_WORDS = {"черновик": "draft", "отправлено": "sent", "отправлен": "sent", "ответили": "replied",
                "ответ": "replied", "собеседование": "interview", "интервью": "interview", "оффер": "offer",
                "отказ": "rejected", "закрыто": "closed", "закрыт": "closed"}
DOTTED = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")


@dataclass
class Application:
    company: str
    sent: date | None  # None only for a draft
    status: str = "sent"  # one of STATUSES; draft = not confirmed sent, closed = you stopped
    channel: str = ""  # how it was sent: email, hh, form, telegram
    track: str = ""  # which resume
    source: str = ""  # where the vacancy was found
    replied: date | None = None  # their first answer; set it when you close a thread that did get one
    updated: date | None = None  # the last message either way, e.g. your reminder
    note: str = ""

    @property
    def answered(self) -> bool:
        return self.status in ANSWERED or self.replied is not None

    @property
    def last_message(self) -> date | None:
        return max((d for d in (self.sent, self.replied, self.updated) if d), default=None)


@dataclass
class Rate:
    key: str
    sent: int = 0
    replied: int = 0
    interviews: int = 0


def working_days(start: date, end: date) -> int:
    return int(np.busday_count(start, end))


def _sent(apps: list[Application]) -> list[Application]:
    return [a for a in apps if a.status != "draft" and a.sent]


def follow_ups(apps: list[Application], today: date, wait: int = DEFAULT_WAIT,
               wait_by_channel: dict[str, int] | None = None):
    """(remind, pending, let_go) as lists of (application, working days since the last message), longest first.
    Remind: sent, no answer, quiet for the channel's wait. Let go: the same after a reminder — one is enough.
    Pending: they answered and the thread is still open, interviews and offers included."""
    waits = wait_by_channel or {}
    remind, pending, let_go = [], [], []
    for a in _sent(apps):
        days = working_days(a.last_message, today)
        if a.status == "sent" and not a.answered:
            if days >= waits.get(a.channel, wait):
                # with no answer yet, a message after the application can only be yours: the reminder
                (let_go if a.updated and a.updated > a.sent else remind).append((a, days))
        elif a.answered and a.status in OPEN:  # a reply date with the status left at sent counts too
            pending.append((a, days))
    return tuple(sorted(group, key=lambda x: -x[1]) for group in (remind, pending, let_go))


def _count(groups: dict[str, Rate], key: str, a: Application) -> None:
    r = groups.setdefault(key, Rate(key))
    r.sent += 1
    r.replied += a.answered
    r.interviews += a.status in INTERVIEWED


def rates(apps: list[Application], field: str) -> list[Rate]:
    groups: dict[str, Rate] = {}
    for a in _sent(apps):
        _count(groups, getattr(a, field) or "unknown", a)
    return sorted(groups.values(), key=lambda r: (-r.sent, r.key))


def by_week(apps: list[Application]) -> list[Rate]:
    groups: dict[str, Rate] = {}
    for a in _sent(apps):
        _count(groups, (a.sent - timedelta(days=a.sent.weekday())).isoformat(), a)
    return sorted(groups.values(), key=lambda r: r.key)


def _date(row: dict, field: str) -> date | None:
    value = (row.get(field) or "").strip()
    if not value:
        return None
    if DOTTED.fullmatch(value):  # Excel in many locales writes 05.10.2026
        day, month, year = value.split(".")
        return date(int(year), int(month), int(day))
    return date.fromisoformat(value)


def load_tracker(path: str | Path) -> tuple[list[Application], list[str]]:
    """Applications and the lines that could not be read, as "line N: why"."""
    apps, skipped = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        if not header.strip():
            return apps, skipped
        delimiter = ";" if header.count(";") > header.count(",") else ","  # Excel writes ; in many locales
        fields = [h.strip().lower() for h in next(csv.reader([header], delimiter=delimiter))]
        if any(c not in fields for c in REQUIRED):
            raise ValueError(f"{Path(path).name}: the header needs columns {', '.join(REQUIRED)}, got {', '.join(fields)}")
        reader = csv.DictReader(f, fieldnames=fields, delimiter=delimiter)
        for row in reader:
            where = f"line {reader.line_num + 1}"
            company = (row.get("company") or "").strip()
            status = (row.get("status") or "").strip().lower() or "sent"
            status = STATUS_WORDS.get(status, status)
            if not company:
                skipped.append(f"{where}: no company")
                continue
            if status not in STATUSES:
                skipped.append(f"{where}: unknown status {status!r}, use one of {', '.join(STATUSES)}")
                continue
            try:
                dates = {f: _date(row, f) for f in ("sent", "replied", "updated")}
            except ValueError as e:
                skipped.append(f"{where}: {e}")
                continue
            if dates["sent"] is None and status != "draft":
                skipped.append(f"{where}: no sent date")
                continue
            apps.append(Application(company, dates["sent"], status, (row.get("channel") or "").strip().lower(),
                                    (row.get("track") or "").strip(), (row.get("source") or "").strip(),
                                    dates["replied"], dates["updated"], (row.get("note") or "").strip()))
    return apps, skipped


def _cell(value) -> str:
    return str(value).replace("|", "/")  # a pipe would split the Markdown table


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "—"


def _rate_table(title: str, rows: list[Rate]) -> list[str]:
    out = [f"| {title} | sent | replied | rate | interviews |", "|---|---|---|---|---|"]
    return out + [f"| {_cell(r.key)} | {r.sent} | {r.replied} | {_pct(r.replied, r.sent)} | {r.interviews} |"
                  for r in rows]


def render_funnel(apps: list[Application], today: date, wait: int = DEFAULT_WAIT,
                  wait_by_channel: dict[str, int] | None = None, skipped: list[str] | tuple = ()) -> str:
    out = [f"# Application funnel — {today.isoformat()}", ""]
    sent, drafts = _sent(apps), [a for a in apps if a.status == "draft"]
    if not apps:
        out.append("No applications yet.")
    else:
        answered = sum(a.answered for a in sent)
        out += [f"Sent {len(sent)} · replied {answered} ({_pct(answered, len(sent))}) · "
                f"interviews {sum(a.status in INTERVIEWED for a in sent)} · "
                f"offers {sum(a.status == 'offer' for a in sent)}. Not confirmed sent: {len(drafts)}.", ""]
        remind, pending, let_go = follow_ups(apps, today, wait, wait_by_channel)
        overrides = ", ".join(f"{c}: {d}" for c, d in sorted((wait_by_channel or {}).items()))
        out += ["## Follow up", "",
                f"No answer for {wait}+ working days since the last message"
                f"{f' ({overrides})' if overrides else ''}. One reminder, then let it go.", ""]
        if remind:
            out += ["| sent | company | channel | track | working days silent |", "|---|---|---|---|---|"]
            out += [f"| {a.sent} | {_cell(a.company)} | {_cell(a.channel)} | {_cell(a.track)} | {d} |"
                    for a, d in remind]
        else:
            out.append("Nobody to remind.")
        if let_go:
            out += ["", "Reminded, still silent, let it go: "
                    + ", ".join(f"{a.company} ({d} working days)" for a, d in let_go) + "."]
        out += ["", "## Replied, next step open", ""]
        if pending:
            out += ["| replied | company | status | channel | working days since the last message | note |",
                    "|---|---|---|---|---|---|"]
            out += [f"| {a.replied or '—'} | {_cell(a.company)} | {a.status} | {_cell(a.channel)} | {d} | {_cell(a.note)} |"
                    for a, d in pending]
        else:
            out.append("No open threads.")
        out += ["", "## By week", "", "| week sent | sent | replied | interviews |", "|---|---|---|---|"]
        out += [f"| {r.key} | {r.sent} | {r.replied} | {r.interviews} |" for r in by_week(apps)]
        for field in ("channel", "track", "source"):
            if any(getattr(a, field) for a in sent):
                out += ["", f"## Reply rate by {field}", ""] + _rate_table(field, rates(apps, field))
        out += ["", "Counts are small: one answer out of four is one answer, not a 25% channel."]
        if drafts:
            out += ["", "## Not confirmed sent", "",
                    "Marked as draft. Sent or not? Until the tracker says, they stay out of every rate.", ""]
            out += [f"- {a.company}" for a in drafts]
        late = [a for a in apps if a.last_message and a.last_message > today]
        if late:
            out += ["", "## Check the dates", ""] + [f"- {a.company}: date after today ({a.last_message})" for a in late]
    if skipped:
        out += ["", "## Skipped rows", ""] + [f"- {s}" for s in skipped]
    return "\n".join(out) + "\n"
