"""Core records: vacancies, profile chunks and labels."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit

REASON_CODE = re.compile(r"[a-z][a-z0-9_]*")  # the candidate's own words: onsite, low_pay, scam
LABELS = frozenset({"fit", "no_fit"})
ACTORS = frozenset({"agent", "human"})

_HH_HOSTS = {"hh.ru", "hh.kz", "hh.uz"}
# query parameters that identify a vacancy (Greenhouse embeds, LinkedIn, generic job boards); the rest is tracking
_ID_PARAMS = {"id", "gh_jid", "jobid", "job_id", "currentjobid", "vacancy", "vacancyid", "posting", "jk"}
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Embedding models read ~512 tokens; a Russian vacancy is often 600-1000. Blocks under these headings go first so the
# requirements and conditions are read and the company story is what gets cut.
_KEY_HEADING = re.compile(r"требовани|обязанност|задач|ожида|ждём|ждем|нужно|предстоит|будешь|стек|навык|услови|"
                          r"предлагаем|requirement|responsibilit|qualification|what you|you will|you'll|must have|"
                          r"nice to have|skills|stack|we offer|benefits|compensation", re.I)
_OTHER_HEADING = re.compile(r"о компании|о нас|кто мы|почему мы|about|who we are|why us", re.I)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60 or s[0] in "-•*·–—":  # a bullet is content, even when it says "задачи" or "about"
        return False
    return s.endswith(":") or bool(_KEY_HEADING.search(s)) or bool(_OTHER_HEADING.match(re.sub(r"^\W+", "", s)))


def _requirements_first(text: str) -> str:
    blocks: list[tuple[bool, list[str]]] = [(False, [])]
    for line in text.splitlines():
        if _is_heading(line):
            blocks.append((bool(_KEY_HEADING.search(line)), [line]))
        else:
            blocks[-1][1].append(line)
    if not any(key for key, _ in blocks):
        return text
    parts = [b for key, b in blocks if key] + [b for key, b in blocks if not key]
    return "\n".join(s for s in ("\n".join(b).strip() for b in parts) if s)


def normalize_url(url: str) -> str:
    """Canonical form for dedup: lowercase host without www, no scheme/fragment, no trailing slash,
    only vacancy-identifying query parameters.

    Every regional hh domain collapses to hh.ru, so krasnoyarsk.hh.ru/vacancy/1?x and hh.kz/vacancy/1 match.
    """
    url = url.strip()
    parts = urlsplit(url if "://" in url else "https://" + url)
    host = parts.netloc.lower().removeprefix("www.")
    if host in _HH_HOSTS or host.endswith(".hh.ru"):
        host = "hh.ru"
    kept = sorted((k.lower(), v) for k, v in parse_qsl(parts.query) if k.lower() in _ID_PARAMS)
    query = f"?{urlencode(kept)}" if kept else ""
    return f"{host}{parts.path.rstrip('/')}{query}"


def run_date(run_id: str) -> str:
    """The YYYY-MM-DD inside a run id ("links-2026-10-12" → "2026-10-12"), or "" when there is none.
    Runs are ordered by this date, never by the raw string: "links-…" vs "results-…" sorts by prefix."""
    m = _DATE.search(run_id)
    return m.group(0) if m else ""


def make_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:16]


def detect_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return ""
    cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return "ru" if cyrillic / len(letters) > 0.3 else "en"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Vacancy:
    id: str
    source: str
    url: str
    title: str
    company: str
    text: str
    salary_text: str = ""
    location: str = ""
    work_format: str = ""
    published_at: str = ""
    lang: str = ""
    run_id: str = ""
    collected_at: str = ""

    @classmethod
    def from_dict(cls, d: dict, run_id: str = "") -> "Vacancy":
        missing = [k for k in ("source", "url", "title") if not str(d.get(k, "")).strip()]
        if missing:
            raise ValueError(f"vacancy record is missing {', '.join(missing)}: {d.get('url') or d.get('title')!r}")
        text = str(d.get("text", "")).strip()
        return cls(
            id=make_id(str(d["url"])),
            source=str(d["source"]).strip(),
            url=str(d["url"]).strip(),
            title=str(d["title"]).strip(),
            company=str(d.get("company", "")).strip(),
            text=text,
            salary_text=str(d.get("salary_text", "")).strip(),
            location=str(d.get("location", "")).strip(),
            work_format=str(d.get("work_format", "")).strip(),
            published_at=str(d.get("published_at", "")).strip(),
            lang=str(d.get("lang") or detect_lang(text)),
            run_id=str(d.get("run_id") or run_id),
            collected_at=str(d.get("collected_at") or now_iso()),
        )

    def embed_text(self) -> str:
        return f"{self.title}\n{_requirements_first(self.text)}"


@dataclass(frozen=True)
class ProfileChunk:
    id: str
    track: str
    lang: str
    section: str
    text: str


@dataclass(frozen=True)
class Label:
    vacancy_id: str
    label: str
    actor: str = "agent"
    track: str = ""
    reasons: tuple[str, ...] = ()
    note: str = ""
    created_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be fit or no_fit, got {self.label!r}")
        if self.actor not in ACTORS:
            raise ValueError(f"actor must be agent or human, got {self.actor!r}")
        bad = [r for r in self.reasons if not REASON_CODE.fullmatch(r)]
        if bad:
            raise ValueError(f"reason codes are lowercase words like low_pay, got: {', '.join(bad)}")
