"""Core records: vacancies, profile chunks and labels."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit

REASON_CODE = re.compile(r"[^\W\d_][\w-]*")  # the candidate's own words, one per reason: onsite, cold-calls, ночь
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


# scripts that name their language; Cyrillic is labelled "ru" (also Ukrainian, Kazakh \u2014 one scale fits them)
_SCRIPTS = [("ja", "\u3040", "\u30ff"), ("ko", "\uac00", "\ud7af"), ("zh", "\u4e00", "\u9fff"), ("ru", "\u0400", "\u04ff"),
            ("ar", "\u0600", "\u06ff"), ("he", "\u0590", "\u05ff"), ("el", "\u0370", "\u03ff"), ("th", "\u0e00", "\u0e7f"),
            ("hi", "\u0900", "\u097f")]
# Latin-script languages tell themselves apart by their most frequent short words
_STOPWORDS = {
    "en": "the and for with you our are will to of in is a this that we on as be",
    "es": "el la los las de que y para con en un una por del se es al lo nuestro",
    "pt": "o a os as de que e para com em um uma por do da dos n\u00e3o \u00e9 nossa",
    "de": "der die das und f\u00fcr mit ist wir sie ein eine den zu von auf im nicht unser",
    "fr": "le la les et pour avec des un une en du est nous vous dans sur au pas",
    "it": "il la di che e per con un una del della \u00e8 sono nel al gli le",
    "nl": "de het een en van voor met is wij je op te in zijn niet ons",
    "pl": "i w na z do jest si\u0119 dla nie \u017ce oraz to od po jak",
    "tr": "ve bir i\u00e7in ile bu da de olarak en \u00e7ok olan biz",
}
_STOPWORD_SETS = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
_WORD = re.compile(r"[^\W\d_]+")


def detect_lang(text: str) -> str:
    """A two-letter guess, no dependencies: by script, then by frequent words for Latin script; "" without letters.
    It only has to be consistent \u2014 vacancies in one language share one fit scale."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return ""
    counts = {lang: sum(1 for c in letters if lo <= c <= hi) for lang, lo, hi in _SCRIPTS}
    script, n = max(counts.items(), key=lambda kv: kv[1])
    cjk = counts["ja"] + counts["zh"]
    if script in ("zh", "ja") and counts["ja"] >= 0.2 * cjk:  # kana is only Japanese, kanji alone reads as Chinese;
        script, n = "ja", cjk                                 # one Japanese company name keeps Chinese text Chinese
    if n / len(letters) > 0.3:
        return script
    words = [w.lower() for w in _WORD.findall(text) if len(w) > 1]  # not the "e" of e-commerce or the "a" of A/B
    votes = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORD_SETS.items()}
    best = max(votes, key=votes.get)
    if votes[best]:
        return best
    return script if n >= 0.1 * len(letters) else "en"  # no Latin sentence: stack names around Russian, Chinese, …


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
            raise ValueError(f"a reason code is one word, e.g. low_pay or cold-calls, got: {', '.join(bad)}")
