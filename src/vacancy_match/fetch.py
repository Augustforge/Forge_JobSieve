"""Collect vacancies from official public job APIs into records for `vm add` — no scraping, no login.

A source is `kind:argument`:
  greenhouse:<board>, lever:<company>, ashby:<org>   one company's open jobs; the name is in its careers URL
      (job-boards.greenhouse.io/<board>, jobs.lever.co/<company>, jobs.ashbyhq.com/<org>)
  remotive:<search>, remoteok[:<tag>], himalayas:<search>, arbeitnow   remote-job aggregators
  hh:<search>   HeadHunter; since 2026 its API needs an application token (dev.hh.ru) in HH_TOKEN
  page:<url>    one vacancy page you found yourself, read from its schema.org JobPosting markup
Every record keeps the job's own URL and its source: the aggregators ask you to link back and credit them if you
republish their jobs, and Remotive asks for no more than a few requests a day."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlsplit
from urllib.request import Request, urlopen

from .jobposting import html_to_text, parse_jobposting

USER_AGENT = "vacancy-match (+https://github.com/Augustforge/vacancy-match)"
_PAY_RANGE = re.compile(r'class="pay-range">(.*?)</div>', re.S)
_EMPLOYMENT = {"FullTime": "Full-time", "PartTime": "Part-time", "Intern": "Internship"}
HH_API = "https://api.hh.ru"
HH_PAUSE = 0.25  # seconds between vacancy requests: one search brings up to 50 of them


class SourceError(Exception):
    """A source answered, but not with jobs: the message says what to do."""


def http_get(url: str, headers: dict | None = None, raw: bool = False):
    """JSON from an API, or the page's text with raw=True."""
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html" if raw else "application/json",
                                **(headers or {})})
    with urlopen(req, timeout=30) as r:
        body = r.read()
        return body.decode(r.headers.get_content_charset() or "utf-8", "replace") if raw else json.loads(body)


def _record(source, url, title, company, text, salary="", published="", work_format="", location="") -> dict:
    fields = {"title": title, "company": company, "text": text, "salary_text": salary, "published_at": published,
              "work_format": work_format, "location": location}
    return {"source": source, "url": url, **{k: (v or "").strip() for k, v in fields.items()}}


def _lines(*parts: str) -> str:
    return "\n".join(line.strip() for part in parts for line in (part or "").splitlines() if line.strip())


def _joined(*parts, sep: str = ", ") -> str:
    return sep.join(p for p in parts if p)


def _epoch(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


def _period(interval: str) -> str:
    s = interval.lower()
    return next((p for key, p in (("year", "per year"), ("annual", "per year"), ("month", "per month"),
                                  ("hour", "per hour")) if key in s), "")


def _range(currency: str, low, high, period: str) -> str:
    """'USD 211400 – 290600 per year', in words the salary parser reads; '' when the source gives no figures."""
    figures = [str(round(x)) for x in (low, high) if x]
    return _joined(currency, " – ".join(dict.fromkeys(figures)), period, sep=" ") if figures else ""


def _greenhouse(board, get):
    out = []
    for j in get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")["jobs"]:
        content = unescape(j.get("content") or "")
        pay = _PAY_RANGE.search(content)
        out.append(_record("greenhouse", j["absolute_url"], j.get("title"), j.get("company_name") or board,
                           html_to_text(content), html_to_text(pay.group(1)) if pay else "",
                           j.get("first_published") or j.get("updated_at") or "", "",
                           (j.get("location") or {}).get("name", "")))
    return out


def _lever(company, get):
    out = []
    for j in get(f"https://api.lever.co/v0/postings/{company}?mode=json"):
        lists = [x for block in j.get("lists") or [] for x in (block.get("text"), html_to_text(block.get("content", "")))]
        cat, pay = j.get("categories") or {}, j.get("salaryRange") or {}
        out.append(_record("lever", j["hostedUrl"], j.get("text"), company,
                           _lines(j.get("descriptionPlain"), *lists, j.get("additionalPlain")),
                           _range(pay.get("currency", ""), pay.get("min"), pay.get("max"), _period(pay.get("interval", ""))),
                           _epoch(j["createdAt"] / 1000) if j.get("createdAt") else "",
                           _joined(cat.get("commitment"), j.get("workplaceType")),
                           _joined(*(cat.get("allLocations") or [cat.get("location")]), sep="; ")))
    return out


def _ashby(org, get):
    out = []
    for j in get(f"https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true")["jobs"]:
        if not j.get("isListed", True):
            continue
        pay = next((c for c in (j.get("compensation") or {}).get("summaryComponents") or []
                    if c.get("compensationType") == "Salary"), {})
        kind = j.get("employmentType") or ""
        out.append(_record("ashby", j["jobUrl"], j.get("title"), org,
                           _lines(j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", ""))),
                           _range(pay.get("currencyCode") or "", pay.get("minValue"), pay.get("maxValue"),
                                  _period(pay.get("interval") or "")),
                           j.get("publishedAt", ""), _joined(_EMPLOYMENT.get(kind, kind), j.get("workplaceType")),
                           _joined(j.get("location"), *(x.get("location") for x in j.get("secondaryLocations") or []),
                                   sep="; ")))
    return out


def _remotive(search, get):
    return [_record("remotive", j["url"], j.get("title"), j.get("company_name"), html_to_text(j.get("description", "")),
                    j.get("salary") or "", j.get("publication_date", ""), (j.get("job_type") or "").replace("_", " "),
                    j.get("candidate_required_location", ""))
            for j in get(f"https://remotive.com/api/remote-jobs?search={quote_plus(search)}")["jobs"]]


def _remoteok(tag, get):
    jobs = get("https://remoteok.com/api" + (f"?tag={quote_plus(tag)}" if tag else ""))
    return [_record("remoteok", j["url"], j.get("position"), j.get("company"), html_to_text(j.get("description", "")),
                    _range("USD", j.get("salary_min"), j.get("salary_max"), "per year"), j.get("date", ""), "",
                    j.get("location", ""))
            for j in jobs if "position" in j]  # the first element is the terms of use


def _himalayas(search, get):
    return [_record("himalayas", j.get("guid") or j.get("applicationLink"), j.get("title"), j.get("companyName"),
                    html_to_text(j.get("description", "")),
                    _range(j.get("currency") or "", j.get("minSalary"), j.get("maxSalary"),
                           _period(j.get("salaryPeriod") or "")),
                    _epoch(j["pubDate"]) if j.get("pubDate") else "", j.get("employmentType") or "",
                    _joined(*(j.get("locationRestrictions") or []), sep="; "))
            for j in get(f"https://himalayas.app/jobs/api/search?q={quote_plus(search)}")["jobs"]]


def _arbeitnow(_, get):
    return [_record("arbeitnow", j["url"], j.get("title"), j.get("company_name"), html_to_text(j.get("description", "")),
                    "", _epoch(j["created_at"]) if j.get("created_at") else "",
                    _joined(*(j.get("job_types") or []), "remote" if j.get("remote") else ""), j.get("location", ""))
            for j in get("https://www.arbeitnow.com/api/job-board-api?page=1")["data"]]


def _hh(search, get):
    token = os.environ.get("HH_TOKEN", "")
    if not token:
        raise SourceError("needs an application token since 2026 — register an app at dev.hh.ru and set HH_TOKEN")
    # hh wants an app name with a contact; set HH_USER_AGENT="YourApp/1.0 (you@mail)" if it refuses the default
    headers = {"Authorization": f"Bearer {token}", "HH-User-Agent": os.environ.get("HH_USER_AGENT", USER_AGENT)}
    out = []
    for item in get(f"{HH_API}/vacancies?text={quote_plus(search)}&per_page=50", headers)["items"]:
        time.sleep(HH_PAUSE)
        full = get(f"{HH_API}/vacancies/{item['id']}", headers)  # the search gives a two-line snippet, not the text
        skills = ", ".join(s["name"] for s in full.get("key_skills") or [])
        pay = item.get("salary_range") or item.get("salary") or {}
        out.append(_record("hh", item["alternate_url"], item.get("name"), (item.get("employer") or {}).get("name"),
                           _lines(html_to_text(full.get("description", "")), f"Key skills: {skills}" if skills else ""),
                           _range(pay.get("currency") or "", pay.get("from"), pay.get("to"), "per month"),
                           item.get("published_at", ""),
                           _joined((item.get("employment") or {}).get("name"), (item.get("schedule") or {}).get("name")),
                           (item.get("area") or {}).get("name", "")))
    return out


def _page(url, get):
    url = url if "://" in url else f"https://{url}"
    found = parse_jobposting(get(url, raw=True))
    if not found:
        raise SourceError("no schema.org JobPosting on the page; for Greenhouse, Lever or Ashby use their own source, "
                          "otherwise add it to the JSONL by hand")
    return [{"source": urlsplit(url).hostname or "page", "url": url, **found}]


# kind → (collector, what the argument must be; None when it is optional)
SOURCES = {
    "greenhouse": (_greenhouse, "a board name, e.g. greenhouse:<board>"),
    "lever": (_lever, "a board name, e.g. lever:<company>"),
    "ashby": (_ashby, "a board name, e.g. ashby:<org>"),
    "remotive": (_remotive, "a search, e.g. remotive:python"),
    "remoteok": (_remoteok, None),
    "himalayas": (_himalayas, "a search, e.g. himalayas:python"),
    "arbeitnow": (_arbeitnow, None),
    "hh": (_hh, "a search, e.g. hh:python"),
    "page": (_page, "a vacancy URL, e.g. page:https://..."),
}


def collect(sources: list[str], get=None, where: str = "", log=None) -> tuple[list[dict], list[str]]:
    """Records from every source that answered, and one message per source that did not.
    `where` keeps jobs whose title, location or work format match the regex (e.g. "remote")."""
    get = get or http_get
    keep = re.compile(where, re.I) if where else None
    records, errors = [], []
    for source in sources:
        kind, _, arg = source.partition(":")
        if kind not in SOURCES:
            errors.append(f"{source}: unknown source; use one of {', '.join(SOURCES)}")
            continue
        collector, needs = SOURCES[kind]
        if needs and not arg.strip():
            errors.append(f"{source}: needs {needs}")
            continue
        try:
            got = [r for r in collector(arg.strip(), get) if r["url"] and r["title"]]
        except HTTPError as e:
            errors.append(f"{source}: HTTP {e.code}" + (" — check the board name in the careers URL" if e.code == 404 else ""))
            continue
        except (SourceError, URLError, TimeoutError, ValueError, KeyError, TypeError) as e:  # offline, odd shape
            errors.append(f"{source}: {e}")
            continue
        kept = [r for r in got if not keep or keep.search(f"{r['title']}\n{r['location']}\n{r['work_format']}")]
        if log:
            log(f"{source}: {len(got)} found" + (f", {len(kept)} kept" if keep else ""))
        records += kept
    return records, errors
