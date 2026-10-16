import json
from urllib.error import HTTPError

import pytest

from vacancy_match import fetch as fetch_mod
from vacancy_match.cli import main
from vacancy_match.fetch import collect
from vacancy_match.models import Vacancy

GREENHOUSE = {"jobs": [{
    "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1", "title": "Backend Engineer", "company_name": "Acme",
    "content": "&lt;h3&gt;Requirements&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;&lt;div class=&quot;"
               "pay-range&quot;&gt;&lt;span&gt;$120,000&lt;/span&gt;&lt;span class=&quot;divider&quot;&gt;&amp;mdash;"
               "&lt;/span&gt;&lt;span&gt;$150,000 USD&lt;/span&gt;&lt;/div&gt;",
    "location": {"name": "Remote, EU"}, "first_published": "2026-10-02T11:31:50-04:00",
    "updated_at": "2026-10-05T10:00:00-04:00"}]}
LEVER = [{"text": "Data Engineer", "hostedUrl": "https://jobs.lever.co/acme/abc", "descriptionPlain": "About us",
          "lists": [{"text": "Requirements", "content": "<li>SQL</li><li>Python</li>"}], "additionalPlain": "Benefits",
          "categories": {"commitment": "Full-time", "location": "Remote", "allLocations": ["Remote", "Berlin"]},
          "workplaceType": "remote", "createdAt": 1786469891368,
          "salaryRange": {"min": 60000, "max": 80000, "currency": "EUR", "interval": "per-year-salary"}}]
ASHBY = {"jobs": [
    {"title": " Security Engineer ", "jobUrl": "https://jobs.ashbyhq.com/acme/1", "descriptionPlain": "Secure things",
     "employmentType": "FullTime", "workplaceType": "Remote", "location": "New York",
     "secondaryLocations": [{"location": "Remote (US)"}], "publishedAt": "2026-04-07T17:12:35.753+00:00",
     "isListed": True, "compensation": {"summaryComponents": [
         {"compensationType": "EquityPercentage", "interval": "NONE", "currencyCode": None, "minValue": None},
         {"compensationType": "Salary", "interval": "1 YEAR", "currencyCode": "USD", "minValue": 211400,
          "maxValue": 290600}]}},
    {"title": "Hidden", "jobUrl": "https://jobs.ashbyhq.com/acme/2", "isListed": False}]}
REMOTIVE = {"0-legal-notice": "link back", "jobs": [
    {"url": "https://remotive.com/remote-jobs/dev/python-dev-1", "title": "Python Dev", "company_name": "Coalition ",
     "description": "<p>Build APIs</p>", "job_type": "full_time", "publication_date": "2026-10-11T20:17:27",
     "candidate_required_location": "Worldwide", "salary": "$35,3k- $52k"}]}
REMOTEOK = [{"legal": "API Terms of Service: link back"},
            {"position": "Principal Engineer", "company": "Vyta", "url": "https://remoteOK.com/remote-jobs/1",
             "description": "<p>Lead</p>", "location": "Kansas", "date": "2026-10-13T00:00:18+00:00",
             "salary_min": 0, "salary_max": 0},
            {"position": "Go Developer", "company": "Gopher", "url": "https://remoteOK.com/remote-jobs/2",
             "description": "<p>Go</p>", "location": "", "date": "2026-10-12T00:00:00+00:00",
             "salary_min": 90000, "salary_max": 120000}]
HIMALAYAS = {"jobs": [{"title": "Account Manager", "companyName": "Growmark",
                       "guid": "https://himalayas.app/companies/growmark/jobs/am", "description": "<p>Sell</p>",
                       "minSalary": 107750, "maxSalary": 148175, "currency": "USD", "salaryPeriod": "annual",
                       "employmentType": "Full Time", "locationRestrictions": ["United States"],
                       "pubDate": 1792058156}]}
ARBEITNOW = {"data": [{"title": "CRM Manager", "company_name": "Juit GmbH", "url": "https://www.arbeitnow.com/jobs/1",
                       "description": "<h2>Your Impact</h2><p>CRM</p>", "remote": True, "job_types": ["Permanent"],
                       "location": "Berlin", "created_at": 1792054871}]}

PAYLOADS = {
    "https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true": GREENHOUSE,
    "https://api.lever.co/v0/postings/acme?mode=json": LEVER,
    "https://api.ashbyhq.com/posting-api/job-board/acme?includeCompensation=true": ASHBY,
    "https://remotive.com/api/remote-jobs?search=python+dev": REMOTIVE,
    "https://remoteok.com/api?tag=golang": REMOTEOK,
    "https://himalayas.app/jobs/api/search?q=account": HIMALAYAS,
    "https://www.arbeitnow.com/api/job-board-api?page=1": ARBEITNOW,
}


HH_SEARCH = {"items": [{"id": "77", "name": "Python-разработчик", "alternate_url": "https://hh.ru/vacancy/77",
                        "employer": {"name": "Бот-Ко"}, "published_at": "2026-10-14T10:00:00+0300",
                        "salary_range": {"from": 150000, "to": 200000, "currency": "RUR", "gross": False},
                        "employment": {"name": "Полная занятость"}, "schedule": {"name": "Удаленная работа"},
                        "area": {"name": "Москва"}}]}
HH_DETAIL = {"id": "77", "description": "<p><strong>Требования:</strong></p><ul><li>Python</li></ul>",
             "key_skills": [{"name": "Django"}, {"name": "SQL"}]}
PAGE = ('<html><script type="application/ld+json">{"@type": "JobPosting", "title": "QA Engineer", '
        '"description": "<p>Tests</p>", "hiringOrganization": {"name": "Acme"}}</script></html>')
PAYLOADS.update({"https://api.hh.ru/vacancies?text=python&per_page=50": HH_SEARCH,
                 "https://api.hh.ru/vacancies/77": HH_DETAIL, "https://jobs.example.com/qa": PAGE,
                 "https://jobs.example.com/blank": "<html>no posting</html>"})
SEEN_HEADERS = []


def fake_get(url, headers=None, raw=False):
    SEEN_HEADERS.append(headers or {})
    if url not in PAYLOADS or isinstance(PAYLOADS[url], str) != raw:
        raise HTTPError(url, 404, "Not Found", None, None)
    return PAYLOADS[url]


def one(source):
    records, errors = collect([source], get=fake_get)
    assert errors == []
    return records


def test_greenhouse_keeps_requirements_as_text_and_the_pay_range_as_salary():
    [r] = one("greenhouse:acme")
    assert r == {"source": "greenhouse", "url": "https://job-boards.greenhouse.io/acme/jobs/1",
                 "title": "Backend Engineer", "company": "Acme", "text": "Requirements\n- Python\n$120,000—$150,000 USD",
                 "salary_text": "$120,000—$150,000 USD", "published_at": "2026-10-02T11:31:50-04:00",
                 "work_format": "", "location": "Remote, EU"}


def test_lever_joins_description_lists_and_salary_range():
    [r] = one("lever:acme")
    assert r["text"] == "About us\nRequirements\n- SQL\n- Python\nBenefits"
    assert (r["salary_text"], r["work_format"], r["location"]) == ("EUR 60000 – 80000 per year", "Full-time, remote",
                                                                   "Remote; Berlin")
    assert r["published_at"].startswith("2026-08-11T") and r["company"] == "acme"


def test_ashby_skips_unlisted_jobs_and_reads_the_salary_component():
    [r] = one("ashby:acme")
    assert (r["title"], r["salary_text"], r["work_format"], r["location"]) == (
        "Security Engineer", "USD 211400 – 290600 per year", "Full-time, Remote", "New York; Remote (US)")


def test_aggregators():
    [rem] = one("remotive:python dev")
    assert (rem["company"], rem["salary_text"], rem["work_format"], rem["location"]) == (
        "Coalition", "$35,3k- $52k", "full time", "Worldwide")
    ok = one("remoteok:golang")
    assert [r["title"] for r in ok] == ["Principal Engineer", "Go Developer"]  # the legal notice is not a job
    assert [r["salary_text"] for r in ok] == ["", "USD 90000 – 120000 per year"]
    [him] = one("himalayas:account")
    assert (him["salary_text"], him["location"], him["published_at"][:10]) == (
        "USD 107750 – 148175 per year", "United States", "2026-10-15")
    [arb] = one("arbeitnow")
    assert (arb["work_format"], arb["text"]) == ("Permanent, remote", "Your Impact\nCRM")


def test_every_record_is_a_valid_vacancy():
    records, _ = collect(["greenhouse:acme", "lever:acme", "ashby:acme", "remotive:python dev", "remoteok:golang",
                          "himalayas:account", "arbeitnow"], get=fake_get)
    assert len(records) == 8 and all(Vacancy.from_dict(r).title for r in records)


def test_bad_sources_are_reported_and_the_rest_still_collected():
    records, errors = collect(["greenhouse:acme", "greenhouse:nosuch", "monster:jobs", "lever", "lever:acme"],
                              get=fake_get)
    assert len(records) == 2
    assert errors == ["greenhouse:nosuch: HTTP 404 — check the board name in the careers URL",
                      "monster:jobs: unknown source; use one of " + ", ".join(fetch_mod.SOURCES),
                      "lever: needs a board name, e.g. lever:<company>"]


def test_where_keeps_only_matching_jobs():
    records, _ = collect(["remoteok:golang", "greenhouse:acme"], get=fake_get, where="remote")
    assert [r["title"] for r in records] == ["Backend Engineer"]  # location "Remote, EU"


def test_cli_fetch_writes_jsonl_that_vm_add_accepts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch_mod, "http_get", fake_get)
    out = tmp_path / "run.jsonl"
    assert main(["fetch", "greenhouse:acme", "lever:nosuch", "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["title"] == "Backend Engineer"
    err = capsys.readouterr().err
    assert "greenhouse:acme: 1" in err and "lever:nosuch: HTTP 404" in err
    assert main(["--db", str(tmp_path / "vm.db"), "--model", "hash-256", "add", str(out), "--run", "2026-10-15"]) == 0
    assert main(["fetch", "lever:nosuch", "--out", str(out)]) == 1  # nothing collected


def test_hh_needs_an_application_token(monkeypatch):
    monkeypatch.delenv("HH_TOKEN", raising=False)
    _, errors = collect(["hh:python"], get=fake_get)
    assert errors == ["hh:python: needs an application token since 2026 — register an app at dev.hh.ru and set HH_TOKEN"]


def test_hh_with_a_token_reads_search_and_full_description(monkeypatch):
    monkeypatch.setenv("HH_TOKEN", "t0ken")
    monkeypatch.setattr(fetch_mod, "HH_PAUSE", 0)
    SEEN_HEADERS.clear()
    [r] = one("hh:python")
    assert r == {"source": "hh", "url": "https://hh.ru/vacancy/77", "title": "Python-разработчик", "company": "Бот-Ко",
                 "text": "Требования:\n- Python\nKey skills: Django, SQL", "salary_text": "RUR 150000 – 200000 per month",
                 "published_at": "2026-10-14T10:00:00+0300", "work_format": "Полная занятость, Удаленная работа",
                 "location": "Москва"}
    assert all(h.get("Authorization") == "Bearer t0ken" and h.get("HH-User-Agent") for h in SEEN_HEADERS)


def test_page_reads_one_vacancy_you_found_yourself():
    [r] = one("page:https://jobs.example.com/qa")
    assert (r["source"], r["url"], r["title"], r["company"], r["text"]) == (
        "jobs.example.com", "https://jobs.example.com/qa", "QA Engineer", "Acme", "Tests")
    _, errors = collect(["page:https://jobs.example.com/blank"], get=fake_get)
    assert errors == ["page:https://jobs.example.com/blank: no schema.org JobPosting on the page; for Greenhouse, "
                      "Lever or Ashby use their own source, otherwise add it to the JSONL by hand"]


@pytest.mark.parametrize("source", ["remotive:", "himalayas:", "page:"])
def test_search_sources_need_a_query(source):
    _, errors = collect([source], get=fake_get)
    assert errors and "needs" in errors[0]


def test_remoteok_location_is_repaired_like_the_title():
    garbled = "دبي, الإمارات".encode("utf-8").decode("latin-1")  # how Remote OK sends it
    job = {"position": "Analyst", "company": "X", "url": "https://remoteOK.com/remote-jobs/9", "description": "<p>A</p>",
           "location": garbled, "date": "2026-10-12T00:00:00+00:00", "salary_min": 0, "salary_max": 0}
    [rec] = collect(["remoteok:x"], get=lambda url: [{"legal": "terms"}, job])[0]
    assert rec["location"] == "دبي, الإمارات"
