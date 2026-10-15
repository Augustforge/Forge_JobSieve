import json

from vacancy_match.jobposting import html_to_text, parse_jobposting


def page(obj):
    return f'<html><head><script type="application/ld+json">{json.dumps(obj)}</script></head></html>'


def test_html_to_text_keeps_lines_and_unescapes():
    assert html_to_text("<h2>Role</h2><ul><li>Python &amp; SQL</li><li>Remote</li></ul>") == "Role\n- Python & SQL\n- Remote"


def test_list_items_stay_bullets_so_they_are_never_read_as_headings():
    assert html_to_text("<p>Requirements</p><ul><li><p>Strong Python skills</p></li></ul>") == \
        "Requirements\n- Strong Python skills"


def test_parse_hh_style_posting():
    rec = parse_jobposting(page({
        "@context": "https://schema.org/", "@type": "JobPosting", "title": "QA &amp; AI",
        "description": "<p>Пишем автотесты</p>", "datePosted": "2026-10-12T16:16:24.267+03:00",
        "hiringOrganization": {"@type": "Organization", "name": "Gear Games"},
        "jobLocation": {"@type": "Place", "address": {"addressLocality": "Москва"}},
    }))
    assert rec == {"title": "QA & AI", "company": "Gear Games", "text": "Пишем автотесты",
                   "published_at": "2026-10-12T16:16:24.267+03:00", "location": "Москва",
                   "work_format": "", "salary_text": ""}


def test_parse_graph_remote_and_salary():
    rec = parse_jobposting(page({"@graph": [{"@type": "WebPage"}, {
        "@type": ["JobPosting"], "title": "Analyst", "description": "x", "jobLocationType": "TELECOMMUTE",
        "baseSalary": {"currency": "USD", "value": {"minValue": 2000, "maxValue": 4000, "unitText": "MONTH"}}}]}))
    assert rec["work_format"] == "remote"
    assert rec["salary_text"] == "from 2000 to 4000 USD per month"


def test_no_posting_returns_none():
    assert parse_jobposting("<html><script type='application/ld+json'>{bad json</script></html>") is None
    assert parse_jobposting(page({"@type": "Organization"})) is None


def test_nulls_do_not_crash_or_become_text():
    rec = parse_jobposting(page({"@type": "JobPosting", "title": None, "description": None,
                                 "hiringOrganization": {"name": None}}))
    assert rec["title"] == "" and rec["company"] == "" and rec["text"] == ""


def test_employment_type_lands_in_work_format():
    page = ('<script type="application/ld+json">{"@type": "JobPosting", "title": "T", "employmentType": ["PART_TIME"],'
            ' "jobLocationType": "TELECOMMUTE"}</script>')
    assert parse_jobposting(page)["work_format"] == "remote, part-time"
