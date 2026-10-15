from datetime import date

import pytest

from vacancy_match.cli import main
from vacancy_match.funnel import Application, by_week, follow_ups, load_tracker, rates, render_funnel, working_days

TODAY = date(2026, 10, 15)  # a Thursday


def app(company, sent, status="sent", channel="email", **kw):
    return Application(company, date.fromisoformat(sent) if sent else None, status, channel, **kw)


def test_working_days_skip_the_weekend():
    assert working_days(date(2026, 10, 9), date(2026, 10, 12)) == 1  # Friday to Monday
    assert working_days(date(2026, 10, 8), TODAY) == 5


def test_silence_is_counted_in_working_days_with_a_wait_per_channel():
    apps = [app("Five", "2026-10-08"), app("Four", "2026-10-09"),
            app("HhSeven", "2026-10-06", channel="hh"), app("HhFive", "2026-10-08", channel="hh")]
    silent, _, _ = follow_ups(apps, TODAY, wait=5, wait_by_channel={"hh": 7})
    assert [(a.company, days) for a, days in silent] == [("HhSeven", 7), ("Five", 5)]


def test_answered_or_unsent_rows_are_not_chased_and_open_threads_include_interviews():
    apps = [app("Answered", "2026-10-01", "replied", replied=date(2026, 10, 2)),
            app("Interview", "2026-10-01", "interview", replied=date(2026, 10, 2), updated=date(2026, 10, 13)),
            app("Offer", "2026-10-01", "offer", updated=date(2026, 10, 14)),
            app("Rejected", "2026-10-01", "rejected"), app("Dropped", "2026-10-01", "closed"),
            app("Draft", None, "draft"), app("DateOnly", "2026-10-01", replied=date(2026, 10, 13))]
    silent, pending, let_go = follow_ups(apps, TODAY)
    assert silent == [] and let_go == []
    assert [(a.company, days) for a, days in pending] == [("Answered", 9), ("Interview", 2), ("DateOnly", 2),
                                                          ("Offer", 1)]


def test_one_reminder_then_let_go():
    apps = [app("Reminded", "2026-10-01", updated=date(2026, 10, 12)),  # 3 working days since the reminder: wait
            app("StillSilent", "2026-09-20", updated=date(2026, 10, 1))]  # 10 working days since the reminder
    silent, _, let_go = follow_ups(apps, TODAY)
    assert silent == [] and [(a.company, days) for a, days in let_go] == [("StillSilent", 10)]


def test_rates_count_any_answer_and_leave_drafts_out():
    apps = [app("A", "2026-10-01", "replied", channel="hh"), app("B", "2026-10-01", channel="hh"),
            app("C", "2026-10-02", "interview", channel="email"),
            app("D", "2026-10-02", "closed", channel="email", replied=date(2026, 10, 3)),  # answered, then we stopped
            app("E", None, "draft", channel="email")]
    got = {r.key: (r.sent, r.replied, r.interviews) for r in rates(apps, "channel")}
    assert got == {"hh": (2, 1, 0), "email": (2, 2, 1)}


def test_weeks_start_on_monday():
    apps = [app("A", "2026-10-04"), app("B", "2026-10-05", "replied"), app("C", "2026-10-11")]
    assert [(w.key, w.sent, w.replied) for w in by_week(apps)] == [("2026-09-28", 1, 0), ("2026-10-05", 2, 1)]


def test_load_tracker_reads_csv_and_names_bad_lines(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("company,sent,status,channel,track,source,replied,updated,note\n"
                 "Acme,2026-10-01,replied,email,backend,example.com,2026-10-03,,asked for a test task\n"
                 "Draftly,,draft,hh,,,,,\n"
                 "NoDate,yesterday,sent,email,,,,,\n"
                 "Weird,2026-10-01,ghosted,email,,,,,\n"
                 ",2026-10-01,sent,email,,,,,\n", encoding="utf-8")
    apps, skipped = load_tracker(p)
    assert [(a.company, a.status, a.replied) for a in apps] == [("Acme", "replied", date(2026, 10, 3)),
                                                                ("Draftly", "draft", None)]
    assert [s.split(":")[0] for s in skipped] == ["line 4", "line 5", "line 6"]
    assert "ghosted" in skipped[1]


def test_tracker_header_is_forgiving_but_needs_company_and_sent(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Company; Sent ;Status\nAcme;2026-10-01;sent\n", encoding="utf-8")  # Excel in many locales
    apps, skipped = load_tracker(p)
    assert [(a.company, a.sent) for a in apps] == [("Acme", date(2026, 10, 1))] and skipped == []
    p.write_text("name,date\nAcme,2026-10-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="company, sent"):
        load_tracker(p)
    p.write_text("", encoding="utf-8")
    assert load_tracker(p) == ([], [])


def test_report_has_queue_rates_and_drafts():
    apps = [app("Silent Co", "2026-10-01", channel="hh"), app("Answered", "2026-10-02", "replied"),
            app("Draft", None, "draft")]
    md = render_funnel(apps, TODAY, skipped=["line 9: bad date"])
    for needle in ["Sent 2 · replied 1 (50%)", "## Follow up", "Silent Co", "## Replied, next step open", "Answered",
                   "## By week", "## Reply rate by channel", "| hh | 1 | 0 |", "Not confirmed sent: 1", "line 9: bad date"]:
        assert needle in md, needle


def test_report_names_dates_after_today_and_keeps_tables_intact():
    md = render_funnel([app("Typo Inc", "2027-10-01"), app("A|B", "2026-10-01", note="x|y")], TODAY)
    assert "Typo Inc: date after today (2027-10-01)" in md
    assert "A/B" in md and "A|B" not in md


def test_empty_tracker_says_so():
    assert "No applications yet" in render_funnel([], TODAY)


def test_cli_funnel(tmp_path, capsys):
    p = tmp_path / "t.csv"
    p.write_text("company,sent,status,channel\nAcme,2026-10-01,sent,hh\n", encoding="utf-8")
    out = tmp_path / "f.md"
    assert main(["funnel", str(p), "--today", "2026-10-15", "--wait-channel", "hh=7", "--out", str(out)]) == 0
    assert "Acme" in out.read_text(encoding="utf-8")
    assert main(["funnel", str(p), "--wait-channel", "hh"]) == 2
    assert "channel=days" in capsys.readouterr().err
    p.write_bytes("company,sent\nКомпания,2026-10-01\n".encode("cp1251"))
    assert main(["funnel", str(p)]) == 2
    assert "UTF-8" in capsys.readouterr().err
