from vacancy_match.models import Vacancy
from vacancy_match.rank import ChunkMatch, Neighbor, Ranked
from vacancy_match.report import render_top
from vacancy_match.rules import Adjustment


def test_render_top_shows_everything_the_agent_needs():
    v = Vacancy.from_dict({"source": "hh", "url": "https://hh.kz/vacancy/1", "title": "AI Engineer",
                           "company": "Acme", "text": "x", "salary_text": "от 150 000 ₽",
                           "published_at": "2026-10-12T10:00:00+03:00"})
    r = Ranked(v, 0.42, 0.84, "backend",
               top_chunks=[ChunkMatch("c1", "backend", "Projects: Order bot", 0.84)],
               neighbors=[Neighbor("n1", "Broker advisor", "no_fit", ("scam",), 0.71)],
               flags=["agency"], duplicate_of="d1", fit=0.84, adjustments=[Adjustment("language", 0.5)])
    md = render_top([r], title="Run 2026-10-14")
    assert md.startswith("# Run 2026-10-14")
    for needle in ["## 1. AI Engineer — Acme", "score 42 · fit 84", "track backend", "от 150 000 ₽", "2026-10-12",
                   "https://hh.kz/vacancy/1", "Projects: Order bot (0.84)", "Flags: agency",
                   "Adjusted: language ×0.5",
                   "duplicate of d1", "no_fit 0.71 Broker advisor [scam]", f"id `{v.id}`"]:
        assert needle in md


def test_render_top_respects_limit():
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "T"})
    md = render_top([Ranked(v, 0.1, 0.1, "A")] * 3, limit=2)
    assert md.count("## ") == 2


def test_cut_vacancy_shows_the_reason_not_a_score():
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "Payment agent"})
    md = render_top([Ranked(v, -2.2, 0.8, "A", cut=[Adjustment("upfront_fee", 0.0)], fit=0.8)])
    assert "CUT: upfront_fee" in md and "fit 80" in md and "score" not in md
