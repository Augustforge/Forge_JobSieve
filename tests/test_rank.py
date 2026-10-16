from datetime import date

import pytest

from vacancy_match.embed import HashEmbedder
from vacancy_match.models import Label, ProfileChunk, Vacancy
from vacancy_match.rank import NO_TEXT_SCORE, Reference, rank
from vacancy_match.rules import RuleSet

E = HashEmbedder(512)
CHUNKS = [
    ProfileChunk("A-en-00", "A", "en", "Projects: bots", "python telegram bots automation api scripts"),
    ProfileChunk("B-en-00", "B", "en", "Projects: design", "figma design landing pages ui brand"),
]
CHUNK_VECS = dict(zip([c.id for c in CHUNKS], E.encode([c.text for c in CHUNKS], "passage")))


def long(body):
    text = body
    while len(text) < 200:
        text += " " + body
    return text


def vac(n, body, company="Co", run="r1", title=None, collected="2026-10-14T00:00:00"):
    return Vacancy.from_dict({"source": "s", "url": f"x.com/{n}", "title": title or f"Job {n}", "company": company,
                              "text": long(body) if body else "", "collected_at": collected}, run_id=run)


def vecs(vs):
    with_text = [v for v in vs if v.text]
    return dict(zip([v.id for v in with_text], E.encode([v.embed_text() for v in with_text], "query")))


def test_orders_by_profile_and_picks_track_with_explanation():
    bot, designer, sales = vac(1, "python telegram bots api"), vac(2, "figma landing pages design"), vac(3, "sales calls clients targets")
    out = rank([sales, bot, designer], vecs([bot, designer, sales]), CHUNKS, CHUNK_VECS)
    assert [r.vacancy.id for r in out][-1] == sales.id
    by_id = {r.vacancy.id: r for r in out}
    assert by_id[bot.id].track == "A" and by_id[designer.id].track == "B"
    assert by_id[bot.id].top_chunks[0].section == "Projects: bots"
    assert len(by_id[bot.id].top_chunks) == 1  # track A has one chunk; another track's chunk would mislead


def test_short_text_goes_to_bottom_with_flag():
    bot, empty = vac(1, "python telegram bots api"), vac(2, "")
    out = rank([empty, bot], vecs([bot]), CHUNKS, CHUNK_VECS)
    assert out[-1].vacancy.id == empty.id
    assert out[-1].score == NO_TEXT_SCORE and "no_text" in out[-1].flags


def test_condition_multiplies_and_red_line_cuts_below_everything():
    rules = RuleSet.from_dict({"conditions": {"language": ["fluent german"]},
                               "red_lines": {"upfront_fee": ["registration fee"]}})
    good, german = vac(1, "python telegram bots api"), vac(2, "python telegram bots api fluent german")
    bad, weak = vac(3, "python telegram bots api registration fee"), vac(4, "sales calls clients targets")
    vs = [good, german, bad, weak]
    out = rank(vs, vecs(vs), CHUNKS, CHUNK_VECS, rules=rules, today=date(2026, 10, 14))
    by = {r.vacancy.id: r for r in out}
    assert by[german.id].score == pytest.approx(by[german.id].fit * 0.5)
    assert [a.rule for a in by[german.id].adjustments] == ["language"]
    assert out[-1].vacancy.id == bad.id and [a.rule for a in out[-1].cut] == ["upfront_fee"]


def test_fit_is_calibrated_to_0_1_and_keeps_order():
    vs = [vac(1, "python telegram bots api"), vac(2, "figma landing pages design"), vac(3, "sales calls clients targets")]
    out = rank(vs, vecs(vs), CHUNKS, CHUNK_VECS)
    assert all(0 < r.fit < 1 for r in out)
    assert [r.fit for r in out] == sorted((r.fit for r in out), reverse=True)
    assert out[1].fit == pytest.approx(0.5)  # the median vacancy sits in the middle


def test_single_vacancy_is_calibrated_against_known_ones():
    known = [vac(n, b, run="r0") for n, b in [(2, "figma landing pages design"), (3, "sales calls clients targets"),
                                              (4, "warehouse loader shifts night")]]
    new = vac(1, "python telegram bots api")
    vv = vecs([new, *known])
    alone = rank([new], vv, CHUNKS, CHUNK_VECS)[0]
    with_refs = rank([new], vv, CHUNKS, CHUNK_VECS, references=[Reference(v, vv[v.id]) for v in known])[0]
    assert alone.fit == 0.5 and with_refs.fit > 0.5


def test_neighbors_vote_moves_score_and_excludes_self():
    target = vac(1, "python telegram bots api")
    twin_bad = vac(2, "python telegram bots api", company="Other", run="r0")
    other = vac(3, "sales calls clients targets", run="r0")  # gives calibration a spread to work with
    vv = vecs([target, twin_bad, other])
    refs = [Reference(twin_bad, vv[twin_bad.id], Label(twin_bad.id, "no_fit", reasons=("scam",))),
            Reference(target, vv[target.id], Label(target.id, "fit")), Reference(other, vv[other.id])]
    plain = rank([target], vv, CHUNKS, CHUNK_VECS, references=refs)[0]
    voted = rank([target], vv, CHUNKS, CHUNK_VECS, references=refs, neighbor_weight=0.5)[0]
    assert [n.vacancy_id for n in voted.neighbors] == [twin_bad.id]
    assert voted.neighbors[0].reasons == ("scam",)
    assert voted.score < plain.score and voted.profile_score == plain.profile_score


def test_duplicate_needs_same_company_and_earlier_run():
    first = vac(1, "python telegram bots api", company="Acme", run="2026-10-01")
    again = vac(2, "python telegram bots api", company="ACME", run="2026-10-02")
    other = vac(3, "python telegram bots api", company="Other", run="2026-10-02")
    vv = vecs([first, again, other])
    refs = [Reference(v, vv[v.id]) for v in (first, again, other)]
    out = {r.vacancy.id: r for r in rank([again, other], vv, CHUNKS, CHUNK_VECS, references=refs)}
    assert out[again.id].duplicate_of == first.id
    assert out[other.id].duplicate_of == ""


def test_hybrid_bm25_changes_scores_but_keeps_signal():
    bot, sales = vac(1, "python telegram bots api"), vac(2, "sales calls clients targets")
    out = rank([bot, sales], vecs([bot, sales]), CHUNKS, CHUNK_VECS, hybrid_alpha=0.5)
    assert out[0].vacancy.id == bot.id


def test_empty_profile_is_a_clear_error():
    with pytest.raises(ValueError, match="index-profile"):
        rank([vac(1, "python")], {}, [], {})


def test_duplicate_inside_one_run_flags_exactly_one_copy():
    a = vac(1, "python telegram bots api", company="Acme", run="2026-10-14")
    b = vac(2, "python telegram bots api", company="Acme", run="2026-10-14")
    vv = vecs([a, b])
    refs = [Reference(v, vv[v.id]) for v in (a, b)]
    dups = {r.vacancy.id: r.duplicate_of for r in rank([a, b], vv, CHUNKS, CHUNK_VECS, references=refs)}
    assert sorted(dups.values()) == ["", min(a.id, b.id)]


def test_duplicate_order_uses_run_date_not_prefix():
    early = vac(1, "python telegram bots api", company="Acme", run="results-2026-10-13")
    late = vac(2, "python telegram bots api", company="Acme", run="links-2026-10-20")
    vv = vecs([early, late])
    refs = [Reference(v, vv[v.id]) for v in (early, late)]
    out = {r.vacancy.id: r for r in rank([early, late], vv, CHUNKS, CHUNK_VECS, references=refs)}
    assert out[late.id].duplicate_of == early.id
    assert out[early.id].duplicate_of == ""


def test_vote_moves_fit_in_fit_space_without_saturating():
    target = vac(1, "python telegram bots api")
    others = [vac(n, b, run="r0") for n, b in [(2, "figma landing pages design"), (3, "sales calls clients targets"),
                                               (4, "warehouse loader shifts night"), (5, "python bots scripts")]]
    vv = vecs([target, *others])
    refs = [Reference(o, vv[o.id], Label(o.id, "no_fit")) for o in others]
    plain = rank([target], vv, CHUNKS, CHUNK_VECS, references=refs)[0]
    voted = rank([target], vv, CHUNKS, CHUNK_VECS, references=refs, neighbor_weight=0.15)[0]
    assert 0 < voted.fit < plain.fit < 1
    assert voted.fit == pytest.approx(plain.fit - 0.15)


def test_hybrid_is_calibrated_on_the_scale_it_is_applied_to():
    vs = [vac(1, "python telegram bots api"), vac(2, "figma landing pages design"), vac(3, "sales calls clients targets")]
    out = rank(vs, vecs(vs), CHUNKS, CHUNK_VECS, hybrid_alpha=0.5)
    assert out[1].fit == pytest.approx(0.5) and all(0 < r.fit < 1 for r in out)


def test_positive_vote_does_not_flatten_the_top():
    best = vac(1, "python telegram bots api")
    weak = [vac(n, b, run="r0") for n, b in [(2, "sales calls clients targets"), (3, "warehouse loader shifts night"),
                                             (4, "accountant tax payroll"), (5, "python bots scripts")]]
    vv = vecs([best, *weak])
    refs = [Reference(w, vv[w.id], Label(w.id, "fit")) for w in weak]
    plain = rank([best], vv, CHUNKS, CHUNK_VECS, references=refs)[0]
    voted = rank([best], vv, CHUNKS, CHUNK_VECS, references=refs, neighbor_weight=0.15)[0]
    assert plain.fit > 0.85 and voted.fit == pytest.approx(plain.fit + 0.15)


def test_given_scale_is_used_as_is():
    from vacancy_match.rank import to_fit
    bot = vac(1, "python telegram bots api")
    r = rank([bot], vecs([bot]), CHUNKS, CHUNK_VECS, scale={"": (0.1, 0.05)})[0]
    assert r.fit == pytest.approx(to_fit(r.profile_score, 0.1, 0.05)) and r.fit != 0.5
