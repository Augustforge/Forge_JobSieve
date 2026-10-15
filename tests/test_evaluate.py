import pytest

from vacancy_match.embed import HashEmbedder, ensure_embeddings
from vacancy_match.evaluate import (auc, average_precision, evaluate, normalized_precision_at_k, precision_at_k,
                                    recall_at_k, references_before, render_eval, weighted_mean)
from vacancy_match.models import Label, ProfileChunk, Vacancy
from vacancy_match.rules import RuleSet
from vacancy_match.store import Store


def test_precision_and_recall():
    ids = ["a", "b", "c", "d"]
    assert precision_at_k(ids, {"a", "c"}, 2) == 0.5
    assert precision_at_k([], {"a"}, 10) == 0.0
    assert recall_at_k(ids, {"a", "c", "z"}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ids, {"a"}, 4) is None  # run of 4 fits inside k=4
    assert recall_at_k(ids, set(), 2) is None


def test_normalized_precision_reaches_one_on_a_perfect_small_run():
    ids = ["f1", "f2", "f3", "n1", "n2", "n3"]
    pos = {"f1", "f2", "f3"}
    assert precision_at_k(ids, pos, 2) == 1.0
    assert normalized_precision_at_k(ids, pos, 2) == 1.0
    assert normalized_precision_at_k(["n1", "f1", "f2", "f3"], pos, 2) == 0.5
    assert normalized_precision_at_k(["n1", "f1"], {"f1"}, 1) == 0.0
    assert normalized_precision_at_k(ids, set(), 2) is None


def test_top_k_metrics_are_undefined_when_the_run_fits_inside_k():
    # 6 vacancies, every fit ranked last: a top-10 metric cannot see the order, AUC can
    worst = ["n1", "n2", "n3", "f1", "f2", "f3"]
    pos = {"f1", "f2", "f3"}
    assert normalized_precision_at_k(worst, pos, 10) is None
    assert recall_at_k(worst, pos, 20) is None
    assert auc(worst, pos) == 0.0


def test_auc_counts_score_ties_as_half():
    assert auc(["f1", "n1"], {"f1"}, scores=[-1.0, -1.0]) == 0.5
    assert auc(["f1", "n1"], {"f1"}, scores=[0.9, 0.1]) == 1.0


def test_auc_counts_correctly_ordered_pairs():
    assert auc(["f1", "n1", "f2", "n2"], {"f1", "f2"}) == pytest.approx(3 / 4)
    assert auc(["f1", "f2"], {"f1", "f2"}) is None  # no negatives
    assert auc(["n1"], set()) is None


def test_average_precision():
    assert average_precision(["f1", "n1", "f2"], {"f1", "f2"}) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(["n1"], set()) is None


def test_weighted_mean_skips_missing_values():
    assert weighted_mean([(1.0, 6), (None, 100), (0.5, 2)]) == pytest.approx((6 + 1) / 8)
    assert weighted_mean([(None, 3)]) is None


def long(body):
    return (body + " ") * (200 // len(body) + 2)


def build(tmp_path, rows):
    s = Store(tmp_path / "e.db")
    s.replace_profile([ProfileChunk("A-en-00", "A", "en", "Projects: bots", "python telegram bots automation api")])
    vs = []
    for run, n, body, lab in rows:
        v = Vacancy.from_dict({"source": "s", "url": f"x.com/{n}", "title": f"Job {n}", "company": f"C{n}",
                               "text": long(body)}, run_id=run)
        vs.append(v)
        s.add_vacancies([v])
        s.add_labels([Label(v.id, lab, track="A" if lab == "fit" else "")])
    return s, vs


ROWS = [("links-2026-10-12", 1, "python telegram bots", "fit"), ("links-2026-10-12", 2, "sales calls clients", "no_fit"),
        ("2026-10-13", 3, "telegram bots python api", "fit"), ("2026-10-13", 4, "warehouse loader shifts", "no_fit")]


def test_references_before_orders_by_run_date_not_prefix(tmp_path):
    rows = [("results-2026-10-13", 1, "a b c", "fit"), ("links-2026-10-20", 2, "d e f", "fit"),
            ("history", 3, "g h i", "fit")]
    s, vs = build(tmp_path, rows)
    vv = ensure_embeddings(s, HashEmbedder(64), "vacancy", {v.id: v.embed_text() for v in vs}, "query")
    labels = s.effective_labels()
    assert {r.vacancy.run_id for r in references_before(vs, vv, labels, "links-2026-10-20")} == {"results-2026-10-13"}
    assert references_before(vs, vv, labels, "results-2026-10-13") == []
    assert references_before(vs, vv, labels, "history") == []  # undated runs never see or give neighbours


def test_evaluate_end_to_end(tmp_path):
    s, _ = build(tmp_path, ROWS)
    res = evaluate(s, HashEmbedder(256))
    assert [r.run_id for r in res.runs] == ["links-2026-10-12", "2026-10-13"]
    assert res.np20 is None and res.r20 is None  # runs of 2 fit inside k: only AUC/AP see the order
    assert res.auc == 1.0 and res.ap == 1.0 and res.track_acc == 1.0
    assert res.runs[0].base_rate == 0.5
    assert {r.vacancy.title for r in res.false_pos} == {"Job 2", "Job 4"}
    md = render_eval([res, evaluate(s, HashEmbedder(256), neighbor_weight=0.15)])
    assert "| hash-256 nw=0.0 hybrid=0.0 stage=1 |" in md and "nw=0.15" in md
    assert "AUC" in md and "base rate" in md and "High score but no_fit" in md


def test_run_without_fits_does_not_drag_the_mean(tmp_path):
    rows = ROWS[:2] + [("2026-10-13", 5, "sales calls clients", "no_fit"), ("2026-10-13", 6, "warehouse", "no_fit")]
    s, _ = build(tmp_path, rows)
    res = evaluate(s, HashEmbedder(256))
    assert res.runs[1].auc is None and res.runs[1].ap is None
    assert res.auc == 1.0 and res.ap == 1.0


def test_stage_two_is_compared_and_rules_are_reported(tmp_path):
    rows = ROWS + [("2026-10-13", 5, "python telegram bots api fluent german", "no_fit")]
    s, _ = build(tmp_path, rows)
    rules = RuleSet.from_dict({"conditions": {"language": ["fluent german"]}})
    one, two = evaluate(s, HashEmbedder(256)), evaluate(s, HashEmbedder(256), rules=rules)
    assert one.config.endswith("stage=1") and two.config.endswith("stage=1+2")
    assert two.auc >= one.auc
    st = two.rule_stats["language"]
    assert (st.effect(), st.on_fit, st.on_no_fit) == ("×0.5", 0, 1)
    assert "| language | ×0.5 | 0 | 1 | 0% | 33% |" in render_eval([one, two])


def test_top_k_metrics_need_negatives():
    ids = [f"v{i}" for i in range(25)]
    assert normalized_precision_at_k(ids, set(ids), 20) is None and recall_at_k(ids, set(ids), 20) is None


def test_stale_is_judged_on_the_run_date_not_today(tmp_path):
    s = Store(tmp_path / "e.db")
    s.replace_profile([ProfileChunk("A-en-00", "A", "en", "Projects: bots", "python telegram bots automation api")])
    for n, (body, lab) in enumerate([("python telegram bots", "fit"), ("sales calls clients", "no_fit")], 1):
        v = Vacancy.from_dict({"source": "s", "url": f"x.com/{n}", "title": f"Job {n}", "company": f"C{n}",
                               "text": long(body), "published_at": "2026-07-25"}, run_id="results-2026-07-26")
        s.add_vacancies([v])
        s.add_labels([Label(v.id, lab)])
    res = evaluate(s, HashEmbedder(256), rules=RuleSet.from_dict({"stale": {"days": 14}}))
    assert "stale" not in res.rule_stats  # one day old on its run, months old today


def test_eval_can_use_human_labels_only(tmp_path):
    s, vs = build(tmp_path, ROWS)
    s.add_labels([Label(vs[0].id, "fit", actor="human", track="A"), Label(vs[1].id, "no_fit", actor="human")])
    res = evaluate(s, HashEmbedder(256), actors={"human"})
    assert sum(m.n for m in res.runs) == 2 and [m.run_id for m in res.runs] == ["links-2026-10-12"]


def test_pooled_auc_with_confidence_interval(tmp_path):
    s, _ = build(tmp_path, ROWS + [("2026-10-13", 5, "python telegram bots api fluent", "no_fit")])
    res = evaluate(s, HashEmbedder(256))
    assert res.pooled_auc is not None and res.auc_lo <= res.pooled_auc <= res.auc_hi
    assert "pooled AUC" in render_eval([res])


def test_every_run_is_scored_on_one_scale(tmp_path):
    rows = [("links-2026-10-12", 1, "python telegram bots", "fit"), ("links-2026-10-12", 2, "sales calls clients", "no_fit"),
            ("2026-10-13", 3, "python telegram bots", "fit"), ("2026-10-13", 4, "python bots api scripts", "no_fit"),
            ("2026-10-13", 5, "warehouse loader shifts", "no_fit")]
    s, _ = build(tmp_path, rows)
    res = evaluate(s, HashEmbedder(256))
    same_text = [r.score for r in res.false_neg if r.vacancy.title in ("Job 1", "Job 3")]
    assert len(same_text) == 2 and same_text[0] == pytest.approx(same_text[1])  # pooled AUC compares like with like
