"""Measure ranking quality on labeled vacancies, run by run, without leaking future labels.

Runs usually hold few labeled vacancies, so raw precision@k mostly measures the share of fits (any ordering of
6 vacancies with 3 fits gives p@10 = 0.5). The headline metrics are therefore order-sensitive:
normalized precision (hits@k / min(k, fits)), AUC (share of fit/no-fit pairs in the right order) and average
precision; each run's base rate is shown as the no-skill baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .embed import ensure_embeddings
from .models import Label, Vacancy, run_date
from .rank import MIN_TEXT, Ranked, Reference, profile_scale, rank
from .rules import RuleSet


def precision_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    k = min(k, len(ranked_ids))
    return sum(1 for i in ranked_ids[:k] if i in positives) / k if k else 0.0


def normalized_precision_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float | None:
    """hits@k / min(k, |positives|): 1.0 means the top holds as many fits as it possibly could.
    None when the run fits inside k or has no negatives: then every ordering scores the same and only AUC/AP see
    the order."""
    if not positives or len(ranked_ids) <= k or all(i in positives for i in ranked_ids):
        return None
    return sum(1 for i in ranked_ids[:k] if i in positives) / min(k, len(positives))


def recall_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float | None:
    # a run inside k is always fully recalled; a run of only fits recalls k/n whatever the order
    if not positives or len(ranked_ids) <= k or all(i in positives for i in ranked_ids):
        return None
    return sum(1 for i in ranked_ids[:k] if i in positives) / len(positives)


def auc(ranked_ids: list[str], positives: set[str], scores: list[float] | None = None) -> float | None:
    """Share of (fit, no_fit) pairs where the fit is ranked higher; None without both classes.
    With scores, a tied pair counts as half, so input order cannot decide it."""
    if scores is None:
        scores = [-float(n) for n in range(len(ranked_ids))]
    pos = [s for i, s in zip(ranked_ids, scores) if i in positives]
    neg = [s for i, s in zip(ranked_ids, scores) if i not in positives]
    if not pos or not neg:
        return None
    correct = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return correct / (len(pos) * len(neg))


def average_precision(ranked_ids: list[str], positives: set[str]) -> float | None:
    if not positives:
        return None
    hits, total = 0, 0.0
    for rank_no, i in enumerate(ranked_ids, 1):
        if i in positives:
            hits += 1
            total += hits / rank_no
    return total / len(positives)


def weighted_mean(pairs) -> float | None:
    """Mean of values weighted by run size, skipping runs where the metric is undefined."""
    pairs = [(v, w) for v, w in pairs if v is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def references_before(vacancies: list[Vacancy], vectors: dict[str, np.ndarray], labels: dict[str, Label],
                      run_id: str) -> list[Reference]:
    """Vacancies from runs strictly earlier by date. Undated runs neither see nor give neighbours."""
    day = run_date(run_id)
    if not day:
        return []
    return [Reference(v, vectors[v.id], labels.get(v.id)) for v in vacancies
            if v.id in vectors and run_date(v.run_id) and run_date(v.run_id) < day]


@dataclass
class RunMetrics:
    run_id: str
    n: int
    positives: int
    base_rate: float
    np10: float | None
    np20: float | None
    r20: float | None
    auc: float | None
    ap: float | None


@dataclass
class RuleStat:
    lo: float
    hi: float
    on_fit: int = 0
    on_no_fit: int = 0

    def effect(self) -> str:
        if self.lo == self.hi:
            return {1.0: "flag", 0.0: "cut"}.get(self.lo, f"×{round(self.lo, 3):g}")
        return f"×{round(self.lo, 2):g}–{round(self.hi, 2):g}"


def _fired(r: Ranked) -> list[tuple[str, float]]:
    return ([(f, 1.0) for f in r.flags if f != "no_text"] + [(a.rule, a.factor) for a in r.adjustments]
            + [(a.rule, 0.0) for a in r.cut])


@dataclass
class EvalResult:
    config: str
    runs: list[RunMetrics]
    np10: float | None
    np20: float | None
    r20: float | None
    auc: float | None
    ap: float | None
    track_acc: float | None
    false_pos: list[Ranked] = field(default_factory=list)
    false_neg: list[Ranked] = field(default_factory=list)
    rule_stats: dict[str, RuleStat] = field(default_factory=dict)
    pooled_auc: float | None = None
    auc_lo: float | None = None
    auc_hi: float | None = None


def pooled_auc_ci(scores: list[float], is_fit: list[bool], n_boot: int = 1000,
                  seed: int = 0) -> tuple[float | None, float | None, float | None]:
    """AUC over all runs at once (the calibrated score is comparable across runs) with a bootstrap 95% interval.
    Per-run AUC drops cross-run pairs and hides how wide the uncertainty is with a handful of rejections."""
    s, y = np.asarray(scores, dtype=float), np.asarray(is_fit, dtype=bool)

    def one(idx: np.ndarray) -> float | None:
        pos, neg = s[idx][y[idx]], s[idx][~y[idx]]
        if not len(pos) or not len(neg):
            return None
        return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())

    point = one(np.arange(len(s)))
    if point is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = np.flatnonzero(y), np.flatnonzero(~y)
    # stratified: fits and no-fits are resampled separately, so no resample loses its few rejections
    boots = [one(np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))]))
             for _ in range(n_boot)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate(store, embedder, *, neighbor_weight: float = 0.0, hybrid_alpha: float = 0.0, k_neighbors: int = 5,
             rules: RuleSet | None = None, actors: set[str] | None = None) -> EvalResult:
    """actors: whose labels count ({"human"} = only the candidate's own decisions); None = all labels."""
    labels = {k: l for k, l in store.effective_labels().items() if actors is None or l.actor in actors}
    chunks = store.profile_chunks()
    chunk_vecs = ensure_embeddings(store, embedder, "profile_chunk", {c.id: c.text for c in chunks}, "passage")
    vacs = [v for v in store.vacancies() if v.id in labels]
    vecs = ensure_embeddings(store, embedder, "vacancy",
                             {v.id: v.embed_text() for v in vacs if len(v.text) >= MIN_TEXT}, "query")
    # one label-free scale for every run (the frozen one, else measured on all vectors), so pooled AUC compares like
    # with like; per-run scales would map each run's median to 0.5
    snap = store.get_calibration(embedder.name)
    scale = None if hybrid_alpha > 0 else (snap[:2] if snap else profile_scale(vecs, chunk_vecs)[:2])
    runs, every, track_pairs = [], [], []
    rule_stats: dict[str, RuleStat] = {}
    for run in sorted({v.run_id for v in vacs}, key=lambda r: (run_date(r), r)):
        in_run = [v for v in vacs if v.run_id == run]
        day = run_date(run)  # replay each run as it was: "stale" means stale on the day it was collected
        ranked = rank(in_run, vecs, chunks, chunk_vecs, references=references_before(vacs, vecs, labels, run),
                      neighbor_weight=neighbor_weight, k_neighbors=k_neighbors, rules=rules, hybrid_alpha=hybrid_alpha,
                      today=date.fromisoformat(day) if day else None, scale=scale)
        ids = [r.vacancy.id for r in ranked]
        positives = {v.id for v in in_run if labels[v.id].label == "fit"}
        runs.append(RunMetrics(run, len(in_run), len(positives), len(positives) / len(in_run),
                               normalized_precision_at_k(ids, positives, 10),
                               normalized_precision_at_k(ids, positives, 20), recall_at_k(ids, positives, 20),
                               auc(ids, positives, [r.score for r in ranked]), average_precision(ids, positives)))
        for r in ranked:
            lab = labels[r.vacancy.id]
            if lab.label == "fit" and lab.track and r.track:
                track_pairs.append((r.track, lab.track))
            for name, factor in _fired(r):
                st = rule_stats.setdefault(name, RuleStat(factor, factor))
                st.lo, st.hi = min(st.lo, factor), max(st.hi, factor)
                if lab.label == "fit":
                    st.on_fit += 1
                else:
                    st.on_no_fit += 1
        every += ranked
    track_acc = sum(p == a for p, a in track_pairs) / len(track_pairs) if track_pairs else None
    false_pos = sorted((r for r in every if labels[r.vacancy.id].label == "no_fit"), key=lambda r: -r.score)[:10]
    false_neg = sorted((r for r in every if labels[r.vacancy.id].label == "fit"), key=lambda r: r.score)[:10]

    def mean(attr: str) -> float | None:
        return weighted_mean((getattr(m, attr), m.n) for m in runs)

    pooled = pooled_auc_ci([r.score for r in every], [labels[r.vacancy.id].label == "fit" for r in every])
    return EvalResult(f"{embedder.name} nw={neighbor_weight} hybrid={hybrid_alpha} stage={'1+2' if rules else '1'}",
                      runs, mean("np10"), mean("np20"), mean("r20"), mean("auc"), mean("ap"), track_acc, false_pos,
                      false_neg, rule_stats, *pooled)


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def render_eval(results: list[EvalResult]) -> str:
    lines = ["# Ranking evaluation", "",
             "Per run over labeled vacancies, averaged with run size as weight. np@k = fits in the top k divided by "
             "min(k, fits). AUC = share of fit/no-fit pairs in the right order (0.5 = random). AP = average "
             "precision. AUC is the main metric; np@k and recall@k are undefined (—) for runs of ≤ k vacancies or "
             "without rejections. Pooled AUC compares every fit with every no-fit across runs, with a bootstrap 95% "
             "interval: read the interval before the number. Neighbours come only from runs with an earlier date.", "",
             "| config | np@10 | np@20 | recall@20 | AUC | AP | track accuracy | pooled AUC [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.config} | {_f(r.np10)} | {_f(r.np20)} | {_f(r.r20)} | {_f(r.auc)} | {_f(r.ap)} | "
              f"{_f(r.track_acc)} | {_f(r.pooled_auc)} [{_f(r.auc_lo)}–{_f(r.auc_hi)}] |" for r in results]
    best = max(results, key=lambda r: (r.auc or 0, r.np20 or 0))
    lines += ["", f"## Per run — {best.config}", "",
              "| run | vacancies | fit | base rate | np@10 | np@20 | recall@20 | AUC | AP |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {m.run_id} | {m.n} | {m.positives} | {_f(m.base_rate)} | {_f(m.np10)} | {_f(m.np20)} | "
              f"{_f(m.r20)} | {_f(m.auc)} | {_f(m.ap)} |" for m in best.runs]
    for title, items in (("High score but no_fit", best.false_pos), ("Low score but fit", best.false_neg)):
        lines += ["", f"## {title}", ""]
        lines += [f"- {r.score:.3f} {r.vacancy.title} — {r.vacancy.company} ({r.track or 'no track'})" for r in items]
    ruled = next((r for r in results if r.rule_stats), None)
    if ruled:
        fits = sum(m.positives for m in ruled.runs)
        no_fits = sum(m.n - m.positives for m in ruled.runs)

        def share(k: int, total: int) -> str:
            return f"{100 * k / total:.0f}%" if total else "—"

        lines += ["", "## Rules on labeled vacancies", "",
                  f"How often each stage-2 rule fired on vacancies you took ({fits}) and on ones you rejected "
                  f"({no_fits}). Compare the shares, not the counts: a penalty or cut should hit a larger share of "
                  "no-fits than of fits — if not, it is too strict; a boost should do the opposite.", "",
                  "| rule | effect | on fit | on no_fit | share of fits | share of no-fits |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {s.effect()} | {s.on_fit} | {s.on_no_fit} | {share(s.on_fit, fits)} | "
                  f"{share(s.on_no_fit, no_fits)} |" for n, s in sorted(ruled.rule_stats.items())]
    return "\n".join(lines) + "\n"
