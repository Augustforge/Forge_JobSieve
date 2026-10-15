"""Stage 1 scores vacancies against profile chunks and labeled neighbours; stage 2 (optional rules) applies the
candidate's own terms as multipliers on the calibrated fit."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .bm25 import bm25_matrix
from .embed import tokenize
from .models import Label, ProfileChunk, Vacancy, run_date
from .rules import Adjustment, Check, RuleSet

MIN_TEXT = 200
NO_TEXT_SCORE = -1.0
CUT_OFFSET = 3.0  # cut vacancies sink below every other one and keep their fit order among themselves


@dataclass
class ChunkMatch:
    chunk_id: str
    track: str
    section: str
    score: float


@dataclass
class Neighbor:
    vacancy_id: str
    title: str
    label: str
    reasons: tuple[str, ...]
    similarity: float


@dataclass
class Reference:
    """An already-known vacancy: a candidate neighbour (when labeled) and a duplicate candidate."""
    vacancy: Vacancy
    vector: np.ndarray
    label: Label | None = None


@dataclass
class Ranked:
    vacancy: Vacancy
    score: float
    profile_score: float
    track: str
    top_chunks: list[ChunkMatch] = field(default_factory=list)
    neighbors: list[Neighbor] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    cut: list[Adjustment] = field(default_factory=list)
    duplicate_of: str = ""
    fit: float = 0.0
    adjustments: list[Adjustment] = field(default_factory=list)


def fit_scale(scores: np.ndarray) -> tuple[float, float]:
    """Centre and width of the logistic map from raw similarity to 0..1: the median known vacancy gets 0.5, the 10th
    and 90th percentiles get 0.1 and 0.9. Raw e5 cosine sits in a narrow band (about 0.84-0.89) where 'a little
    lower' and 'much lower' look the same, so stage-2 multipliers need this scale."""
    p10, p50, p90 = np.percentile(scores, [10, 50, 90])
    return float(p50), float(p90 - p10) / (2 * math.log(9))


def profile_scale(vectors: dict[str, np.ndarray], chunk_vectors: dict[str, np.ndarray]) -> tuple[float, float, int]:
    """The fit scale measured on every known vacancy: (centre, width, how many vacancies it was measured on)."""
    if not vectors or not chunk_vectors:
        return 0.0, 0.0, 0
    scores = (np.stack(list(vectors.values())) @ np.stack(list(chunk_vectors.values())).T).max(axis=1)
    return (*fit_scale(scores), len(scores))


def to_fit(x: float, centre: float, width: float) -> float:
    if width <= 1e-9:  # one known vacancy, or all alike: nothing to compare with
        return 0.5
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, (x - centre) / width))))


def order_key(v: Vacancy) -> tuple[str, str, str]:
    """Which of two vacancies came first: run date, then collection time, then id as a stable tie-break
    (two copies collected in the same second still get exactly one 'first')."""
    return (run_date(v.run_id), v.collected_at, v.id)


def rank(vacancies: list[Vacancy], vectors: dict[str, np.ndarray], chunks: list[ProfileChunk],
         chunk_vectors: dict[str, np.ndarray], *, references: list[Reference] = (), neighbor_weight: float = 0.0,
         k_neighbors: int = 5, rules: RuleSet | None = None, today: date | None = None,
         hybrid_alpha: float = 0.0, dup_threshold: float = 0.95,
         scale: tuple[float, float] | None = None) -> list[Ranked]:
    """scale: a frozen (centre, width) for the fit; without it the scale is measured on this batch plus references."""
    if not chunks:
        raise ValueError("profile is empty: run `vm index-profile` first")
    today = today or date.today()
    scored = [v for v in vacancies if len(v.text) >= MIN_TEXT]
    missing = [v.id for v in scored if v.id not in vectors]
    if missing:
        raise ValueError(f"missing embeddings for {len(missing)} vacancies, e.g. {missing[0]}")

    out: list[Ranked] = []
    if scored:
        C = np.stack([chunk_vectors[c.id] for c in chunks])
        V = np.stack([vectors[v.id] for v in scored])
        S = V @ C.T
        if hybrid_alpha > 0:
            B = bm25_matrix([tokenize(v.embed_text()) for v in scored], [tokenize(c.text) for c in chunks])
            if B.max() > 0:
                S = S + hybrid_alpha * (B / B.max())
        labeled = [r for r in references if r.label is not None]
        L = np.stack([r.vector for r in labeled]) if labeled else None
        R = np.stack([r.vector for r in references]) if references else None
        # calibrate on the scale the fit is computed on: with the hybrid bonus only this batch has that scale
        known = {v.id: float(s) for v, s in zip(scored, S.max(axis=1))}
        if R is not None and hybrid_alpha <= 0:
            ref_raw = (R @ C.T).max(axis=1)
            for j, ref in enumerate(references):
                known.setdefault(ref.vacancy.id, float(ref_raw[j]))
        centre, width = scale if scale else fit_scale(np.array(list(known.values())))

        for i, v in enumerate(scored):
            row = S[i]
            best_by_track: dict[str, float] = {}
            for j, c in enumerate(chunks):
                best_by_track[c.track] = max(best_by_track.get(c.track, -np.inf), float(row[j]))
            track = max(best_by_track, key=best_by_track.get)
            profile_score = best_by_track[track]
            top = [ChunkMatch(chunks[j].id, chunks[j].track, chunks[j].section, float(row[j]))
                   for j in np.argsort(-row)[:2]]

            neighbors: list[Neighbor] = []
            vote = 0.0
            if L is not None:
                sims = L @ vectors[v.id]
                for j in np.argsort(-sims):
                    ref = labeled[j]
                    if ref.vacancy.id == v.id:
                        continue
                    neighbors.append(Neighbor(ref.vacancy.id, ref.vacancy.title, ref.label.label,
                                              ref.label.reasons, float(sims[j])))
                    if len(neighbors) == k_neighbors:
                        break
                weights = np.array([max(n.similarity, 0.0) for n in neighbors])
                if neighbors and weights.sum() > 0:
                    signs = np.array([1.0 if n.label == "fit" else -1.0 for n in neighbors])
                    vote = float((weights * signs).sum() / weights.sum())

            duplicate_of = ""
            if R is not None:
                sims = R @ vectors[v.id]
                for j in np.argsort(-sims):
                    if sims[j] <= dup_threshold:
                        break
                    other = references[j].vacancy
                    if (other.id != v.id and other.company and other.company.lower() == v.company.lower()
                            and order_key(other) < order_key(v)):
                        duplicate_of = other.id
                        break

            check = rules.check(v, today) if rules else Check()
            # the vote moves the calibrated fit, not raw cosine: 0.15 raw is ~12 calibration widths and saturates it
            # not clipped at 1: a cap would tie every top vacancy and erase its stage-1 order
            fit = max(0.0, to_fit(profile_score, centre, width) + neighbor_weight * vote)
            score = fit * check.factor - (CUT_OFFSET if check.cut else 0.0)
            out.append(Ranked(v, score, profile_score, track, top, neighbors, check.flags, check.cut, duplicate_of,
                              fit, check.adjustments))

    for v in vacancies:
        if len(v.text) < MIN_TEXT:
            check = rules.check(v, today) if rules else Check()
            out.append(Ranked(v, NO_TEXT_SCORE - (CUT_OFFSET if check.cut else 0.0), 0.0, "",
                              flags=["no_text", *check.flags], cut=check.cut, adjustments=check.adjustments))

    out.sort(key=lambda r: -r.score)
    return out
