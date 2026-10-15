"""Command line: vm add | index-profile | rank | calibrate | label | eval | funnel."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from .embed import DEFAULT_MODEL, EmbedderUnavailable, ensure_embeddings, get_embedder
from .evaluate import evaluate, render_eval
from .funnel import DEFAULT_WAIT, load_tracker, render_funnel
from .labels import label_from_dict, load_label_file
from .models import Vacancy, make_id, run_date
from .profile import load_profile
from .rank import MIN_TEXT, Reference, profile_scale, rank
from .report import render_top
from .rules import RuleSet
from .store import Store


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _paths(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        p = Path(item)
        out += sorted(p.glob("*.md")) if p.is_dir() else [p]
    return out


def _vacancy_ref(ref: str) -> str:
    return make_id(ref) if ("/" in ref or "." in ref) else ref


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


MIN_FREEZE = 50  # a scale measured on fewer vacancies is used for this call but never stored


def frozen_scale(store: Store, model: str, vecs: dict, chunk_vecs: dict,
                 refresh: bool = False) -> tuple[float, float, int]:
    """The fit scale stored for this model, measured once and kept: otherwise the same vacancy would get a different
    fit every time the base grows. Frozen on first use once the base has MIN_FREEZE vacancies; `vm calibrate`
    (refresh) measures and stores it on purpose. Returns (centre, width, vacancies it was measured on)."""
    snap = None if refresh else store.get_calibration(model)
    if snap is None:
        snap = profile_scale(vecs, chunk_vecs)
        if refresh or snap[2] >= MIN_FREEZE:
            store.put_calibration(model, *snap)
    return snap


def cmd_calibrate(args) -> int:
    store = Store(args.db)
    chunks = store.profile_chunks()
    if not chunks:
        _err("profile is empty: run `vm index-profile` first")
        return 1
    try:
        embedder = get_embedder(args.model)
    except EmbedderUnavailable as e:
        _err(str(e))
        return 2
    chunk_vecs = ensure_embeddings(store, embedder, "profile_chunk", {c.id: c.text for c in chunks}, "passage")
    vecs = ensure_embeddings(store, embedder, "vacancy",
                             {v.id: v.embed_text() for v in store.vacancies() if len(v.text) >= MIN_TEXT}, "query")
    centre, width, n = frozen_scale(store, embedder.name, vecs, chunk_vecs, refresh=True)
    print(f"fit scale for {embedder.name}: centre {centre:.4f}, width {width:.4f}, measured on {n} vacancies")
    if n < MIN_FREEZE:
        _err(f"warning: {n} vacancies is a thin base for a scale; run `vm calibrate` again once you have {MIN_FREEZE}+")
    return 0


def cmd_add(args) -> int:
    store = Store(args.db)
    vacancies = []
    for n, line in enumerate(Path(args.file).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            v = Vacancy.from_dict(json.loads(line), run_id=args.run or "")
        except ValueError as e:
            _err(f"skip line {n}: {e}")
            continue
        if not run_date(v.run_id):
            _err(f"skip line {n}: run id {v.run_id!r} has no date — runs are ordered by date, use e.g. 2026-10-14 "
                 f"or links-2026-10-14 (pass --run or set run_id in the record)")
            continue
        vacancies.append(v)
    res = store.add_vacancies(vacancies)
    print(f"added {res.new} new, {res.known} already known")
    try:
        stored = [store.get_vacancy(v.id) for v in vacancies]  # the store may keep a longer text than this file
        ensure_embeddings(store, get_embedder(args.model), "vacancy",
                          {v.id: v.embed_text() for v in stored if len(v.text) >= MIN_TEXT}, "query")
    except EmbedderUnavailable as e:
        _err(f"warning: embeddings skipped ({e}); `vm rank` computes them later")
    return 0


def cmd_index_profile(args) -> int:
    store = Store(args.db)
    chunks = load_profile(_paths(args.paths), exclude={t for t in args.exclude.split(",") if t})
    if not chunks:
        _err("no resume chunks found")
        return 1
    store.replace_profile(chunks)
    try:
        ensure_embeddings(store, get_embedder(args.model), "profile_chunk", {c.id: c.text for c in chunks}, "passage")
    except EmbedderUnavailable as e:
        _err(f"warning: embeddings skipped ({e}); `vm rank` computes them later")
    tracks = sorted({c.track for c in chunks})
    print(f"indexed {len(chunks)} chunks across {len(tracks)} tracks: {', '.join(tracks)}")
    return 0


def cmd_rank(args) -> int:
    store = Store(args.db)
    run_vacs = store.vacancies(args.run)
    if not run_vacs:
        print(f"no vacancies for run {args.run}")
        return 1
    chunks = store.profile_chunks()
    if not chunks:
        _err("profile is empty: run `vm index-profile` first")
        return 1
    try:
        embedder = get_embedder(args.model)
    except EmbedderUnavailable as e:
        _err(str(e))
        return 2
    chunk_vecs = ensure_embeddings(store, embedder, "profile_chunk", {c.id: c.text for c in chunks}, "passage")
    every = store.vacancies()
    vecs = ensure_embeddings(store, embedder, "vacancy",
                             {v.id: v.embed_text() for v in every if len(v.text) >= MIN_TEXT}, "query")
    snap = frozen_scale(store, embedder.name, vecs, chunk_vecs) if args.hybrid <= 0 else None
    labels = store.effective_labels()
    refs = [Reference(v, vecs[v.id], labels.get(v.id)) for v in every if v.id in vecs]
    n_labeled = sum(1 for r in refs if r.label)
    n_no_fit = sum(1 for r in refs if r.label and r.label.label == "no_fit")
    # a vote from one class only says "everything is a fit": it needs both sides to mean anything
    weight = args.neighbor_weight if n_labeled >= args.min_labels and n_no_fit >= args.min_no_fits else 0.0
    rules = RuleSet.load(args.rules) if args.rules else None
    today = date.fromisoformat(args.today) if args.today else date.today()
    ranked = rank(run_vacs, vecs, chunks, chunk_vecs, references=refs, neighbor_weight=weight, k_neighbors=args.k,
                  rules=rules, today=today, hybrid_alpha=args.hybrid, scale=snap[:2] if snap else None)
    scale_note = f", fit scale from {snap[2]} vacancies" if snap else ", fit scale from this batch (hybrid)"
    md = render_top(ranked, args.top, title=f"Run {args.run}: {len(ranked)} vacancies, "
                                            f"neighbour weight {weight} ({n_labeled} labeled, {n_no_fit} no_fit)"
                                            + scale_note)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(md)
    return 0


def cmd_label(args) -> int:
    store = Store(args.db)
    try:
        if args.file:
            labels = load_label_file(args.file)
        elif args.ref and args.label:
            labels = [label_from_dict({"vacancy_id": _vacancy_ref(args.ref), "label": args.label, "actor": args.actor,
                                       "track": args.track, "reasons": args.reasons, "note": args.note})]
        else:
            _err("give REF and LABEL, or --file")
            return 1
    except ValueError as e:
        _err(f"nothing stored: {e}")
        return 1
    unknown = sum(1 for l in labels if store.get_vacancy(l.vacancy_id) is None)
    store.add_labels(labels)
    print(f"stored {len(labels)} labels" + (f", {unknown} for vacancies not in the store yet" if unknown else ""))
    return 0


def cmd_eval(args) -> int:
    store = Store(args.db)
    actors = None if args.labels == "all" else {"human"}
    labels = store.effective_labels()
    if not labels:
        print("no labeled vacancies yet")
        return 1
    if actors and not any(l.actor in actors for l in labels.values()):
        print("no human labels yet: eval counts your own decisions by default; use --labels all to include the agent's")
        return 1
    rules = RuleSet.load(args.rules) if args.rules else None
    stages = [None, rules] if rules else [None]  # with rules, every config is measured without and with stage 2
    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            embedder = get_embedder(model)
        except EmbedderUnavailable as e:
            _err(f"skip {model}: {e}")
            continue
        for nw in _floats(args.neighbor_weights):
            for ha in _floats(args.hybrid_alphas):
                for rs in stages:
                    results.append(evaluate(store, embedder, neighbor_weight=nw, hybrid_alpha=ha, rules=rs,
                                            actors=actors))
    if not results:
        return 2
    md = render_eval(results)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(md)
    return 0


def cmd_funnel(args) -> int:
    waits = {}
    for item in args.wait_channel:
        channel, _, days = item.partition("=")
        if not channel or not days.isdigit():
            _err(f"--wait-channel takes channel=days, got {item!r}")
            return 2
        waits[channel] = int(days)
    try:
        apps, skipped = load_tracker(args.tracker)
    except UnicodeDecodeError:
        _err(f"{args.tracker}: not UTF-8 text; save the tracker as CSV UTF-8")
        return 2
    except ValueError as e:
        _err(str(e))
        return 2
    md = render_funnel(apps, date.fromisoformat(args.today) if args.today else date.today(), args.wait, waits, skipped)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vm", description="Rank vacancies against a profile and measure the ranking.")
    p.add_argument("--db", default=os.environ.get("VM_DB", "vm.db"))
    p.add_argument("--model", default=os.environ.get("VM_MODEL", DEFAULT_MODEL))
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="store vacancies from a JSONL file")
    a.add_argument("file")
    a.add_argument("--run", default="")
    a.set_defaults(func=cmd_add)

    ip = sub.add_parser("index-profile", help="chunk and embed resume Markdown files or folders")
    ip.add_argument("paths", nargs="+")
    ip.add_argument("--exclude", default="", help="comma-separated tracks to skip")
    ip.set_defaults(func=cmd_index_profile)

    r = sub.add_parser("rank", help="rank one run and print or write the Markdown top")
    r.add_argument("--run", required=True)
    r.add_argument("--top", type=int, default=50)
    r.add_argument("--out", default="")
    r.add_argument("--rules", default=os.environ.get("VM_RULES", ""))
    r.add_argument("--neighbor-weight", type=float, default=0.15)
    r.add_argument("--min-labels", type=int, default=100)
    r.add_argument("--min-no-fits", type=int, default=20)
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--hybrid", type=float, default=0.0)
    r.add_argument("--today", default="")
    r.set_defaults(func=cmd_rank)

    cal = sub.add_parser("calibrate", help="measure the fit scale again on the whole base and freeze it")
    cal.set_defaults(func=cmd_calibrate)

    lb = sub.add_parser("label", help="record fit/no_fit decisions")
    lb.add_argument("ref", nargs="?", help="vacancy id or url")
    lb.add_argument("label", nargs="?", choices=["fit", "no_fit"])
    lb.add_argument("--file", default="")
    lb.add_argument("--actor", default="agent", choices=["agent", "human"])
    lb.add_argument("--track", default="")
    lb.add_argument("--reasons", default="")
    lb.add_argument("--note", default="")
    lb.set_defaults(func=cmd_label)

    e = sub.add_parser("eval", help="measure ranking quality on labeled vacancies")
    e.add_argument("--models", default=os.environ.get("VM_MODEL", DEFAULT_MODEL))
    e.add_argument("--neighbor-weights", default="0")
    e.add_argument("--hybrid-alphas", default="0")
    e.add_argument("--labels", default="human", choices=["human", "all"],
                   help="whose labels count: your own decisions (default) or the agent's too")
    e.add_argument("--rules", default=os.environ.get("VM_RULES", ""))
    e.add_argument("--out", default="")
    e.set_defaults(func=cmd_eval)

    f = sub.add_parser("funnel", help="answers per week, channel and resume; whom to remind")
    f.add_argument("tracker", help="CSV: company, sent, status, channel, track, source, replied, updated, note")
    f.add_argument("--wait", type=int, default=DEFAULT_WAIT, help="working days of silence before a reminder")
    f.add_argument("--wait-channel", action="append", default=[], help="channel=days, e.g. hh=7; repeatable")
    f.add_argument("--today", default="")
    f.add_argument("--out", default="")
    f.set_defaults(func=cmd_funnel)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
