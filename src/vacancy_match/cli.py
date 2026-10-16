"""Command line: vm fetch | add | index-profile | rank | calibrate | label | eval | funnel."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from .embed import DEFAULT_MODEL, EmbedderUnavailable, ensure_embeddings, get_embedder
from .evaluate import evaluate, render_eval
from .fetch import SOURCES, collect
from .funnel import DEFAULT_WAIT, load_tracker, render_funnel
from .labels import label_from_dict, load_label_file
from .models import Vacancy, make_id, run_date
from .profile import load_profile
from .rank import MIN_LANG_GROUP, MIN_TEXT, Reference, profile_scales, rank, vacancy_lang
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


def frozen_scales(store: Store, model: str, vecs: dict, chunk_vecs: dict, langs: dict[str, str],
                  refresh: bool = False) -> dict[str, tuple[float, float, int]]:
    """The fit scales stored for this model — all vacancies pooled ("") and one per language — measured once and
    kept: otherwise the same vacancy would get a different fit every time the base grows. Frozen on first use once the
    base has MIN_FREEZE vacancies; a language gets its own scale once it has MIN_LANG_GROUP, until then it uses the
    pooled one. `vm calibrate` (refresh) measures and stores them all on purpose. Values: (centre, width, n)."""
    stored = {} if refresh else store.get_calibrations(model)
    measured = profile_scales(vecs, chunk_vecs, langs, MIN_LANG_GROUP)
    if "" not in stored:
        if measured and (refresh or measured[""][2] >= MIN_FREEZE):
            for lang, snap in measured.items():
                store.put_calibration(model, *snap, lang=lang)
        return measured
    for lang, snap in measured.items():  # a language that grew big enough since: frozen now, the others stay put
        if lang and lang not in stored:
            store.put_calibration(model, *snap, lang=lang)
            stored[lang] = snap
    return stored


def _scale_note(scales: dict) -> str:
    per_lang = [f"{lang} {snap[2]}" for lang, snap in sorted(scales.items()) if lang]
    if per_lang:
        return f", fit scale per language: {', '.join(per_lang)}"
    return f", fit scale from {scales[''][2]} vacancies" if scales else ""


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
    every = [v for v in store.vacancies() if len(v.text) >= MIN_TEXT]
    vecs = ensure_embeddings(store, embedder, "vacancy", {v.id: v.embed_text() for v in every}, "query")
    scales = frozen_scales(store, embedder.name, vecs, chunk_vecs, {v.id: vacancy_lang(v) for v in every},
                           refresh=True)
    print(f"fit scale for {embedder.name}:")
    for lang, (centre, width, n) in sorted(scales.items()):
        print(f"  {lang or 'all'}: centre {centre:.4f}, width {width:.4f}, measured on {n} vacancies")
    if len(scales) == 1:
        print(f"  (a language gets its own scale once it has {MIN_LANG_GROUP} vacancies)")
    n = scales[""][2] if scales else 0
    if n < MIN_FREEZE:
        _err(f"warning: {n} vacancies is a thin base for a scale; run `vm calibrate` again once you have {MIN_FREEZE}+")
    return 0


def cmd_add(args) -> int:
    store = Store(args.db)
    vacancies, undated = [], 0
    # split on \n only: str.splitlines() also breaks on U+2028 and U+0085, which JSON keeps inside strings
    for n, line in enumerate(Path(args.file).read_text(encoding="utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if args.run:
                record["run_id"] = args.run  # --run wins over the run a record was fetched with
            v = Vacancy.from_dict(record)
        except (ValueError, AttributeError) as e:
            _err(f"skip line {n}: {e}")
            continue
        if not run_date(v.run_id):
            undated += 1
            continue
        vacancies.append(v)
    if undated:
        _err(f"skipped: {undated} records have no dated run id — runs are ordered by date: pass --run 2026-10-14 "
             f"(or links-2026-10-14), or set run_id in the records")
    if not vacancies:
        return 1
    res = store.add_vacancies(vacancies)
    print(f"added {res.new} new, {res.known} already known — run {', '.join(sorted({v.run_id for v in vacancies}))}")
    try:
        stored = [store.get_vacancy(v.id) for v in vacancies]  # the store may keep a longer text than this file
        ensure_embeddings(store, get_embedder(args.model), "vacancy",
                          {v.id: v.embed_text() for v in stored if len(v.text) >= MIN_TEXT}, "query")
    except EmbedderUnavailable as e:
        _err(f"warning: embeddings skipped ({e}); `vm rank` computes them later")
    return 0


def cmd_index_profile(args) -> int:
    store = Store(args.db)
    notes: list[str] = []
    chunks = load_profile(_paths(args.paths), exclude={t for t in args.exclude.split(",") if t}, notes=notes)
    for note in notes:
        _err(f"note: {note}")
    if not chunks:
        _err("no resume chunks found")
        return 1
    dropped = sorted({c.track for c in store.profile_chunks()} - {c.track for c in chunks})
    if dropped:
        _err(f"note: the profile is replaced as a whole; no longer indexed: {', '.join(dropped)}. "
             f"Index every resume together, e.g. `vm index-profile path/to/resumes/`")
    store.replace_profile(chunks)
    try:
        ensure_embeddings(store, get_embedder(args.model), "profile_chunk", {c.id: c.text for c in chunks}, "passage")
    except EmbedderUnavailable as e:
        _err(f"warning: embeddings skipped ({e}); `vm rank` computes them later")
    tracks = sorted({c.track for c in chunks})
    per_track = ", ".join(f"{t} ({sum(c.track == t for c in chunks)})" for t in tracks)
    print(f"indexed {len(chunks)} chunks across {len(tracks)} tracks: {per_track}")
    return 0


def cmd_rank(args) -> int:
    store = Store(args.db)
    run_vacs = store.vacancies(args.run)
    if not run_vacs:
        print(f"no vacancies for run {args.run}")
        return 1
    try:
        rules = RuleSet.load(args.rules) if args.rules else None  # a typo must not cost a model load first
    except ValueError as e:
        _err(str(e))
        return 2
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
    scales = (frozen_scales(store, embedder.name, vecs, chunk_vecs, {v.id: vacancy_lang(v) for v in every})
              if args.hybrid <= 0 else None)
    labels = store.effective_labels()
    refs = [Reference(v, vecs[v.id], labels.get(v.id)) for v in every if v.id in vecs]
    n_labeled = sum(1 for r in refs if r.label)
    n_no_fit = sum(1 for r in refs if r.label and r.label.label == "no_fit")
    # a vote from one class only says "everything is a fit": it needs both sides to mean anything
    weight = args.neighbor_weight if n_labeled >= args.min_labels and n_no_fit >= args.min_no_fits else 0.0
    today = date.fromisoformat(args.today) if args.today else date.today()
    ranked = rank(run_vacs, vecs, chunks, chunk_vecs, references=refs, neighbor_weight=weight, k_neighbors=args.k,
                  rules=rules, today=today, hybrid_alpha=args.hybrid, scale=scales or None)
    scale_note = _scale_note(scales) if scales else ", fit scale from this batch (hybrid)"
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
    runs: dict[str, set[str]] = {}
    for vid, lab in labels.items():
        v = store.get_vacancy(vid)
        if v and (actors is None or lab.actor in actors):
            runs.setdefault(v.run_id, set()).add(lab.label)
    if not any(kinds == {"fit", "no_fit"} for kinds in runs.values()):
        print("nothing to measure yet: a run needs at least one fit and one no_fit label — label the ones you "
              "skipped too (`vm label <url> no_fit --reasons ...`)")
        return 1
    try:
        rules = RuleSet.load(args.rules) if args.rules else None
    except ValueError as e:
        _err(str(e))
        return 2
    stages = [None, rules] if rules else [None]  # with rules, every config is measured without and with stage 2
    results = []
    for model in [m.strip() for m in (args.models or args.model).split(",") if m.strip()]:
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


def cmd_fetch(args) -> int:
    run = args.run or date.today().isoformat()
    if not run_date(run):
        _err(f"run id {run!r} has no date — runs are ordered by date: use e.g. 2026-10-15 or python-2026-10-15")
        return 2
    records, errors = collect(args.sources, where=args.where, log=_err)
    for e in errors:
        _err(e)
    for r in records:
        r["run_id"] = run  # `vm add` orders runs by this date
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    if not records:
        _err("nothing collected")
    elif args.out:
        Path(args.out).write_text(lines, encoding="utf-8")
        _err(f"wrote {len(records)} vacancies to {args.out}, run {run}")
    else:
        sys.stdout.write(lines)
    return 0 if records else 1


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


WORKFLOW = """typical use:
  vm index-profile resumes/                       resume-<track>-<LANG>.md files, all of them at once
  vm fetch remotive:python greenhouse:gitlab --where remote --out run.jsonl
  vm add run.jsonl                                 into the base (vm.db, or --db / VM_DB)
  vm calibrate                                     once you have 50+ vacancies
  vm rank --run 2026-10-15 --rules my-rules.yaml --out top.md
  vm label <url> fit | no_fit --reasons onsite,low_pay
  vm eval --rules my-rules.yaml                    after a few runs with fits and no_fits
  vm funnel tracker.csv                            after you apply: whom to remind, reply rates
each command has --help; the README explains the rules file and the score."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vm", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Rank vacancies against your resumes and your own terms, and measure it.",
                                epilog=WORKFLOW)
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
    lb.add_argument("--actor", default="human", choices=["agent", "human"],
                    help="human (default): your decision; agent: a suggestion, left out of eval by default")
    lb.add_argument("--track", default="")
    lb.add_argument("--reasons", default="")
    lb.add_argument("--note", default="")
    lb.set_defaults(func=cmd_label)

    e = sub.add_parser("eval", help="measure ranking quality on labeled vacancies")
    e.add_argument("--models", default="", help="comma-separated models to compare; default: --model")
    e.add_argument("--neighbor-weights", default="0")
    e.add_argument("--hybrid-alphas", default="0")
    e.add_argument("--labels", default="human", choices=["human", "all"],
                   help="whose labels count: your own decisions (default) or the agent's too")
    e.add_argument("--rules", default=os.environ.get("VM_RULES", ""))
    e.add_argument("--out", default="")
    e.set_defaults(func=cmd_eval)

    fe = sub.add_parser("fetch", help="collect vacancies from public job APIs into JSONL for vm add")
    fe.add_argument("sources", nargs="+", metavar="SOURCE",
                    help=f"kind:argument, e.g. greenhouse:gitlab remotive:python; kinds: {', '.join(SOURCES)}")
    fe.add_argument("--where", default="", help="keep jobs whose title, location or work format match this regex")
    fe.add_argument("--out", default="", help="JSONL file; stdout when omitted")
    fe.add_argument("--run", default="", help="run id with a date for `vm add`; default: today's date")
    fe.set_defaults(func=cmd_fetch)

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
