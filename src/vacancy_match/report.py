"""Markdown view of a ranking, written for the agent that does the final selection."""
from __future__ import annotations

from .rank import Ranked


def render_top(ranked: list[Ranked], limit: int = 50, title: str = "") -> str:
    lines = [f"# {title or 'Ranked vacancies'}", ""]
    for n, r in enumerate(ranked[:limit], 1):
        v = r.vacancy
        lines += [f"## {n}. {v.title} — {v.company or '?'}", ""]
        meta = []
        if r.track:
            meta.append(f"fit {100 * r.fit:.0f}" if r.cut else f"score {100 * r.score:.0f} · fit {100 * r.fit:.0f}")
            meta.append(f"track {r.track}")
        if v.salary_text:
            meta.append(v.salary_text)
        if v.published_at:
            meta.append(v.published_at[:10])
        if meta:
            lines.append("- " + " · ".join(meta))
        lines.append(f"- {v.url}")
        if r.top_chunks:
            lines.append("- Why: " + "; ".join(f"{c.section} ({c.score:.2f})" for c in r.top_chunks))
        if r.flags:
            lines.append("- Flags: " + ", ".join(r.flags))
        if r.adjustments:
            lines.append("- Adjusted: " + "; ".join(map(str, r.adjustments)))
        if r.cut:
            lines.append("- CUT: " + "; ".join(map(str, r.cut)))
        if r.duplicate_of:
            lines.append(f"- Possible duplicate of {r.duplicate_of}")
        if r.neighbors:
            lines.append("- Similar labeled: " + "; ".join(
                f"{nb.label} {nb.similarity:.2f} {nb.title}" + (f" [{','.join(nb.reasons)}]" if nb.reasons else "")
                for nb in r.neighbors))
        lines += [f"- id `{v.id}`", ""]
    return "\n".join(lines)
