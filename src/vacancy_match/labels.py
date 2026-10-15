"""Label records from CLI arguments or JSONL batch files."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Label, make_id


def label_from_dict(d: dict) -> Label:
    vacancy_id = d.get("vacancy_id") or (make_id(d["url"]) if d.get("url") else "")
    if not vacancy_id:
        raise ValueError("label needs vacancy_id or url")
    reasons = d.get("reasons") or ()
    if isinstance(reasons, str):
        reasons = [r.strip() for r in reasons.split(",") if r.strip()]
    return Label(vacancy_id=vacancy_id, label=d.get("label", ""), actor=d.get("actor", "human"),
                 track=d.get("track", ""), reasons=tuple(reasons), note=d.get("note", ""))


def load_label_file(path: str | Path) -> list[Label]:
    labels = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").split("\n"), 1):  # not splitlines: JSON keeps U+2028 inside strings
        if not line.strip():
            continue
        try:
            labels.append(label_from_dict(json.loads(line)))
        except (ValueError, KeyError, AttributeError, TypeError) as e:  # bad JSON, non-object line, bad field
            raise ValueError(f"{Path(path).name}:{n}: {e}") from e
    return labels
