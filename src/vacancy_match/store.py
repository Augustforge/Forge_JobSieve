"""SQLite storage for vacancies, sightings, embeddings, profile chunks and labels."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import Label, ProfileChunk, Vacancy, detect_lang

SCHEMA = """
CREATE TABLE IF NOT EXISTS vacancies (
  id TEXT PRIMARY KEY, source TEXT, url TEXT, title TEXT, company TEXT, text TEXT,
  salary_text TEXT, location TEXT, work_format TEXT, published_at TEXT, lang TEXT,
  run_id TEXT, collected_at TEXT);
CREATE TABLE IF NOT EXISTS sightings (
  vacancy_id TEXT, run_id TEXT, seen_at TEXT, PRIMARY KEY (vacancy_id, run_id));
CREATE TABLE IF NOT EXISTS embeddings (
  owner_type TEXT, owner_id TEXT, model TEXT, vector BLOB, text_hash TEXT DEFAULT '',
  PRIMARY KEY (owner_type, owner_id, model));
CREATE TABLE IF NOT EXISTS profile_chunks (id TEXT PRIMARY KEY, track TEXT, lang TEXT, section TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS calibration (model TEXT PRIMARY KEY, centre REAL, width REAL, n INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS labels (
  vacancy_id TEXT, label TEXT, actor TEXT, track TEXT, reasons TEXT, note TEXT, created_at TEXT);
"""
_COLS = ("id", "source", "url", "title", "company", "text", "salary_text", "location",
         "work_format", "published_at", "lang", "run_id", "collected_at")


@dataclass
class AddResult:
    new: int
    known: int


class Store:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(embeddings)")}
        if "text_hash" not in cols:  # bases created before vectors were tied to their text
            with self.db:
                self.db.execute("ALTER TABLE embeddings ADD COLUMN text_hash TEXT DEFAULT ''")

    def close(self) -> None:
        self.db.close()

    # vacancies
    def add_vacancies(self, vacancies: list[Vacancy]) -> AddResult:
        new = known = 0
        placeholders = ",".join("?" * len(_COLS))
        with self.db:
            for v in vacancies:
                cur = self.db.execute(f"INSERT OR IGNORE INTO vacancies VALUES ({placeholders})",
                                      tuple(getattr(v, c) for c in _COLS))
                if cur.rowcount:
                    new += 1
                else:
                    known += 1
                    # a full description replaces a snippet; its cached vectors were built from the snippet
                    cur = self.db.execute(
                        "UPDATE vacancies SET text = ?, lang = ? WHERE id = ? AND length(text) < ?",
                        (v.text, detect_lang(v.text), v.id, len(v.text)))
                    if cur.rowcount:
                        self.db.execute("DELETE FROM embeddings WHERE owner_type = 'vacancy' AND owner_id = ?", (v.id,))
                self.db.execute("INSERT OR IGNORE INTO sightings VALUES (?, ?, ?)", (v.id, v.run_id, v.collected_at))
        return AddResult(new, known)

    def get_vacancy(self, vacancy_id: str) -> Vacancy | None:
        row = self.db.execute("SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)).fetchone()
        return Vacancy(*row) if row else None

    def vacancies(self, run_id: str | None = None) -> list[Vacancy]:
        if run_id is None:
            rows = self.db.execute("SELECT * FROM vacancies ORDER BY run_id, collected_at").fetchall()
        else:
            rows = self.db.execute(
                "SELECT v.* FROM vacancies v JOIN sightings s ON s.vacancy_id = v.id "
                "WHERE s.run_id = ? ORDER BY v.collected_at", (run_id,)).fetchall()
        return [Vacancy(*r) for r in rows]

    # embeddings
    def put_embeddings(self, owner_type: str, model: str, vectors: dict[str, np.ndarray],
                       hashes: dict[str, str] | None = None) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings (owner_type, owner_id, model, vector, text_hash) VALUES (?, ?, ?, ?, ?)",
                [(owner_type, oid, model, np.asarray(v, dtype=np.float32).tobytes(), (hashes or {}).get(oid, ""))
                 for oid, v in vectors.items()])

    def get_embeddings(self, owner_type: str, model: str, ids: list[str] | None = None,
                       hashes: dict[str, str] | None = None) -> dict[str, np.ndarray]:
        """With hashes, only vectors computed from exactly that text come back; the rest count as missing."""
        rows = self.db.execute("SELECT owner_id, vector, text_hash FROM embeddings WHERE owner_type = ? AND model = ?",
                               (owner_type, model)).fetchall()
        out = {oid: np.frombuffer(blob, dtype=np.float32) for oid, blob, h in rows
               if hashes is None or hashes.get(oid) == h}
        return out if ids is None else {i: out[i] for i in ids if i in out}

    # calibration: the frozen fit scale per model
    def get_calibration(self, model: str) -> tuple[float, float, int] | None:
        row = self.db.execute("SELECT centre, width, n FROM calibration WHERE model = ?", (model,)).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def put_calibration(self, model: str, centre: float, width: float, n: int) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO calibration VALUES (?, ?, ?, ?, ?)",
                            (model, centre, width, n, datetime.now(timezone.utc).isoformat(timespec="seconds")))

    # profile
    def replace_profile(self, chunks: list[ProfileChunk]) -> None:
        with self.db:
            self.db.execute("DELETE FROM embeddings WHERE owner_type = 'profile_chunk'")
            self.db.execute("DELETE FROM calibration")  # the scale is measured against this profile
            self.db.execute("DELETE FROM profile_chunks")
            self.db.executemany("INSERT INTO profile_chunks VALUES (?, ?, ?, ?, ?)",
                                [(c.id, c.track, c.lang, c.section, c.text) for c in chunks])

    def profile_chunks(self) -> list[ProfileChunk]:
        return [ProfileChunk(*r) for r in self.db.execute("SELECT * FROM profile_chunks ORDER BY id")]

    # labels
    def add_labels(self, labels: list[Label]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(l.vacancy_id, l.label, l.actor, l.track, ",".join(l.reasons), l.note, l.created_at) for l in labels])

    def labels(self) -> list[Label]:
        rows = self.db.execute("SELECT * FROM labels").fetchall()
        return [Label(vid, lab, actor, track, tuple(r for r in reasons.split(",") if r), note, created)
                for vid, lab, actor, track, reasons, note, created in rows]

    def effective_labels(self) -> dict[str, Label]:
        """One label per vacancy: a human decision beats the agent's; within the same actor the latest wins."""
        best: dict[str, Label] = {}
        for lab in sorted(self.labels(), key=lambda l: (l.actor == "human", l.created_at)):
            best[lab.vacancy_id] = lab
        return best
