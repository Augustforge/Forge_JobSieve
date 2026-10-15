import numpy as np

from vacancy_match.models import Label, ProfileChunk, Vacancy
from vacancy_match.store import Store


def vac(url, text="t" * 250, run="r1", **kw):
    return Vacancy.from_dict({"source": "s", "url": url, "title": "T", "text": text, **kw}, run_id=run)


def test_add_and_get(tmp_path):
    s = Store(tmp_path / "a.db")
    v = vac("x.com/1", company="Acme")
    assert s.add_vacancies([v]).new == 1
    assert s.get_vacancy(v.id).company == "Acme"
    assert s.get_vacancy("nope") is None


def test_add_same_vacancy_twice_counts_known(tmp_path):
    s = Store(tmp_path / "a.db")
    s.add_vacancies([vac("https://krasnoyarsk.hh.ru/vacancy/7?query=AI", run="r1")])
    res = s.add_vacancies([vac("https://hh.kz/vacancy/7", run="r2")])
    assert (res.new, res.known) == (0, 1)
    assert len(s.vacancies()) == 1
    assert [v.run_id for v in s.vacancies("r2")] == ["r1"]  # seen in r2, first run kept


def test_known_vacancy_gets_missing_text(tmp_path):
    s = Store(tmp_path / "a.db")
    first = vac("x.com/1", text="")
    s.add_vacancies([first])
    s.add_vacancies([vac("x.com/1", text="full description " * 20, run="r2")])
    assert s.get_vacancy(first.id).text.startswith("full description")


def test_embeddings_roundtrip_per_model(tmp_path):
    s = Store(tmp_path / "a.db")
    s.put_embeddings("vacancy", "m1", {"a": np.array([1, 0], dtype=np.float32)})
    s.put_embeddings("vacancy", "m2", {"a": np.array([0, 1], dtype=np.float32)})
    assert s.get_embeddings("vacancy", "m1")["a"].tolist() == [1.0, 0.0]
    assert s.get_embeddings("vacancy", "m2", ["a", "missing"]).keys() == {"a"}


def test_reindex_profile_keeps_vacancy_vectors(tmp_path):
    s = Store(tmp_path / "a.db")
    s.replace_profile([ProfileChunk("A-en-00", "A", "en", "Summary", "python bots")])
    s.put_embeddings("profile_chunk", "m", {"A-en-00": np.ones(2, dtype=np.float32)})
    s.put_embeddings("vacancy", "m", {"v": np.ones(2, dtype=np.float32)})
    s.replace_profile([ProfileChunk("B-en-00", "B", "en", "Summary", "design")])
    assert [c.id for c in s.profile_chunks()] == ["B-en-00"]
    assert s.get_embeddings("profile_chunk", "m") == {}
    assert "v" in s.get_embeddings("vacancy", "m")


def test_effective_labels_human_beats_agent_and_latest_wins(tmp_path):
    s = Store(tmp_path / "a.db")
    s.add_labels([
        Label("v1", "fit", actor="agent", created_at="2026-10-01T00:00:00"),
        Label("v1", "no_fit", actor="human", reasons=("language",), created_at="2026-10-02T00:00:00"),
        Label("v1", "fit", actor="agent", created_at="2026-10-03T00:00:00"),
        Label("v2", "no_fit", actor="agent", created_at="2026-10-01T00:00:00"),
        Label("v2", "fit", actor="agent", track="C", created_at="2026-10-05T00:00:00"),
    ])
    eff = s.effective_labels()
    assert eff["v1"].label == "no_fit" and eff["v1"].reasons == ("language",)
    assert eff["v2"].label == "fit" and eff["v2"].track == "C"


def test_longer_text_replaces_snippet_and_drops_stale_vectors(tmp_path):
    s = Store(tmp_path / "a.db")
    snippet = vac("x.com/1", text="Short snippet")
    s.add_vacancies([snippet])
    s.put_embeddings("vacancy", "m", {snippet.id: np.ones(2, dtype=np.float32)})
    s.add_vacancies([vac("x.com/1", text="Полное описание вакансии " * 20, run="r2")])
    v = s.get_vacancy(snippet.id)
    assert v.text.startswith("Полное описание") and v.lang == "ru"
    assert s.get_embeddings("vacancy", "m") == {}


def test_shorter_text_does_not_overwrite(tmp_path):
    s = Store(tmp_path / "a.db")
    full = vac("x.com/1", text="full description " * 20)
    s.add_vacancies([full])
    s.add_vacancies([vac("x.com/1", text="snippet", run="r2")])
    assert s.get_vacancy(full.id).text == full.text


def test_old_base_without_hash_column_is_migrated(tmp_path):
    import sqlite3
    p = tmp_path / "old.db"
    db = sqlite3.connect(p)
    db.execute("CREATE TABLE embeddings (owner_type TEXT, owner_id TEXT, model TEXT, vector BLOB, "
               "PRIMARY KEY (owner_type, owner_id, model))")
    db.execute("INSERT INTO embeddings VALUES ('vacancy', 'a', 'm', ?)", (np.ones(2, dtype=np.float32).tobytes(),))
    db.commit()
    db.close()
    s = Store(p)
    assert "a" in s.get_embeddings("vacancy", "m")  # old vectors stay readable
    assert s.get_embeddings("vacancy", "m", hashes={"a": "h1"}) == {}  # but never match a text hash
