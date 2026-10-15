import numpy as np

from vacancy_match.embed import HashEmbedder, ensure_embeddings, get_embedder, tokenize
from vacancy_match.store import Store


def test_tokenize_handles_cyrillic():
    assert tokenize("Python-разработчик, БОТЫ!") == ["python", "разработчик", "боты"]


def test_hash_embedder_is_normalised_and_deterministic():
    e = HashEmbedder(64)
    a = e.encode(["python bots", "python bots", ""], kind="query")
    assert a.shape == (3, 64)
    assert np.allclose(np.linalg.norm(a[0]), 1.0)
    assert np.allclose(a[0], a[1])
    assert np.allclose(a[2], 0.0)


def test_hash_embedder_overlap_means_similarity():
    e = HashEmbedder(512)
    q, near, far = e.encode(["python telegram bots", "telegram bots in python", "sales manager calls"], "query")
    assert q @ near > q @ far


def test_get_embedder_hash_dim():
    assert get_embedder("hash-256").name == "hash-256"
    assert get_embedder("hash").name == "hash-512"


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__(32)
        self.calls = 0

    def encode(self, texts, kind="passage"):
        self.calls += len(texts)
        return super().encode(texts, kind)


def test_ensure_embeddings_only_computes_missing(tmp_path):
    s = Store(tmp_path / "a.db")
    e = CountingEmbedder()
    first = ensure_embeddings(s, e, "vacancy", {"a": "python", "b": "design"}, "query")
    again = ensure_embeddings(s, e, "vacancy", {"a": "python", "b": "design", "c": "sales"}, "query")
    assert e.calls == 3
    assert set(again) == {"a", "b", "c"}
    assert np.allclose(first["a"], again["a"])


def test_changed_text_is_re_embedded(tmp_path):
    s, e = Store(tmp_path / "e.db"), HashEmbedder(64)
    first = ensure_embeddings(s, e, "vacancy", {"a": "python telegram bots"}, "query")["a"]
    again = ensure_embeddings(s, e, "vacancy", {"a": "python telegram bots"}, "query")["a"]
    changed = ensure_embeddings(s, e, "vacancy", {"a": "figma landing design"}, "query")["a"]
    assert np.allclose(first, again) and not np.allclose(first, changed)
