"""Embedders: a deterministic hashing one for tests, sentence-transformers for real use."""
from __future__ import annotations

import hashlib
import re

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
_TOKEN = re.compile(r"\w+", re.U)


class EmbedderUnavailable(RuntimeError):
    pass


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector. No notion of synonyms; for tests and offline smoke runs."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
                m[i, h % self.dim] += 1.0
        return _normalize(m)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable("sentence-transformers is not installed: run `uv sync --extra model`") from e
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:  # network, disk, broken cache
            raise EmbedderUnavailable(
                f"could not load {model_name}: {e}. Download it once with network access, or point HF_HOME "
                f"to an ASCII-only folder if the cache path is the problem.") from e
        e5 = "e5" in model_name.lower()
        self._prefix = {"query": "query: " if e5 else "", "passage": "passage: " if e5 else ""}

    def encode(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self._model.encode([self._prefix[kind] + t for t in texts], batch_size=16, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))


def get_embedder(name: str):
    if name.startswith("hash"):
        return HashEmbedder(int(name.split("-", 1)[1]) if "-" in name else 512)
    return SentenceTransformerEmbedder(name)


def ensure_embeddings(store, embedder, owner_type: str, items: dict[str, str], kind: str) -> dict[str, np.ndarray]:
    # a vector is reused only for the exact text it was computed from: changing what is embedded re-embeds
    hashes = {i: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for i, t in items.items()}
    have = store.get_embeddings(owner_type, embedder.name, list(items), hashes=hashes)
    missing = [i for i in items if i not in have]
    if missing:
        new = dict(zip(missing, embedder.encode([items[i] for i in missing], kind=kind)))
        store.put_embeddings(owner_type, embedder.name, new, hashes=hashes)
        have.update(new)
    return have
