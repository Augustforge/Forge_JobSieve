"""Plain BM25, used only for the hybrid ranking experiment."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def bm25_matrix(queries: list[list[str]], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    n = len(docs)
    out = np.zeros((len(queries), n), dtype=np.float32)
    if n == 0:
        return out
    avgdl = sum(map(len, docs)) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    tfs = [Counter(d) for d in docs]
    for i, q in enumerate(queries):
        for j, (tf, d) in enumerate(zip(tfs, docs)):
            score = 0.0
            for t in set(q):
                f = tf.get(t)
                if f:
                    score += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
            out[i, j] = score
    return out
