"""Candidate generation: multi-view sparse TF-IDF top-K retrieval + union.

All retrieval runs inside a country partition (country-label equality holds for 100% of
true train pairs; the partition key is whatever label strings appear, so unseen labels
such as France are handled identically — nothing is hard-coded).

Views (each is a sparse L2-normalised TF-IDF matrix; cosine = dot product):
  name   char_wb 3-grams of the normalised name            (typos, reorder, translit)
  cat    char 3-grams of the name with spaces removed      (handles, domains, run-ons)
  addr   word unigrams of the normalised address (numbers kept)
  combo  [sqrt(w)*name | sqrt(1-w)*addr]                   (hybrid; also run T->S1)

Top-K per view uses sparse_dot_topn (multithreaded C++ sparse matmul with top-n pruning).
Vectorizers are fit on the union of S1+targets of the partition (label-free).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


def _vectorizer(kind: str, max_df: float, min_df: int = 2):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if kind == "char_wb":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=min_df, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
    if kind == "char":
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=min_df, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
    if kind == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", min_df=min_df, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32, lowercase=False)
    raise ValueError(kind)


def _transform_parallel(vec, texts: List[str], n_jobs: int, chunk: int = 200_000):
    from joblib import Parallel, delayed

    parts = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    if len(parts) == 1 or n_jobs == 1:
        return vec.transform(texts)
    mats = Parallel(n_jobs=n_jobs, backend="loky")(delayed(vec.transform)(p) for p in parts)
    return sp.vstack(mats).tocsr()


def tfidf_pair(q_texts: List[str], t_texts: List[str], kind: str, max_df: float,
               fit_sample: int = 3_000_000, n_jobs: int = 16, seed: int = 0):
    """Fit on (a sample of) the union of both sides, transform both sides."""
    vec = _vectorizer(kind, max_df)
    allt = q_texts + t_texts
    if len(allt) > fit_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(allt), size=fit_sample, replace=False)
        vec.fit([allt[i] for i in idx])
    else:
        vec.fit(allt)
    Q = _transform_parallel(vec, q_texts, n_jobs)
    T = _transform_parallel(vec, t_texts, n_jobs)
    return Q.astype(np.float32), T.astype(np.float32), len(vec.vocabulary_)


def hstack_views(views: List[Tuple[sp.csr_matrix, float]]) -> sp.csr_matrix:
    """Weighted concat of L2-normalised views -> cosine = sum_w w * cos_view."""
    mats = [m.multiply(np.float32(np.sqrt(w))).tocsr() for m, w in views]
    return sp.hstack(mats).tocsr()


def topk(Q: sp.csr_matrix, T: sp.csr_matrix, k: int, threshold: float = 0.0,
         n_threads: int = 32, chunk: int = 250_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of Q: top-k rows of T by dot product. Returns (qi, ti, score) arrays,
    sorted by (qi, -score)."""
    from sparse_dot_topn import sp_matmul_topn

    TT = T.T.tocsr()
    qs, ts, vs = [], [], []
    for s in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[s:s + chunk], TT, top_n=k, threshold=threshold, sort=True, n_threads=n_threads)
        C = C.tocoo()
        qs.append(C.row.astype(np.int64) + s)
        ts.append(C.col.astype(np.int64))
        vs.append(C.data.astype(np.float32))
    qi = np.concatenate(qs) if qs else np.zeros(0, np.int64)
    ti = np.concatenate(ts) if ts else np.zeros(0, np.int64)
    sc = np.concatenate(vs) if vs else np.zeros(0, np.float32)
    order = np.lexsort((-sc, qi))
    return qi[order], ti[order], sc[order]


def ranks_from_sorted(qi: np.ndarray) -> np.ndarray:
    """Rank (0-based) within each query group of an array already sorted by (qi, -score)."""
    if len(qi) == 0:
        return np.zeros(0, np.int32)
    start = np.r_[0, np.flatnonzero(np.diff(qi)) + 1]
    counts = np.diff(np.r_[start, len(qi)])
    return (np.arange(len(qi)) - np.repeat(start, counts)).astype(np.int32)


def recall_at(pairs_q: np.ndarray, pairs_t: np.ndarray, rank: np.ndarray,
              true_q: np.ndarray, true_t: np.ndarray, ks=(1, 3, 5, 10, 20, 30, 50, 100)) -> Dict[str, float]:
    """Pair recall of a retrieval list at several cut-offs (ids must be global ints)."""
    key = pairs_q.astype(np.int64) * (1 << 32) + pairs_t.astype(np.int64)
    tkey = true_q.astype(np.int64) * (1 << 32) + true_t.astype(np.int64)
    out = {}
    for k in ks:
        m = rank < k
        out[f"R@{k}"] = float(np.isin(tkey, key[m]).mean()) if len(tkey) else float("nan")
    return out
