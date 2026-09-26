"""Decision layer: turn pair probabilities into per-S1 match sets.

Rules implemented (all compared on validation in EXPERIMENT_PLAN / MODEL_COMPARISON):
  threshold        keep pairs with p >= tau
  exclusive        many-to-one: each target is kept only for its best-scoring S1
                   (train GT: every target belongs to <= 1 S1)
  expected_f       per-S1 Bayes-optimal set under the per-entity F0.5 utility:
                   choose k* = argmax_k E[F0.5(top-k)], with k=0 (empty) allowed.
                   Under independent Bernoulli labels the optimal set is a top-k set
                   (Jansche 2007; Ye et al. ICML 2012; Waegeman et al. JMLR 2014), and
                   E[F] is computed exactly with Poisson-binomial DPs.

Vectorised macro-F0.5 scoring (`macro_f05_frame`) reproduces ber.metric exactly given
n_true per S1 (which includes true matches that blocking missed).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - numba optional locally
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


# ----------------------------------------------------------------------------- scoring

def macro_f05_frame(pred_q, pred_y, all_q, n_true, beta: float = 0.5):
    """Exact macro F0.5 from arrays.

    pred_q: q ids of predicted pairs; pred_y: 1 if the predicted pair is a true match.
    all_q:  every S1 id in the evaluation set; n_true: #true matches per all_q (aligned).
    """
    import polars as pl

    b2 = beta * beta
    P = pl.DataFrame({"q": pred_q, "y": pred_y}).group_by("q").agg(
        pl.len().alias("n_pred"), pl.col("y").sum().alias("tp"))
    A = pl.DataFrame({"q": all_q, "n_true": n_true}).join(P, on="q", how="left").fill_null(0)
    tp = A["tp"].to_numpy().astype(np.float64)
    npred = A["n_pred"].to_numpy().astype(np.float64)
    nt = A["n_true"].to_numpy().astype(np.float64)
    f = np.where(nt == 0, (npred == 0).astype(np.float64),
                 np.where(tp > 0, (1 + b2) * tp / np.maximum((1 + b2) * tp + b2 * (nt - tp) + (npred - tp), 1e-12), 0.0))
    return float(f.mean()), f


# ----------------------------------------------------------------------------- simple rules

def exclusive_mask(q: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for pairs whose q is the (first) argmax-p S1 for their target t."""
    import polars as pl

    df = pl.DataFrame({"i": np.arange(len(q)), "q": q, "t": t, "p": p})
    # deterministic: highest p, ties broken by the smaller S1 row id
    best = df.sort(["t", "p", "q"], descending=[False, True, False]).group_by("t", maintain_order=True).first()
    m = np.zeros(len(q), dtype=bool)
    m[best["i"].to_numpy()] = True
    return m


# ----------------------------------------------------------------------------- expected F

@njit(cache=True)
def _poibin(p):
    n = p.shape[0]
    d = np.zeros(n + 1)
    d[0] = 1.0
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1.0 - pi) + d[j - 1] * pi
        d[0] = d[0] * (1.0 - pi)
    return d


@njit(cache=True)
def _best_k(ps, extra_miss, beta2):
    """ps sorted descending. Returns (k*, E[F] at k*). extra_miss = expected # true
    matches outside the candidate list (added to the 'rest' as a fixed mean)."""
    n = ps.shape[0]
    best_k = 0
    # k = 0: F = 1 iff no positives anywhere
    p_none = 1.0
    for i in range(n):
        p_none *= (1.0 - ps[i])
    best_e = p_none * np.exp(-extra_miss)
    for k in range(1, n + 1):
        dA = _poibin(ps[:k])
        dB = _poibin(ps[k:])
        e = 0.0
        for a in range(1, k + 1):
            if dA[a] < 1e-12:
                continue
            for b in range(0, n - k + 1):
                if dB[b] < 1e-12:
                    continue
                fn = b + extra_miss
                e += dA[a] * dB[b] * ((1 + beta2) * a / ((1 + beta2) * a + beta2 * fn + (k - a)))
        if e > best_e:
            best_e = e
            best_k = k
    return best_k, best_e


@njit(cache=True)
def _expected_f_all(qptr, p_sorted, extra_miss, beta2, kmax):
    nq = qptr.shape[0] - 1
    ks = np.zeros(nq, np.int32)
    es = np.zeros(nq)
    for g in range(nq):
        s, e = qptr[g], qptr[g + 1]
        n = min(e - s, kmax)
        if n == 0:
            ks[g] = 0
            es[g] = np.exp(-extra_miss)
            continue
        k, v = _best_k(p_sorted[s:s + n], extra_miss, beta2)
        ks[g] = k
        es[g] = v
    return ks, es


def expected_f_select(q: np.ndarray, p: np.ndarray, extra_miss: float = 0.0, beta: float = 0.5,
                      kmax: int = 25, p_floor: float = 1e-3) -> np.ndarray:
    """Boolean mask of selected pairs (per-q Bayes-optimal top-k under E[F_beta])."""
    order = np.lexsort((-p, q))
    qs, ps = q[order], p[order].astype(np.float64)
    ps = np.where(ps < p_floor, 0.0, ps)
    starts = np.r_[0, np.flatnonzero(np.diff(qs)) + 1, len(qs)].astype(np.int64)
    ks, _ = _expected_f_all(starts, ps, float(extra_miss), beta * beta, kmax)
    sel_sorted = np.zeros(len(qs), dtype=bool)
    rank = np.arange(len(qs)) - np.repeat(starts[:-1], np.diff(starts))
    kk = np.repeat(ks, np.diff(starts))
    sel_sorted = rank < kk
    mask = np.zeros(len(q), dtype=bool)
    mask[order] = sel_sorted
    return mask


@njit(cache=True)
def _best_k_nonempty(ps, extra_miss, beta2):
    """max_{k>=1} E[F(top-k)] (no empty option). ps sorted descending."""
    n = ps.shape[0]
    best_k, best_e = 1, -1.0
    for k in range(1, n + 1):
        dA = _poibin(ps[:k])
        dB = _poibin(ps[k:])
        e = 0.0
        for a in range(1, k + 1):
            if dA[a] < 1e-12:
                continue
            for b in range(0, n - k + 1):
                if dB[b] < 1e-12:
                    continue
                fn = b + extra_miss
                e += dA[a] * dB[b] * ((1 + beta2) * a / ((1 + beta2) * a + beta2 * fn + (k - a)))
        if e > best_e:
            best_e = e
            best_k = k
    return best_k, best_e


@njit(cache=True)
def _pi_select_all(qptr, p_sorted, pi, extra_miss, beta2, kmax):
    """Hurdle rule: E[F | Y!=empty] from conditional probs q_i = min(p_i/pi, 1);
    predict non-empty top-k* iff pi * E* > 1 - pi (empty scores 1 exactly when Y is empty)."""
    nq = qptr.shape[0] - 1
    ks = np.zeros(nq, np.int32)
    for g in range(nq):
        s, e = qptr[g], qptr[g + 1]
        n = min(e - s, kmax)
        if n == 0:
            continue
        pg = pi[g]
        if pg <= 1e-9:
            continue
        qs = np.empty(n)
        for i in range(n):
            v = p_sorted[s + i] / pg
            qs[i] = v if v < 1.0 else 1.0
        k, est = _best_k_nonempty(qs, extra_miss, beta2)
        if pg * est > 1.0 - pg:
            ks[g] = k
    return ks


def pi_select(q: np.ndarray, p: np.ndarray, pi_by_q: dict, extra_miss: float = 0.0, beta: float = 0.5,
              kmax: int = 25, p_floor: float = 1e-3) -> np.ndarray:
    """Selection with an explicit match-exists probability pi(q) (hurdle model)."""
    order = np.lexsort((-p, q))
    qs, ps = q[order], p[order].astype(np.float64)
    ps = np.where(ps < p_floor, 0.0, ps)
    starts = np.r_[0, np.flatnonzero(np.diff(qs)) + 1, len(qs)].astype(np.int64)
    uq = qs[starts[:-1]]
    pi = np.array([pi_by_q.get(int(x), 0.0) for x in uq], dtype=np.float64)
    # pi can never be below the largest marginal
    mx = np.maximum.reduceat(ps, starts[:-1]) if len(ps) else np.zeros(0)
    pi = np.clip(np.maximum(pi, mx), 1e-9, 1.0)
    ks = _pi_select_all(starts, ps, pi, float(extra_miss), beta * beta, kmax)
    rank = np.arange(len(qs)) - np.repeat(starts[:-1], np.diff(starts))
    sel_sorted = rank < np.repeat(ks, np.diff(starts))
    mask = np.zeros(len(q), dtype=bool)
    mask[order] = sel_sorted
    return mask


def s1_aggregates(q: np.ndarray, p: np.ndarray, extra: Optional[dict] = None):
    """S1-level features over the candidate list for the match-exists model."""
    import polars as pl

    df = pl.DataFrame({"q": q, "p": p, **(extra or {})})
    aggs = [
        pl.col("p").max().alias("p_max"),
        pl.col("p").sort(descending=True).get(1, null_on_oob=True).fill_null(0.0).alias("p_2nd"),
        pl.col("p").sort(descending=True).get(2, null_on_oob=True).fill_null(0.0).alias("p_3rd"),
        pl.col("p").sum().alias("p_sum"),
        (pl.col("p") > 0.5).sum().alias("n_p50"),
        (pl.col("p") > 0.2).sum().alias("n_p20"),
        (pl.col("p") > 0.05).sum().alias("n_p05"),
        pl.len().alias("n_cand"),
    ]
    for k in (extra or {}):
        aggs.append(pl.col(k).max().alias(f"{k}_max"))
        aggs.append(pl.col(k).sort_by("p", descending=True).first().alias(f"{k}_at_best"))
    return df.group_by("q").agg(aggs)


def _selftest():
    # E[F] sanity: one candidate with p=0.9 -> select (0.9) vs empty (0.1)
    m = expected_f_select(np.array([0]), np.array([0.9]))
    assert m.tolist() == [True]
    m = expected_f_select(np.array([0]), np.array([0.1]))
    assert m.tolist() == [False]
    # F0.5 threshold intuition: p=0.3 alone -> empty (E=0.7) beats select (0.3)
    m = expected_f_select(np.array([0, 0, 0]), np.array([0.95, 0.9, 0.2]))
    assert m.tolist() == [True, True, False], m
    f, _ = macro_f05_frame(np.array([1, 1, 1]), np.array([1, 0, 1]), np.array([1, 2]), np.array([2, 0]))
    assert abs(f - (0.7142857 + 1) / 2) < 1e-6, f
    print("decide selftest OK")


if __name__ == "__main__":
    _selftest()
