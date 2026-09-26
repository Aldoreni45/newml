"""Pairwise features for (S1 record q, target record t) candidate pairs.

Groups (see FEATURE_CATALOG.md for the full list and rationale):
  A. string similarities on normalized names / addresses (RapidFuzz, C++ multithreaded)
  B. exact TF-IDF cosines per view (name char3, name-concat char3, name word, address word)
  C. token-level set features (Jaccard, IDF-weighted overlap, rare-token hits, initials,
     phonetic keys, numbers / house-number agreement)
  D. structural: script of target, source, empty fields, name frequency (ambiguity)
  E. context: within-query ranks/gaps (per q) and within-target competition (per t)
     -> the many-to-one signal "is q the best S1 for this t?"
All features are label-free functions of the records, so they can be computed identically
for train, validation and test.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .normalize import phonetic

# ----------------------------------------------------------------------------- IDF tables


def token_idf(docs: Sequence[str], min_df: int = 1) -> Dict[str, float]:
    df: Counter = Counter()
    for d in docs:
        df.update(set(d.split()))
    n = max(len(docs), 1)
    return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items() if c >= min_df}


# ----------------------------------------------------------------------------- token features

_GENERIC_CACHE: Dict[str, set] = {}


def _initials(toks: List[str]) -> str:
    return "".join(w[0] for w in toks if w and not w.isdigit())


def token_feats_batch(args) -> np.ndarray:
    """Pure-python token features for a batch of pairs.

    args: (q_nm, t_nm, t_nm_a, t_nm_b, q_num, t_num, q_ad, t_ad, idf_name, idf_addr, generic)
    Returns float32 array [n, F_TOK].
    """
    (q_nm, t_nm, t_a, t_b, q_num, t_num, q_ad, t_ad, idf_n, idf_a, generic) = args
    n = len(q_nm)
    out = np.zeros((n, F_TOK), dtype=np.float32)
    dflt_n = max(idf_n.values()) if idf_n else 1.0
    dflt_a = max(idf_a.values()) if idf_a else 1.0
    for i in range(n):
        qt = q_nm[i].split()
        tt = t_nm[i].split()
        qs, ts = set(qt), set(tt)
        inter = qs & ts
        uni = qs | ts
        # name token overlap
        out[i, 0] = len(inter) / len(uni) if uni else 0.0
        wi = sum(idf_n.get(w, dflt_n) for w in inter)
        wq = sum(idf_n.get(w, dflt_n) for w in qs)
        wt = sum(idf_n.get(w, dflt_n) for w in ts)
        out[i, 1] = wi / (wq + wt - wi) if (wq + wt - wi) > 0 else 0.0
        out[i, 2] = wi / wq if wq > 0 else 0.0            # fraction of S1 name mass covered
        out[i, 3] = wi / wt if wt > 0 else 0.0            # fraction of target mass explained
        out[i, 4] = max((idf_n.get(w, dflt_n) for w in inter), default=0.0)  # rarest shared token
        qc = [w for w in qt if w not in generic] or qt
        tc = [w for w in tt if w not in generic] or tt
        qcs, tcs = set(qc), set(tc)
        out[i, 5] = len(qcs & tcs) / len(qcs | tcs) if (qcs | tcs) else 0.0   # core-token Jaccard
        out[i, 6] = float(" ".join(sorted(qcs)) == " ".join(sorted(tcs)))       # core-set equal
        out[i, 7] = float(bool(qt) and bool(tt) and qt[0] == tt[0])
        ini_q = _initials(qc)
        tj = "".join(tt)
        out[i, 8] = float(len(ini_q) >= 2 and (tj == ini_q or t_nm[i].replace(" ", "") == ini_q))
        pq = {phonetic(w) for w in qc}
        pt = {phonetic(w) for w in tc}
        out[i, 9] = len(pq & pt) / len(pq | pt) if (pq | pt) else 0.0
        # alias parts: best core Jaccard against either part of "X dba Y"
        best = 0.0
        for part in (t_a[i], t_b[i]):
            if part and part != t_nm[i]:
                ps = set(part.split()) - generic or set(part.split())
                if ps:
                    best = max(best, len(qcs & ps) / len(qcs | ps))
        out[i, 10] = best
        out[i, 11] = len(qt)
        out[i, 12] = len(tt)
        out[i, 13] = len(qcs - tcs)                         # S1 core tokens missing in target
        out[i, 14] = len(tcs - qcs)                         # extra target core tokens
        # prefix/truncation: target core tokens are a prefix of S1 core tokens
        out[i, 15] = float(len(tc) < len(qc) and qc[:len(tc)] == tc)
        # numbers in address
        qn = q_num[i].split()
        tn = t_num[i].split()
        qns, tns = set(qn), set(tn)
        out[i, 16] = len(qns & tns) / len(qns | tns) if (qns | tns) else -1.0
        out[i, 17] = float(bool(qn) and bool(tn) and qn[0] == tn[0]) if (qn and tn) else -1.0
        out[i, 18] = len(qns & tns)
        out[i, 19] = float(bool(qns - tns)) if tns else -1.0   # S1 has a number the target lacks
        big_q = {x for x in qns if len(x) >= 3}
        big_t = {x for x in tns if len(x) >= 3}
        out[i, 20] = (len(big_q & big_t) / len(big_q | big_t)) if (big_q and big_t) else -1.0
        # address tokens (alpha only) IDF overlap
        qa = {w for w in q_ad[i].split() if not w.isdigit()}
        ta = {w for w in t_ad[i].split() if not w.isdigit()}
        ia = qa & ta
        wi = sum(idf_a.get(w, dflt_a) for w in ia)
        wq = sum(idf_a.get(w, dflt_a) for w in qa)
        wt = sum(idf_a.get(w, dflt_a) for w in ta)
        out[i, 21] = wi / (wq + wt - wi) if (wq + wt - wi) > 0 else (-1.0 if not ta else 0.0)
        out[i, 22] = wi / wt if wt > 0 else -1.0
        out[i, 23] = max((idf_a.get(w, dflt_a) for w in ia), default=0.0)
        out[i, 24] = float(not t_ad[i])
        # house-number geometry: generator applies both digit typos and small numbering
        # offsets to TRUE copies, and builds sibling distractors with nearby numbers.
        nq = [int(x) for x in qn if 1 <= len(x) <= 7]
        nt = [int(x) for x in tn if 1 <= len(x) <= 7]
        if nq and nt:
            best_abs, best_rel, best_ed = 1e9, 1e9, 99
            for a in nq[:4]:
                for b in nt[:4]:
                    d = abs(a - b)
                    if d < best_abs:
                        best_abs = d
                    r = d / max(a, b, 1)
                    if r < best_rel:
                        best_rel = r
                    ed = _digit_edit(str(a), str(b))
                    if ed < best_ed:
                        best_ed = ed
            out[i, 25] = min(best_abs, 1e6)
            out[i, 26] = best_rel
            out[i, 27] = best_ed
            out[i, 28] = float(nq[0] == nt[0])
            out[i, 29] = abs(nq[0] - nt[0]) if abs(nq[0] - nt[0]) < 1e6 else 1e6
        else:
            out[i, 25:30] = -1.0
    return out


def _digit_edit(a: str, b: str) -> int:
    """Levenshtein distance between two short digit strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


TOK_NAMES = [
    "n_jacc", "n_idf_jacc", "n_idf_cov_q", "n_idf_cov_t", "n_max_shared_idf", "n_core_jacc",
    "n_core_eq", "n_first_tok_eq", "n_initials", "n_phon_jacc", "n_alias_core_jacc", "n_len_q",
    "n_len_t", "n_core_missing", "n_core_extra", "n_trunc_prefix",
    "a_num_jacc", "a_first_num_eq", "a_num_shared", "a_q_num_missing", "a_bignum_jacc",
    "a_idf_jacc", "a_idf_cov_t", "a_max_shared_idf", "a_t_empty",
    "a_num_min_absdiff", "a_num_min_reldiff", "a_num_min_digit_edit", "a_num_first_eq", "a_num_first_absdiff",
]
F_TOK = len(TOK_NAMES)


# ----------------------------------------------------------------------------- rapidfuzz features

def fuzz_feats(q_nm, t_nm, q_cat, t_cat, q_ad, t_ad, workers: int = -1) -> Dict[str, np.ndarray]:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    from rapidfuzz.process import cpdist

    def cp(a, b, scorer, **kw):
        return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32, **kw).astype(np.float32)

    f = {
        "n_ratio": cp(q_nm, t_nm, fuzz.ratio),
        "n_tsort": cp(q_nm, t_nm, fuzz.token_sort_ratio),
        "n_tset": cp(q_nm, t_nm, fuzz.token_set_ratio),
        "n_partial": cp(q_nm, t_nm, fuzz.partial_ratio),
        "n_wratio": cp(q_nm, t_nm, fuzz.WRatio),
        "n_jw": cp(q_nm, t_nm, JaroWinkler.normalized_similarity),
        "n_cat_ratio": cp(q_cat, t_cat, fuzz.ratio),
        "n_cat_partial": cp(q_cat, t_cat, fuzz.partial_ratio),
        "n_cat_lev": cp(q_cat, t_cat, Levenshtein.normalized_similarity),
        "a_ratio": cp(q_ad, t_ad, fuzz.ratio),
        "a_tsort": cp(q_ad, t_ad, fuzz.token_sort_ratio),
        "a_tset": cp(q_ad, t_ad, fuzz.token_set_ratio),
        "a_partial": cp(q_ad, t_ad, fuzz.partial_ratio),
    }
    return f


# ----------------------------------------------------------------------------- tf-idf cosines

def rowwise_cos(Q, T, qi: np.ndarray, ti: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """cos(Q[qi[k]], T[ti[k]]) for all k (rows are L2-normalised)."""
    out = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        a = Q[qi[s:s + chunk]]
        b = T[ti[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


# ----------------------------------------------------------------------------- collective features

COLL_NAMES = ["col_p_anchor", "col_a_tset", "col_n_tset", "col_a_eq", "col_n_eq", "col_same_src",
              "col_num_jacc", "col_n_sib_addr90", "col_n_sib_name90", "col_n_conf",
              "col_wsupport_addr", "col_wsupport_name", "col_num_agree_conf", "col_best_sib_same_src",
              "col_best_sib_other_src", "col_isolated"]


def collective_batch(args) -> np.ndarray:
    """Support from the S1's co-candidates, weighted by stage-1 probabilities p1.

    args: (q, t, p1, nm, ad, adnum, src) with q/t/p1 arrays for a contiguous block of
    WHOLE q-groups (sorted by q), and record-level arrays nm/ad/adnum/src indexed by rid.
    Copies of one entity share a corrupted base record, and sibling distractor entities
    form their own coherent copy cluster. So agreement between t and the S1's confident
    co-candidates (same house number, near-identical address/name) separates true copies
    from lookalikes.
    """
    from rapidfuzz import fuzz

    q, t, p1, nm, ad, adnum, src = args
    n = len(q)
    out = np.zeros((n, len(COLL_NAMES)), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(np.diff(q)) + 1, n]
    for g in range(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        idx = np.arange(s, e)
        order = idx[np.argsort(-p1[s:e])]
        conf = [i for i in order if p1[i] >= 0.5]
        support_pool = [i for i in order if p1[i] >= 0.05][:15]
        for i in idx:
            anc = order[0] if order[0] != i else (order[1] if len(order) > 1 else -1)
            ti = t[i]
            nums_i = set(adnum[ti].split())
            if anc >= 0:
                ta = t[anc]
                out[i, 0] = p1[anc]
                out[i, 1] = fuzz.token_set_ratio(ad[ti], ad[ta]) if (ad[ti] and ad[ta]) else -1.0
                out[i, 2] = fuzz.token_set_ratio(nm[ti], nm[ta])
                out[i, 3] = float(bool(ad[ti]) and ad[ti] == ad[ta])
                out[i, 4] = float(nm[ti] == nm[ta])
                out[i, 5] = float(src[ti] == src[ta])
                b = set(adnum[ta].split())
                out[i, 6] = len(nums_i & b) / len(nums_i | b) if (nums_i | b) else -1.0
            else:
                out[i, 0:7] = -1.0
            ns_a = ns_n = 0
            agree = n_num = 0
            best_same = best_other = -1.0
            for j in conf:
                if j == i:
                    continue
                tj = t[j]
                sa = fuzz.token_set_ratio(ad[ti], ad[tj]) if (ad[ti] and ad[tj]) else -1.0
                if sa >= 90:
                    ns_a += 1
                if fuzz.token_set_ratio(nm[ti], nm[tj]) >= 90:
                    ns_n += 1
                nj = set(adnum[tj].split())
                if nums_i and nj:
                    n_num += 1
                    agree += bool(nums_i & nj)
                if src[ti] == src[tj]:
                    best_same = max(best_same, sa)
                else:
                    best_other = max(best_other, sa)
            out[i, 7] = ns_a
            out[i, 8] = ns_n
            out[i, 9] = len(conf) - (1 if i in conf else 0)
            wa = wn = wsum = 0.0
            for j in support_pool:
                if j == i:
                    continue
                tj = t[j]
                w = float(p1[j])
                wsum += w
                if ad[ti] and ad[tj]:
                    wa += w * fuzz.token_set_ratio(ad[ti], ad[tj]) / 100.0
                wn += w * fuzz.token_set_ratio(nm[ti], nm[tj]) / 100.0
            out[i, 10] = wa / wsum if wsum > 0 else -1.0
            out[i, 11] = wn / wsum if wsum > 0 else -1.0
            out[i, 12] = agree / n_num if n_num else -1.0
            out[i, 13] = best_same
            out[i, 14] = best_other
            out[i, 15] = float(ns_a == 0 and ns_n == 0 and len(conf) > (1 if i in conf else 0))
    return out


# ----------------------------------------------------------------------------- context features

def context_feats(df, score_cols: Sequence[str]):
    """Add per-q and per-t competition features for each score column (polars)."""
    import polars as pl

    exprs = []
    for c in score_cols:
        exprs += [
            pl.col(c).rank("ordinal", descending=True).over("q").cast(pl.Int16).alias(f"{c}_qrank"),
            (pl.col(c).max().over("q") - pl.col(c)).alias(f"{c}_qgap"),
            pl.col(c).rank("ordinal", descending=True).over("t").cast(pl.Int16).alias(f"{c}_trank"),
            (pl.col(c).max().over("t") - pl.col(c)).alias(f"{c}_tgap"),
        ]
    exprs += [pl.len().over("q").cast(pl.Int32).alias("q_ncand"), pl.len().over("t").cast(pl.Int32).alias("t_nq")]
    df = df.with_columns(exprs)
    # margin vs the best OTHER S1 competing for the same target (many-to-one signal):
    # > 0 only when this q is the unique best S1 for t; equals the score when t has one S1.
    for c in score_cols:
        g = df.group_by("t").agg(pl.col(c).max().alias("_b1"), pl.col(c).top_k(2).min().alias("_b2"),
                                 pl.len().alias("_n"))
        df = df.join(g, on="t", how="left").with_columns(
            pl.when(pl.col(c) < pl.col("_b1")).then(pl.col(c) - pl.col("_b1"))
            .when(pl.col("_n") >= 2).then(pl.col(c) - pl.col("_b2"))
            .otherwise(pl.col(c)).alias(f"{c}_tmargin")
        ).drop(["_b1", "_b2", "_n"])
    return df
