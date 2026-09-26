"""Exact reproduction of the challenge metric + diagnostics.

Official definition (problem statement, "Evaluation Criteria"):
  * F_0.5 = 1.25*P*R / (0.25*P + R), computed PER Source-1 entity, then macro-averaged
    over ALL Source-1 entities in the evaluation set.
  * Singletons (no true matches): 1.0 if the predicted list is empty, else 0.0.
  * Worked example: pred {S2-47,S2-193,S3-812}, truth {S2-47,S3-812} -> 0.714.

Conventions for the undefined cases (standard, and the only ones consistent with the
statement): truth non-empty & prediction empty -> 0 (recall 0); no overlap -> 0.

Equivalent count form used below (avoids division-by-zero branches):
  F_0.5 = 1.25*TP / (1.25*TP + 0.25*FN + FP)
i.e. one false positive costs 4x one false negative in the denominator.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Optional, Set


def f05(pred: Set[str], true: Set[str], beta: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    b2 = beta * beta
    fp = len(pred) - tp
    fn = len(true) - tp
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def macro_f05(
    pred: Mapping[str, Set[str]],
    truth: Mapping[str, Set[str]],
    ids: Optional[Iterable[str]] = None,
) -> float:
    ids = list(truth.keys()) if ids is None else list(ids)
    if not ids:
        return float("nan")
    return sum(f05(pred.get(i, set()), truth.get(i, set())) for i in ids) / len(ids)


def _bucket(n: int) -> str:
    if n == 0:
        return "0(singleton)"
    if n <= 5:
        return str(n)
    if n <= 10:
        return "6-10"
    return "11+"


def evaluate(
    pred: Mapping[str, Set[str]],
    truth: Mapping[str, Set[str]],
    ids: Optional[Iterable[str]] = None,
    groups: Optional[Mapping[str, str]] = None,
) -> Dict:
    """Full diagnostic report.

    groups: optional {s1_id: label} (e.g. country) for per-group macro scores.
    """
    ids = list(truth.keys()) if ids is None else list(ids)
    n = len(ids)
    tot = 0.0
    by_bucket = defaultdict(lambda: [0.0, 0])
    by_group = defaultdict(lambda: [0.0, 0])
    sum_p = sum_r = 0.0
    n_p = n_r = 0
    tp_all = fp_all = fn_all = 0
    singles = single_ok = 0
    nonsingle_empty_pred = 0
    perfect = 0
    for i in ids:
        p = pred.get(i, set())
        t = truth.get(i, set())
        s = f05(p, t)
        tot += s
        perfect += s == 1.0
        b = by_bucket[_bucket(len(t))]
        b[0] += s
        b[1] += 1
        if groups is not None:
            g = by_group[groups.get(i, "?")]
            g[0] += s
            g[1] += 1
        tp = len(p & t)
        tp_all += tp
        fp_all += len(p) - tp
        fn_all += len(t) - tp
        if p:
            sum_p += tp / len(p)
            n_p += 1
        if t:
            sum_r += tp / len(t)
            n_r += 1
            if not p:
                nonsingle_empty_pred += 1
        else:
            singles += 1
            single_ok += not p
    rep = {
        "macro_f05": tot / n if n else float("nan"),
        "n_entities": n,
        "exact_entity_rate": perfect / n if n else float("nan"),
        "macro_precision_over_nonempty_preds": sum_p / n_p if n_p else float("nan"),
        "macro_recall_over_nonsingletons": sum_r / n_r if n_r else float("nan"),
        "pair_precision": tp_all / (tp_all + fp_all) if tp_all + fp_all else float("nan"),
        "pair_recall": tp_all / (tp_all + fn_all) if tp_all + fn_all else float("nan"),
        "pair_tp": tp_all,
        "pair_fp": fp_all,
        "pair_fn": fn_all,
        "singleton_count": singles,
        "singleton_accuracy": single_ok / singles if singles else float("nan"),
        "nonsingleton_empty_pred": nonsingle_empty_pred,
        "by_true_count": {k: {"f05": v[0] / v[1], "n": v[1]} for k, v in sorted(by_bucket.items())},
    }
    if groups is not None:
        rep["by_group"] = {k: {"f05": v[0] / v[1], "n": v[1]} for k, v in sorted(by_group.items())}
    return rep


def candidate_report(
    cands: Mapping[str, Set[str]],
    truth: Mapping[str, Set[str]],
    ids: Optional[Iterable[str]] = None,
    n_targets: Optional[int] = None,
) -> Dict:
    """Blocking diagnostics: recall ceiling, candidate volume, oracle F0.5.

    oracle_f05 = best achievable macro F0.5 if the matcher picked exactly the true
    matches inside the candidate set (and predicted empty for singletons).
    n_targets: total # of S2+S3 records, for the reduction ratio.
    """
    ids = list(truth.keys()) if ids is None else list(ids)
    tot_c = tp = tt = 0
    full = nonsingle = 0
    oracle = 0.0
    for i in ids:
        c = cands.get(i, set())
        t = truth.get(i, set())
        tot_c += len(c)
        hit = len(c & t)
        tp += hit
        tt += len(t)
        if t:
            nonsingle += 1
            full += hit == len(t)
            oracle += f05(c & t, t)
        else:
            oracle += 1.0
    n = len(ids)
    rep = {
        "pair_recall": tp / tt if tt else float("nan"),
        "entity_full_recall": full / nonsingle if nonsingle else float("nan"),
        "oracle_macro_f05": oracle / n if n else float("nan"),
        "mean_candidates": tot_c / n if n else float("nan"),
        "total_candidates": tot_c,
    }
    if n_targets:
        rep["reduction_ratio"] = 1.0 - tot_c / (n * n_targets)
    return rep


def _selftest() -> None:
    ex = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(ex - 0.7142857) < 1e-6, ex
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0
    assert f05({"S2-2"}, {"S2-1"}) == 0.0
    assert f05({"S2-1"}, {"S2-1", "S2-2"}) == 1.25 * 0.5 / (0.25 + 0.5)
    # macro over singleton + example
    truth = {"a": set(), "b": {"S2-00047", "S3-00812"}}
    pred = {"a": set(), "b": {"S2-00047", "S2-00193", "S3-00812"}}
    assert abs(macro_f05(pred, truth) - (1 + ex) / 2) < 1e-12
    print("metric selftest OK; spec example =", round(ex, 3))


if __name__ == "__main__":
    _selftest()
