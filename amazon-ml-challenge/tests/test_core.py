"""Unit tests for the parts that must be exactly right (metric, submission writer,
normalization invariants, decision rules). Run: python -m pytest tests -q  (or python tests/test_core.py)."""
from __future__ import annotations

import os
import random
import sys
import tempfile

for _p in (os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "src")):
    sys.path.insert(0, _p)  # repo layout (src/ber) or package layout (src/tests next to src/ber)

from ber import metric  # noqa: E402
from ber.io import read_id_list_file, write_id_list_file  # noqa: E402
from ber.normalize import basic_clean, phonetic, translit_indic  # noqa: E402
from ber import records as R  # noqa: E402


def test_metric_spec_example():
    f = metric.f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(f - 0.7142857142857143) < 1e-12


def test_metric_singletons_and_edges():
    assert metric.f05(set(), set()) == 1.0
    assert metric.f05({"S2-1"}, set()) == 0.0
    assert metric.f05(set(), {"S2-1"}) == 0.0
    assert metric.f05({"S2-9"}, {"S2-1"}) == 0.0


def test_metric_count_form_matches_pr_form():
    rnd = random.Random(0)
    for _ in range(2000):
        t = {f"S2-{i}" for i in range(rnd.randint(1, 8))}
        p = {f"S2-{i}" for i in rnd.sample(range(12), rnd.randint(1, 10))}
        tp = len(p & t)
        if tp == 0:
            assert metric.f05(p, t) == 0.0
            continue
        P, Rc = tp / len(p), tp / len(t)
        assert abs(metric.f05(p, t) - 1.25 * P * Rc / (0.25 * P + Rc)) < 1e-12


def test_writer_rules():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.tsv")
        write_id_list_file(path, ["S1-1", "S1-2", "S1-3"],
                           {"S1-1": ["S2-5", "S2-5", "S3-7", "S1-9"], "S1-2": []}, kind="matching")
        lines = open(path, encoding="utf-8").read().split("\n")
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-1\tS2-5,S3-7"          # dedup + S1 self-match removed
        assert lines[2] == "S1-2\t" and lines[3] == "S1-3\t"  # empty rows, every S1 present
        back = read_id_list_file(path)
        assert back == {"S1-1": {"S2-5", "S3-7"}, "S1-2": set(), "S1-3": set()}
        try:
            write_id_list_file(path, ["S1-1", "S1-1"], {}, kind="candidate")
            raise AssertionError("duplicate S1 rows must raise")
        except ValueError:
            pass


def test_normalization_invariants():
    assert translit_indic("राम") == "ram"
    assert basic_clean("Smith Lane, <NULL>, N/A") == "smith lane"
    assert basic_clean("E.E. Johnson L.L.C.") == "ee johnson llc"
    assert basic_clean("N°37 Rue") == "n 37 rue"
    assert phonetic("shree") == phonetic("sri")
    assert phonetic("laxmi") == phonetic("lakshmi")
    nm, nm_a, nm_b, dom, scr = R.norm_name_fields("Kelonylalum a/k/a Allied Federation")
    assert nm_b == "allied federation" and nm_a == "kelonylalum"
    assert R.norm_name_fields("@parabolalaw")[3] == "parabolalaw"


def test_decision_rules():
    try:
        import numpy as np
    except ImportError:
        return
    from ber.decide import expected_f_select

    q = np.array([0, 0, 0])
    assert expected_f_select(q, np.array([0.95, 0.9, 0.2])).tolist() == [True, True, False]
    assert expected_f_select(np.array([0]), np.array([0.4])).tolist() == [False]  # empty beats a 40% guess
    try:
        import polars  # noqa: F401
    except ImportError:
        return
    from ber.decide import exclusive_mask, macro_f05_frame

    m = exclusive_mask(np.array([1, 2, 3]), np.array([10, 10, 11]), np.array([0.9, 0.8, 0.5]))
    assert m.tolist() == [True, False, True]
    f, _ = macro_f05_frame(np.array([1, 1, 1]), np.array([1, 0, 1]), np.array([1, 2]), np.array([2, 0]))
    assert abs(f - (0.7142857142857143 + 1) / 2) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
