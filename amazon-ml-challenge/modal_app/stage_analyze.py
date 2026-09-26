"""Error analysis on validation selections.

  modal run modal_app/stage_analyze.py --tag cv_v1 --feat-tag v1

Inputs : /vol/artifacts/models/<tag>_valsel.parquet (q, t, y, p, sel) + feat parquet + records
Outputs: /vol/reports/errors_<tag>.json (taxonomy counts) and errors_<tag>_samples.txt
Taxonomy detectors are the label-free features listed in ERROR_ANALYSIS.md.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, ensure_dirs, ml_image, setup_path, split_paths, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-analyze")


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=16, memory=131072, timeout=3 * 3600)
def analyze(tag: str, feat_tag: str, n_samples: int = 150, tau: float = 0.7) -> dict:
    setup_path()
    import numpy as np
    import polars as pl

    from ber.decide import exclusive_mask

    ensure_dirs()
    if os.path.exists(f"{ARTIFACTS}/models/{tag}_valsel.parquet"):
        S = pl.read_parquet(f"{ARTIFACTS}/models/{tag}_valsel.parquet")
    else:  # single-split run: rebuild the selection with threshold + exclusivity
        S = pl.read_parquet(f"{ARTIFACTS}/models/{tag}_valpred.parquet")
        m = exclusive_mask(S["q"].to_numpy(), S["t"].to_numpy(), S["p"].to_numpy()) & (S["p"].to_numpy() >= tau)
        S = S.with_columns(pl.Series("sel", m))
    cols = ["q", "t", "t_nm_scr", "a_t_empty", "t_has_alias", "t_has_dom", "q_nm_freq_s1", "n_initials",
            "n_trunc_prefix", "n_tset", "a_tset", "c_combo_tmargin", "c_combo_trank", "n_core_eq", "t_src"]
    F = pl.read_parquet(f"{ARTIFACTS}/train/feat_{feat_tag}.parquet", columns=cols)
    D = S.join(F, on=["q", "t"], how="left")
    rec = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet", columns=["rid", "entity_id", "name", "address", "country"])
    D = D.join(rec.select(pl.col("rid").alias("q"), "country"), on="q", how="left")
    fp = D.filter(pl.col("sel") & (pl.col("y") == 0))
    fn_in = D.filter(~pl.col("sel") & (pl.col("y") == 1))
    tp = D.filter(pl.col("sel") & (pl.col("y") == 1))
    det = {
        "t_addr_empty": pl.col("a_t_empty") > 0.5,
        "cross_script": pl.col("t_nm_scr").is_in([1, 2]),
        "alias": pl.col("t_has_alias") > 0,
        "handle_domain": pl.col("t_has_dom") > 0,
        "shared_s1_name": pl.col("q_nm_freq_s1") > 1,
        "acronym": pl.col("n_initials") > 0.5,
        "truncated": pl.col("n_trunc_prefix") > 0.5,
        "low_name_sim(<50)": pl.col("n_tset") < 50,
        "low_addr_sim(<50)": pl.col("a_tset") < 50,
        "not_best_for_target": pl.col("c_combo_trank") > 1,
        "core_name_equal": pl.col("n_core_eq") > 0.5,
    }
    rep = {"tag": tag, "n_fp": fp.height, "n_fn_in_candidates": fn_in.height, "n_tp": tp.height, "taxonomy": {}}
    for name, e in det.items():
        rep["taxonomy"][name] = {
            "fp": int(fp.filter(e).height), "fn": int(fn_in.filter(e).height), "tp": int(tp.filter(e).height),
            "fp_share": fp.filter(e).height / max(fp.height, 1), "fn_share": fn_in.filter(e).height / max(fn_in.height, 1),
        }
    rep["by_country"] = {str(k[0]): {"fp": int((v["sel"] & (v["y"] == 0)).sum()),
                                     "fn": int((~v["sel"] & (v["y"] == 1)).sum()),
                                     "tp": int((v["sel"] & (v["y"] == 1)).sum())} for k, v in D.group_by(["country"])}
    rep["by_src"] = {str(k[0]): {"fp": int((v["sel"] & (v["y"] == 0)).sum()),
                                 "fn": int((~v["sel"] & (v["y"] == 1)).sum())} for k, v in D.group_by(["t_src"])}
    # singleton false merges: S1 with no true match but >=1 prediction
    per_q = D.group_by("q").agg(pl.col("y").sum().alias("ny"), pl.col("sel").sum().alias("ns"))
    gt = pl.read_parquet(split_paths("train")["gt"]).select(
        pl.col("source1_entity_id").alias("entity_id"),
        (pl.col("matched_entity_ids").str.strip_chars() != "").alias("has_match"))
    per_q = per_q.join(rec.select(pl.col("rid").alias("q"), "entity_id").join(gt, on="entity_id"), on="q", how="left")
    rep["singletons_with_pred"] = int(per_q.filter(~pl.col("has_match") & (pl.col("ns") > 0)).height)
    rep["s1_with_matches_but_empty_pred"] = int(per_q.filter((pl.col("ny") > 0) & (pl.col("ns") == 0)).height)
    # samples
    names = rec.select(pl.col("rid"), "name", "address")

    def dump(frame, title, fh):
        fh.write(f"\n==================== {title} ({frame.height}) ====================\n")
        smp = frame.sample(n=min(n_samples, frame.height), seed=3) if frame.height else frame
        smp = (smp.join(names.rename({"rid": "q", "name": "q_name", "address": "q_addr"}), on="q")
               .join(names.rename({"rid": "t", "name": "t_name", "address": "t_addr"}), on="t"))
        for r in smp.iter_rows(named=True):
            fh.write(f"p={r['p']:.3f} y={r['y']} tset={r['n_tset']:.0f} atset={r['a_tset']:.0f} "
                     f"trank={r['c_combo_trank']} freq={r['q_nm_freq_s1']} [{r['country']}]\n"
                     f"  S1: {r['q_name']} | {r['q_addr']}\n  T : {r['t_name']} | {r['t_addr']}\n")
            if r["y"] == 0:  # show what the true matches of this S1 look like
                pass

    with open(f"{REPORTS}/errors_{tag}_samples.txt", "w", encoding="utf-8") as fh:
        dump(fp, "FALSE POSITIVES", fh)
        dump(fn_in, "FALSE NEGATIVES (in candidates)", fh)
    with open(f"{REPORTS}/errors_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def main(tag: str = "cv_v1", feat_tag: str = "v1"):
    print(json.dumps(analyze.remote(tag, feat_tag), indent=1))
