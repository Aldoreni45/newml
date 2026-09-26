"""Stage 5: score test candidates, apply the decision rule, write + validate submission.

  modal run modal_app/stage_predict.py --feat-tag v1 --model-tag lgb_v1 --rule '{"type":"expF_excl","extra_miss":0.05}'

Writes /vol/submissions/<out_tag>/output/{matching_results.tsv,candidate_pairs.tsv},
runs the official validator (utils/validate_submission.py --check-ids) and returns stats.
candidate_pairs.tsv = exactly the (q, t) set scored by the model (spec: the LAST candidate set).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, DATA, REPORTS, VOL, ensure_dirs, ml_image, setup_path, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-predict")
SUB = f"{VOL}/submissions"


def apply_rule(q, t, p, rule):
    import numpy as np

    from ber.decide import exclusive_mask, expected_f_select

    kind = rule["type"]
    if kind == "thr":
        return p >= rule["tau"]
    excl = exclusive_mask(q, t, p)
    if kind == "thr_excl":
        return (p >= rule["tau"]) & excl
    if kind == "expF":
        return expected_f_select(q, p, extra_miss=rule.get("extra_miss", 0.0))
    if kind == "expF_excl":
        return expected_f_select(q, np.where(excl, p, 0.0), extra_miss=rule.get("extra_miss", 0.0))
    raise ValueError(kind)


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=196608, timeout=6 * 3600)
def predict(feat_tag: str, model_tag: str, rule_json: str, out_tag: str) -> dict:
    setup_path()
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from ber.io import write_id_list_file

    ensure_dirs()
    t0 = time.time()
    rule = json.loads(rule_json)
    rec = pl.read_parquet(f"{ARTIFACTS}/test/records.parquet", columns=["rid", "entity_id", "src", "country"])
    F = pl.read_parquet(f"{ARTIFACTS}/test/feat_{feat_tag}.parquet")
    tags = [x for x in model_tag.split(",") if x]  # several tags -> average (fold ensemble)
    p = np.zeros(F.height)
    from ber.modelio import model_features
    for mt in tags:
        model = lgb.Booster(model_file=f"{ARTIFACTS}/models/{mt}.txt")
        feats = model_features(model, f"{ARTIFACTS}/models", mt)
        p += model.predict(F.select(feats).to_numpy().astype(np.float32)) / len(tags)
    q, t = F["q"].to_numpy(), F["t"].to_numpy()
    sel = apply_rule(q, t, p, rule)
    ids = rec["entity_id"].to_numpy()
    s1 = rec.filter(pl.col("src") == 1)
    s1_ids = s1["entity_id"].to_list()
    P = pl.DataFrame({"q": ids[q], "t": ids[t], "p": p.astype(np.float32), "sel": sel})
    cand = {k: v for k, v in P.sort("p", descending=True).group_by("q", maintain_order=True)
            .agg(pl.col("t")).iter_rows()}
    match = {k: v for k, v in P.filter(pl.col("sel")).sort("p", descending=True)
             .group_by("q", maintain_order=True).agg(pl.col("t")).iter_rows()}
    out = f"{SUB}/{out_tag}/output"
    os.makedirs(out, exist_ok=True)
    write_id_list_file(f"{out}/matching_results.tsv", s1_ids, match, kind="matching")
    write_id_list_file(f"{out}/candidate_pairs.tsv", s1_ids, cand, kind="candidate")
    P.write_parquet(f"{SUB}/{out_tag}/test_scores.parquet")
    n_pred = P.filter(pl.col("sel")).group_by("q").len()
    s1c = s1.select(pl.col("entity_id").alias("q"), "country").join(n_pred, on="q", how="left").fill_null(0)
    rep = {
        "rule": rule, "model": model_tag, "feat": feat_tag, "n_s1": len(s1_ids), "n_cand_pairs": P.height,
        "n_pred_pairs": int(sel.sum()), "mean_pred_per_s1": float(sel.sum() / len(s1_ids)),
        "empty_rate": float((s1c["len"] == 0).mean()),
        "by_country": {str(k[0]): {"n": v.height, "mean_pred": float(v["len"].mean()),
                                   "empty_rate": float((v["len"] == 0).mean())}
                       for k, v in s1c.group_by(["country"])},
        "p_sum_per_s1": float(p.sum() / len(s1_ids)),
    }
    val = subprocess.run([sys.executable, f"{DATA}/utils/validate_submission.py", "--matching",
                          f"{out}/matching_results.tsv", "--candidate", f"{out}/candidate_pairs.tsv",
                          "--test-dir", f"{DATA}/test", "--check-ids"], capture_output=True, text=True)
    rep["validator_exit"] = val.returncode
    rep["ok"] = val.returncode == 0
    rep["validator_out"] = val.stdout[-4000:] + val.stderr[-2000:]
    rep["t_total"] = time.time() - t0
    with open(f"{SUB}/{out_tag}/predict_report.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def main(feat_tag: str = "v1", model_tag: str = "lgb_v1", rule: str = '{"type":"thr_excl","tau":0.5}',
         out_tag: str = "sub_v1"):
    rep = predict.remote(feat_tag, model_tag, rule, out_tag)
    print(json.dumps(rep, indent=1))
    if rep.get("validator_exit") != 0:
        raise SystemExit("OFFICIAL VALIDATOR FAILED - do not submit")
