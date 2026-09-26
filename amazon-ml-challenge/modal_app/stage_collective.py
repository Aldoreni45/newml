"""Stage 3b (stacking): add stage-1 probability p1 + collective/support features.

  modal run modal_app/stage_collective.py --split train --feat-tag v1 --s1-tag cv_v1
  modal run modal_app/stage_collective.py --split test  --feat-tag v1 --s1-tag cv_v1

train: p1 = OOF predictions on the cv folds + fold-averaged predictions on val folds
       (both written by stage_train.train_cv) -> no in-sample leakage into stage 2.
test : p1 = average of the stage-1 fold models' predictions.
Output: feat_<feat_tag>_col.parquet (all original columns + p1 + COLL_NAMES).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, ensure_dirs, ml_image, setup_path, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-collective")
G = {}


def _worker(rng):
    from ber.features import collective_batch

    s, e = rng
    return collective_batch((G["q"][s:e], G["t"][s:e], G["p1"][s:e], G["nm"], G["ad"], G["adnum"], G["src"]))


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=229376, timeout=10 * 3600)
def collective(split: str, feat_tag: str, s1_tag: str, cv_folds: str = "1,2,3", out_suffix: str = "",
               p1_competition: bool = False) -> dict:
    setup_path()
    import multiprocessing as mp

    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from ber.features import COLL_NAMES

    ensure_dirs()
    t0 = time.time()
    rec = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet", columns=["rid", "nm", "ad", "ad_num", "src"])
    F = pl.read_parquet(f"{ARTIFACTS}/{split}/feat_{feat_tag}.parquet")
    from ber.modelio import model_features

    folds_cv = [int(x) for x in cv_folds.split(",")]
    files = [f"{ARTIFACTS}/models/{s1_tag}_f{k}.txt" for k in folds_cv]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"stage-1 fold models missing: {missing}")

    def ensemble_p1(frame):
        p = np.zeros(frame.height)
        for fpath in files:
            m = lgb.Booster(model_file=fpath)
            fl = model_features(m, f"{ARTIFACTS}/models", os.path.basename(fpath)[:-4])
            p += m.predict(frame.select(fl).to_numpy().astype(np.float32)) / len(files)
        return p.astype(np.float32)

    if split == "train":
        # OOF p1 for the cv folds, fold-averaged p1 for the eval fold (both written by train_cv);
        # any OTHER fold present in the feature file (never trained on) gets the fold-model ensemble,
        # exactly like test, so per-target competition sees every S1 of the graph.
        P = pl.concat([pl.read_parquet(f"{ARTIFACTS}/models/{s1_tag}_oof.parquet"),
                       pl.read_parquet(f"{ARTIFACTS}/models/{s1_tag}_valpred.parquet")]).select(
            ["q", "t", pl.col("p").alias("p1")])
        F = F.join(P, on=["q", "t"], how="left")
        need = F["p1"].is_null()
        if need.any():
            Fx = F.filter(need)
            F = pl.concat([F.filter(~need), Fx.with_columns(pl.Series("p1", ensemble_p1(Fx)))])
    else:
        F = F.with_columns(pl.Series("p1", ensemble_p1(F)))
    folds_present = sorted(F["fold"].unique().to_list()) if "fold" in F.columns else None
    # per-target p1 competition is only meaningful when the file holds the WHOLE graph (all S1 folds);
    # with a fold-restricted train file it would differ from test -> skip it (consistently for test too).
    full_graph = (split == "test") or (folds_present is not None and len(folds_present) == 10)
    if p1_competition and not full_graph:
        raise ValueError("p1_competition needs train features for ALL folds (stage_features --folds ''), "
                         f"got folds {folds_present}; otherwise train and test would disagree")
    full_graph = full_graph and p1_competition
    F = F.sort(["q", "t"])
    G.update(q=F["q"].to_numpy(), t=F["t"].to_numpy(), p1=F["p1"].to_numpy(),
             nm=rec["nm"].to_numpy().astype(object), ad=rec["ad"].to_numpy().astype(object),
             adnum=rec["ad_num"].to_numpy().astype(object), src=rec["src"].to_numpy())
    q = G["q"]
    # chunk boundaries aligned to whole q-groups
    starts = np.r_[0, np.flatnonzero(np.diff(q)) + 1]
    step = max(1, len(starts) // 256)
    bounds = list(starts[::step]) + [len(q)]
    ranges = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    with mp.get_context("fork").Pool(32) as pool:
        parts = pool.map(_worker, ranges, chunksize=1)
    X = np.vstack(parts)
    F = F.with_columns([pl.Series(k, X[:, j]) for j, k in enumerate(COLL_NAMES)])
    # per-S1 p1 context (always consistent: all candidates of a q share its fold)
    F = F.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Int16).alias("p1_qrank"),
        (pl.col("p1").max().over("q") - pl.col("p1")).alias("p1_qgap"),
        pl.col("p1").sum().over("q").alias("p1_qsum"),
    )
    if full_graph:  # per-target competition with the refined score (needs every S1 of the graph)
        g = F.group_by("t").agg(pl.col("p1").max().alias("_b1"), pl.col("p1").top_k(2).min().alias("_b2"),
                                pl.len().alias("_n"))
        F = F.join(g, on="t", how="left").with_columns(
            pl.when(pl.col("p1") < pl.col("_b1")).then(pl.col("p1") - pl.col("_b1"))
            .when(pl.col("_n") >= 2).then(pl.col("p1") - pl.col("_b2")).otherwise(pl.col("p1")).alias("p1_tmargin"),
        ).drop(["_b1", "_b2", "_n"])
    out = f"{ARTIFACTS}/{split}/feat_{feat_tag}_col{out_suffix}.parquet"
    F.write_parquet(out)
    rep = {"split": split, "n": F.height, "t": time.time() - t0, "full_graph_p1_competition": full_graph,
           "folds_present": folds_present,
           "cols_added": ["p1"] + COLL_NAMES + ["p1_qrank", "p1_qgap", "p1_qsum"] + (["p1_tmargin"] if full_graph else [])}
    with open(f"{REPORTS}/collective_{split}_{feat_tag}{out_suffix}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def main(split: str = "train", feat_tag: str = "v1", s1_tag: str = "cv_v1", out_suffix: str = "",
         p1_competition: bool = False):
    print(json.dumps(collective.remote(split, feat_tag, s1_tag, "1,2,3", out_suffix, p1_competition), indent=1))
