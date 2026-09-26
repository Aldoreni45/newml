"""Stage 2: multi-view candidate generation (blocking) + recall diagnostics.

  modal run modal_app/stage_block.py --split train
  modal run modal_app/stage_block.py --split test

Output: /vol/artifacts/{split}/cands_{tag}.parquet with one row per (S1 rid q, target rid t)
        and per-view retrieval score s_<view> / rank r_<view> (0 / 999 when not retrieved).
        /vol/reports/block_{split}_{tag}.json (+ missed-pair samples for train).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, ensure_dirs, ml_image, setup_path, split_paths, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-block")

DEFAULT_CFG = {
    "K": {"name": 40, "cat": 20, "addr": 40, "combo": 40},
    "K_rev": 8,
    "max_df": {"name": 0.05, "cat": 0.05, "addr": 0.05},
    "w_name": 0.5,
    "tag": "v1",
}
VIEWS = ["name", "cat", "addr", "combo", "rev"]


def gt_rid_pairs(records):
    import polars as pl

    gt = pl.read_parquet(split_paths("train")["gt"])
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("t"))
             .explode("t").with_columns(pl.col("t").str.strip_chars()).filter(pl.col("t") != "")
             .select(pl.col("source1_entity_id").alias("s1"), "t"))
    ids = records.select(["entity_id", "rid"])
    return (pairs.join(ids.rename({"entity_id": "s1", "rid": "q"}), on="s1")
            .join(ids.rename({"entity_id": "t", "rid": "t_rid"}), on="t")
            .select(["q", pl.col("t_rid").alias("t")]))


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=131072, timeout=10 * 3600)
def block(split: str, cfg_json: str = "{}") -> dict:
    setup_path()
    import numpy as np
    import polars as pl

    from ber.blocking import hstack_views, ranks_from_sorted, tfidf_pair, topk

    ensure_dirs()
    cfg = {**DEFAULT_CFG, **json.loads(cfg_json)}
    tag = cfg["tag"]
    rep = {"cfg": cfg, "countries": {}}
    rec = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet",
                          columns=["rid", "entity_id", "src", "country", "nm", "ad"])
    part_paths = []
    t_all = time.time()
    for c in sorted(rec["country"].unique().to_list()):
        t0 = time.time()
        q = rec.filter((pl.col("src") == 1) & (pl.col("country") == c))
        t = rec.filter((pl.col("src") != 1) & (pl.col("country") == c))
        q_rid, t_rid = q["rid"].to_numpy(), t["rid"].to_numpy()
        crep = {"n_q": q.height, "n_t": t.height, "timing": {}}
        qn, tn = q["nm"].to_list(), t["nm"].to_list()
        Qn, Tn, vn = tfidf_pair(qn, tn, "char_wb", cfg["max_df"]["name"])
        Qc, Tc, vc = tfidf_pair([s.replace(" ", "") for s in qn], [s.replace(" ", "") for s in tn], "char",
                                cfg["max_df"]["cat"])
        Qa, Ta, va = tfidf_pair(q["ad"].to_list(), t["ad"].to_list(), "word", cfg["max_df"]["addr"])
        w = cfg["w_name"]
        Qx, Tx = hstack_views([(Qn, w), (Qa, 1 - w)]), hstack_views([(Tn, w), (Ta, 1 - w)])
        crep["vocab"] = {"name": vn, "cat": vc, "addr": va}
        crep["timing"]["vectorize"] = time.time() - t0
        print(f"[{c}] n_q={q.height} n_t={t.height} vectorized in {crep['timing']['vectorize']:.0f}s", flush=True)
        frames = []
        for view, (A, B) in (("name", (Qn, Tn)), ("cat", (Qc, Tc)), ("addr", (Qa, Ta)), ("combo", (Qx, Tx))):
            t1 = time.time()
            qi, ti, sc = topk(A, B, cfg["K"][view])
            rk = ranks_from_sorted(qi)
            frames.append(pl.DataFrame({"q": q_rid[qi], "t": t_rid[ti], f"s_{view}": sc, f"r_{view}": rk}))
            crep["timing"][view] = time.time() - t1
            print(f"[{c}] view {view}: {len(qi)} pairs in {crep['timing'][view]:.0f}s", flush=True)
        t1 = time.time()
        ti, qi, sc = topk(Tx, Qx, cfg["K_rev"])  # target-side retrieval of S1 records
        rk = ranks_from_sorted(ti)
        frames.append(pl.DataFrame({"q": q_rid[qi], "t": t_rid[ti], "s_rev": sc, "r_rev": rk}))
        crep["timing"]["rev"] = time.time() - t1
        U = frames[0]
        for f in frames[1:]:
            U = U.join(f, on=["q", "t"], how="full", coalesce=True)
        U = U.with_columns(
            [pl.col(f"s_{v}").fill_null(0.0) for v in VIEWS] + [pl.col(f"r_{v}").fill_null(999).cast(pl.Int16) for v in VIEWS]
        )
        crep["n_cands"] = U.height
        crep["cands_per_q"] = U.height / max(q.height, 1)
        p = f"{ARTIFACTS}/{split}/cands_{tag}_{c}.parquet"
        U.write_parquet(p)
        part_paths.append(p)
        crep["timing"]["total"] = time.time() - t0
        rep["countries"][c] = crep
        del Qn, Tn, Qc, Tc, Qa, Ta, Qx, Tx, frames, U
    C = pl.concat([pl.read_parquet(p) for p in part_paths])
    C.write_parquet(f"{ARTIFACTS}/{split}/cands_{tag}.parquet")
    for p in part_paths:
        os.remove(p)
    rep["n_cands"] = C.height
    rep["n_q"] = int((rec["src"] == 1).sum())
    rep["cands_per_q"] = C.height / rep["n_q"]
    rep["time_total"] = time.time() - t_all

    if split == "train":
        G = gt_rid_pairs(rec).with_columns(pl.lit(1).cast(pl.Int8).alias("y"))
        J = G.join(C, on=["q", "t"], how="left")
        rec_rep = {}
        for v in VIEWS:
            r = J[f"r_{v}"].fill_null(999).to_numpy()
            rec_rep[v] = {f"R@{k}": float((r < k).mean()) for k in (1, 3, 5, 10, 20, 30, 40)}
        hit = J["s_name"].is_not_null().to_numpy()
        rec_rep["union"] = float(hit.mean())
        # union recall / volume under smaller uniform K grids
        grid = {}
        for k in (5, 10, 15, 20, 30, 40):
            m = pl.lit(False)
            for v in ("name", "cat", "addr", "combo"):
                m = m | (pl.col(f"r_{v}") < min(k, cfg["K"][v]))
            m = m | (pl.col("r_rev") < cfg["K_rev"])
            n_c = C.filter(m).height
            jr = J.filter(pl.col("s_name").is_not_null()).filter(m).height
            grid[str(k)] = {"pair_recall": jr / G.height, "cands_per_q": n_c / rep["n_q"]}
        rec_rep["grid_uniformK"] = grid
        # entity-level recall and oracle macro F0.5
        per = (J.group_by("q").agg(pl.len().alias("n_true"), pl.col("s_name").is_not_null().sum().alias("hit")))
        s1 = rec.filter(pl.col("src") == 1).select(pl.col("rid").alias("q"), "country")
        per = s1.join(per, on="q", how="left").fill_null(0)
        R = (per["hit"] / per["n_true"]).fill_nan(0.0)
        f = pl.when(pl.col("n_true") == 0).then(1.0).otherwise(
            1.25 * (pl.col("hit") / pl.col("n_true")) / (0.25 + pl.col("hit") / pl.col("n_true")))
        per = per.with_columns(f.fill_nan(0.0).alias("oracle"))
        rec_rep["entity_full_recall"] = float(per.filter(pl.col("n_true") > 0).select(
            (pl.col("hit") == pl.col("n_true")).mean()).item())
        rec_rep["oracle_macro_f05"] = float(per["oracle"].mean())
        rec_rep["oracle_by_country"] = {str(k[0]): float(v["oracle"].mean()) for k, v in per.group_by(["country"])}
        by_src = J.join(
            rec.select(pl.col("rid").alias("t"), pl.col("src").alias("tsrc"), "country"), on="t")
        rec_rep["recall_by_src_country"] = {
            f"S{k[0]}|{k[1]}": float(v["s_name"].is_not_null().mean()) for k, v in by_src.group_by(["tsrc", "country"])}
        rep["recall"] = rec_rep
        # sample misses for error analysis
        miss = J.filter(pl.col("s_name").is_null()).sample(n=min(300, J.filter(pl.col("s_name").is_null()).height), seed=1)
        full = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet", columns=["rid", "name", "address", "nm", "ad"])
        mm = (miss.join(full.rename({"rid": "q", "name": "q_name", "address": "q_addr", "nm": "q_nm", "ad": "q_ad"}), on="q")
              .join(full.rename({"rid": "t", "name": "t_name", "address": "t_addr", "nm": "t_nm", "ad": "t_ad"}), on="t"))
        with open(f"{REPORTS}/block_{split}_{tag}_misses.txt", "w", encoding="utf-8") as fh:
            for r in mm.iter_rows(named=True):
                fh.write(f"S1: {r['q_name']} | {r['q_addr']}\n  -> {r['q_nm']} | {r['q_ad']}\n"
                         f"T : {r['t_name']} | {r['t_addr']}\n  -> {r['t_nm']} | {r['t_ad']}\n\n")
    with open(f"{REPORTS}/block_{split}_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=65536, timeout=3 * 3600)
def bench(country: str = "India", frac: float = 0.1, cfg_json: str = "{}") -> dict:
    """Timing + recall of each view on a random `frac` of one country's records (train)."""
    setup_path()
    import numpy as np
    import polars as pl

    from ber.blocking import hstack_views, ranks_from_sorted, tfidf_pair, topk

    cfg = {**DEFAULT_CFG, **json.loads(cfg_json)}
    rec = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet",
                          columns=["rid", "entity_id", "src", "country", "nm", "ad"])
    G = gt_rid_pairs(rec)
    s1 = rec.filter((pl.col("src") == 1) & (pl.col("country") == country)).sample(fraction=frac, seed=0)
    Gs = G.join(s1.select(pl.col("rid").alias("q")), on="q")
    # targets: all true targets of the sampled S1 + the same fraction of all other targets (density kept)
    tt = rec.filter((pl.col("src") != 1) & (pl.col("country") == country))
    t = pl.concat([tt.join(Gs.select(pl.col("t").alias("rid")), on="rid"),
                   tt.sample(fraction=frac, seed=1)]).unique("rid")
    q_rid, t_rid = s1["rid"].to_numpy(), t["rid"].to_numpy()
    out = {"n_q": s1.height, "n_t": t.height, "timing": {}, "recall": {}}
    t0 = time.time()
    qn, tn = s1["nm"].to_list(), t["nm"].to_list()
    Qn, Tn, _ = tfidf_pair(qn, tn, "char_wb", cfg["max_df"]["name"])
    Qc, Tc, _ = tfidf_pair([s.replace(" ", "") for s in qn], [s.replace(" ", "") for s in tn], "char", cfg["max_df"]["cat"])
    Qa, Ta, _ = tfidf_pair(s1["ad"].to_list(), t["ad"].to_list(), "word", cfg["max_df"]["addr"])
    w = cfg["w_name"]
    Qx, Tx = hstack_views([(Qn, w), (Qa, 1 - w)]), hstack_views([(Tn, w), (Ta, 1 - w)])
    out["timing"]["vectorize"] = time.time() - t0
    gkey = set(zip(Gs["q"].to_list(), Gs["t"].to_list()))
    union = set()
    for view, (A, B, k) in (("name", (Qn, Tn, cfg["K"]["name"])), ("cat", (Qc, Tc, cfg["K"]["cat"])),
                            ("addr", (Qa, Ta, cfg["K"]["addr"])), ("combo", (Qx, Tx, cfg["K"]["combo"])),
                            ("rev", (Tx, Qx, cfg["K_rev"]))):
        t1 = time.time()
        qi, ti, sc = topk(A, B, k)
        out["timing"][view] = time.time() - t1
        if view == "rev":
            pairs = set(zip(q_rid[ti].tolist(), t_rid[qi].tolist()))
        else:
            pairs = set(zip(q_rid[qi].tolist(), t_rid[ti].tolist()))
        union |= pairs
        out["recall"][view] = len(gkey & pairs) / max(len(gkey), 1)
        print(view, out["timing"][view], out["recall"][view], flush=True)
    out["recall"]["union"] = len(gkey & union) / max(len(gkey), 1)
    out["cands_per_q"] = len(union) / max(s1.height, 1)
    return out


@app.local_entrypoint()
def benchmark(country: str = "India", frac: float = 0.1, cfg: str = "{}"):
    print(json.dumps(bench.remote(country, frac, cfg), indent=1))


@app.local_entrypoint()
def main(split: str = "train", cfg: str = "{}"):
    rep = block.remote(split, cfg)
    print(json.dumps(rep, indent=1)[:12000])
