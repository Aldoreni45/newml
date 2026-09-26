"""Stage 3: pair features for candidate pairs.

  modal run modal_app/stage_features.py --split train --folds 0,1,2,3
  modal run modal_app/stage_features.py --split test

Steps
 1. rebuild TF-IDF views per country (same settings as blocking) and compute EXACT
    cosines for every candidate pair (name char3, concat char3, name word, address word).
 2. competition/context features over the FULL candidate graph (label-free).
 3. restrict to the requested S1 folds (fold = numeric S1 id % 10; train only).
 4. RapidFuzz + token-level features, labels (train).
Output: /vol/artifacts/{split}/feat_{tag}.parquet
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, ensure_dirs, ml_image, setup_path, split_paths, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-features")

# globals used by forked workers
G = {}


def _tok_worker(rng):
    import numpy as np
    from ber.features import token_feats_batch

    s, e = rng
    qi, ti = G["qi"][s:e], G["ti"][s:e]
    R = G["R"]
    args = (R["nm"][qi].tolist(), R["nm"][ti].tolist(), R["nm_a"][ti].tolist(), R["nm_b"][ti].tolist(),
            R["ad_num"][qi].tolist(), R["ad_num"][ti].tolist(), R["ad"][qi].tolist(), R["ad"][ti].tolist(),
            G["idf_n"], G["idf_a"], G["generic"])
    return token_feats_batch(args)


def token_feats_parallel(qi, ti, workers=32, chunk=200_000):
    import multiprocessing as mp

    import numpy as np

    G["qi"], G["ti"] = qi, ti
    ranges = [(s, min(s + chunk, len(qi))) for s in range(0, len(qi), chunk)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_tok_worker, ranges, chunksize=1)
    return np.vstack(parts) if parts else np.zeros((0, 0), np.float32)


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=196608, timeout=12 * 3600)
def features(split: str, cands_tag: str = "v1", tag: str = "v1", folds: str = "", w_name: float = 0.5,
             max_df: float = 0.05, prune_json: str = "", use_emb: bool = True, drop_json: str = "") -> dict:
    setup_path()
    import numpy as np
    import polars as pl

    from ber.blocking import hstack_views, tfidf_pair
    from ber.features import TOK_NAMES, context_feats, fuzz_feats, rowwise_cos, token_idf

    ensure_dirs()
    t0 = time.time()
    rep = {"split": split, "cands_tag": cands_tag, "tag": tag, "folds": folds, "timing": {}}
    rec = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet")
    C = pl.read_parquet(f"{ARTIFACTS}/{split}/cands_{cands_tag}.parquet")
    rep["n_pairs_all"] = C.height
    if prune_json:  # per-view rank caps, e.g. {"rev": 8, "combo": 20, "name": 10, "addr": 10, "cat": 5}
        caps = json.loads(prune_json)
        m = pl.lit(False)
        for v, k in caps.items():
            m = m | (pl.col(f"r_{v}") < k)
        C = C.filter(m)
        rep["prune"] = caps
        rep["n_pairs_pruned"] = C.height
    if drop_json:  # P-dense: remove S1 records (from non-eval folds) so their targets become orphans
        dcfg = json.loads(drop_json)
        s1f = rec.filter(pl.col("src") == 1).select(
            "rid", (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 10).alias("fold"),
            ((pl.col("entity_id").str.slice(3).cast(pl.Int64) // 10) % 1000).alias("h"))
        dropped = s1f.filter(pl.col("fold").is_in(dcfg["folds"]) & (pl.col("h") < int(1000 * dcfg["frac"])))["rid"]
        C = C.filter(~pl.col("q").is_in(dropped))
        # the reverse view ranked S1s per target INCLUDING the dropped owner: re-rank among survivors
        C = C.with_columns(
            pl.when(pl.col("r_rev") < 999)
            .then(pl.col("s_rev").rank("ordinal", descending=True).over(
                pl.col("t"), pl.col("r_rev") < 999).cast(pl.Int16) - 1)
            .otherwise(pl.col("r_rev")).alias("r_rev"))
        dropped_rids = set(dropped.to_list())
        rep["dropped_s1"] = int(dropped.len())
        rep["n_pairs_after_drop"] = C.height
    else:
        dropped_rids = set()
    # rid == row index in records (records.parquet is sorted by rid, rid = 0..n-1)
    assert (rec["rid"].to_numpy() == np.arange(rec.height)).all()
    R = {k: rec[k].to_numpy().astype(object) for k in ("nm", "nm_a", "nm_b", "ad", "ad_num", "nm_dom")}
    G["R"] = R
    country = rec["country"].to_numpy()
    src = rec["src"].to_numpy()

    # ---------------- 1. exact cosines per country --------------------------------------
    qa_all, ta_all = C["q"].to_numpy(), C["t"].to_numpy()
    cos = {k: np.zeros(C.height, np.float32) for k in ("c_name", "c_cat", "c_nword", "c_addr")}
    for c in np.unique(country):
        m_pairs = country[qa_all] == c
        if not m_pairs.any():
            continue
        q_rid = np.flatnonzero((src == 1) & (country == c))
        t_rid = np.flatnonzero((src != 1) & (country == c))
        pos_q = np.full(rec.height, -1, np.int64)
        pos_q[q_rid] = np.arange(len(q_rid))
        pos_t = np.full(rec.height, -1, np.int64)
        pos_t[t_rid] = np.arange(len(t_rid))
        qn, tn = R["nm"][q_rid].tolist(), R["nm"][t_rid].tolist()
        views = {
            "c_name": tfidf_pair(qn, tn, "char_wb", max_df),
            "c_cat": tfidf_pair([s.replace(" ", "") for s in qn], [s.replace(" ", "") for s in tn], "char", max_df),
            "c_nword": tfidf_pair(qn, tn, "word", 0.5),
            "c_addr": tfidf_pair(R["ad"][q_rid].tolist(), R["ad"][t_rid].tolist(), "word", max_df),
        }
        idx = np.flatnonzero(m_pairs)
        qi, ti = pos_q[qa_all[idx]], pos_t[ta_all[idx]]
        for k, (Q, T, _) in views.items():
            cos[k][idx] = rowwise_cos(Q, T, qi, ti)
        del views
    C = C.with_columns([pl.Series(k, v) for k, v in cos.items()])
    C = C.with_columns((w_name * pl.col("c_name") + (1 - w_name) * pl.col("c_addr")).alias("c_combo"))
    # optional static multilingual embedding cosines (potion-multilingual-128M, MIT)
    ctx_cols = ["c_combo", "c_name", "c_addr"]
    if use_emb and os.path.exists(f"{ARTIFACTS}/{split}/emb_pot_full.npy"):
        for kind in ("name", "addr", "full"):
            E = np.load(f"{ARTIFACTS}/{split}/emb_pot_{kind}.npy", mmap_mode="r")
            E = np.asarray(E)
            v = np.empty(C.height, np.float32)
            for s in range(0, C.height, 5_000_000):
                a = E[qa_all[s:s + 5_000_000]].astype(np.float32)
                b = E[ta_all[s:s + 5_000_000]].astype(np.float32)
                v[s:s + 5_000_000] = (a * b).sum(1)
            C = C.with_columns(pl.Series(f"e_{kind}", v))
            del E
        ctx_cols.append("e_full")
        rep["embeddings"] = "potion-multilingual-128M"
    rep["timing"]["cosines"] = time.time() - t0
    print("cosines done", round(time.time() - t0), "s", flush=True)

    # ---------------- 2. context over the full graph --------------------------------------
    C = context_feats(C, ctx_cols)
    rep["timing"]["context"] = time.time() - t0
    print("context done", round(time.time() - t0), "s", flush=True)

    # ---------------- 3. fold restriction --------------------------------------------------
    s1_fold = rec.select(pl.col("rid").alias("q"),
                         (pl.col("entity_id").str.slice(3).cast(pl.Int64) % 10).cast(pl.Int8).alias("fold"))
    C = C.join(s1_fold, on="q", how="left")
    if folds:
        fl = [int(x) for x in folds.split(",")]
        C = C.filter(pl.col("fold").is_in(fl))
    rep["n_pairs"] = C.height
    qi, ti = C["q"].to_numpy(), C["t"].to_numpy()

    # ---------------- 4. string + token features ------------------------------------------
    cat = np.array([s.replace(" ", "") for s in R["nm"]], dtype=object)
    ff = fuzz_feats(R["nm"][qi].tolist(), R["nm"][ti].tolist(), cat[qi].tolist(), cat[ti].tolist(),
                    R["ad"][qi].tolist(), R["ad"][ti].tolist())
    C = C.with_columns([pl.Series(k, v) for k, v in ff.items()])
    rep["timing"]["fuzz"] = time.time() - t0
    print("fuzz done", round(time.time() - t0), "s", flush=True)
    tok = np.zeros((C.height, len(TOK_NAMES)), np.float32)
    pc = country[qi]
    for c in np.unique(pc):
        m = np.flatnonzero(pc == c)
        rid_c = np.flatnonzero(country == c)
        G["idf_n"] = token_idf(R["nm"][rid_c].tolist())
        G["idf_a"] = token_idf(R["ad"][rid_c].tolist())
        n_docs = len(rid_c)
        # generic name tokens: document frequency > 1% within this country's names (data-driven)
        from collections import Counter
        dfc = Counter()
        for s in R["nm"][rid_c]:
            dfc.update(set(s.split()))
        G["generic"] = {w for w, k in dfc.items() if k / n_docs > 0.01}
        rep.setdefault("generic_tokens", {})[str(c)] = sorted(G["generic"])[:300]
        tok[m] = token_feats_parallel(qi[m], ti[m])
    C = C.with_columns([pl.Series(k, tok[:, j]) for j, k in enumerate(TOK_NAMES)])
    rep["timing"]["tokens"] = time.time() - t0
    print("tokens done", round(time.time() - t0), "s", flush=True)

    # ---------------- structural ----------------------------------------------------------
    scr_map = {"latin": 0, "brahmic": 1, "mixed": 2, "other": 3, "empty": 4}
    nm_scr = np.array([scr_map.get(s, 3) for s in rec["nm_scr"].to_list()], np.int8)
    ad_scr = np.array([scr_map.get(s, 3) for s in rec["ad_scr"].to_list()], np.int8)
    has_dom = np.array([bool(s) for s in R["nm_dom"]], np.int8)
    has_alias = np.array([bool(s) for s in R["nm_b"]], np.int8)
    # name frequency (ambiguity): how many S1 / targets share the exact normalized name
    nm_s = rec.select(["rid", "nm", "src", "country"])
    s1_alive = nm_s.filter((pl.col("src") == 1) & ~pl.col("rid").is_in(list(dropped_rids)))  # P-dense: hide dropped S1
    f1 = s1_alive.group_by(["country", "nm"]).len().rename({"len": "nm_freq_s1"})
    ft = nm_s.filter(pl.col("src") != 1).group_by(["country", "nm"]).len().rename({"len": "nm_freq_t"})
    freq = nm_s.join(f1, on=["country", "nm"], how="left").join(ft, on=["country", "nm"], how="left").fill_null(0)
    freq_s1 = freq.sort("rid")["nm_freq_s1"].to_numpy()
    freq_t = freq.sort("rid")["nm_freq_t"].to_numpy()
    C = C.with_columns([
        pl.Series("t_src", src[ti].astype(np.int8)),
        pl.Series("t_nm_scr", nm_scr[ti]), pl.Series("t_ad_scr", ad_scr[ti]),
        pl.Series("t_has_dom", has_dom[ti]), pl.Series("t_has_alias", has_alias[ti]),
        pl.Series("q_nm_freq_s1", freq_s1[qi].astype(np.int32)),
        pl.Series("t_nm_freq_s1", freq_s1[ti].astype(np.int32)),
        pl.Series("t_nm_freq_t", freq_t[ti].astype(np.int32)),
    ])
    # domain/handle vs S1 concatenated name: prefix coverage
    dom_t = R["nm_dom"][ti]
    qc = cat[qi]
    dom_cov = np.array([(len(d) / max(len(q), 1)) if (d and q.startswith(d[:max(4, len(d) // 2)])) else 0.0
                        for d, q in zip(dom_t, qc)], np.float32)
    C = C.with_columns(pl.Series("n_dom_prefix_cov", dom_cov))

    # ---------------- labels ---------------------------------------------------------------
    if split == "train":
        gt = pl.read_parquet(split_paths("train")["gt"])
        pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
                 .explode("m").with_columns(pl.col("m").str.strip_chars()).filter(pl.col("m") != "")
                 .select(pl.col("source1_entity_id").alias("s1"), pl.col("m").alias("t_id")))
        ids = rec.select(["entity_id", "rid"])
        pairs = (pairs.join(ids.rename({"entity_id": "s1", "rid": "q"}), on="s1")
                 .join(ids.rename({"entity_id": "t_id", "rid": "t"}), on="t_id").select(["q", "t"])
                 .with_columns(pl.lit(1).cast(pl.Int8).alias("y")))
        C = C.join(pairs, on=["q", "t"], how="left").with_columns(pl.col("y").fill_null(0))
        rep["pos_rate"] = float(C["y"].mean())
        rep["n_pos"] = int(C["y"].sum())
    out = f"{ARTIFACTS}/{split}/feat_{tag}.parquet"
    C.write_parquet(out)
    rep["n_cols"] = C.width
    rep["columns"] = C.columns
    rep["timing"]["total"] = time.time() - t0
    with open(f"{REPORTS}/features_{split}_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def main(split: str = "train", cands_tag: str = "v1", tag: str = "v1", folds: str = "", prune: str = "",
         use_emb: bool = True, drop: str = ""):
    rep = features.remote(split, cands_tag, tag, folds, 0.5, 0.05, prune, use_emb, drop)
    print(json.dumps({k: v for k, v in rep.items() if k != "generic_tokens"}, indent=1)[:6000])
