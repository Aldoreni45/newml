"""Stage 1: learn normalization maps from train pairs, normalize every record.

  modal run modal_app/stage_normalize.py

Outputs (volume):
  /vol/artifacts/maps/native_map.json, abbrev_name_<country>.json, abbrev_addr_<country>.json
  /vol/artifacts/{train,test}/records.parquet
  /vol/reports/normalize_report.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, cpu_image, ensure_dirs, setup_path, split_paths, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-normalize")


def load_records(split: str):
    import polars as pl

    p = split_paths(split)
    frames = []
    for src, key in ((1, "s1"), (2, "s2"), (3, "s3")):
        df = pl.read_parquet(p[key]).with_columns(pl.lit(src).cast(pl.Int8).alias("src"))
        frames.append(df)
    df = pl.concat(frames).rename({"business_name": "name", "business_address": "address"})
    return df.with_row_index("rid")


def gt_pairs(records):
    import polars as pl

    gt = pl.read_parquet(split_paths("train")["gt"])
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("t"))
        .explode("t").with_columns(pl.col("t").str.strip_chars()).filter(pl.col("t") != "")
        .select(pl.col("source1_entity_id").alias("s1"), "t")
    )
    ids = records.select(["entity_id", "rid"])
    return (pairs.join(ids.rename({"entity_id": "s1", "rid": "s1_rid"}), on="s1")
            .join(ids.rename({"entity_id": "t", "rid": "t_rid"}), on="t"))


@app.function(image=cpu_image, volumes={MOUNT: vol}, cpu=32, memory=131072, timeout=6 * 3600)
def run(sample_report: int = 200_000, exclude_folds: str = "0") -> dict:
    setup_path()
    import numpy as np
    import polars as pl
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist

    from ber import records as R
    from ber.aliases import learn_abbrev_map, learn_native_map

    ensure_dirs()
    os.makedirs(f"{ARTIFACTS}/maps", exist_ok=True)
    rep: dict = {}
    t0 = time.time()
    tr = load_records("train")
    te = load_records("test")
    pairs_all = gt_pairs(tr)
    # learn maps WITHOUT the evaluation fold(s) (fold = numeric S1 id % 10) so validation stays clean
    ex = [int(x) for x in exclude_folds.split(",") if x != ""]
    pairs = pairs_all.filter(~(pl.col("s1").str.slice(3).cast(pl.Int64) % 10).is_in(ex)) if ex else pairs_all
    rep["n_pairs"] = pairs.height
    rep["map_learning_excluded_folds"] = ex

    # ---- pass 1: S1 names/addresses with seed maps only --------------------------------
    R.set_maps()
    s1 = tr.filter(pl.col("src") == 1)
    c1 = R.normalize_parallel(s1["name"].to_list(), s1["address"].to_list())
    s1n = s1.select("rid").with_columns(pl.Series("nm", c1["nm"]), pl.Series("ad", c1["ad"]))
    P = (pairs.join(s1n.rename({"rid": "s1_rid"}), on="s1_rid")
         .join(tr.select(["rid", "name", "address", "country"]).rename({"rid": "t_rid"}), on="t_rid"))
    nat_name = P.filter(pl.col("name").str.contains(r"[ऀ-ൿ]"))
    nat_addr = P.filter(pl.col("address").str.contains(r"[ऀ-ൿ]"))
    map_name, st_name = learn_native_map(zip(nat_name["nm"].to_list(), nat_name["name"].to_list()))
    map_addr, st_addr = learn_native_map(zip(nat_addr["ad"].to_list(), nat_addr["address"].to_list()),
                                         min_dice=0.5)
    native = dict(map_addr)
    native.update(map_name)  # name evidence wins on conflicts
    rep["native_map"] = {"name": st_name, "addr": st_addr, "merged_size": len(native)}
    with open(f"{ARTIFACTS}/maps/native_map.json", "w", encoding="utf-8") as f:
        json.dump(native, f, ensure_ascii=False)
    rep["t_pass1"] = time.time() - t0

    # ---- pass 2: normalize the records of a pair sample with the native map -----------
    R.set_maps(native=native)
    samp = pairs.sample(n=min(1_500_000, pairs.height), seed=11)
    need = pl.concat([samp.select(pl.col("s1_rid").alias("rid")), samp.select(pl.col("t_rid").alias("rid"))]).unique()
    sub = tr.join(need, on="rid")
    c2 = R.normalize_parallel(sub["name"].to_list(), sub["address"].to_list())
    subn = sub.select(["rid", "country"]).with_columns(pl.Series("nm", c2["nm"]), pl.Series("ad", c2["ad"]))
    S = (samp.join(subn.rename({"rid": "s1_rid", "nm": "s_nm", "ad": "s_ad"}), on="s1_rid")
         .join(subn.drop("country").rename({"rid": "t_rid", "nm": "t_nm", "ad": "t_ad"}), on="t_rid"))
    abbrev = {}
    for c in S["country"].unique().to_list():
        Sc = S.filter(pl.col("country") == c)
        mn, ev_n = learn_abbrev_map(zip(Sc["s_nm"].to_list(), Sc["t_nm"].to_list()))
        ma, ev_a = learn_abbrev_map(zip(Sc["s_ad"].to_list(), Sc["t_ad"].to_list()))
        abbrev[c] = (mn, ma)
        with open(f"{ARTIFACTS}/maps/abbrev_name_{c}.json", "w") as f:
            json.dump({"map": mn, "evidence": ev_n[:400]}, f)
        with open(f"{ARTIFACTS}/maps/abbrev_addr_{c}.json", "w") as f:
            json.dump({"map": ma, "evidence": ev_a[:400]}, f)
        rep[f"abbrev_{c}"] = {"name_size": len(mn), "addr_size": len(ma),
                              "name_top": ev_n[:40], "addr_top": ev_a[:60]}
    rep["t_pass2"] = time.time() - t0

    # ---- pass 3: final normalization of every record, per country ---------------------
    for split, df in (("train", tr), ("test", te)):
        outs = []
        for c in df["country"].unique().to_list():
            d = df.filter(pl.col("country") == c)
            mn, ma = abbrev.get(c, ({}, {}))  # unseen country: seed + native maps only
            name_alias = dict(R.SEED_TOKEN_MAP) if not mn else {**R.SEED_TOKEN_MAP, **mn}
            R.set_maps(native=native, name_alias=name_alias, addr_alias=ma)
            cols = R.normalize_parallel(d["name"].to_list(), d["address"].to_list())
            outs.append(d.with_columns([pl.Series(k, v) for k, v in cols.items()]))
        out = pl.concat(outs).sort("rid")
        os.makedirs(f"{ARTIFACTS}/{split}", exist_ok=True)
        out.write_parquet(f"{ARTIFACTS}/{split}/records.parquet")
        rep[f"{split}_records"] = out.height
        if split == "test":
            nat_tokens = {}
            for col in ("name", "address"):
                toks = (out.select(pl.col(col).str.split(" ")).explode(col)
                        .filter(pl.col(col).str.contains(r"[ऀ-ൿ]"))[col]
                        .str.replace_all(r"^[^\w]+|[^\w]+$", ""))
                vc = toks.value_counts()
                tot = int(vc["count"].sum())
                cov = int(vc.filter(pl.col(col).is_in(list(native.keys())))["count"].sum())
                nat_tokens[col] = {"native_token_occurrences": tot, "covered_by_train_map": cov,
                                   "coverage": cov / tot if tot else None}
            rep["test_native_coverage"] = nat_tokens
    rep["t_pass3"] = time.time() - t0

    # ---- quality report: similarity of true pairs before/after --------------------------
    tr_n = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet")
    q = pairs_all.sample(n=min(sample_report, pairs_all.height), seed=5)
    Q = (q.join(tr_n.select(["rid", "nm", "ad", "name", "address", "country"]).rename(
        {"rid": "s1_rid", "nm": "s_nm", "ad": "s_ad", "name": "s_name", "address": "s_address"}), on="s1_rid")
         .join(tr_n.select(["rid", "nm", "ad", "name", "address", "nm_scr"]).rename(
             {"rid": "t_rid", "nm": "t_nm", "ad": "t_ad", "name": "t_name", "address": "t_address"}), on="t_rid"))

    def low(x):
        return [s.lower() for s in x]

    raw_name = cpdist(low(Q["s_name"].to_list()), low(Q["t_name"].to_list()), scorer=fuzz.token_set_ratio, workers=-1)
    new_name = cpdist(Q["s_nm"].to_list(), Q["t_nm"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    raw_addr = cpdist(low(Q["s_address"].to_list()), low(Q["t_address"].to_list()), scorer=fuzz.token_set_ratio, workers=-1)
    new_addr = cpdist(Q["s_ad"].to_list(), Q["t_ad"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    Q = Q.with_columns(pl.Series("raw_name", raw_name), pl.Series("new_name", new_name),
                       pl.Series("raw_addr", raw_addr), pl.Series("new_addr", new_addr))
    qs = [0.01, 0.05, 0.1, 0.25, 0.5]
    rep["quality"] = {}
    for key, sub in Q.group_by(["country", "nm_scr"]):
        if sub.height < 200:
            continue
        rep["quality"]["|".join(map(str, key))] = {
            "n": sub.height,
            **{f"{c}_q": [float(np.quantile(sub[c].to_numpy(), x)) for x in qs]
               for c in ("raw_name", "new_name", "raw_addr", "new_addr")},
        }
    ex = Q.filter(pl.col("nm_scr") != "latin").head(60)
    rep["examples_nonlatin"] = [
        [r["s_name"], r["t_name"], r["s_nm"], r["t_nm"], round(r["new_name"])] for r in ex.iter_rows(named=True)]
    rep["t_total"] = time.time() - t0
    with open(f"{REPORTS}/normalize_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
    commit()
    return rep


@app.local_entrypoint()
def main():
    rep = run.remote()
    print(json.dumps({k: v for k, v in rep.items() if k not in ("examples_nonlatin",)}, indent=1,
                     ensure_ascii=False, default=str)[:20000])
