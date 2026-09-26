"""Phase 1: extract the dataset on Modal and compute full-data statistics.

Run:  modal run modal_app/profile_data.py
Outputs (volume ber-data): /vol/reports/profile.json, /vol/reports/samples_*.txt
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, DATA, REPORTS, VOL  # noqa: E402,F401
from common import cpu_image, ensure_dirs, setup_path, split_paths, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-profile")

SCRIPTS = {
    "devanagari": r"\p{Devanagari}",
    "bengali": r"\p{Bengali}",
    "gurmukhi": r"\p{Gurmukhi}",
    "gujarati": r"\p{Gujarati}",
    "oriya": r"\p{Oriya}",
    "tamil": r"\p{Tamil}",
    "telugu": r"\p{Telugu}",
    "kannada": r"\p{Kannada}",
    "malayalam": r"\p{Malayalam}",
    "arabic": r"\p{Arabic}",
    "latin_accented": r"[À-ɏ]",
}

PATTERNS = {
    "leading_junk": r"^[^\p{L}\p{N}]",
    "trailing_junk": r"[^\p{L}\p{N}.)\]]$",
    "has_dba": r"(?i)\b(dba|d/b/a|doing business as|t/a|aka)\b",
    "has_domain": r"(?i)\b[\w-]+\.(com|in|net|org|fr|co|biz|info|io)\b",
    "has_pipe": r"\|",
    "has_paren": r"[()\[\]{}]",
    "has_amp": r"&",
    "has_and": r"(?i)\band\b|\bet\b",
    "all_upper": r"^[^\p{Ll}]*\p{Lu}[^\p{Ll}]*$",
    "all_lower": r"^[^\p{Lu}]*\p{Ll}[^\p{Lu}]*$",
    "double_space": r"  ",
    "digit": r"\d",
    "legal_us": r"(?i)\b(llc|l\.l\.c|inc|incorporated|corp|corporation|co|company|ltd|limited|pc|pllc|llp|lp|plc)\b",
    "legal_in": r"(?i)\b(pvt|private|ltd|limited|llp|opc)\b",
    "legal_fr": r"(?i)\b(sarl|sas|sasu|sa|sci|eurl|snc|scop|selarl)\b",
    "null_token": r"(?i)\b(null|none|nan|n/a)\b",
    "near_landmark": r"(?i)\b(near|opp|opposite|behind|beside|next to|nr\.?)\b",
    "po_box": r"(?i)\bp\.?\s?o\.?\s?box\b",
    "zip5": r"\b\d{5}\b",
    "pin6": r"\b\d{6}\b",
    "pin_spaced": r"\b\d{3}\s\d{3}\b",
    "unit": r"(?i)\b(unit|apt|apartment|suite|ste|fl|floor|#)\b",
    "hash_junk": r"##",
}


def _q(s, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    return {str(q): float(s.quantile(q)) for q in qs}


@app.function(image=cpu_image, volumes={MOUNT: vol}, cpu=8, memory=32768, timeout=3600)
def extract(data_root: str = None) -> dict:
    """Unzip once or copy pre-extracted data, scan raw lines for format anomalies, convert TSV -> parquet."""
    import zipfile
    import shutil

    setup_path()
    from ber.io import read_source

    import polars as pl

    ensure_dirs()
    report = {}
    
    if data_root:
        # Kaggle mode: copy pre-extracted TSV files from data_root
        print(f"Using pre-extracted data from: {data_root}")
        for split in ("train", "test"):
            src_split = os.path.join(data_root, split)
            dst_split = f"{DATA}/{split}"
            os.makedirs(dst_split, exist_ok=True)
            for fn in os.listdir(src_split):
                if fn.endswith(".tsv"):
                    src_path = os.path.join(src_split, fn)
                    dst_path = os.path.join(dst_split, fn)
                    print(f"Copying {src_path} -> {dst_path}")
                    shutil.copy(src_path, dst_path)
    else:
        # Original mode: extract from zip
        zpath = f"{VOL}/raw/student_resource.zip"
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                n = info.filename
                if n.startswith("__MACOSX") or n.endswith("/") or n.endswith(".DS_Store"):
                    continue
                rel = n.split("student_resource/", 1)[1]
                if rel.startswith("dataset/"):
                    rel = rel[len("dataset/"):]
                dst = f"{DATA}/{rel}"
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(info) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, 64 * 1024 * 1024)
    for split in ("train", "test"):
        for fn in sorted(os.listdir(f"{DATA}/{split}")):
            if not fn.endswith(".tsv"):
                continue
            path = f"{DATA}/{split}/{fn}"
            want = 1 if "ground_truth" in fn else 3
            bad = 0
            bad_ex = []
            n = 0
            cr = 0
            with open(path, "rb") as f:
                f.readline()
                for line in f:
                    n += 1
                    if line.endswith(b"\r\n"):
                        cr += 1
                    t = line.count(b"\t")
                    if t != want:
                        bad += 1
                        if len(bad_ex) < 5:
                            bad_ex.append(line[:200].decode("utf-8", "replace"))
            report[f"{split}/{fn}"] = {"rows": n, "bad_tab_rows": bad, "bad_examples": bad_ex, "crlf_rows": cr}
            if "ground_truth" in fn:
                df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                                 missing_utf8_is_empty_string=True)
            else:
                df = read_source(path)
            df.write_parquet(path.replace(".tsv", ".parquet"))
            report[f"{split}/{fn}"]["parquet_rows"] = df.height
    with open(f"{REPORTS}/extract.json", "w") as f:
        json.dump(report, f, indent=2)
    commit()
    return report


@app.function(image=cpu_image, volumes={MOUNT: vol}, cpu=8, memory=65536, timeout=7200)
def profile() -> dict:
    setup_path()
    import numpy as np
    import polars as pl
    from rapidfuzz import fuzz

    ensure_dirs()
    out: dict = {}

    def load(split, key):
        return pl.read_parquet(split_paths(split)[key])

    # ------------------------------------------------------------------ per-file stats
    files = {}
    for split in ("train", "test"):
        for key in ("s1", "s2", "s3"):
            df = load(split, key)
            st = {"rows": df.height, "unique_ids": df["entity_id"].n_unique()}
            st["id_prefix_ok"] = int(df["entity_id"].str.starts_with(key.upper() + "-").sum())
            st["country_counts"] = {r[0]: r[1] for r in df["country"].value_counts(sort=True).head(30).iter_rows()}
            by_c = {}
            for c in df["country"].unique().to_list():
                sub = df.filter(pl.col("country") == c)
                if sub.height < 100:
                    continue
                cs = {"rows": sub.height}
                for col in ("business_name", "business_address"):
                    s = sub[col]
                    lens = s.str.len_chars()
                    toks = s.str.split(" ").list.len()
                    cc = {
                        "empty_rate": float((s.str.strip_chars() == "").mean()),
                        "len_chars_q": _q(lens),
                        "n_tokens_q": _q(toks),
                        "unique_rate": s.n_unique() / sub.height,
                    }
                    cc["scripts"] = {k: float(s.str.contains(v).mean()) for k, v in SCRIPTS.items()}
                    cc["non_ascii"] = float(s.str.contains(r"[^\x00-\x7F]").mean())
                    cc["patterns"] = {k: float(s.str.contains(v).mean()) for k, v in PATTERNS.items()}
                    cs[col] = cc
                cs["name_addr_unique_rate"] = sub.select(["business_name", "business_address"]).n_unique() / sub.height
                cs["top_names"] = [r[0] for r in sub["business_name"].value_counts(sort=True).head(15).iter_rows()]
                by_c[c] = cs
            st["by_country"] = by_c
            files[f"{split}_{key}"] = st
    out["files"] = files

    # ------------------------------------------------------------------ ground truth
    s1 = load("train", "s1").with_row_index("row_s1")
    s2 = load("train", "s2").with_row_index("row_t")
    s3 = load("train", "s3").with_row_index("row_t")
    gt = pl.read_parquet(split_paths("train")["gt"])
    g = {"rows": gt.height, "unique_s1": gt["source1_entity_id"].n_unique()}
    s1_ids = set(s1["entity_id"].to_list())
    gt_ids = set(gt["source1_entity_id"].to_list())
    g["s1_in_file_not_in_gt"] = len(s1_ids - gt_ids)
    g["gt_not_in_s1_file"] = len(gt_ids - s1_ids)
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m")
        .with_columns(pl.col("m").str.strip_chars())
        .filter(pl.col("m") != "")
        .select(pl.col("source1_entity_id").alias("s1"), pl.col("m").alias("t"))
        .with_columns(pl.col("t").str.slice(0, 2).alias("src"))
    )
    g["n_pairs"] = pairs.height
    g["pairs_by_src"] = {r[0]: r[1] for r in pairs["src"].value_counts().iter_rows()}
    cnt = gt.select("source1_entity_id").join(
        pairs.group_by("s1").agg(
            pl.len().alias("n"),
            (pl.col("src") == "S2").sum().alias("n2"),
            (pl.col("src") == "S3").sum().alias("n3"),
        ),
        left_on="source1_entity_id", right_on="s1", how="left",
    ).fill_null(0)
    cnt = cnt.join(s1.select(["entity_id", "country"]), left_on="source1_entity_id", right_on="entity_id", how="left")
    g["match_count_dist"] = {str(r[0]): r[1] for r in cnt["n"].value_counts(sort=True).sort("n").iter_rows()}
    g["s2_count_dist"] = {str(r[0]): r[1] for r in cnt["n2"].value_counts().sort("n2").iter_rows()}
    g["s3_count_dist"] = {str(r[0]): r[1] for r in cnt["n3"].value_counts().sort("n3").iter_rows()}
    g["singleton_rate"] = float((cnt["n"] == 0).mean())
    g["by_country"] = {}
    for c in cnt["country"].unique().to_list():
        sub = cnt.filter(pl.col("country") == c)
        g["by_country"][str(c)] = {
            "n_s1": sub.height,
            "singleton_rate": float((sub["n"] == 0).mean()),
            "mean_matches": float(sub["n"].mean()),
            "mean_s2": float(sub["n2"].mean()),
            "mean_s3": float(sub["n3"].mean()),
            "match_count_dist": {str(r[0]): r[1] for r in sub["n"].value_counts().sort("n").iter_rows()},
        }
    mult = pairs.group_by("t").agg(pl.len().alias("k"))
    g["target_multiplicity_dist"] = {str(r[0]): r[1] for r in mult["k"].value_counts().sort("k").iter_rows()}
    for key, df in (("S2", s2), ("S3", s3)):
        ids_in_gt = pairs.filter(pl.col("src") == key).select("t").unique()
        in_gt = df.with_columns(pl.col("entity_id").is_in(ids_in_gt["t"]).alias("matched"))
        g[f"{key}_coverage"] = float(in_gt["matched"].mean())
        g[f"{key}_coverage_by_country"] = {
            str(r[0]): float(r[1]) for r in in_gt.group_by("country").agg(pl.col("matched").mean()).iter_rows()
        }
        g[f"{key}_gt_ids_missing_from_file"] = int((~ids_in_gt["t"].is_in(df["entity_id"])).sum())
    out["ground_truth"] = g

    # ------------------------------------------------------------------ pair-level similarity
    tgt = pl.concat([s2, s3]).rename({"entity_id": "t", "business_name": "t_name",
                                      "business_address": "t_addr", "country": "t_country"})
    s1r = s1.rename({"entity_id": "s1", "business_name": "s1_name", "business_address": "s1_addr",
                     "country": "s1_country"})
    P = pairs.join(s1r, on="s1", how="left").join(tgt, on="t", how="left")
    out["pair_country_agreement"] = float((P["s1_country"] == P["t_country"]).mean())
    out["pair_country_confusion"] = {
        f"{r[0]}->{r[1]}": r[2] for r in P.group_by(["s1_country", "t_country"]).len().iter_rows()
    }
    rng = np.random.default_rng(0)
    samp = P.sample(n=min(200_000, P.height), seed=0)

    def norm(x: str) -> str:
        import re
        x = x.lower()
        x = re.sub(r"[^\w\s]", " ", x)
        return " ".join(x.split())

    import re
    num_re = re.compile(r"\d+")
    z5 = re.compile(r"\b\d{5}\b")
    z6 = re.compile(r"\b\d{6}\b|\b\d{3}\s\d{3}\b")

    rows = samp.select(["src", "s1_country", "s1_name", "t_name", "s1_addr", "t_addr", "row_s1", "row_t"]).iter_rows()
    rec = []
    for src, c, a, b, aa, ba, r1, rt in rows:
        a = a or ""
        b = b or ""
        aa = aa or ""
        ba = ba or ""
        na, nb = norm(a), norm(b)
        naa, nba = norm(aa), norm(ba)
        za = set(z5.findall(aa)) | set(x.replace(" ", "") for x in z6.findall(aa))
        zb = set(z5.findall(ba)) | set(x.replace(" ", "") for x in z6.findall(ba))
        nums_a = set(num_re.findall(aa))
        nums_b = set(num_re.findall(ba))
        rec.append((
            src, c,
            a == b, na == nb,
            fuzz.token_set_ratio(na, nb), fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb),
            naa == nba, fuzz.token_set_ratio(naa, nba), fuzz.partial_ratio(naa, nba),
            bool(re.search(r"[^\x00-\x7F]", b)) and not re.search(r"[^\x00-\x7F]", a),
            len(za) > 0, len(zb) > 0, bool(za & zb),
            len(nums_a & nums_b) > 0, len(nums_a) > 0, len(nums_b) > 0,
            ba.strip() == "",
            r1, rt,
        ))
    cols = ["src", "country", "name_raw_eq", "name_norm_eq", "name_tset", "name_ratio", "name_tsort",
            "addr_norm_eq", "addr_tset", "addr_partial", "t_nonascii_s1_ascii",
            "s1_has_zip", "t_has_zip", "zip_eq", "num_overlap", "s1_has_num", "t_has_num", "t_addr_empty",
            "row_s1", "row_t"]
    R = pl.DataFrame(rec, schema=cols, orient="row")
    pair_stats = {}
    for (src, c), sub in R.group_by(["src", "country"]):
        d = {"n": sub.height}
        for col in ("name_raw_eq", "name_norm_eq", "addr_norm_eq", "t_nonascii_s1_ascii", "s1_has_zip",
                    "t_has_zip", "num_overlap", "t_addr_empty"):
            d[col] = float(sub[col].mean())
        both = sub.filter(pl.col("s1_has_zip") & pl.col("t_has_zip"))
        d["zip_eq_given_both"] = float(both["zip_eq"].mean()) if both.height else None
        for col in ("name_tset", "name_ratio", "name_tsort", "addr_tset", "addr_partial"):
            d[col + "_q"] = _q(sub[col].cast(pl.Float64), qs=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75))
        pair_stats[f"{src}|{c}"] = d
    out["positive_pair_stats"] = pair_stats

    # random same-country negatives for contrast
    neg = []
    s1_s = s1.sample(n=100_000, seed=1)
    t_s = tgt.sample(n=100_000, seed=2)
    for (a, aa, c), (b, ba, tc) in zip(
        s1_s.select(["business_name", "business_address", "country"]).iter_rows(),
        t_s.select(["t_name", "t_addr", "t_country"]).iter_rows(),
    ):
        neg.append((c, fuzz.token_set_ratio(norm(a), norm(b)), fuzz.token_set_ratio(norm(aa), norm(ba))))
    N = pl.DataFrame(neg, schema=["country", "name_tset", "addr_tset"], orient="row")
    out["random_negative_stats"] = {
        str(c): {"name_tset_q": _q(sub["name_tset"].cast(pl.Float64), qs=(0.5, 0.9, 0.99, 0.999)),
                 "addr_tset_q": _q(sub["addr_tset"].cast(pl.Float64), qs=(0.5, 0.9, 0.99, 0.999))}
        for (c,), sub in N.group_by(["country"])
    }

    # ------------------------------------------------------------------ leakage probes
    leak = {}
    idnum = lambda s: s.str.slice(3).cast(pl.Int64)  # noqa: E731
    L = P.select([idnum(pl.col("s1")).alias("a"), idnum(pl.col("t")).alias("b"),
                  pl.col("row_s1").cast(pl.Float64), pl.col("row_t").cast(pl.Float64), "src"])
    leak["spearman_id_s1_vs_t"] = float(L.select(pl.corr("a", "b", method="spearman")).item())
    for src in ("S2", "S3"):
        sub = L.filter(pl.col("src") == src)
        leak[f"spearman_row_s1_vs_row_{src}"] = float(sub.select(pl.corr("row_s1", "row_t", method="spearman")).item())
    # consecutive-rows: do the matches of one S1 sit next to each other in the target file?
    gaps = (P.filter(pl.col("src") == "S2").group_by("s1").agg(pl.col("row_t").sort().diff().drop_nulls().alias("d"))
            .explode("d").drop_nulls())
    leak["S2_intra_entity_row_gap_q"] = _q(gaps["d"].cast(pl.Float64), qs=(0.01, 0.1, 0.5))
    leak["id_len_dist_s1"] = {str(r[0]): r[1] for r in s1["entity_id"].str.len_chars().value_counts().sort("entity_id").iter_rows()}
    # train/test overlap
    te1 = load("test", "s1")
    te2 = load("test", "s2")
    te3 = load("test", "s3")
    leak["id_overlap_train_test_s1"] = int(te1["entity_id"].is_in(s1["entity_id"]).sum())
    leak["id_overlap_train_test_s2"] = int(te2["entity_id"].is_in(s2["entity_id"]).sum())
    leak["id_overlap_train_test_s3"] = int(te3["entity_id"].is_in(s3["entity_id"]).sum())
    k1 = s1.select(pl.concat_str(["business_name", "business_address"], separator="|").alias("k"))["k"]
    kt = te1.select(pl.concat_str(["business_name", "business_address"], separator="|").alias("k"))["k"]
    leak["test_s1_exact_name_addr_in_train_s1"] = int(kt.is_in(k1).sum())
    leak["test_s1_exact_name_in_train_s1"] = int(te1["business_name"].is_in(s1["business_name"]).sum())
    out["leakage"] = leak

    # ------------------------------------------------------------------ human-readable samples
    os.makedirs(REPORTS, exist_ok=True)
    cl = cnt.sample(n=400, seed=7)
    tgt_map = tgt.select(["t", "t_name", "t_addr", "t_country"])
    lines = []
    for sid, n, n2, n3, c in cl.select(["source1_entity_id", "n", "n2", "n3", "country"]).iter_rows():
        r = s1r.filter(pl.col("s1") == sid).row(0, named=True)
        lines.append(f"### {sid} [{c}] matches={n} (S2={n2}, S3={n3})")
        lines.append(f"  S1 | {r['s1_name']} | {r['s1_addr']}")
        ms = pairs.filter(pl.col("s1") == sid)["t"].to_list()
        for t in ms:
            tr = tgt_map.filter(pl.col("t") == t)
            if tr.height:
                tr = tr.row(0, named=True)
                lines.append(f"  {t[:2]} | {tr['t_name']} | {tr['t_addr']} | {tr['t_country']}")
        lines.append("")
    with open(f"{REPORTS}/samples_train_clusters.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # unmatched S2/S3 records (distractors) sample
    matched_t = pairs["t"]
    um = tgt.filter(~pl.col("t").is_in(matched_t)).sample(n=min(300, tgt.height), seed=3)
    with open(f"{REPORTS}/samples_unmatched_targets.txt", "w", encoding="utf-8") as f:
        for t, a, b, c in um.select(["t", "t_name", "t_addr", "t_country"]).iter_rows():
            f.write(f"{t} | {a} | {b} | {c}\n")
    # test samples incl. France
    with open(f"{REPORTS}/samples_test.txt", "w", encoding="utf-8") as f:
        for key, df in (("S1", te1), ("S2", te2), ("S3", te3)):
            for c in df["country"].unique().to_list():
                sub = df.filter(pl.col("country") == c)
                for r in sub.sample(n=min(40, sub.height), seed=5).iter_rows():
                    f.write(f"{key} [{c}] | {r[1]} | {r[2]}\n")
    with open(f"{REPORTS}/profile.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    commit()
    return out


@app.local_entrypoint()
def main(skip_extract: bool = False):
    if not skip_extract:
        rep = extract.remote()
        print(json.dumps(rep, indent=1))
    res = profile.remote()
    print(json.dumps({k: res[k] for k in ("ground_truth", "leakage")}, indent=1, default=str))
