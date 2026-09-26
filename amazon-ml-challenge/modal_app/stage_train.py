"""Stage 4: train the pair scorer, evaluate decision rules with the EXACT metric.

  modal run modal_app/stage_train.py --feat-tag v1 --tag lgb_v1

Split protocol (S1-grouped, fold = numeric S1 id % 10; ids are random -> random split):
  train folds  : --train-folds   (default 1,2)
  early stop   : --es-folds      (default 3)
  evaluation   : --val-folds     (default 0)  -> never used for fitting or early stopping
Evaluation population = EVERY S1 of the val folds (incl. singletons and S1s whose true
matches were missed by blocking), so the score is directly comparable to the leaderboard.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, ensure_dirs, ml_image, setup_path, split_paths, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-train")

NON_FEATURES = {"q", "t", "y", "fold"}
DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 400,
    "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 127, "num_threads": 32, "verbose": -1, "seed": 42, "deterministic": True,
    "force_row_wise": True,
}


def s1_truth(rec, folds, country=None):
    """(q array, n_true array, country array) for every S1 in the given folds."""
    import polars as pl

    gt = pl.read_parquet(split_paths("train")["gt"]).with_columns(
        pl.when(pl.col("matched_entity_ids").str.strip_chars() == "").then(0)
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("n_true"))
    s1 = (rec.filter(pl.col("src") == 1)
          .with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) % 10).alias("fold"))
          .filter(pl.col("fold").is_in(folds)))
    if country is not None:
        s1 = s1.filter(pl.col("country").is_in(country))
    s1 = s1.join(gt.select(pl.col("source1_entity_id").alias("entity_id"), "n_true"), on="entity_id", how="left")
    return s1["rid"].to_numpy(), s1["n_true"].fill_null(0).to_numpy(), s1["country"].to_numpy()


def evaluate_rules(V, all_q, n_true, extra_miss_mean: float):
    """V: polars frame with q,t,y,p (+features for baselines). Returns dict of rule -> score."""
    import numpy as np
    import polars as pl

    from ber.decide import exclusive_mask, expected_f_select, macro_f05_frame

    q, t, y, p = (V[c].to_numpy() for c in ("q", "t", "y", "p"))
    res = {}

    def score(mask, name):
        f, _ = macro_f05_frame(q[mask], y[mask], all_q, n_true)
        res[name] = round(f, 5)
        return f

    excl = exclusive_mask(q, t, p)
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        score(p >= tau, f"thr_{tau}")
        score((p >= tau) & excl, f"thr_{tau}_excl")
    for em in (0.0, extra_miss_mean):
        m = expected_f_select(q, p, extra_miss=em)
        score(m, f"expF_miss{em:.3f}")
        m2 = expected_f_select(q, np.where(excl, p, 0.0), extra_miss=em)
        score(m2, f"expF_excl_miss{em:.3f}")
    # deterministic baselines (no ML): exact core-name set equality, optionally best-for-target
    ce = V["n_core_eq"].to_numpy() > 0.5
    best_t = V["c_combo_trank"].to_numpy() == 1
    score(ce, "B0_core_name_equal")
    score(ce & best_t, "B1_core_equal_and_best_for_target")
    cc = V["c_combo"].to_numpy()
    for th in (0.5, 0.6, 0.7, 0.8):
        score((cc >= th) & best_t, f"B2_combo>={th}_best_for_target")
    return res


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=196608, timeout=8 * 3600)
def train(feat_tag: str = "v1", tag: str = "lgb_v1", train_folds: str = "1,2", es_folds: str = "3",
          val_folds: str = "0", params_json: str = "{}", train_countries: str = "", val_countries: str = "",
          drop_features: str = "", num_boost_round: int = 3000) -> dict:
    setup_path()
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    ensure_dirs()
    t0 = time.time()
    params = {**DEFAULT_PARAMS, **json.loads(params_json)}
    tf = [int(x) for x in train_folds.split(",")]
    ef = [int(x) for x in es_folds.split(",")]
    vf = [int(x) for x in val_folds.split(",")]
    rec = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet", columns=["rid", "entity_id", "src", "country"])
    df = pl.read_parquet(f"{ARTIFACTS}/train/feat_{feat_tag}.parquet")
    cmap = rec.select(pl.col("rid").alias("q"), pl.col("country").alias("_country"))
    df = df.join(cmap, on="q", how="left")
    drop = set(x for x in drop_features.split(",") if x)
    feats = [c for c in df.columns if c not in NON_FEATURES and c != "_country" and c not in drop]
    tc = [x for x in train_countries.split(",") if x] or None
    vc = [x for x in val_countries.split(",") if x] or None

    def part(folds, countries):
        d = df.filter(pl.col("fold").is_in(folds))
        if countries:
            d = d.filter(pl.col("_country").is_in(countries))
        return d

    Tr, Es, Va = part(tf, tc), part(ef, tc), part(vf, vc)
    rep = {"tag": tag, "feat_tag": feat_tag, "params": params, "n_features": len(feats), "features": feats,
           "n_train": Tr.height, "n_es": Es.height, "n_val": Va.height, "train_countries": tc, "val_countries": vc}
    dtr = lgb.Dataset(Tr.select(feats).to_numpy().astype(np.float32), label=Tr["y"].to_numpy(), feature_name=feats,
                      free_raw_data=True)
    des = lgb.Dataset(Es.select(feats).to_numpy().astype(np.float32), label=Es["y"].to_numpy(), reference=dtr)
    log = {}
    model = lgb.train(params, dtr, num_boost_round=num_boost_round, valid_sets=[des], valid_names=["es"],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200),
                                 lgb.record_evaluation(log)])
    rep["best_iteration"] = model.best_iteration
    rep["es_logloss"] = float(min(log["es"]["binary_logloss"]))
    rep["t_train"] = time.time() - t0
    os.makedirs(f"{ARTIFACTS}/models", exist_ok=True)
    model.save_model(f"{ARTIFACTS}/models/{tag}.txt")
    p = model.predict(Va.select(feats).to_numpy().astype(np.float32), num_iteration=model.best_iteration)
    V = Va.with_columns(pl.Series("p", p.astype(np.float32)))
    all_q, n_true, ctry = s1_truth(rec, vf, vc)
    # expected #true matches per S1 that blocking missed (val-measured mean)
    hits = V.group_by("q").agg(pl.col("y").sum().alias("h"))
    tot_true = n_true.sum()
    extra = float((tot_true - hits["h"].sum()) / max(len(all_q), 1))
    rep["val_blocking_pair_recall"] = float(hits["h"].sum() / max(tot_true, 1))
    rep["extra_miss_mean"] = extra
    rep["rules"] = evaluate_rules(V, all_q, n_true, extra)
    best_rule = max(rep["rules"], key=rep["rules"].get)
    rep["best_rule"] = [best_rule, rep["rules"][best_rule]]
    # per-country breakdown of a few key rules
    from ber.decide import exclusive_mask, expected_f_select, macro_f05_frame
    q, tt, y, pp = (V[c].to_numpy() for c in ("q", "t", "y", "p"))
    excl = exclusive_mask(q, tt, pp)
    masks = {"thr_0.5_excl": (pp >= 0.5) & excl, "expF_excl": expected_f_select(q, np.where(excl, pp, 0.0), extra)}
    rep["by_country"] = {}
    for c in np.unique(ctry):
        sel = ctry == c
        qs = set(all_q[sel].tolist())
        inq = np.array([x in qs for x in q])
        rep["by_country"][str(c)] = {
            k: round(macro_f05_frame(q[m & inq], y[m & inq], all_q[sel], n_true[sel])[0], 5) for k, m in masks.items()}
    imp = model.feature_importance("gain")
    rep["importance_gain"] = sorted([[f, float(g)] for f, g in zip(feats, imp)], key=lambda r: -r[1])
    V.select(["q", "t", "y", "p"]).write_parquet(f"{ARTIFACTS}/models/{tag}_valpred.parquet")
    rep["t_total"] = time.time() - t0
    with open(f"{REPORTS}/train_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


def _reliability(y, p, bins=(0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0001)):
    import numpy as np

    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append([round(lo, 3), round(hi, 3), int(m.sum()), round(float(p[m].mean()), 4), round(float(y[m].mean()), 4)])
    return out


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=229376, timeout=12 * 3600)
def train_cv(feat_tag: str = "v1", tag: str = "cv_v1", cv_folds: str = "1,2,3", val_folds: str = "0",
             params_json: str = "{}", rounds: int = 0, drop_features: str = "") -> dict:
    """Cross-fitted pair model (OOF on cv folds, fold-averaged predictions on val) +
    S1-level match-exists model pi trained on OOF aggregates. Evaluates all decision rules."""
    setup_path()
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from ber.decide import (exclusive_mask, expected_f_select, macro_f05_frame, pi_select,
                            s1_aggregates)

    ensure_dirs()
    t0 = time.time()
    params = {**DEFAULT_PARAMS, **json.loads(params_json)}
    cvf = [int(x) for x in cv_folds.split(",")]
    vf = [int(x) for x in val_folds.split(",")]
    rec = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet", columns=["rid", "entity_id", "src", "country"])
    df = pl.read_parquet(f"{ARTIFACTS}/train/feat_{feat_tag}.parquet")
    drop = set(x for x in drop_features.split(",") if x)
    feats = [c for c in df.columns if c not in NON_FEATURES and c not in drop]
    rep = {"tag": tag, "feat_tag": feat_tag, "params": params, "n_features": len(feats), "features": feats,
           "cv_folds": cvf, "val_folds": vf}
    Va = df.filter(pl.col("fold").is_in(vf))
    Xv = Va.select(feats).to_numpy().astype(np.float32)
    if not rounds:  # pick the number of rounds once, by early stopping on the last cv fold
        tr = df.filter(pl.col("fold").is_in(cvf[:-1]))
        es = df.filter(pl.col("fold") == cvf[-1])
        d1 = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), label=tr["y"].to_numpy())
        d2 = lgb.Dataset(es.select(feats).to_numpy().astype(np.float32), label=es["y"].to_numpy(), reference=d1)
        m = lgb.train(params, d1, 5000, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        rounds = int(m.best_iteration * 1.1)
        del d1, d2, tr, es, m
    rep["rounds"] = rounds
    with open(f"{ARTIFACTS}/models/{tag}_features.json", "w") as fh:
        json.dump(feats, fh)
    Ot = df.filter(~pl.col("fold").is_in(cvf + vf))  # folds neither trained on nor evaluated
    Xo = Ot.select(feats).to_numpy().astype(np.float32) if Ot.height else None
    oof_parts, pv, po = [], np.zeros(Va.height), np.zeros(Ot.height)
    for f in cvf:
        tr = df.filter(pl.col("fold").is_in([x for x in cvf if x != f]))
        ho = df.filter(pl.col("fold") == f)
        d1 = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), label=tr["y"].to_numpy(),
                         feature_name=feats)
        m = lgb.train(params, d1, rounds)
        m.save_model(f"{ARTIFACTS}/models/{tag}_f{f}.txt")
        ho_p = m.predict(ho.select(feats).to_numpy().astype(np.float32))
        oof_parts.append(ho.select(["q", "t", "y"] + [c for c in ("c_combo", "n_tset", "a_tset", "q_nm_freq_s1",
                                                                  "c_combo_tmargin", "q_ncand") if c in feats])
                         .with_columns(pl.Series("p", ho_p.astype(np.float32))))
        pv += m.predict(Xv) / len(cvf)
        if Xo is not None:
            po += m.predict(Xo) / len(cvf)
        del d1, tr, ho, m
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    rep["t_models"] = time.time() - t0
    OOF = pl.concat(oof_parts)
    V = Va.select(["q", "t", "y"] + [c for c in ("c_combo", "n_tset", "a_tset", "q_nm_freq_s1", "c_combo_tmargin",
                                                  "q_ncand") if c in feats]).with_columns(pl.Series("p", pv.astype(np.float32)))
    OOF.select(["q", "t", "y", "p"]).write_parquet(f"{ARTIFACTS}/models/{tag}_oof.parquet")
    V.select(["q", "t", "y", "p"]).write_parquet(f"{ARTIFACTS}/models/{tag}_valpred.parquet")
    rep["val_reliability"] = _reliability(V["y"].to_numpy(), V["p"].to_numpy())
    rep["oof_reliability"] = _reliability(OOF["y"].to_numpy(), OOF["p"].to_numpy())

    # ---------------- match-exists model pi (S1 level) -------------------------------------
    extra_cols = [c for c in ("c_combo", "n_tset", "a_tset", "q_nm_freq_s1", "c_combo_tmargin") if c in OOF.columns]

    def agg(D):
        ex = {c: D[c].to_numpy() for c in extra_cols}
        return s1_aggregates(D["q"].to_numpy(), D["p"].to_numpy(), ex)

    tr_q, tr_nt, _ = s1_truth(rec, cvf)
    va_q, va_nt, va_c = s1_truth(rec, vf)
    A_tr = pl.DataFrame({"q": tr_q, "has": (tr_nt > 0).astype(np.int8)}).join(agg(OOF), on="q", how="left").fill_null(0)
    A_va = pl.DataFrame({"q": va_q, "has": (va_nt > 0).astype(np.int8)}).join(agg(V), on="q", how="left").fill_null(0)
    pf = [c for c in A_tr.columns if c not in ("q", "has")]
    pparams = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
               "feature_fraction": 0.9, "verbose": -1, "seed": 7, "num_threads": 32}
    pm = lgb.train(pparams, lgb.Dataset(A_tr.select(pf).to_numpy().astype(np.float32), label=A_tr["has"].to_numpy()), 400)
    pi_va = pm.predict(A_va.select(pf).to_numpy().astype(np.float32))
    pm.save_model(f"{ARTIFACTS}/models/{tag}_pi.txt")
    rep["pi_features"] = pf
    rep["pi_val_logloss"] = float(-np.mean(A_va["has"].to_numpy() * np.log(np.clip(pi_va, 1e-6, 1)) +
                                          (1 - A_va["has"].to_numpy()) * np.log(np.clip(1 - pi_va, 1e-6, 1))))
    rep["pi_val_singleton_rate_true"] = float(1 - A_va["has"].mean())
    rep["pi_val_singleton_rate_pred"] = float((pi_va < 0.5).mean())
    pi_by_q = dict(zip(va_q.tolist(), pi_va.tolist()))

    # ---------------- exclusivity over the WHOLE predicted graph --------------------------------
    # test applies exclusivity among all S1s; validation must see the same competitors, so the
    # pool = OOF (cv folds) + val + label-free ensemble predictions on any other folds present.
    parts = [OOF.select(["q", "t", "p"]), V.select(["q", "t", "p"])]
    if Ot.height:
        parts.append(Ot.select(["q", "t"]).with_columns(pl.Series("p", po.astype(np.float32))))
    Pool = pl.concat(parts)
    ex_all = exclusive_mask(Pool["q"].to_numpy(), Pool["t"].to_numpy(), Pool["p"].to_numpy())
    n_oof = OOF.height
    excl_oof, excl = ex_all[:n_oof], ex_all[n_oof:n_oof + V.height]
    rep["exclusivity_pool"] = {"oof": n_oof, "val": V.height, "other": Ot.height}

    # ---------------- decision parameters chosen on OOF (never on the eval fold) -------------
    qo, yo, pov = OOF["q"].to_numpy(), OOF["y"].to_numpy(), OOF["p"].to_numpy()
    grid = [round(x, 2) for x in np.arange(0.40, 0.951, 0.05)]
    oof_scores = {tau: macro_f05_frame(qo[(pov >= tau) & excl_oof], yo[(pov >= tau) & excl_oof], tr_q, tr_nt)[0]
                  for tau in grid}
    tau_oof = max(oof_scores, key=oof_scores.get)
    rep["oof_threshold_curve"] = {str(k): round(v, 5) for k, v in oof_scores.items()}
    rep["tau_oof"] = tau_oof
    extra = float((tr_nt.sum() - yo.sum()) / max(len(tr_q), 1))  # blocking-miss rate from OOF, not eval
    rep["extra_miss_mean"] = extra

    # ---------------- decision rules on val ------------------------------------------------
    q, t, y, p = (V[c].to_numpy() for c in ("q", "t", "y", "p"))
    tot_true = va_nt.sum()
    rep["val_blocking_pair_recall"] = float(y.sum() / max(tot_true, 1))
    rules = {}

    def score(mask, name):
        rules[name] = round(macro_f05_frame(q[mask], y[mask], va_q, va_nt)[0], 5)

    for tau in grid:
        score(p >= tau, f"thr_{tau}")
        score((p >= tau) & excl, f"thr_{tau}_excl")
    pe = np.where(excl, p, 0.0)
    for em in (0.0, extra):
        score(expected_f_select(q, p, extra_miss=em), f"expF_m{em:.3f}")
        score(expected_f_select(q, pe, extra_miss=em), f"expF_excl_m{em:.3f}")
        score(pi_select(q, p, pi_by_q, extra_miss=em), f"pi_m{em:.3f}")
        score(pi_select(q, pe, pi_by_q, extra_miss=em), f"pi_excl_m{em:.3f}")
    # oracle upper bound given candidates: select exactly the true ones
    score(y == 1, "ORACLE_given_candidates")
    rep["rules"] = rules
    best = max((k for k in rules if not k.startswith("ORACLE")), key=rules.get)
    rep["best_rule_on_val_optimistic"] = [best, rules[best]]  # selected on the eval fold: reference only
    # HEADLINE: rule chosen on OOF, reported on the eval fold (no selection bias)
    kind = f"thr_{tau_oof}_excl"
    rep["best_rule"] = [kind, rules[kind]]
    rep["val_at_tau_oof"] = rules[kind]
    if kind.startswith("pi_excl"):
        mbest = pi_select(q, pe, pi_by_q, extra_miss=extra if kind.endswith(f"{extra:.3f}") else 0.0)
    elif kind.startswith("pi_"):
        mbest = pi_select(q, p, pi_by_q, extra_miss=extra if kind.endswith(f"{extra:.3f}") else 0.0)
    elif kind.startswith("expF_excl"):
        mbest = expected_f_select(q, pe, extra_miss=extra if kind.endswith(f"{extra:.3f}") else 0.0)
    elif kind.startswith("expF"):
        mbest = expected_f_select(q, p, extra_miss=extra if kind.endswith(f"{extra:.3f}") else 0.0)
    else:
        tau = float(kind.split("_")[1])
        mbest = (p >= tau) & (excl if kind.endswith("excl") else True)
    V.select(["q", "t", "y", "p"]).with_columns(pl.Series("sel", mbest)).write_parquet(
        f"{ARTIFACTS}/models/{tag}_valsel.parquet")
    f_all, f_per = macro_f05_frame(q[mbest], y[mbest], va_q, va_nt)
    strata = pl.DataFrame({"q": va_q, "f": f_per, "country": va_c, "n_true": va_nt})
    rep["best_by_country"] = {str(k[0]): [round(float(v["f"].mean()), 5), v.height] for k, v in strata.group_by(["country"])}
    rep["best_by_ntrue"] = {str(k[0]): [round(float(v["f"].mean()), 5), v.height]
                            for k, v in strata.with_columns(pl.col("n_true").clip(0, 7)).group_by(["n_true"])}
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        boot.append(f_per[rng.integers(0, len(f_per), len(f_per))].mean())
    rep["best_bootstrap_se"] = float(np.std(boot))
    tp = int(y[mbest].sum())
    rep["best_pair_precision"] = tp / max(int(mbest.sum()), 1)
    rep["best_pair_recall"] = tp / max(int(tot_true), 1)
    rep["t_total"] = time.time() - t0
    with open(f"{REPORTS}/train_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=229376, timeout=6 * 3600)
def train_loco(feat_tag: str = "v1", tag: str = "loco", train_country: str = "US", target_country: str = "India",
               country_norm: bool = False, pseudo: bool = False, pseudo_hi: float = 0.9, pseudo_lo: float = 0.1,
               params_json: str = "{}") -> dict:
    """Unseen-country protocol. Train on `train_country` only, evaluate on `target_country` fold 0.

    country_norm: replace each feature by its within-country percentile rank (label-free,
                  transductive standardisation computed over all candidate pairs of that country).
    pseudo      : self-training. Pseudo-label the target country's folds 1-3 with the source model
                  (y=1 if p>=pseudo_hi and exclusive-best for its target; y=0 if p<=pseudo_lo; rest
                  dropped), add them to training, retrain. Target labels are never used for training.
    """
    setup_path()
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from ber.decide import exclusive_mask, macro_f05_frame

    ensure_dirs()
    t0 = time.time()
    params = {**DEFAULT_PARAMS, **json.loads(params_json)}
    rec = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet", columns=["rid", "entity_id", "src", "country"])
    df = pl.read_parquet(f"{ARTIFACTS}/train/feat_{feat_tag}.parquet").join(
        rec.select(pl.col("rid").alias("q"), pl.col("country").alias("_c")), on="q", how="left")
    df = df.filter(pl.col("_c").is_in([train_country, target_country]))
    feats = [c for c in df.columns if c not in NON_FEATURES and c != "_c"]
    if country_norm:
        df = df.with_columns([(pl.col(f).rank("average").over("_c") / pl.len().over("_c")).cast(pl.Float32).alias(f)
                              for f in feats])
    rep = {"tag": tag, "train_country": train_country, "target_country": target_country,
           "country_norm": country_norm, "pseudo": pseudo}

    def fit(Tr, Es):
        d1 = lgb.Dataset(Tr.select(feats).to_numpy().astype(np.float32), label=Tr["y"].to_numpy(), feature_name=feats)
        d2 = lgb.Dataset(Es.select(feats).to_numpy().astype(np.float32), label=Es["y"].to_numpy(), reference=d1)
        return lgb.train(params, d1, 3000, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False)])

    src = df.filter(pl.col("_c") == train_country)
    Tr, Es = src.filter(pl.col("fold").is_in([1, 2])), src.filter(pl.col("fold") == 3)
    Va = df.filter((pl.col("_c") == target_country) & (pl.col("fold") == 0))
    va_q, va_nt, _ = s1_truth(rec, [0], [target_country])

    def score(model, name):
        p = model.predict(Va.select(feats).to_numpy().astype(np.float32))
        q, t, y = Va["q"].to_numpy(), Va["t"].to_numpy(), Va["y"].to_numpy()
        ex = exclusive_mask(q, t, p)
        rules = {f"thr_{tau}_excl": round(macro_f05_frame(q[(p >= tau) & ex], y[(p >= tau) & ex], va_q, va_nt)[0], 5)
                 for tau in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)}
        rep[name] = {"rules": rules, "best": max(rules.items(), key=lambda kv: kv[1]),
                     "reliability": _reliability(y, p)[4:13]}

    m0 = fit(Tr, Es)
    rep["iter0"] = m0.best_iteration
    score(m0, "source_only")
    if pseudo:
        U = df.filter((pl.col("_c") == target_country) & pl.col("fold").is_in([1, 2, 3]))
        pu = m0.predict(U.select(feats).to_numpy().astype(np.float32))
        ex = exclusive_mask(U["q"].to_numpy(), U["t"].to_numpy(), pu)
        yl = np.full(len(pu), -1, np.int8)
        yl[(pu >= pseudo_hi) & ex] = 1
        yl[pu <= pseudo_lo] = 0
        keep = yl >= 0
        true_y = U["y"].to_numpy()
        rep["pseudo_n"] = int(keep.sum())
        rep["pseudo_pos"] = int((yl == 1).sum())
        rep["pseudo_label_precision_pos"] = float(true_y[yl == 1].mean()) if (yl == 1).any() else None  # diagnostic only
        rep["pseudo_label_acc_neg"] = float(1 - true_y[yl == 0].mean()) if (yl == 0).any() else None
        P = U.filter(pl.Series(keep)).with_columns(pl.Series("y", yl[keep]).cast(pl.Int8))
        m1 = fit(pl.concat([Tr, P.select(Tr.columns)]), Es)
        rep["iter1"] = m1.best_iteration
        score(m1, "self_trained")
    rep["t"] = time.time() - t0
    with open(f"{REPORTS}/loco_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def loco(feat_tag: str = "v1", tag: str = "loco", train_country: str = "US", target_country: str = "India",
         country_norm: bool = False, pseudo: bool = False):
    rep = train_loco.remote(feat_tag, tag, train_country, target_country, country_norm, pseudo)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))


@app.function(image=ml_image, volumes={MOUNT: vol}, cpu=32, memory=131072, timeout=4 * 3600)
def evaluate_models(feat_tag: str, model_tags: str, val_folds: str = "0", tag: str = "") -> dict:
    """Score existing models (averaged) on a feature file; evaluate all threshold rules."""
    setup_path()
    import lightgbm as lgb
    import numpy as np
    import polars as pl

    from ber.decide import exclusive_mask, macro_f05_frame

    vf = [int(x) for x in val_folds.split(",")]
    rec = pl.read_parquet(f"{ARTIFACTS}/train/records.parquet", columns=["rid", "entity_id", "src", "country"])
    V = pl.read_parquet(f"{ARTIFACTS}/train/feat_{feat_tag}.parquet").filter(pl.col("fold").is_in(vf))
    p = np.zeros(V.height)
    tags = [x for x in model_tags.split(",") if x]
    from ber.modelio import model_features
    for mt in tags:
        m = lgb.Booster(model_file=f"{ARTIFACTS}/models/{mt}.txt")
        fl = model_features(m, f"{ARTIFACTS}/models", mt)
        p += m.predict(V.select(fl).to_numpy().astype(np.float32)) / len(tags)
    q, t, y = V["q"].to_numpy(), V["t"].to_numpy(), V["y"].to_numpy()
    va_q, va_nt, va_c = s1_truth(rec, vf)
    present = np.isin(va_q, np.unique(q)) | (va_nt == 0)
    rep = {"feat_tag": feat_tag, "models": tags, "n_pairs": V.height, "rules": {}}
    excl = exclusive_mask(q, t, p)
    for tau in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        for use_ex in (False, True):
            m = (p >= tau) & (excl if use_ex else True)
            rep["rules"][f"thr_{tau}{'_excl' if use_ex else ''}"] = round(macro_f05_frame(q[m], y[m], va_q, va_nt)[0], 5)
    best = max(rep["rules"], key=rep["rules"].get)
    rep["best_rule"] = [best, rep["rules"][best]]
    rep["n_val_s1"] = int(len(va_q))
    rep["reliability"] = _reliability(y, p)
    rep["mean_p_sum_per_s1"] = float(p.sum() / len(va_q))
    rep["mean_true_per_s1"] = float(va_nt.mean())
    if tag:
        with open(f"{REPORTS}/eval_{tag}.json", "w") as fh:
            json.dump(rep, fh, indent=1)
        commit()
    return rep


@app.local_entrypoint()
def evaluate(feat_tag: str = "v1_dense", model_tags: str = "cv_v1_f1,cv_v1_f2,cv_v1_f3", val_folds: str = "0",
             tag: str = ""):
    rep = evaluate_models.remote(feat_tag, model_tags, val_folds, tag)
    print(json.dumps({k: v for k, v in rep.items() if k != "reliability"}, indent=1))
    print("reliability", rep["reliability"])


@app.local_entrypoint()
def cv(feat_tag: str = "v1", tag: str = "cv_v1", cv_folds: str = "1,2,3", val_folds: str = "0", params: str = "{}",
       rounds: int = 0, drop_features: str = ""):
    rep = train_cv.remote(feat_tag, tag, cv_folds, val_folds, params, rounds, drop_features)
    show = {k: v for k, v in rep.items() if k not in ("features", "pi_features")}
    print(json.dumps(show, indent=1))


@app.local_entrypoint()
def main(feat_tag: str = "v1", tag: str = "lgb_v1", train_folds: str = "1,2", es_folds: str = "3",
         val_folds: str = "0", params: str = "{}", train_countries: str = "", val_countries: str = "",
         drop_features: str = ""):
    rep = train.remote(feat_tag, tag, train_folds, es_folds, val_folds, params, train_countries, val_countries,
                       drop_features)
    show = {k: v for k, v in rep.items() if k not in ("features", "importance_gain")}
    show["top_importance"] = rep["importance_gain"][:30]
    print(json.dumps(show, indent=1))
