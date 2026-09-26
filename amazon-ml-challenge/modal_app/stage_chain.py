"""Server-side chaining of stages in ONE detached container (robust to laptop network drops).

  modal run --detach modal_app/stage_chain.py::dense_stack

dense_stack:
  1. collective features on train (v1_dense, stage-1 = cv_v1d)       -> feat_v1_dense_col
  2. stage-2 cross-fit on feat_v1_dense_col                           -> cv_v2sd_f{1,2,3}
  3. collective features on test (stage-1 = cv_v1d, suffix _d)        -> test feat_v1_col_d
  4. pick the best threshold+exclusivity rule on the dense validation fold
  5. predict test with the stage-2 fold ensemble                      -> /submissions/sub_v2
Each step calls the stage's plain Python function (`.local()` = same container).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, ARTIFACTS, REPORTS, VOL, commit, ml_image, vol  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-chain")
chain_image = ml_image.add_local_dir(os.path.dirname(os.path.abspath(__file__)), "/root/modal_app")


@app.function(image=chain_image, volumes={MOUNT: vol}, cpu=32, memory=229376, timeout=20 * 3600)
def dense_stack(s1_tag: str = "cv_v1d", train_feat: str = "v1_dense", test_feat: str = "v1",
                s2_tag: str = "cv_v2sd", out_tag: str = "sub_v2", skip: str = "") -> dict:
    sys.path.insert(0, "/root/modal_app")
    from stage_collective import collective
    from stage_predict import predict
    from stage_train import train_cv

    done = set(x for x in skip.split(",") if x)
    log = {}
    if "col_train" not in done:
        log["col_train"] = collective.local("train", train_feat, s1_tag)
        commit()
        print("collective train done", flush=True)
    if "s2" not in done:
        r = train_cv.local(f"{train_feat}_col", s2_tag, "1,2,3", "0", "{}", 0, "")
        log["s2"] = {k: r.get(k) for k in ("rounds", "best_rule", "tau_oof", "val_at_tau_oof", "rules",
                                           "best_bootstrap_se", "best_pair_precision", "best_pair_recall")}
        commit()
        print("stage-2 done", r["best_rule"], flush=True)
    else:
        with open(f"{REPORTS}/train_{s2_tag}.json") as fh:
            r = json.load(fh)
    tau = float(r["tau_oof"])  # chosen on out-of-fold predictions, never on the eval fold
    log["chosen_rule"] = {"rule": f"thr_{tau}_excl", "tau": tau, "val_at_tau_oof": r.get("val_at_tau_oof")}
    if "col_test" not in done:
        log["col_test"] = collective.local("test", test_feat, s1_tag, "1,2,3", "_d")
        commit()
        print("collective test done", flush=True)
    models = ",".join(f"{s2_tag}_f{k}" for k in (1, 2, 3))
    rep = predict.local(f"{test_feat}_col_d", models, json.dumps({"type": "thr_excl", "tau": tau}), out_tag)
    log["predict"] = {k: v for k, v in rep.items() if k != "validator_out"}
    log["validator_out"] = rep.get("validator_out", "")[-800:]
    with open(f"{REPORTS}/chain_{out_tag}.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    commit()
    return log


@app.local_entrypoint()
def main(skip: str = "", out_tag: str = "sub_v2"):
    print(json.dumps(dense_stack.remote(skip=skip, out_tag=out_tag), indent=1, default=str)[:6000])
