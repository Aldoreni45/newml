#!/usr/bin/env python3
"""End-to-end pipeline WITHOUT Modal (any Linux box / Kaggle / Colab / Lightning).

The Modal stage files in modal_app/ contain the real logic. Every stage is a plain Python
function; `.local()` runs it in this process. Storage goes to --workdir (BER_ROOT).

Usage (recommended: >= 32 vCPU, >= 128 GB RAM for the full-size data):
    python run_pipeline.py --zip /path/to/student_resource.zip --workdir /data/ber --config configs/final.json
    python run_pipeline.py --data-root /kaggle/input/mlchallenge/data/raw --workdir /kaggle/working/ber --config configs/final.json

Outputs:
    <workdir>/submissions/<out_tag>/output/matching_results.tsv
    <workdir>/submissions/<out_tag>/output/candidate_pairs.tsv
    <workdir>/reports/*.json   (every stage writes a report)
Steps can be skipped with --skip (comma list of step names) to resume after a failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = ["extract", "normalize", "block_train", "block_test", "embed_train", "embed_test",
         "features_train", "features_test", "stage1", "collective_train", "collective_test", "stage2", "predict"]


def validate_dataset(data_root: str) -> bool:
    """Validate that all required TSV files exist in the Kaggle dataset structure."""
    data_root = Path(data_root)
    print(f"\n=== Dataset Validation ===")
    print(f"Dataset root: {data_root}")
    
    required_files = {
        "train": [
            "train_ground_truth.tsv",
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv",
        ],
        "test": [
            "test_source1.tsv",
            "test_source2.tsv",
            "test_source3.tsv",
        ]
    }
    
    all_ok = True
    
    for split, files in required_files.items():
        print(f"\n{split.capitalize()} files:")
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  ✗ Directory not found: {split_dir}")
            all_ok = False
            continue
        
        for fn in files:
            fp = split_dir / fn
            if fp.exists():
                print(f"  ✓ {fn}")
            else:
                print(f"  ✗ {fn} - NOT FOUND")
                all_ok = False
    
    if not all_ok:
        print("\n❌ Dataset validation failed - missing required files")
        return False
    
    print("\n✅ Dataset validation passed")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=False, help="student_resource.zip (only needed for the extract step)")
    ap.add_argument("--data-root", required=False, help="pre-extracted dataset root directory (e.g., /kaggle/input/mlchallenge/data/raw)")
    ap.add_argument("--workdir", required=True, help="working directory (BER_ROOT)")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "final.json"))
    ap.add_argument("--skip", default="", help="comma list of steps to skip")
    ap.add_argument("--only", default="", help="comma list of steps to run (default: all)")
    args = ap.parse_args()
    
    # Validate dataset if --data-root is provided
    if args.data_root:
        if not validate_dataset(args.data_root):
            print("Error: Dataset validation failed. Please check the paths above.")
            return 1

    os.environ["BER_ROOT"] = os.path.abspath(args.workdir)  # must be set before importing common
    for p in (HERE, os.path.join(HERE, "src"), os.path.join(HERE, "modal_app")):  # repo or package layout
        if os.path.isdir(p):
            sys.path.insert(0, p)
    with open(args.config) as fh:
        cfg = json.load(fh)
    skip = set(x for x in args.skip.split(",") if x)
    only = set(x for x in args.only.split(",") if x)
    run = [s for s in STEPS if s not in skip and (not only or s in only)]

    from common import ensure_dirs
    ensure_dirs()
    root = os.environ["BER_ROOT"]
    log = {}
    t0 = time.time()

    def step(name, fn):
        if name not in run:
            return
        t = time.time()
        print(f"=== {name}", flush=True)
        out = fn()
        log[name] = {"seconds": round(time.time() - t, 1)}
        with open(os.path.join(root, "reports", "pipeline_log.json"), "w") as fh:
            json.dump(log, fh, indent=1)
        return out

    def _extract():
        os.makedirs(os.path.join(root, "raw"), exist_ok=True)
        if args.zip:
            shutil.copy(args.zip, os.path.join(root, "raw", "student_resource.zip"))
        from profile_data import extract
        return extract.local(data_root=args.data_root)

    from stage_block import block
    from stage_collective import collective
    from stage_embed_static import embed_static
    from stage_features import features
    from stage_normalize import run as normalize
    from stage_predict import predict
    from stage_train import train_cv

    prune = json.dumps(cfg["prune"])
    drop = json.dumps(cfg["dense_drop"]) if cfg.get("dense_drop") else ""
    s1, s2 = cfg["stage1_tag"], cfg["stage2_tag"]
    ftr = cfg["train_feat_tag"]
    step("extract", _extract)
    step("normalize", lambda: normalize.local())
    step("block_train", lambda: block.local("train", json.dumps(cfg.get("blocking", {}))))
    step("block_test", lambda: block.local("test", json.dumps(cfg.get("blocking", {}))))
    step("embed_train", lambda: embed_static.local("train"))
    step("embed_test", lambda: embed_static.local("test"))
    step("features_train", lambda: features.local("train", "v1", ftr, cfg["train_folds_features"], 0.5, 0.05,
                                                  prune, True, drop))
    step("features_test", lambda: features.local("test", "v1", "v1", "", 0.5, 0.05, prune, True, ""))
    step("stage1", lambda: train_cv.local(ftr, s1, cfg["cv_folds"], cfg["val_folds"], "{}", 0, ""))
    p1c = bool(cfg.get("p1_competition", False))
    step("collective_train", lambda: collective.local("train", ftr, s1, cfg["cv_folds"], "", p1c))
    step("collective_test", lambda: collective.local("test", "v1", s1, cfg["cv_folds"], "_d", p1c))
    rep2 = step("stage2", lambda: train_cv.local(f"{ftr}_col", s2, cfg["cv_folds"], cfg["val_folds"], "{}", 0, ""))

    def _predict():
        tau = cfg.get("tau")
        if tau is None:  # threshold chosen on stage-2 out-of-fold predictions (never on the eval fold)
            with open(os.path.join(root, "reports", f"train_{s2}.json")) as fh:
                tau = float(json.load(fh)["tau_oof"])
        models = ",".join(f"{s2}_f{k}" for k in cfg["cv_folds"].split(","))
        return predict.local("v1_col_d", models, json.dumps({"type": "thr_excl", "tau": tau}), cfg["out_tag"])

    rep = step("predict", _predict)
    if rep:
        print(json.dumps({k: v for k, v in rep.items() if k != "validator_out"}, indent=1))
        print(rep.get("validator_out", ""))
        if rep.get("validator_exit") != 0:
            print("OFFICIAL VALIDATOR FAILED - do not submit")
            return 1
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
