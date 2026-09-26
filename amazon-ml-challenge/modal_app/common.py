"""Shared Modal definitions: volume, images, paths.

All heavy compute runs on Modal (never on the dev laptop). Data lives on the
`ber-data` volume:
  /vol/raw/student_resource.zip      uploaded once
  /vol/data/{train,test}/*.parquet   extracted + converted by extract()
  /vol/reports/                      profiling / experiment outputs
  /vol/artifacts/                    candidates, features, models
"""
from __future__ import annotations

import os
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
# repo layout: <root>/src/ber ; submission-package layout: <root>=src/ with src/ber next to modal_app/
LOCAL_SRC = str(ROOT / "src") if (ROOT / "src" / "ber").is_dir() else str(ROOT)

VOL_NAME = "ber-data"
# Storage root: the Modal volume mount in the cloud; any local directory for Modal-free runs
# (export BER_ROOT=/path/to/workdir, see README "Running without Modal").
MOUNT = "/vol"                              # Modal volume mount point (POSIX; used in decorators)
VOL = os.environ.get("BER_ROOT", MOUNT)     # data root actually read/written by the stages
DATA = f"{VOL}/data"
REPORTS = f"{VOL}/reports"
ARTIFACTS = f"{VOL}/artifacts"

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

BASE_PKGS = [
    "polars==1.9.0",
    "pyarrow==17.0.0",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scipy==1.13.1",
    "rapidfuzz==3.10.0",
    "scikit-learn==1.5.2",
    "anyascii==0.3.3",
]

ML_PKGS = [
    "sparse_dot_topn==1.1.5",
    "lightgbm==4.6.0",
    "joblib==1.4.2",
    "numba==0.60.0",
]


def with_src(img: "modal.Image") -> "modal.Image":
    """Attach the local package + this module (must be the last image step)."""
    return img.add_local_file(str(ROOT / "modal_app" / "common.py"), "/root/common.py").add_local_dir(
        LOCAL_SRC, "/root/src")


base_cpu = modal.Image.debian_slim(python_version="3.11").pip_install(*BASE_PKGS)
cpu_image = with_src(base_cpu)
ml_image = with_src(base_cpu.pip_install(*ML_PKGS))


def setup_path() -> None:
    import sys

    for p in ("/root/src", LOCAL_SRC):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def commit() -> None:
    """Persist volume writes when running inside Modal; no-op for local (Modal-free) runs."""
    if not modal.is_local():
        vol.commit()


def split_paths(split: str) -> dict:
    d = f"{DATA}/{split}"
    return {
        "s1": f"{d}/{split}_source1.parquet",
        "s2": f"{d}/{split}_source2.parquet",
        "s3": f"{d}/{split}_source3.parquet",
        "gt": f"{d}/{split}_ground_truth.parquet",
        "gt_tsv": f"{d}/{split}_ground_truth.tsv",
    }


def ensure_dirs() -> None:
    for p in (DATA, REPORTS, ARTIFACTS, f"{ARTIFACTS}/models"):
        os.makedirs(p, exist_ok=True)
