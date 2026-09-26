"""E-9 (CPU variant): static multilingual embeddings with Model2Vec — no GPU required.

  modal run --detach modal_app/stage_embed_static.py --split train

Model: minishlab/potion-multilingual-128M (MIT; distilled from BAAI/bge-m3, MIT). Resolved
commit sha is recorded in the report. Output: /vol/artifacts/{split}/emb_pot_{name,addr,full}.npy
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, base_cpu, ensure_dirs, setup_path, vol, with_src  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-embed-static")
hf_cache = modal.Volume.from_name("ber-hf-cache", create_if_missing=True)

STATIC_ID = "minishlab/potion-multilingual-128M"  # MIT; static Model2Vec distilled from BAAI/bge-m3 (MIT)
STATIC_REV = "73908c3438cf03b6a01bcb9611d62b23d0726f08"  # revision used for all reported results
static_image = with_src(base_cpu.pip_install("model2vec==0.6.0", "huggingface_hub==0.25.1"))


@app.function(image=static_image, volumes={MOUNT: vol, "/hf": hf_cache}, cpu=32, memory=131072, timeout=6 * 3600)
def embed_static(split: str) -> dict:
    """CPU embeddings with a static multilingual model (no GPU needed)."""
    setup_path()
    if os.path.isdir("/hf"):
        os.environ["HF_HOME"] = "/hf"
    import numpy as np
    import polars as pl
    from huggingface_hub import HfApi
    from model2vec import StaticModel

    ensure_dirs()
    t0 = time.time()
    sha = STATIC_REV
    try:
        model = StaticModel.from_pretrained(STATIC_ID, revision=STATIC_REV)
    except TypeError:  # older model2vec without the revision argument
        model = StaticModel.from_pretrained(STATIC_ID)
    rec = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet", columns=["rid", "name", "address"])
    names, addrs = rec["name"].to_list(), rec["address"].to_list()
    rep = {"split": split, "model": STATIC_ID, "revision": sha, "n": rec.height, "timing": {}}
    for kind, texts in (("name", names), ("addr", addrs), ("full", [f"{n} , {a}" for n, a in zip(names, addrs)])):
        t1 = time.time()
        E = model.encode(texts, batch_size=4096, show_progress_bar=False, use_multiprocessing=True)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        np.save(f"{ARTIFACTS}/{split}/emb_pot_{kind}.npy", E.astype(np.float16))
        rep["timing"][kind] = time.time() - t1
        rep["dim"] = int(E.shape[1])
        print(kind, rep["timing"][kind], flush=True)
        commit()
    rep["t_total"] = time.time() - t0
    with open(f"{REPORTS}/embed_static_{split}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def main(split: str = "train"):
    print(json.dumps(embed_static.remote(split), indent=1))


