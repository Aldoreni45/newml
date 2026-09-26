"""E-9: multilingual sentence embeddings (GPU) -> dense cosine features / dense retrieval view.

  modal run modal_app/stage_embed.py --split train
  modal run modal_app/stage_embed.py --split test

Model: intfloat/multilingual-e5-small (MIT license, 118M params; license verified from the HF
card in research/MODEL_LICENSES.md). The resolved commit sha is recorded in the report.
Three texts per record are embedded (L2-normalised, float16):
  name  : "query: <raw name>"            (raw so native scripts reach the multilingual model)
  addr  : "query: <raw address>"
  full  : "query: <raw name> , <raw address>"
Output: /vol/artifacts/{split}/emb_e5s_{name,addr,full}.npy  (rows aligned with records.rid)
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import MOUNT, commit, ARTIFACTS, REPORTS, VOL, base_cpu, ensure_dirs, setup_path, vol, with_src  # noqa: E402

import modal  # noqa: E402

app = modal.App("ber-embed")
MODEL_ID = "intfloat/multilingual-e5-small"
gpu_image = with_src(base_cpu.pip_install("torch==2.4.1", "transformers==4.44.2",
                                          "sentence-transformers==3.1.1", "huggingface_hub==0.25.1"))
hf_cache = modal.Volume.from_name("ber-hf-cache", create_if_missing=True)


@app.function(image=gpu_image, gpu="A10G", volumes={MOUNT: vol, "/hf": hf_cache}, cpu=8, memory=65536,
              timeout=6 * 3600)
def embed(split: str, batch: int = 1024, max_len: int = 64) -> dict:
    setup_path()
    os.environ["HF_HOME"] = "/hf"
    import numpy as np
    import polars as pl
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    ensure_dirs()
    t0 = time.time()
    sha = HfApi().model_info(MODEL_ID).sha
    model = SentenceTransformer(MODEL_ID, revision=sha, device="cuda")
    model.max_seq_length = max_len
    model.half()
    rec = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet", columns=["rid", "name", "address"])
    names = rec["name"].to_list()
    addrs = rec["address"].to_list()
    rep = {"split": split, "model": MODEL_ID, "revision": sha, "n": rec.height, "timing": {}}
    for kind, texts in (("name", [f"query: {n}" for n in names]),
                        ("addr", [f"query: {a}" for a in addrs]),
                        ("full", [f"query: {n} , {a}" for n, a in zip(names, addrs)])):
        t1 = time.time()
        # sort by length for throughput, restore order after
        order = np.argsort([len(x) for x in texts])
        out = np.zeros((len(texts), 384), dtype=np.float16)
        with torch.inference_mode():
            for s in range(0, len(texts), batch * 64):
                idx = order[s:s + batch * 64]
                e = model.encode([texts[i] for i in idx], batch_size=batch, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
                out[idx] = e.astype(np.float16)
        np.save(f"{ARTIFACTS}/{split}/emb_e5s_{kind}.npy", out)
        rep["timing"][kind] = time.time() - t1
        print(kind, rep["timing"][kind], flush=True)
        commit()
    rep["t_total"] = time.time() - t0
    with open(f"{REPORTS}/embed_{split}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.function(image=gpu_image, gpu="A10G", volumes={MOUNT: vol}, cpu=8, memory=131072, timeout=4 * 3600)
def dense_topk(split: str, kind: str = "full", k: int = 20, k_rev: int = 5, tag: str = "e5s") -> dict:
    """Exact dense top-k (S1->T and T->S1) per country on GPU; recall report on train."""
    setup_path()
    import numpy as np
    import polars as pl
    import torch

    t0 = time.time()
    rec = pl.read_parquet(f"{ARTIFACTS}/{split}/records.parquet", columns=["rid", "src", "country"])
    E = np.load(f"{ARTIFACTS}/{split}/emb_e5s_{kind}.npy", mmap_mode="r")
    frames = []

    def topk(Qm, Tm, kk, qc=4096, tb=1_000_000):
        T = torch.from_numpy(np.ascontiguousarray(Tm)).cuda()
        vals, idxs = [], []
        for s in range(0, Qm.shape[0], qc):
            Q = torch.from_numpy(np.ascontiguousarray(Qm[s:s + qc])).cuda()
            bv, bi = None, None
            for b in range(0, T.shape[0], tb):
                sc = Q @ T[b:b + tb].T
                v, i = torch.topk(sc.float(), min(kk, sc.shape[1]), dim=1)
                i = i + b
                if bv is None:
                    bv, bi = v, i
                else:
                    cv = torch.cat([bv, v], 1)
                    ci = torch.cat([bi, i], 1)
                    bv, j = torch.topk(cv, kk, dim=1)
                    bi = torch.gather(ci, 1, j)
            vals.append(bv.cpu().numpy())
            idxs.append(bi.cpu().numpy())
        del T
        torch.cuda.empty_cache()
        return np.vstack(vals), np.vstack(idxs)

    for c in rec["country"].unique().to_list():
        q_rid = rec.filter((pl.col("src") == 1) & (pl.col("country") == c))["rid"].to_numpy()
        t_rid = rec.filter((pl.col("src") != 1) & (pl.col("country") == c))["rid"].to_numpy()
        Qe, Te = E[q_rid], E[t_rid]
        v, i = topk(Qe, Te, k)
        rk = np.tile(np.arange(k, dtype=np.int16), len(q_rid))
        frames.append(pl.DataFrame({"q": np.repeat(q_rid, k), "t": t_rid[i.ravel()], "s_dense": v.ravel(),
                                    "r_dense": rk}))
        v2, i2 = topk(Te, Qe, k_rev)
        rk2 = np.tile(np.arange(k_rev, dtype=np.int16), len(t_rid))
        frames.append(pl.DataFrame({"q": q_rid[i2.ravel()], "t": np.repeat(t_rid, k_rev), "s_dense_rev": v2.ravel(),
                                    "r_dense_rev": rk2}))
        print(c, time.time() - t0, flush=True)
    D1 = pl.concat([f for f in frames if "r_dense" in f.columns])
    D2 = pl.concat([f for f in frames if "r_dense_rev" in f.columns])
    D = D1.join(D2, on=["q", "t"], how="full", coalesce=True).with_columns(
        pl.col("s_dense").fill_null(0.0), pl.col("s_dense_rev").fill_null(0.0),
        pl.col("r_dense").fill_null(999).cast(pl.Int16), pl.col("r_dense_rev").fill_null(999).cast(pl.Int16))
    D.write_parquet(f"{ARTIFACTS}/{split}/cands_dense_{tag}.parquet")
    rep = {"split": split, "kind": kind, "k": k, "k_rev": k_rev, "n": D.height, "t": time.time() - t0}
    with open(f"{REPORTS}/dense_{split}_{tag}.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    commit()
    return rep


@app.local_entrypoint()
def main(split: str = "train", step: str = "embed"):
    if step == "embed":
        print(json.dumps(embed.remote(split), indent=1))
    else:
        print(json.dumps(dense_topk.remote(split), indent=1))
