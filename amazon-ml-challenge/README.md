# Business Entity Resolution — Amazon ML Challenge 2026

Multi-view retrieval + stacked gradient boosting with a learned cross-script dictionary,
many-to-one competition features and copy-cluster ("sibling") support features.
Everything is trained only on the provided data. No external lookups, no external datasets.
The one pretrained weight used (`minishlab/potion-multilingual-128M`, MIT) is a static
multilingual embedding model that provides optional cosine features.

## Pipeline

```
extract → normalize (learned native→Latin dictionary + learned abbreviation maps)
        → block (5 sparse TF-IDF views per country partition, incl. reverse target→S1)
        → static embeddings (CPU)
        → pair features (exact cosines, RapidFuzz, token/IDF/number, competition features)
        → stage-1 LightGBM (3-fold cross-fit, density-matched training)
        → collective / sibling-support features from out-of-fold stage-1 probabilities
        → stage-2 LightGBM (3-fold cross-fit)
        → threshold + many-to-one exclusivity → matching_results.tsv, candidate_pairs.tsv
```
Details: `SOLUTION_ARCHITECTURE.md`, `FEATURE_CATALOG.md`, `BLOCKING_STRATEGIES.md`,
`VALIDATION_STRATEGY.md`, `MODEL_COMPARISON.md`.

## Reproduce

### Option A — Modal (what we used)
```bash
pip install -r requirements.txt
modal volume create ber-data
modal volume put ber-data student_resource.zip /raw/student_resource.zip
modal run modal_app/profile_data.py
modal run --detach modal_app/stage_normalize.py
modal run --detach modal_app/stage_block.py --split train
modal run --detach modal_app/stage_block.py --split test
modal run --detach modal_app/stage_embed_static.py --split train
modal run --detach modal_app/stage_embed_static.py --split test
modal run --detach modal_app/stage_features.py --split train --tag v1_dense --folds 0,1,2,3 \
    --prune '{"rev":8,"combo":10,"name":5,"cat":5,"addr":5}' --drop '{"folds":[4,5,6,7,8,9],"frac":0.311}'
modal run --detach modal_app/stage_features.py --split test --tag v1 \
    --prune '{"rev":8,"combo":10,"name":5,"cat":5,"addr":5}'
modal run --detach modal_app/stage_train.py::cv --feat-tag v1_dense --tag cv_v1d
modal run --detach modal_app/stage_chain.py::main        # collective → stage 2 → test → predict → validator
modal volume get ber-data /submissions/sub_v2/output ./output
```

### Option B — any machine, no Modal (Kaggle / Colab / Lightning / workstation)
Needs Python 3.11 and ≥ 32 vCPU / ≥ 128 GB RAM for full-size data (the heaviest steps are
sparse top-K retrieval and feature extraction over ~90M candidate pairs).
```bash
pip install -r requirements.txt
python run_pipeline.py --zip /path/to/student_resource.zip --workdir /path/to/work --config configs/final.json
# resume after an interruption:
python run_pipeline.py --workdir /path/to/work --skip extract,normalize,block_train,block_test
```
Outputs: `<workdir>/submissions/final/output/{matching_results.tsv,candidate_pairs.tsv}`.
In the submission zip the runner lives at `code/business_entity_resolution/src/run_pipeline.py`. Run it from that folder;
it finds `ber/`, `modal_app/` and `configs/` next to itself.
The predict step runs the official validator (`--check-ids`) automatically.

## Determinism
Fold assignment is `int(S1 id) % 10`. TF-IDF fit samples use seed 0. LightGBM runs with `seed=42`,
`deterministic=True`, `force_row_wise=True`. Every stage writes a JSON report with its
configuration and timings to `<root>/reports/`.

## Layout
```
src/ber/          library: io, metric (exact F0.5), normalize, aliases (learned maps), records,
                  blocking, features, decide (thresholds / exclusivity / expected-F), modelio
modal_app/        stage entry points (also callable in-process via .local())
configs/          pipeline configurations
run_pipeline.py   Modal-free end-to-end runner
experiments/      mirrored stage reports and logs
*.md              research + engineering artifacts (start with PROJECT_STATE.md)
```

## Licences
Code: the project's own code. Libraries: polars (MIT), pyarrow (Apache-2.0), numpy/scipy/pandas/scikit-learn (BSD),
rapidfuzz (MIT), anyascii (ISC), sparse_dot_topn (Apache-2.0), LightGBM (MIT), numba (BSD), model2vec (MIT),
modal client (Apache-2.0). Model weights: potion-multilingual-128M (MIT). All other models are trained from scratch
on the challenge training data.
"# amazon-ml-challenge" 
