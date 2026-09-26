# IMPLEMENTATION_PLAN.md

## Code layout (current → final submission layout)

```
amazon-ml-er/
├── src/ber/                    # library (becomes code/business_entity_resolution/src/ber)
│   ├── io.py                   # TSV readers/writers (submission format guarantees)
│   ├── metric.py               # exact per-S1 macro F0.5 + blocking diagnostics
│   ├── normalize.py            # script-aware translit, cleaning, phonetic keys
│   ├── aliases.py              # LEARNED maps: native->latin dictionary, abbreviation miner
│   ├── records.py              # record-level normalization (parallel)
│   ├── blocking.py             # multi-view TF-IDF top-K (sparse_dot_topn), union, recall@K
│   ├── features.py             # pair features (RapidFuzz, cosines, token/number, context)
│   └── decide.py               # threshold / exclusivity / expected-F0.5 set selection, fast scorer
├── modal_app/                  # cloud entry points (Modal); each stage persists to volume ber-data
│   ├── common.py               # images, volume, paths
│   ├── profile_data.py         # extract + profile
│   ├── stage_normalize.py      # learn maps + normalize all records
│   ├── stage_block.py          # candidates + recall report
│   ├── stage_features.py       # pair features (+labels on train)
│   ├── stage_train.py          # LightGBM + rule evaluation with the exact metric
│   └── stage_predict.py        # TODO: test features -> model -> decision -> TSVs -> validator
└── experiments/{reports,logs}  # mirrored JSON reports and run logs
```

For the final zip, a `run_pipeline.py` CLI will call the same stage functions **locally**, without Modal, on a machine with ≥ 128 GB RAM. The Modal wrappers stay in the repo as an optional launcher. The README will document both paths.

## Stage contracts
| stage | input | output | runtime (32 vCPU) |
|---|---|---|---|
| normalize | raw parquet | records.parquet (+maps) | ~5 min both splits |
| block | records | cands_<tag>.parquet (q, t, s_*, r_*) | TBD |
| features | records + cands (+GT) | feat_<tag>.parquet | TBD |
| train | feat (train) | model.txt, val predictions, report | TBD |
| predict | feat (test) + model | matching_results.tsv, candidate_pairs.tsv | TBD |

## Reproducibility
- Pinned packages in `modal_app/common.py` (to be mirrored into `requirements.txt`): polars 1.9.0, pyarrow 17.0.0, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, rapidfuzz 3.10.0, scikit-learn 1.5.2, unidecode 1.3.8, sparse_dot_topn 1.1.5, lightgbm 4.6.0, joblib 1.4.2, numba 0.60.0. Python 3.11.
- Seeds: TF-IDF fit sample seed 0, LightGBM seed 42 with `deterministic=True`, fold assignment = id mod 10.
- Every stage writes a JSON report with its config and timings.

## Submission-size concern
With about 70 candidates per S1, `candidate_pairs.tsv` for 1.73M S1 would be about 1.6 GB. Plan: a two-stage scorer where stage 1 (cheap LightGBM on retrieval + cosine features) prunes to top-N per S1. Stage 2 (full features) scores that set, which is then the documented `candidate_pairs.tsv` (the last set fed to the model, as the spec requires).
