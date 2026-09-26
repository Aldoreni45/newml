# COMPETITION_PLAYBOOK.md

Operational guide: how to go from code to a scored leaderboard upload and the final zip, safely.

## 1. End-to-end run (Modal)
```bash
# one-time: upload data
modal volume create ber-data
modal volume put ber-data <student_resource.zip> /raw/student_resource.zip
modal run modal_app/profile_data.py                    # extract -> parquet (+ profile)
# pipeline (use --detach for long stages so a laptop disconnect doesn't kill them)
modal run --detach modal_app/stage_normalize.py
modal run --detach modal_app/stage_block.py --split train
modal run --detach modal_app/stage_block.py --split test
modal run --detach modal_app/stage_features.py --split train --folds 0,1,2,3
modal run --detach modal_app/stage_features.py --split test
modal run --detach modal_app/stage_train.py::cv --feat-tag v1 --tag cv_v1
modal run --detach modal_app/stage_predict.py --feat-tag v1 --model-tag <final> --rule '<json>' --out-tag sub_v1
modal volume get ber-data /submissions/sub_v1/output ./output
```

## 2. Pre-upload checklist (every leaderboard upload)
1. `predict_report.json`: `validator_exit == 0` (official validator ran with `--check-ids`).
2. Row count = 1,732,544 and each country present (France 259,452 rows).
3. Mean predicted matches per S1 and empty-rate per country are plausible (train: 5.58% singletons, 3.46 matches per S1; test has more targets per S1, so expect ≥ 3.46 if the generator is the same).
4. Matches ⊆ candidates (the validator warns otherwise).
5. Log the upload in `MODEL_COMPARISON.md` with its public score next to the local estimate.

## 3. Leaderboard discipline
- The public LB is a subset of test; the private LB is the rest. **Never tune a parameter to the public LB.** Use it only to check that the local estimate (VALIDATION_STRATEGY §3) tracks reality.
- Final selection: the submission with the best **local, density-matched, bootstrap-robust** score, provided its public score is not anomalous. If two candidates are within noise, pick the simpler one.
- Keep two final candidates ready: (a) best-local and (b) most-conservative (higher precision). Choose between them based on how public-LB and local deltas agree.

## 4. Final package (zip)
```
<team>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
```
- `src/` = `src/ber` + a local CLI (`run_pipeline.py`) with no Modal dependency, plus the optional Modal launchers.
- `requirements.txt` pinned (see IMPLEMENTATION_PLAN).
- Check the size of `candidate_pairs.tsv`. If it is too large, use the two-stage pruning (IMPLEMENTATION_PLAN §submission-size).
- Re-run COMPLIANCE_CHECKLIST.md end to end on the final package.

## 5. Risk register
| risk | mitigation |
|---|---|
| France unseen (15% of test) | country-agnostic features; LOCO validation; self-trained maps (E-8); conservative decision on low-evidence countries |
| Test distractor density higher than train | density-matched validation (P-dense); π model; prior-shift check on test predictions |
| Laptop disconnect kills Modal jobs | `modal run --detach`; all stages persist to the volume |
| Model-license violation | only license-verified weights (research/MODEL_LICENSES.md); GBDT trained from scratch |
| External-data accusation | no external data; learned maps from provided data only; documented in methodology |
| Submission rejected | official validator in the predict stage; checklist above |
