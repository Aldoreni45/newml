# PROJECT_STATE.md — read this first in every session

_Last updated: 2026-09-25 (session 1, evening)_

## 1. Where everything is
- Project root: `D:\ML Project\hackathon\amazon-ml-er\` (sub-folder of the RegimeGate git repo; `.gitignore` excludes data and outputs).
- Raw challenge bundle: `student_resource/` (from `C:\Users\ayanp\Downloads\6ab10eb3b23ba_student_resource.zip`). Spec PDF is in Downloads.
- **Compute: Modal CPU only** (profile `ayanpthan768`). **Modal GPU is BLOCKED**: "Please add a payment method to use A10G GPU functions". The user must add a payment method, or approve a Kaggle GPU route (that needs the data uploaded as a private Kaggle dataset, so ask first). The laptop has 5.9 GB RAM and the user forbids running any AI/ML model on it.
- **Always launch long stages with `modal run --detach`.** Laptop heartbeat drops were seen; detached apps survive them.
- Modal volume `ber-data`:
  - `/data/{train,test}/*.parquet|tsv`, `/data/utils/validate_submission.py`
  - `/artifacts/maps/` learned maps
  - `/artifacts/{split}/records.parquet` (rid = row index)
  - `/artifacts/{split}/cands_v1.parquet` (full union, 112–118 per S1)
  - `/artifacts/{split}/emb_pot_{name,addr,full}.npy` (potion-multilingual-128M, 256-d fp16)
  - `/artifacts/train/feat_v1.parquet` (folds 0–3, 38.0M pairs, 87 features + fold, y)
  - `/artifacts/train/feat_v1_dense.parquet` (P-dense variant)
  - `/artifacts/train/feat_v1_col.parquet` (v1 + p1 + 20 collective/stacking features)
  - `/artifacts/test/feat_v1.parquet` (88.4M pairs, 51 per S1); `/artifacts/test/feat_v1_col.parquet` and `train/feat_v1_dense_col.parquet` exist but may be PARTIAL (writer died without a report). Regenerate them.
  - `/artifacts/models/`: lgb_v1, cv_v1_f{1,2,3} + cv_v1_features.json + cv_v1_pi, OOF/val predictions
  - `/submissions/<tag>/output/{matching_results.tsv,candidate_pairs.tsv}` + predict_report.json
  - `/reports/*.json|txt`, mirrored in `experiments/reports/`
- Code: `src/ber/` (library), `modal_app/` (stages: profile_data → stage_normalize → stage_block → [stage_embed_static] → stage_features → stage_train(main|cv|evaluate) → stage_collective → stage_train cv (stage 2) → stage_predict; plus stage_analyze).

## 2. Completed
- Spec → `COMPLIANCE_CHECKLIST.md`. Data profile → `DATA_ANALYSIS.md`. Exact evaluator (`ber.metric`, `ber.decide.macro_f05_frame`).
- Normalization v1 with learned cross-script dictionary + learned abbreviation maps.
- Blocking v1 (5 views incl. reverse): recall 0.9905, oracle 0.9972. Budget pruned to 43/S1 (train) and 51/S1 (test).
- Features v1 (87), LightGBM single and cross-fitted, deterministic baselines, P-dense evaluation, first error analysis.
- Research workflow #1 (9 topics): notes in `research/notes/*.md` (synthesis `LITERATURE_REVIEW.md` pending). Research workflow #2 (open-scope sweep) is running.

## 3. Best validation results (fold 0, 220,683 S1, exact metric)
| system | P-main | P-dense |
|---|---|---|
| **cv_v2s: stage-2 stacked (v1 + p1 + collective), expected-F / thr 0.7 + excl** | **0.98703 ± 0.00016** | – (dense variant blocked by spend limit) |
| cv_v1d: stage 1 trained on dense features, thr 0.7 + excl | – | **0.98479** |
| cv_v1 (3-fold LightGBM ensemble, 87 feats) + thr 0.7 + excl | 0.98492 ± 0.00016 | 0.98311 |
| cv_v1 + thr 0.8 + excl | 0.98467 | **0.98343** |
| best no-ML rule | 0.76697 | – |
| blocking oracle (given candidates) | 0.99645 | – |

## 3b. Caveats from the code review (2026-09-25)
- cv_v2s 0.98703 used `p1_tmargin` computed on a fold-restricted graph, so the gain may not transfer to test as-is. The re-run must use the fixed collective stage (p1 competition off, or all-fold features).
- P-dense and LOCO numbers are slightly optimistic (rank/frequency trace; target-country maps). Details in the RESEARCH_LOG "code review" entry.

## 4. Key facts learned
- The many-to-one competition features (c_combo_tmargin, trank) dominate importance. Hard exclusivity adds ≈0 on top.
- Probabilities are well calibrated at train density. The expected-F and π-hurdle decision rules do NOT beat a tuned threshold (negative results).
- The generator plants **sibling distractor entities** (same name, nearby house number, same street), and also shifts house numbers of TRUE copies. The remaining errors are mostly (a) empty-address targets with shared names (FN, often undecidable) and (b) sibling lookalikes (FP).
- Test density is higher, so the optimal threshold shifts 0.7 → 0.8 and mid-range probabilities become over-confident.

## 4b. BLOCKER (2026-09-25 evening)
**Modal workspace spend limit exceeded.** No Modal compute (CPU or GPU) until the user raises the limit or adds billing. Options: (1) raise the Modal limit, which resumes instantly with `modal run --detach modal_app/stage_chain.py::main`; (2) Kaggle/Colab/Lightning notebooks using the Modal-free path (`BER_ROOT=<dir>`, stage functions via `.local()`), which needs ≥ 64–128 GB RAM for the full-scale stages or chunked variants; (3) wait for the monthly credit reset.
Offline work that needs no compute: docs, packaging, notebooks.

## 5. Running now (before the blocker)
- `cv_v1d`: cross-fitted model trained on the dense-regime features.
- `cv_v2s`: stage-2 stacked model on v1 + collective features.
- `sub_v1`: first test submission (cv_v1, thr 0.7 + excl) + official validator.
- Test collective features (needed for a stage-2 test submission).

## 6. Next actions
1. Compare cv_v2s vs cv_v1 (stacking gain), and cv_v1d on dense.
2. Choose the submission config under P-dense. Produce sub_v2 (stacked, dense-tuned threshold).
3. Features v2: house-number geometry features (already coded in `features.py`: a_num_min_absdiff etc.) plus normalization v2 (anyascii, ° fix, name-map filter). Needs a rerun of features for train, train-dense and test.
4. LOCO (US↔India) for the France proxy (E-4). Self-trained France maps (E-8).
5. Learning curve: more training folds (E-11).
6. When research workflow #1 completes: review `LITERATURE_REVIEW.md` and fold it into the plan.
7. Final package: refactor stages into a Modal-free CLI (`ber.cli`), README, requirements.txt, Documentation_template.md.
