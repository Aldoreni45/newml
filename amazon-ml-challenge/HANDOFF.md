# HANDOFF.md — resume guide (checkpoint 2026-09-25, session 1)

Read `PROJECT_STATE.md` first, then the tail of `RESEARCH_LOG.md`. This file is the short version.

## What was completed
- Spec parsed and compliance checklist written. Full-data profile (DATA_ANALYSIS.md). Exact evaluator.
- Normalization with **learned** cross-script dictionary (1,347 Indic→Latin tokens) and learned abbreviation maps.
- Blocking v1: 5 sparse TF-IDF views incl. **reverse target→S1 retrieval** (R@1 0.972). Union recall 0.9905, oracle 0.9972. Pruned budget 43–51 per S1.
- Features v1 (87): exact cosines, RapidFuzz, token/IDF/number, static multilingual embedding cosines (potion-multilingual-128M, MIT, CPU), per-S1 and **per-target competition features**.
- LightGBM stage 1 (cross-fitted) + **stage-2 stacking with collective/sibling-support features**.
- Density-matched validation (P-dense) and LOCO (unseen-country) protocols.
- First submission **sub_v1** (validator PASS with --check-ids): `output/sub_v1/matching_results.tsv`, not uploaded yet.

## Current best validation scores (fold 0, 220,683 S1)
- P-main: **cv_v2s (stage 2 stacked) 0.98703 ± 0.00016**; cv_v1 (stage 1) 0.98492.
- P-dense: cv_v1d (trained dense) 0.98479; cv_v1 (trained normal) 0.98343.
- LOCO: US→India 0.94475, India→US 0.97390 (in-domain ≈0.984).

## Current best architecture
normalize (learned maps) → country partition → 5-view TF-IDF retrieval (incl. reverse) → prune → 87 features → **stage-1 LightGBM (dense-regime training)** → OOF p1 → collective/sibling features → **stage-2 LightGBM** → threshold (0.7–0.8, tuned under P-dense) + exclusivity (or expected-F; equivalent here).

## Important discoveries
1. The many-to-one structure (every target ↔ ≤1 S1) drives both blocking (reverse view) and scoring (competition margins dominate importance).
2. The generator plants **sibling distractors** (same name, same street, nearby house number) and shifts house numbers of true copies. Copy-cluster consistency (stacking) is the fix: +0.0021.
3. Test is denser in distractors, so tune the threshold under P-dense and train on dense features.
4. Unseen country costs 0.01–0.04. France = 15% of test, so this is the biggest remaining lever.
5. Decision theory (expected-F, π hurdle) ≈ tuned threshold here (well-separated, calibrated p). Negative results are documented.

## Rejected / blocked
- Modal GPU: blocked until the user adds a payment method. The Kaggle GPU route needs user approval to upload data as a private dataset.
- Postcode blocking: postcodes too sparse. Name-only blocking: flooded by same-name entities.
- π hurdle / independence expected-F as the default rule: no gain over the threshold.

## In flight at checkpoint time
- stage_collective train (v1_dense, cv_v1d) → then `stage_train.py::cv --feat-tag v1_dense_col --tag cv_v2sd` (stage 2 dense).
- stage_collective test (cv_v1d, suffix _d) → test feat `feat_v1_col_d` → stage_predict with cv_v2sd fold models.
- LOCO: us2in_norm, us2in_pseudo, in2us_pseudo (per-country normalization and self-training for unseen countries).
- Research workflows #1 (resumed) and #2 open-scope (resumed).

## BLOCKER
Modal spend limit exceeded (2026-09-25). Resume with `modal run --detach modal_app/stage_chain.py::main` once the limit is raised. That single command produces sub_v2 (dense-regime stacked model, P-dense-tuned threshold) and writes reports/chain_sub_v2.json.

## Exact next actions
1. When the dense collective finishes: `modal run --detach modal_app/stage_train.py::cv --feat-tag v1_dense_col --tag cv_v2sd`.
2. Evaluate cv_v2sd on dense fold 0 (its own report). Pick the threshold.
3. Predict test: `modal run --detach modal_app/stage_predict.py --feat-tag v1_col_d --model-tag "cv_v2sd_f1,cv_v2sd_f2,cv_v2sd_f3" --rule '{"type":"thr_excl","tau":<τ>}' --out-tag sub_v2`.
4. If LOCO self-training helps: implement it for France in the final model (pseudo-label France test pairs; retrain).
5. Features v2 (house-number geometry already coded) + normalization v2, then a full clean end-to-end run for reproducibility.
6. Final package: Modal-free CLI, README, requirements.txt, Documentation_template.md, zip.

## Risks
- Laptop network drops kill `modal run` clients. Always use `--detach`, and verify that every output has its JSON report (a parquet without a report may be partial).
- The France gap is unmeasurable without labels. Rely on LOCO evidence and conservative thresholds for unseen countries.
- `candidate_pairs.tsv` for test is about 88M IDs (~1.2 GB). Check the zip size limits of the final package.

## Compliance status
No external data. Model weights: potion-multilingual-128M (MIT, rev 73908c34). Everything else is trained from scratch on the provided data. Library licences are listed in IMPLEMENTATION_PLAN. The GPL `unidecode` fallback was replaced by `anyascii` (ISC) in the code; this takes effect in normalization v2.

## STOPPED BY USER (2026-09-25, end of session 1)
Everything was stopped at the user's request. No Modal jobs are running (the spend limit was hit anyway).
Saved state:
- All 14 artifacts + DATA_ANALYSIS.md + HANDOFF.md + README.md + Documentation_template.md are current.
- Code includes every code-review fix (see the RESEARCH_LOG "code review" entries); the fixes take effect on the next run.
- Research notes: `research/notes/*.md` (8 topics; competitions_repos note missing), `research/open_scope/*.md` (4 of 8 sweeps).
  LITERATURE_REVIEW.md and OPEN_SCOPE_REVIEW.md were NOT synthesized yet.
- Submission ready: `output/sub_v1/matching_results.tsv` (validator PASS).

## HOW TO RESUME
1. Research: resume both workflows from their run IDs (completed agents are cached):
   - wf_fb338127-211 (script: ~/.claude/projects/D--ML-Project-hackathon-amazon-ml-er-modal-app/.../ber-literature-research-wf_fb338127-211.js)
   - wf_76dac818-5a5 (script: ~/.claude/projects/D--ML-Project-hackathon-amazon-ml-er-research-notes/.../ber-open-scope-research-wf_76dac818-5a5.js)
2. Compute (after the Modal spend limit is raised): re-run normalization (fixed), blocking, features, then `modal run --detach modal_app/stage_chain.py::main`. Alternatively run `python run_pipeline.py --workdir <dir> --zip <zip>` on a machine with ≥ 128 GB RAM.
