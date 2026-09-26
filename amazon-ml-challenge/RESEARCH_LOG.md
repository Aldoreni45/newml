# RESEARCH_LOG.md

Chronological log. Newest entries at the bottom. Every entry: date, what was done, what was learned, decisions.

---

## 2026-09-25 — Session 1: kickoff

**Environment constraints (from the user, binding):**
- Dev laptop: Ryzen 5 5500H, 5.9 GB RAM (≈0.4 GB free), RTX 2050 4 GB. **No AI/ML model may run on the laptop** (user instruction). Laptop is used only for editing code, stdlib/streaming sanity checks, and the pure-Python evaluator.
- All heavy compute → cloud. Modal CLI is authenticated (profile `ayanpthan768`); Kaggle credentials exist. Colab / Lightning AI are alternatives.

**Phase 0 — spec parsed** (PDF + `student_resource/README.md` + `utils/validate_submission.py` + `Documentation_template.md`). Key rules captured in `COMPLIANCE_CHECKLIST.md`. Notable details:
- Validator says the full test set has "~1.7M entities" — matches the 1,732,544 test S1 rows.
- ID-existence check is optional (`--check-ids`) and a nonexistent ID only lowers the score.
- `candidate_pairs.tsv` must be the LAST candidate list fed to the model; matches ⊆ candidates.

**Phase 1 (partial) — raw sizes (wc -l, minus header):**

| file | rows |
|---|---|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

First-look observations from file heads (to be quantified by the profiler):
- Records are synthetic with injected noise. S2 style: UPPERCASE addresses with USPS-like abbreviations (ST, AVE), some Indic-script names; S3 style: full US state names ("Texas"), "Unit", "PO Box", abbreviations (Ct, Dr, Ln).
- Name noise: legal suffix moved to the front ("LLC Moncada Léarning Center", "Pvt. EFS Print Ventures Ltd."), junk prefixes (`--`, `<<`), `##` in addresses, injected diacritics, typos ("Hospirlg", "Hurricanne"), doubled spaces, web domains as names ("heassociates.com"), " | www.x.com" suffixes, DBA ("Ectolumdrex dba X+ Madison Inc"), repeated tokens ("VIDYALAYA VIDYALAYA").
- **Cross-script Indian names**: Devanagari/Tamil/Kannada full names ("राम मार्केटिंग प्राइवेट लिमिटेड") and mixed-script ("Sun पावर Provision"); native-script state names in addresses ("ಕರ್ನಾಟಕ", "महाराष्ट्र"); literal "null" in addresses.
- GT: typical S1 has 2–6 matches split across S2 and S3 → S2/S3 each hold several copies of one entity.
- Test France examples: "R. DE DIEPPE", "5 bis Rue Pierre Dignac", SARL/SAS/SASU/SCI, lowercase names, accents injected/removed.

**Infra:** dataset uploaded to Modal volume `ber-data` (`/raw/student_resource.zip`); `modal_app/profile_data.py` extracts → parquet and profiles full data (8 CPU / 64 GB container).

**Evaluator:** `src/ber/metric.py` — exact per-S1 F0.5 macro; self-test reproduces the spec example (0.714). Includes blocking diagnostics (pair recall, entity full-recall, oracle F0.5, reduction ratio).

**Research:** launched a 9-topic research workflow (classical RL, blocking, neural EM, name/address/transliteration, license-verified models, F-beta decision theory, multi-source clustering, contrastive/augmentation, competitions+repos), each followed by an independent citation/license auditor and a synthesis step → `LITERATURE_REVIEW.md`, `research/notes/*.md`, `research/REFERENCES.md`, `research/MODEL_LICENSES.md`.

## 2026-09-25 — Session 1: profiling + normalization results

**Profiling (Modal, 8 CPU/64 GB, ~6 min)** → `DATA_ANALYSIS.md`. Decisions that follow from the data:
- GT is many-to-one (no target linked to 2 S1) → evaluate a target-exclusivity constraint.
- Country agreement 100% on true pairs → block within country partitions (label-agnostic).
- Postcodes are rare → no postcode-centric blocking; address word TF-IDF + house numbers instead.
- 26–36% of S1 names are non-unique → the address must carry the decision; name-frequency features.
- Cross-script names in India targets (~19–28%) → need a cross-script bridge.
- Test is denser in distractors than train → density-matched validation.

**Normalization v1 (Modal, 32 CPU/128 GB, 5.2 min for 24M records):**
- Rule-based Brahmic→Latin transliteration (ISCII-shared block layout: one table covers 9 scripts; virama, matras, anusvara, word-final schwa deletion) + phonetic skeleton.
- **Learned native→Latin token dictionary** from 551k train pairs (position alignment when token counts agree: 549,951 of 551,240 pairs aligned) → 1,347 tokens. Effect on true India pairs whose target is native-script: name token-set quantiles (p1, p5, p10) 5/7/8 → 82/95/100. Mixed-script: 19/26/32 → 89/100/100. Test coverage of native tokens: 96.4% (name), 100% (address).
- **Learned abbreviation maps** (tokens that differ between the two sides of true pairs + abbreviation-shape test + Dice/precision filter): US address 83 entries (street/road/drive/avenue + 50 state codes), India address 19 (state codes incl. tg→telangana). The US name map picked up "md→mdpc"-style artefacts caused by collapsing runs of single letters (M.D., P.C.). They are applied consistently to both sides, so harmless, but v2 should filter prefix-extensions.
- Effect on true pairs, address token-set p10: US 70 → 84, India-latin 70 → 86.
- Generic cleaning: NFKC, diacritics folded, null tokens removed, `&`→and, apostrophes dropped, runs of single letters collapsed (L.L.C.→llc), leetspeak repaired in name tokens with ≥3 letters, alias split (dba/aka/fka), domain/handle core extraction.
- Bug found and fixed along the way: a regex `\b` written via a Python non-raw string became a literal backspace (0x08). Unit checks caught it before the full run.

## 2026-09-25 — Session 1: blocking benchmark, pipeline code, parallel tracks

**Blocking benchmark** (India 10% sample; details in BLOCKING_STRATEGIES.md): name-only top-40 recall 0.826; address top-40 0.909; hybrid top-40 0.984; **reverse target→S1 top-8 0.996**; union 0.997 at 124 candidates/S1. Conclusion: the many-to-one structure makes target-side retrieval the strongest single blocker. The name views need smaller K.

**Code written (all stages persist to the Modal volume):**
- `stage_features.py`: exact cosines, RapidFuzz, token/IDF/number features, competition features over the full graph, per-view candidate pruning (`--prune`).
- `stage_train.py`: `train` (single split) and `train_cv` (cross-fitted OOF + fold-averaged val predictions + S1-level **match-exists model π**). Evaluates thresholds, exclusivity, expected-F (independence DP), π-hurdle rule (`pi_select`), with a bootstrap SE.
- `stage_collective.py`: stacking stage that adds stage-1 p1, collective/sibling-support features and p1 margins.
- `stage_analyze.py`: FP/FN taxonomy counts and samples.
- `stage_predict.py`: test inference, decision rule, both TSVs, **official validator with --check-ids**.
- `stage_embed.py`: multilingual-e5-small (MIT, revision sha recorded) embeddings on A10G, plus exact dense top-k per country on GPU.

**Compliance hygiene:** replaced the GPL-2 `unidecode` fallback with `anyascii` (ISC). Degree/numero signs are mapped to spaces ("N°37" must not become "ndeg37"). This takes effect on the next normalization run.

**Parallel tracks running:** blocking (train, test; about 1.5 h each), embeddings (A10G, train and test), research workflow #1 (9 topics) and #2 (open-scope sweep of any venue or source type, per user instruction).

## 2026-09-25 — Session 1: full blocking results (E-B1, E-B2 done)
- Train, full pool: union pair recall 0.9905, oracle macro F0.5 0.9972, 112 candidates/S1. **Reverse target→S1 retrieval R@1 = 0.972**, R@8 = 0.9875. On the full pool the name view collapses to R@40 = 0.677 (same-name flooding).
- Budget decision: rev8 ∪ combo10 ∪ name/cat/addr top-5, about 45 candidates/S1 at ≈0.989 recall.
- Misses: mostly empty-address targets with generic shared names. Left out deliberately (unsafe to match).
- GPU on Modal is blocked ("add a payment method"). The embedding track switched to a CPU static model (potion-multilingual-128M, MIT): train took 36 min on 32 vCPU for 3×12.5M texts, rev 73908c34. Test is running.
- Launched train features v1 (folds 0–3, pruned, with embedding cosines).

## 2026-09-25 — E-0 / E-1 first validated scores
- Features v1: 38.0M pairs (folds 0–3), 87 features, 31.5 min on 32 vCPU. Pruned graph: 95.2M pairs over all 2.2M S1 (43/S1).
- **LightGBM v1 = 0.98456 macro F0.5** on fold 0 (220,683 S1): threshold 0.7 + exclusivity. Early stop at 777 rounds, 5 min training.
- Deterministic baselines on the same candidates: core-name equality 0.422; + best-for-target 0.714; combo ≥ 0.7 + best-for-target 0.767. Learning adds about 0.22 absolute F0.5.
- Exclusivity adds only +0.00002 on top of LightGBM: the competition features (tmargin/trank) already encode the many-to-one constraint. The independence expected-F rule (0.98415) is slightly below the tuned threshold. Next step is the π hurdle (cv_v1).
- Headroom vs the blocking oracle: 0.9972 − 0.9846 = 0.0126 → error analysis next.

## 2026-09-25 — E-1b cross-fitted model, E-3/E-7 decision rules, first error analysis
- **cv_v1 = 0.98492 ± 0.00016** (3 fold models averaged; thr 0.7 + exclusivity). Pair P 0.9959, R 0.9649. Oracle given candidates 0.99645.
- Calibration is good (reliability table in train_cv_v1.json), so decision-theoretic rules are meaningful. Still, neither expected-F (0.98457) nor the π hurdle (0.98466) beats the tuned threshold. With probabilities this well separated (most mass in <0.02 or >0.98), per-entity set optimisation has little to reclaim. The independence assumption also misprices correlated copies. Kept as documented negative results; threshold + exclusivity stays.
- Per stratum: n_true=1 is the weakest (0.954), singletons 0.980, n_true ≥ 4 ≥ 0.988.
- **Error analysis (lgb_v1, thr 0.7+excl)**: 3,332 FP vs 17,885 FN-in-candidates. FN: 49% have an empty target address, 59% a shared S1 name, 46% are not best-for-target. FP: 57% core-name equal, 38% shared S1 name, 21% empty target address.
  - **Key discovery:** the generator plants **sibling distractor entities**: same or near-same name, same street, nearby house number ("Amber Inc 340 vs 345 Olympic Park Dr", "Quex Enhanced Group 1927 vs 1931 Ray Leonard Rd"). It also applies small numbering offsets to TRUE copies ("Downtown Deli 104 vs 103 Main St", "69171 vs 69169 Primrose Rd"). Number offsets alone are therefore ambiguous. The discriminating signal has to be copy-cluster consistency.
  - Response: house-number geometry features (min abs/rel offset, digit edit distance) for features v2, and collective features v2 (p1-weighted support, same- vs cross-source sibling confirmation, house-number agreement with the confident cluster, isolation flag). Launched the stacking stage.

## 2026-09-25 — E-5 density-matched validation (P-dense)
- Features v1_dense: 410,795 S1 from folds 4–9 removed from the candidate graph (their ~1.4M targets become orphan distractors), so train targets/S1 = 5.74 ≈ test 5.75. Fold 0 keeps all its S1s, so the evaluation population is unchanged.
- **cv_v1 models evaluated on v1_dense fold 0: best 0.98343 at thr 0.8 + excl** (vs 0.98492 at thr 0.7 on normal density). Δ = −0.0015.
- Optimal threshold shifts 0.7 → 0.8. Middle-bin calibration degrades (pred 0.547 → obs 0.437 in [0.5, 0.6)). The mean predicted matches per S1 (Σp) stays 3.44 vs 3.46 true, so the shift is concentrated in the ambiguous pairs.
- Implication: tune the test decision threshold under P-dense, not P-main. Train on dense-regime features so competition features see orphan targets (running: cv_v1d).

## 2026-09-25 — first test submission sub_v1
- cv_v1 ensemble, thr 0.7 + exclusivity, on test features v1 (88.4M candidate pairs, 51/S1).
- **Official validator with --check-ids: PASS** (1,732,544 rows; 97,464 empty; 5.86M links). Re-validated locally with the stdlib validator: PASS.
- Test prediction profile: 3.38 predicted matches per S1; empty rate 5.63%; Σp per S1 3.47. Train has 3.46 matches per S1 and 5.58% singletons, so the generator seems consistent across splits and the extra test targets are mostly distractors.
- France: 3.45 matches per S1, empty 5.10%, a plausible profile for an unseen country.
- File: `output/sub_v1/matching_results.tsv` (98 MB). Local estimate: P-main 0.9849 / P-dense 0.9831. Not yet uploaded (the user uploads via the portal).

## 2026-09-25 — E-6 stacking + collective features: significant gain
- Stage 2 (`cv_v2s`): v1 features + OOF stage-1 probability p1 + 16 collective/sibling-support features + p1-based competition (tmargin/qrank/qgap/qsum). 3-fold cross-fit, 172 rounds (ES logloss 0.00452 vs 0.00572 for stage 1).
- **P-main 0.98703 ± 0.00016** (expected-F rule; thr 0.7 + excl 0.98697) vs 0.98492 → **+0.0021 (≈13 SE)**. Pair precision 0.9959 → 0.9973; recall 0.9649 → 0.9680.
- Every n_true stratum improves: n_true=1 0.954 → 0.957; ≥ 4 → ≥ 0.990. Singletons 0.980 → 0.979 (flat).
- Leakage check: fold 0 (eval) never enters any training. Its p1 comes from stage-1 models trained on folds 1–3 only, so the eval score is clean. The stage-2 training folds carry a mild second-order leak from cross-fitting; it does not affect the eval fold.
- E-5b: **cv_v1d** (stage 1 trained on v1_dense), evaluated on dense fold 0 = 0.98479 vs 0.98343 for the normally trained model. Density-matched training restores calibration: mid-bin reliability is back within about 0.02.
- Decision: final pipeline = dense-regime stage 1 → collective features → dense-regime stage 2. Running: collective on train (v1_dense, cv_v1d) and test (cv_v1d, suffix _d). Also queued E-4 LOCO and E-11 learning curve.

## 2026-09-25 — E-4 LOCO + E-11 learning curve
- US→India 0.94475 (best at thr 0.9), India→US 0.97390 (thr 0.8). In-domain ≈0.984. **The unseen-country gap is the largest remaining lever** (France = 15% of test). OOD models are over-confident, so the optimal threshold rises.
- Learning curve: 1 fold 0.9838 → 2 folds 0.98456 (+0.0008).
- Launched the LOCO testbed (`stage_train.py::loco`): (a) per-country percentile standardisation of features (label-free), (b) self-training with pseudo-labels on the target country (p ≥ 0.9 & exclusive-best → 1, p ≤ 0.1 → 0; target labels never used for fitting).
- Research workflows #1 and #2 had died when my usage limit hit (agents marked failed at 02:52). Stopped and resumed both from their journals; cached agents are reused.

## 2026-09-25 — unseen-country fixes (LOCO) + compute blocker
- **Per-country percentile standardisation of features: rejected.** US→India 0.93369 vs 0.94475 baseline; it destroys informative absolute similarity levels.
- **Self-training on the unseen country: mixed.**
  - India→US: 0.9739 → **0.97606 (+0.0022)**; pseudo-positive precision 99.3%.
  - US→India: 0.94475 → 0.94301 (−0.0017); pseudo-positive precision only 96.5% (label noise).
  - Conclusion: it helps only with very clean pseudo-labels. For France, use a stricter cut (p ≥ 0.97 + exclusive) if applied, and weigh its risk. Priority lowered.
- Infra: repeated laptop network drops (getaddrinfo / connection reset) killed several launches. Wrote `modal_app/stage_chain.py` to run collective → stage-2 → test collective → predict in ONE detached container.
- **BLOCKER: Modal returned "Workspace has exceeded its spend limit". No further cloud compute is possible until the user raises the Modal spend limit or adds billing, or picks another provider (Kaggle / Colab / Lightning AI).** The Modal volume still holds every artifact.
- Pending runs blocked: `stage_chain.py::main` (dense stacking → sub_v2), features v2, final clean end-to-end run.

## 2026-09-25 — adversarial code review (round 1) and fixes
Four review dimensions (metric/decision, leakage/validation, submission/compliance, features/normalization). Each finding was checked by 2 skeptical verifiers; the session limit interrupted part of the verification, so the review was resumed. Actions taken (code only, no compute):

**Validation-integrity findings. They change how earlier numbers must be read:**
1. `p1_tmargin` (stage 2, per-target competition on stage-1 probabilities) was computed over fold 0–3 S1s only in train but over all S1s at test. It is a train/test feature shift. The validation fold had the same truncation, so **cv_v2s = 0.98703 is internally consistent but may not transfer to test as-is**. Fix: `stage_collective` now gives p1 to every S1 in the file (OOF / val / label-free ensemble for other folds). It computes `p1_tmargin` only with `--p1-competition`, which requires all-fold train features, and otherwise skips it for both train and test.
2. **P-dense leaked a trace of the dropped owner.** `r_rev`/`s_rev` (target-side ranks) and `q/t_nm_freq_s1` were computed with the dropped S1s present. Fix: re-rank `r_rev` among surviving S1s and compute name frequencies without the dropped S1s. **P-dense numbers so far (0.98343 / 0.98479) are therefore slightly optimistic.**
3. **Exclusivity at validation only saw fold-0 competitors**, while test sees all S1s. Fix: `train_cv` builds exclusivity over the whole predicted pool (OOF + val + label-free predictions on any other folds present).
4. **τ was chosen and reported on fold 0** (small selection bias, ≈ SE level). Fix: τ is chosen on stage-level OOF predictions (`tau_oof`, 0.40–0.95 grid). The headline is `val_at_tau_oof`; the val-selected rule is kept only as `best_rule_on_val_optimistic`. The extra-miss rate is now also estimated from OOF.
5. **LOCO used normalization maps learned with the target country's labels.** The measured unseen-country gaps (−0.011 / −0.040) are therefore **lower bounds**; France (no learned maps) may lose more. The final plan must lean on conservative unseen-country handling.

**Bugs fixed:**
- Native-map keys kept ZWNJ (U+200C), but lookups strip it. Frequent Telugu/Kannada tokens such as "ఇన్‌ఫ్రాస్ట్రక్చర్" never matched the learned dictionary. The map is now learned on the canonical (NFKC, ZW-stripped) form.
- Abbreviation maps were not closed under composition (kl→kerala→keralam); `close_map` fixes that.
- Gurmukhi tippi/addak and Malayalam chillu were dropped by `translit_indic`.
- `phonetic()` merged "ch" with "k".
- Exclusivity ties were non-deterministic.
- `s1_aggregates` p_2nd/p_3rd were wrong for short lists.
- Error-analysis singleton count used candidate labels instead of GT.
- Zip had an extra top-level folder (the spec requires `output/`, `code/`, `Documentation_template.md` at the root).
- Validator failure did not fail the run.
- The embedding revision was not pinned; HF_HOME was hard-coded.
- `BER_ROOT` was reused as the Modal mount path, which crashes with Windows paths. There are now separate `MOUNT` and data roots.
- Test p1 ignored `cv_folds` and silently fell back to 0.

**Submitted file sub_v1** (stage-1 cv_v1, τ 0.7 + exclusivity) is unaffected by 1 and 2 (no stage-2 features; normal-density training). It is mildly affected by 4 (τ chosen on fold 0; the 0.6/0.7/0.8 scores differ by ≤ 0.0004) and by 3: test exclusivity is stricter than validated, which removes pairs whose target has a higher-scoring S1. That is expected to be precision-positive.
- Review round 1 is complete (70 agents; 11 findings confirmed by both verifiers). Every confirmed finding is fixed in code. Several "rejected" items were real bugs that the verifiers checked after I had already fixed them (ZWNJ map, P-dense trace, zip layout, validator gating, mount path, tie-break, p_2nd).
- The last confirmed item is fixed as well: `stage_normalize` now learns the native/abbreviation maps **without fold 0** (`--exclude-folds 0`). The next normalization run keeps the evaluation fold clean.
- **All fixes take effect only on the next compute run. Existing artifacts/scores were produced by the pre-fix code; the caveats are listed above.**
