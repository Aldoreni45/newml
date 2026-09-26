# MODEL_COMPARISON.md — experiment leaderboard

All scores: exact macro F0.5 on the evaluation population defined in `VALIDATION_STRATEGY.md` (P-main = eval fold 0, every S1 of the fold counted). "±" = bootstrap SE over S1 (200 resamples). Only measured numbers appear here; anything not yet run is marked "—".

## Leaderboard

| rank | ID | date | architecture | candidates | features | model | decision | macro F0.5 | pair P | pair R | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **E-6 cv_v2s** | 2026-09-25 | E: B + collective/stacking | B-v1-pruned | v1 + p1 + 20 collective (108) | stage-2 3-fold LightGBM on OOF p1 (172 rounds) | expected-F (≈ thr 0.7+excl 0.98697) | **0.98703 ± 0.00016** | 0.9973 | 0.9680 | +0.0021 vs cv_v1 (≈13 SE) |
| – | E-5 cv_v1d on dense | 2026-09-25 | B trained at test-like density | same | v1_dense | 3-fold (855 rounds) | thr 0.7 + excl | P-dense: **0.98479** | 0.9958 | 0.9647 | vs cv_v1 on dense 0.98343: dense training recovers +0.0014 |
| 2 | E-1b cv_v1 | 2026-09-25 | B | B-v1-pruned | v1 (87) | 3-fold cross-fitted LightGBM (folds 1,2,3; 855 rounds), fold-averaged on val | p ≥ 0.7 + excl | **0.98492 ± 0.00016** | 0.9959 | 0.9649 | calibrated; singletons 0.980; n_true=1 0.954 |
| – | E-7 cv_v1 | 2026-09-25 | B | same | same | same + S1-level π model | π-hurdle expected-F (+excl) | 0.98466 | – | – | π does NOT beat threshold (negative result) |
| – | E-3 cv_v1 | 2026-09-25 | B | same | same | same | expected-F independence DP (+excl) | 0.98457 | – | – | negative vs threshold |
| 2 | E-1 lgb_v1 | 2026-09-25 | B: retrieval + GBDT | B-v1-pruned (43/S1) | v1 (87 feats) | LightGBM lr .08, 255 leaves, 777 rounds (ES fold 3), train folds 1–2 (19.0M pairs) | p ≥ 0.7 + exclusivity | **0.98456** | – | – | val = fold 0 (220,683 S1; 9.49M pairs); blocking pair recall 0.9883 |
| – | E-1 lgb_v1 | 2026-09-25 | B | same | same | same | p ≥ 0.8 + excl | 0.98450 | – | – | |
| 3 | E-3 lgb_v1 | 2026-09-25 | B | same | same | same | expected-F0.5 top-k* (+excl) | 0.98415 | – | – | independence DP slightly below the best threshold |
| 4 | E-1 lgb_v1 | 2026-09-25 | B | same | same | same | p ≥ 0.5 + excl | 0.98266 | – | – | |
| – | E-0 B2 | 2026-09-25 | A: no ML | same | – | rule | combo ≥ 0.7 ∧ best S1 for target | 0.76697 | – | – | best deterministic rule |
| – | E-0 B1 | 2026-09-25 | A | same | – | rule | core-name equal ∧ best S1 for target | 0.71376 | – | – | |
| – | E-0 B0 | 2026-09-25 | A | same | – | rule | core-name equal | 0.42236 | – | – | |

Threshold curve for lgb_v1 (exclusive): τ 0.2 → 0.9714, 0.3 → 0.9771, 0.4 → 0.9804, 0.5 → 0.9827, 0.6 → 0.9840, **0.7 → 0.9846**, 0.8 → 0.9845, 0.9 → 0.9829. The optimum sits well above 0.5, as F0.5 decision theory predicts: with calibrated p, the optimal plug-in threshold is F*/(1+β²) ≈ 0.79 at F* ≈ 0.98.
Top gain features: c_combo_tmargin ≫ c_combo_trank > r_rev > a_q_num_missing > c_combo_tgap > n_idf_cov_t > a_num_jacc > a_tset > n_tsort. The **many-to-one competition features dominate**.

## Blocking comparison

| ID | views / K | pair recall | entity full recall | oracle macro F0.5 | cands per S1 | runtime |
|---|---|---|---|---|---|---|
| B-v1 | name40, cat20, addr40, combo40, rev8 | 0.9905 | 0.967 | 0.9972 | 112.3 | 2h10m (train, 32 vCPU) |
| B-v1-pruned | rev8 ∪ combo10 ∪ name5 ∪ cat5 ∪ addr5 | ≈0.989 | — | — | ≈45 | (subset of B-v1) |
| B-bench | rev only, top-8 | 0.9875 | — | — | ≤ 8·(targets/S1) | — |

## Ablations (component removed → Δ vs. full system)

| component | Δ macro F0.5 | evidence |
|---|---|---|
| learned native dictionary | — | — |
| learned abbreviation maps | — | — |
| exclusivity (many-to-one) | — | — |
| expected-F / π decision vs. best threshold | — | — |
| competition features | — | — |
| reverse (target→S1) retrieval view | — | — |


## Unseen-country robustness (E-4 LOCO; France proxy)
Train on one country's folds 1–2 (ES fold 3), evaluate the other country's fold 0. Features v1, single LightGBM.

| ID | train → eval | best rule | macro F0.5 | in-domain reference (lgb_v1, same country) | gap |
|---|---|---|---|---|---|
| loco_us2in | US → India | thr 0.9 + excl | 0.94475 | ≈0.984 | **−0.040** |
| loco_in2us | India → US | thr 0.8 + excl | 0.97390 | ≈0.984 | −0.011 |

The out-of-domain optimum threshold moves up to 0.8–0.9 (over-confidence). US → India is much worse because cross-script names and India-specific address structure are never seen in US-only training. France is Latin-script and will be scored by a US+India model, so its gap is expected to be closer to the India→US case. It is still material: at 15% of test, a −0.01 gap costs 0.0015 overall.

## Learning curve (E-11)
| training folds | pairs | P-main (thr 0.7 + excl) |
|---|---|---|
| 1 | 9.5M | 0.98380 |
| 2 | 19.0M | 0.98456 |
Doubling the data gives +0.0008, so more folds are worth adding in the final model (diminishing).
