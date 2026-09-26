# FINAL_RECOMMENDATION.md (interim, 2026-09-25; update after the dense stacked run)

## Executive summary
Build a **retrieval + stacked gradient-boosting** entity resolver:
- learned, country-agnostic normalization (a native→Latin dictionary learned from training pairs);
- five-view sparse TF-IDF blocking, including **target→S1 reverse retrieval**;
- ~90 engineered pair features, dominated by **many-to-one competition features**;
- a stage-1 LightGBM trained under **test-like distractor density**;
- **collective / sibling-support features** from out-of-fold stage-1 probabilities, then a stage-2 LightGBM;
- a threshold + exclusivity decision tuned under the density-matched protocol.

This is the strongest evidence-backed architecture from our experiments. We make no claim of winning; the numbers below are measured validation scores.

## Why (evidence)
| decision | evidence |
|---|---|
| country partition for blocking | 100% of 7.64M true pairs share the country label (label-agnostic, so it works for unseen labels) |
| learned cross-script dictionary | native-script true pairs: name token-set p10 8 → 100; covers 96.4% of test native tokens |
| reverse (target→S1) retrieval | R@1 = 0.972 alone; name-only retrieval R@40 = 0.677 |
| pruned budget (≈43–51 per S1) | recall 0.989 vs 0.9905 at 112 per S1; oracle 0.9965 |
| GBDT instead of rules | best rule 0.767 → LightGBM 0.9846 |
| competition features | top gain features (c_combo_tmargin ≫ others); exclusivity adds ≈0 on top |
| cross-fit ensemble | +0.0004 (0.98492) |
| stacking + collective features | **+0.0021 → 0.98703 ± 0.00016** (13 SE) |
| density-matched training | under test-like density, 0.98343 → 0.98479 |
| threshold ≈ 0.7–0.8 (+ exclusivity) | expected-F and π-hurdle rules did not beat the tuned threshold |

## Data analysis (short)
Train 2.2M/5.0M/5.3M (S1/S2/S3), test 1.73M/4.9M/5.1M; France is 15% of test and unseen. Every target belongs to ≤ 1 S1. 26% of targets are distractors, and the test has a higher distractor density. 5.6% singletons, 3.46 matches per S1. The noise includes planted sibling entities (same name, nearby house number). See DATA_ANALYSIS.md.

## Best blocking
rev (target→S1 hybrid) top-8 ∪ hybrid name+address top-10 ∪ name char3 top-5 ∪ name-concat char3 top-5 ∪ address words top-5. Details in BLOCKING_STRATEGIES.md.

## Best features / model
FEATURE_CATALOG.md groups B–G + collective (H). LightGBM (lr 0.08, 255 leaves, min_data 400, ff/bf 0.7, max_bin 127). 3-fold cross-fit at both stages; stage 2 adds p1 + 16 collective features + p1 competition.

## Thresholding / singletons
Threshold + many-to-one exclusivity, with τ chosen on the density-matched validation fold (0.7 at train density, 0.8 under P-dense for a normally trained model). Singletons are handled implicitly (no candidate above τ → empty list). Singleton accuracy ≈ 0.98. A dedicated π model was tested and gave no gain.

## Validation protocol
Fold = S1 id mod 10. Train folds 1–3, eval fold 0 (220,683 S1, exact metric), bootstrap SE ≈ 0.00016. P-dense for density shift and LOCO for unseen countries (VALIDATION_STRATEGY.md).

## Open risks and next steps
1. **Unseen country (France)**: LOCO gap 0.01–0.04. Self-training helps only with ≥ 99% clean pseudo-labels (+0.002 India→US, −0.002 US→India). Next: a stricter pseudo-label cut, France-specific self-trained abbreviation maps, and a conservative τ for unseen labels.
2. Compute: the Modal spend limit stopped the final dense stacked run (sub_v2). One command resumes it (HANDOFF.md).
3. Features v2 (house-number geometry) and normalization v2 have not been measured yet.

## Submission status
- `output/sub_v1/matching_results.tsv`: stage-1 cv_v1, τ 0.7 + excl. Official validator PASS with --check-ids. Local estimate 0.983–0.985.
- sub_v2 (dense stacked) is pending compute.
