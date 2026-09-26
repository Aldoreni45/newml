# VALIDATION_STRATEGY.md

Goal: a local score that ranks design choices the same way the **private leaderboard** will, and does not overfit one split.

## 0. Invariants (all protocols)
- **Metric = exact challenge metric** (`ber.metric.f05` / `ber.decide.macro_f05_frame`): per-S1 F0.5, macro over **every** S1 in the evaluation population. This includes singletons and S1s whose true matches blocking missed; those count with the true `n_true` from GT.
- **Grouping unit = S1 entity.** A target belongs to at most one S1, so grouping by S1 also groups its targets. No true pair spans two folds.
- **Full target pool kept.** Validation S1 queries are retrieved against all train S2/S3 records. Candidate density, hard negatives and competition features therefore stay realistic. Queries are never retrieved against a filtered pool.
- **Label-free transductive features** (IDF, name frequency, competition ranks over the candidate graph) are computed on the whole split, including the evaluation fold. That is legitimate because the test set gets identical, label-free treatment. Labels of evaluation folds are never used by any feature, map or model.
- Learned normalization maps (native dictionary, abbreviations) are currently learned from a 1.5M-pair sample of ALL train pairs, which includes validation-fold pairs. ⚠ This is a mild optimistic leak for the val score (the maps are then also applied to test, where the same generator vocabulary is expected). Check E-V3 relearns the maps on the training folds only and measures the difference.

## 1. Fold assignment
`fold = int(entity_id[3:]) % 10` for S1 records. IDs are random (Spearman with anything ≈ 0), so this is a random, reproducible, S1-grouped 10-fold split.
Default protocol **P-main**: train folds {1,2} (≈441k S1), early-stopping fold {3} (≈221k S1), evaluation fold {0} (≈221k S1). Final model: retrain on folds 1–9 with the number of rounds fixed from P-main (fold 0 stays as the untouched check).

## 2. Protocols

| ID | Protocol | Purpose | When |
|---|---|---|---|
| P-main | train {1,2}, ES {3}, eval {0} | primary model/rule selection | every experiment |
| P-rep | same model, eval on {9} and on {4,5} | variance check across folds | each accepted change |
| P-boot | bootstrap over eval S1 (1,000 resamples) | CI of macro F0.5; a change is "real" only if ΔF exceeds CI noise | reported for key changes |
| P-loco | train US only → eval India; train India only → eval US | **unseen-country proxy for France** (15% of test) | features/normalization changes |
| P-dense | drop a random 19% of train S1 from the whole graph; their targets become unlabeled distractors (test has ~24% more targets per S1) → recompute competition features → eval | decision-rule/threshold robustness under test-like distractor density | threshold/decision experiments |
| P-strata | breakdown by country, target script (latin/brahmic/mixed), true-count bucket, singleton vs not, S1 name ambiguity (nm_freq_s1>1), empty target address | error analysis; detects gains that only help easy strata | every experiment |
| P-hard | eval restricted to S1 with non-unique normalized names | hardest slice (chains, generic names) | model changes |
| P-lb | public leaderboard | sanity check only; never tune to it (private = the rest of test) | per submission |

## 3. Which protocol predicts the leaderboard best? (to be measured)
Hypothesis: P-main overestimates test for two reasons: (i) France is unseen, and (ii) test has a higher distractor density. The leaderboard-predictive estimate is therefore
`E[test] ≈ 0.85 · F(P-dense, US+India) + 0.15 · F(P-loco-like France proxy)`.
After the first leaderboard submission, compare this estimate with the public score and keep whichever protocol tracks it best.

## 4. Anti-overfitting rules
- Thresholds and other decision parameters are chosen on the ES fold or with P-dense, then **reported** on fold 0. Never choose and report on the same fold.
- Per-group thresholds (per country/source) are allowed only if they win on P-rep as well as P-main. Group-level tuning is the easiest place to overfit.
- A change is accepted only if Δmacro-F0.5 > 2× the bootstrap SE, or if it is neutral on score but simplifies the system.

## 5. Protocol corrections after the code review (2026-09-25)
- **Decision parameters (τ, extra-miss) are chosen on stage-level OOF predictions** of the training folds and *reported* on fold 0 (`val_at_tau_oof`). The fold-0-selected value is kept only as `best_rule_on_val_optimistic`.
- **Exclusivity is evaluated over the whole predicted graph** (OOF + val + label-free predictions of any other folds in the file), the same competitor set test sees.
- **P-dense** now re-ranks the target-side reverse view among surviving S1s and computes S1 name frequencies without the dropped S1s. Before this fix, the dropped owner left a trace, which made P-dense optimistic.
- **Per-target features of stacked probabilities** (`p1_tmargin`) are only allowed when train features cover all S1 folds. Otherwise they are skipped for both train and test.
- **LOCO caveat:** normalization maps are learned from all train pairs (incl. the target country), so LOCO measures a *lower bound* on the unseen-country gap. A strict LOCO would relearn the maps without the target country's pairs (E-V3b).
