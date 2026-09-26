# EXPERIMENT_PLAN.md

Priority: **P0** likely high impact · **P1** promising · **P2** exploratory · **P3** low value.
Expected gain / compute cost drives the order. All results go to `MODEL_COMPARISON.md` (leaderboard) and `RESEARCH_LOG.md`.
Protocols are defined in `VALIDATION_STRATEGY.md`.

## Experiment record template
```
ID | date | hypothesis | split/protocol | candidates (tag) | features (tag) | model + params | decision rule |
macro F0.5 | P | R | cand. recall | runtime | memory | observations | decision
```

## Queue

| ID | P | Hypothesis | Protocol | Cost | Status |
|---|---|---|---|---|---|
| E-B1 | P0 | Multi-view TF-IDF blocking (name/cat/addr/combo + reverse) reaches ≥ 99% pair recall at ≤ 80 cands/S1 | train, all folds | 32 CPU ≈ 1 h | **done**: 0.9905 @112; oracle 0.9972 |
| E-B2 | P0 | Recall@K grid picks per-view K that keeps ≥ 99.5% of the union recall with fewer candidates | from E-B1 report | 0 | **done**: 0.989 @ ~43 |
| E-0 | P0 | Deterministic baselines (exact core-name; + best-for-target; combo ≥ θ + best-for-target) | P-main | CPU min | **done**: 0.422 / 0.714 / 0.767 |
| E-1 | P0 | LightGBM on features A–G beats the best deterministic baseline by a wide margin | P-main | 32 CPU ≈ 30 min | **done**: 0.98456; cv 0.98492 |
| E-2 | P0 | Many-to-one exclusivity improves F0.5 (precision ↑, small recall loss) | P-main | 0 | **done**: +0.00002 (subsumed by competition features) |
| E-3 | P0 | Per-S1 expected-F0.5 set selection ≥ best global threshold | P-main, P-dense | 0 | **done**: ≈ threshold (−0.0004 stage 1; +0.00006 stage 2) |
| E-4 | P0 | LOCO (US→India, India→US) quantifies the unseen-country drop; country-agnostic features keep it small | P-loco | 2× E-1 | **done**: −0.040 / −0.011; per-country standardisation hurts (0.934) |
| E-5 | P0 | Density-matched validation (drop 19% S1) lowers F0.5 and shifts the optimal threshold upward | P-dense | features rerun | **done**: −0.0015, τ 0.7→0.8; dense training recovers +0.0014 |
| E-6 | P1 | Collective features (t vs q's top-1 candidate; near-duplicate co-candidates) help alias/pure-address targets | P-main | features rerun | **done**: +0.0021 → 0.98703 |
| E-7 | P1 | Singleton model P(no match \| S1-level aggregates) improves the empty-set decision beyond independence-based E[F] | P-main | small | **done**: no gain (0.98466 vs 0.98492) |
| E-8 | P1 | Self-trained abbreviation maps for unseen countries (mine from high-confidence test pairs) improve France without labels; simulate on LOCO (India maps mined without labels) | P-loco | CPU | todo |
| E-9 | P1 | Multilingual embedding cosine features (MIT/Apache encoder, GPU) add recall on cross-script/alias cases | P-main | GPU ≈ 1 h | todo |
| E-10 | P1 | Fine-tuned small cross-encoder on hard pairs (p ∈ [0.05, 0.95]) stacked into LightGBM | P-main | GPU 2–4 h | todo |
| E-11 | P1 | More training S1 (folds 1–9) → +ΔF (learning curve with 1, 2, 4, 8 folds) | P-main | CPU | partial: 1→2 folds +0.0008 |
| E-12 | P2 | CatBoost / XGBoost vs LightGBM on the same features; rank-average ensemble | P-main | CPU/GPU | todo |
| E-13 | P2 | Per-country / per-source thresholds (overfitting risk) | P-main + P-rep | 0 | todo |
| E-14 | P2 | Graph clustering of targets (S2/S3 near-duplicate components) with ≤1 S1 per cluster | P-main | CPU | todo |
| E-15 | P2 | Synthetic noise augmentation (French-like abbreviations, accent injection) for robustness to France | P-loco | CPU | todo |
| E-V3 | P2 | Relearn normalization maps on training folds only → measure the optimistic leak in val | P-main | CPU | todo |
| E-16 | P3 | LLM pairwise judge (≤8B, Apache/MIT) on the most ambiguous pairs; too slow for 100M pairs, only for a tiny slice | P-main | GPU | later |
| E-8b | P1 | Self-training on unseen country (pseudo-labels p≥0.9 & exclusive) | P-loco | CPU | **done**: India→US +0.0022 (clean labels), US→India −0.0017 (noisy) → try a stricter cut |
| E-17 | P0 | Dense-regime stacking (collective on v1_dense + stage 2) → sub_v2 | P-dense | CPU ≈ 1.5 h | **blocked (Modal spend limit)**; `stage_chain.py::main` ready |
| E-18 | P1 | Features v2 (house-number geometry) | P-main/P-dense | features rerun | coded, not run |
