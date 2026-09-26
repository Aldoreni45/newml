# SOLUTION_ARCHITECTURE.md

_Status: v1 under construction. Components move from "hypothesis" to "proven" only after an ablation in MODEL_COMPARISON.md._

## 1. Pipeline (v1)

```
raw TSV (S1, S2, S3; any country label)
   │
   ▼
[1] NORMALIZE (country-agnostic, script-aware)                       src/ber/normalize.py, records.py
    NFKC · diacritic folding · Brahmic→Latin rules (9 scripts, one table)
    · LEARNED native→Latin token dictionary (train pairs)            src/ber/aliases.py
    · LEARNED abbreviation maps (address/name; per country present in train)
    · alias split (dba/aka/fka) · handle/domain core · leetspeak · null tokens
   │
   ▼
[2] PARTITION by country label (lossless on train: 100% of true pairs share the label)
   │
   ▼
[3] MULTI-VIEW RETRIEVAL (sparse TF-IDF top-K, sparse_dot_topn)      src/ber/blocking.py
    name char_wb3 (K40) · space-less name char3 (K20) · address words (K40)
    · hybrid name+address (K40) · reverse target→S1 hybrid (K8)
    → union → candidate set C (+ per-view scores/ranks)
   │
   ▼
[4] PAIR FEATURES (≈90)                                              src/ber/features.py
    exact cosines · RapidFuzz similarities · token/IDF/number/initials/phonetic
    · structural (script, source, empties, name frequency)
    · COMPETITION features over the whole graph (per-S1 and per-target rank/gap/margin)
   │
   ▼
[5] PAIR SCORER: LightGBM (binary, early-stopped on a held-out S1 fold)
   │
   ▼
[6] DECISION LAYER                                                   src/ber/decide.py
    many-to-one exclusivity (each target → its best S1 only)
    · per-S1 expected-F0.5-optimal set (top-k*, k*=0 allowed = singleton)
   │
   ▼
[7] OUTPUT: matching_results.tsv (selected) + candidate_pairs.tsv (= exact set scored in [5])
```

## 2. Competing architectures (to be compared on P-main; numbers in MODEL_COMPARISON.md)

| ID | Architecture | Hypothesis / role |
|---|---|---|
| A | **Classical**: normalization + exact/fuzzy rules (core-name equality, combo cosine ≥ θ, best-for-target), no learning | floor. Shows how much ML adds |
| B | **Hybrid retrieval + GBDT** (pipeline above) | expected main workhorse: strong engineered features + calibrated probabilities |
| C | **Neural EM**: fine-tuned multilingual cross-encoder (MIT/Apache, ≤ 8B) on candidate pairs | captures patterns features miss; costly on ~100M pairs |
| D | **Retrieval + neural reranking**: B's scorer prunes to top-N, cross-encoder rescored, stacked | C's accuracy at a fraction of the cost |
| E | **Graph-enhanced**: B + collective features (target–target near-duplicate support, stage-1 probabilities of co-candidates) + optional constrained clustering (≤1 S1 per cluster) | exploits "copies of one entity share the same corrupted base" |
| F | **Best practical ensemble**: B (+E features) + D score as a feature, with the expected-F decision | final candidate if each part pays for itself |

Selection rule: the simplest architecture within the bootstrap noise of the best one wins.

### Measured comparison (P-main, fold 0, 220,683 S1; SE ≈ 0.00016)
| ID | measured macro F0.5 | status |
|---|---|---|
| A classical rules | 0.767 (best of 6 rules) | done; floor |
| B retrieval + GBDT | 0.98492 (3-fold cross-fit) | done |
| C neural cross-encoder | – | not run: Modal GPU unavailable (payment method) and compute paused |
| D retrieval + neural rerank | – | not run (same reason) |
| E graph-enhanced (B + collective / sibling features, stacked) | **0.98703** | done; +0.0021 over B (≈13 SE) |
| F ensemble | = E so far | a neural member would be added only if it pays for itself |

Also measured: static multilingual embeddings (potion-multilingual-128M) are used as B features; their gain has not been ablated in isolation yet. Density-matched training for B: 0.98479 on P-dense vs 0.98343 with normal training.
**Current choice: E, trained under density-matched conditions** (the dense stage-2 run is pending compute).

## 3. Why these choices (evidence so far)
- Country partition: measured 100% label agreement on 7.64M true pairs (DATA_ANALYSIS §3).
- Learned cross-script dictionary: name similarity on native-script true pairs p10 from 8 to 100 (normalize_report).
- Exclusivity: GT target multiplicity = 1 for all targets (DATA_ANALYSIS §3).
- Expected-F decision: metric is per-entity F0.5 with an empty-set option. Decision theory for F-measures (see LITERATURE_REVIEW) gives the Bayes-optimal top-k rule under calibrated independent probabilities.

## 4. Open-set country handling
- No component branches on a hard-coded label. Partitioning, IDF, generic-token sets and abbreviation maps are all computed per label found in the data.
- For a country without training labels (France), learned abbreviation maps are empty (seed only). Planned E-8 mines them from high-confidence test pairs without labels (self-training). LOCO experiments quantify the unseen-country gap.
