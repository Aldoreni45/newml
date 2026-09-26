# BLOCKING_STRATEGIES.md

Objective: pair recall (candidate completeness) as close to 100% as possible, with a candidate volume that keeps feature extraction for 1.73M test S1 tractable. Blocking sets the recall ceiling. Precision is the matcher's job.

## Constraints from the data (DATA_ANALYSIS.md)
- Country label equality holds for 100% of true pairs → partition by country label. The label set is discovered from the data, so unseen labels are handled the same way.
- Postcodes appear in only ~11% (US) / <2% (India) of addresses, so postcode keys are too sparse.
- Names are noisy: typos, reorder, truncation, aliases, handles, acronyms, cross-script. Addresses are the more stable field (true-pair address token-set p50 ≈ 94–100 after normalization), but 3% of target addresses are empty.
- Many S1 names are shared by several entities (26–36%) → name-only retrieval returns many same-name distractors. The address has to break ties.

## Views (v1, `stage_block.py`, tag `v1`)

| view | representation | K (S1→T) | what it catches |
|---|---|---|---|
| name | char_wb 3-gram TF-IDF (sublinear tf, max_df 0.05, min_df 2) on normalized, transliterated name | 40 | typos, reorder, legal-form noise, cross-script after dictionary |
| cat | char 3-gram TF-IDF on the space-less name | 20 | handles/domains ("@parabolalaw"), run-together names, spacing noise |
| addr | word TF-IDF (numbers kept) on normalized address | 40 | alias-only / truncated / acronym names with a stable address |
| combo | [√0.5·name ‖ √0.5·addr] → cosine = 0.5·cos_name + 0.5·cos_addr | 40 | hybrid ranking. Needs both fields to agree, which is the most precise view |
| rev | combo, target → S1 top-8 | 8 per target | S1s whose list is flooded by same-name distractors; the target's own view of its best S1 |

Union of all views → `cands_v1.parquet`, with per-view scores and ranks kept as features.

Implementation: `sparse_dot_topn` (multithreaded C++ sparse matmul with top-n heap), queries in 250k-row chunks. Vectorizers are fit on a ≤3M sample of S1 ∪ targets of the partition (label-free).

## Results

### Benchmark B-bench-India10 (2026-09-25; 10% of India train S1 = 88,318 queries; targets = their true targets + 10% of all other India targets = 688,517)
| view | recall | time (s) |
|---|---|---|
| name char_wb3, top-40 | 0.826 | 4.2 |
| cat char3, top-20 | 0.758 | 4.1 |
| addr words, top-40 | 0.909 | 2.5 |
| combo name+addr, top-40 | 0.984 | 6.3 |
| **rev target→S1 combo, top-8** | **0.996** | 4.6 |
| union | **0.997** at 124 cands/S1 | vectorize 31 |

Readings:
- Name-only retrieval is weak (0.83). 26–36% of S1 names are shared, so the top-40 fills up with same-name distractors.
- The hybrid view is strong (0.984). The **reverse view is the strongest single blocker (0.996 with only 8 S1s per target)**. This follows from the many-to-one structure: each target has exactly one S1, and in hybrid similarity that S1 is almost always among the target's top few.
- The union costs 124 candidates per S1 for +0.1pt over rev alone. K for name/cat/addr must be cut (full-data grid below).
- Caveat: the sample has 10× lower distractor density than the full data, so full-data recall will be somewhat lower.

### Full train (B-v1), 2026-09-25. 2,206,821 S1 queries vs the full 10.32M-target pool; 32 vCPU, 2h10m
| view | R@1 | R@3 | R@5 | R@10 | R@20 | R@40 |
|---|---|---|---|---|---|---|
| name (char_wb3) | 0.150 | 0.371 | 0.478 | 0.566 | 0.627 | 0.677 |
| cat (space-less char3) | 0.164 | 0.392 | 0.499 | 0.589 | 0.656 | (K=20) |
| addr (words) | 0.234 | 0.573 | 0.731 | 0.820 | 0.862 | 0.886 |
| combo (0.5 name + 0.5 addr) | 0.259 | 0.668 | 0.881 | 0.960 | 0.970 | 0.975 |
| **rev (target→S1 combo)** | **0.972** | 0.983 | 0.986 | 0.988 (K=8) | – | – |

- **Union (K 40/20/40/40 + rev 8): pair recall 0.9905, entity full-recall 0.967, oracle macro F0.5 = 0.9972** (US 0.9974, India 0.9968). 112 candidates/S1, 248M pairs.
- Recall by target source/country: S2|India 0.9905, S3|India 0.9877, S2|US 0.9908, S3|US 0.9921.
- On the full pool, name-only recall is much lower than in the 10% benchmark (0.677 vs 0.826). Full density floods name lists with same-name entities. The reverse view is robust to that because it asks the question the GT is structured around: which S1 does this target belong to?
- Test (1,732,544 S1, 9.97M targets): 118 candidates/S1, 205M pairs, 1h55m. France partition: 259k S1 × 1.43M targets, 4.4 min.

## Candidate budget analysis (full train)
Uniform cap K on name/cat/addr/combo, rev fixed at 8:

| K | pair recall | cands/S1 |
|---|---|---|
| 5 | 0.9882 | 41.9 |
| 10 | 0.9890 | 51.6 |
| 15 | 0.9895 | 63.1 |
| 20 | 0.9900 | 75.2 |
| 30 | 0.9903 | 93.5 |
| 40 | 0.9905 | 112.3 |

Going from K=5 to K=40 adds only +0.23pt pair recall for 2.7× the candidates. **Chosen budget v1: rev<8 ∪ combo<10 ∪ name<5 ∪ cat<5 ∪ addr<5**, about 45 per S1 and ≈0.989 recall. The oracle ceiling stays ≈0.997, so blocking is not the bottleneck.

## What the misses look like (300 sampled, `block_train_v1_misses.txt`)
The large majority are **targets with an empty (or near-empty) address and a generic, non-unique name**: "Sunrise Trading Private Limited", "Ace Properties Private Limited", "Modern Consultancy", "Ridge Fund | www.ridgefund.com". Several S1s share such names, so retrieval by name cannot rank the right one first, and a matcher could not safely pick it either. The rest are pure aliases with only a house number ("Gildonyx | 5, KOLKATA…") and cross-script records whose address is truncated to one token. Chasing these costs precision; they are left out on purpose.

## Rejected / deferred
- Postcode + name-prefix keys: postcodes too sparse.
- Pure exact-key blocking: fails on typos, reorders and cross-script.
- Dense embedding ANN: deferred to E-9. It would need GPU and a license-verified multilingual encoder, and sparse TF-IDF is known to be a very strong blocker (Sparkly, PVLDB 2023 — see LITERATURE_REVIEW).
