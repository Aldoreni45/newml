# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
We resolve entities with **multi-view sparse retrieval plus stacked gradient boosting**.
- A learned, country-agnostic normalizer bridges Indic scripts to Latin with a token dictionary learned from the training pairs.
- Five TF-IDF retrieval views, including a **reverse target→S1 view** (R@1 = 0.972), reach a 0.9965 oracle macro-F0.5 ceiling with about 45 candidates per S1.
- A LightGBM on ~90 features is dominated by **many-to-one competition features**. A second LightGBM adds **copy-cluster ("sibling") support features** built from out-of-fold first-stage probabilities.
- Validation macro F0.5 (exact metric, 220,683 held-out S1): **0.9849** (stage 1, submitted) and **0.9870** (stacked).

---

## 2. Methodology

### 2.1 Problem Analysis
Full-data profiling (all numbers measured on the provided files):
- **Scale:** train 2.21M S1 / 5.03M S2 / 5.29M S3; test 1.73M / 4.89M / 5.08M. France (test only) is 15% of test S1.
- **Label structure:** every S2/S3 record matches **at most one** S1 (7.64M true pairs, multiplicity 1). 26% of S2/S3 records match no S1 (distractors). 5.58% of S1 are singletons, with a mean of 3.46 matches per S1. The test has ~24% more targets per S1, i.e. more distractors.
- **Country** labels agree on 100% of true pairs.
- **Names:** 26–36% of S1 names are shared by several entities ("Primary Care Group", "Shree Trading Private Limited"), so the address must break ties.
- **Cross-script:** 19–28% of India targets carry Devanagari/Tamil/Telugu/Kannada/Bengali/Gujarati/Malayalam names. Native-script state names appear in addresses.
- **Other noise:** typos, leetspeak, injected diacritics, token reorder, moved/added legal forms, truncation, aliases ("X a/k/a Y", "dba"), web-domain/@handle names, acronyms, abbreviation and state-code variants, missing components, "null" tokens, number offsets.
- **Postcodes** appear in only ≈11% (US) / <2% (India) of addresses.
- **Generator traps:** the data contains **planted sibling entities** (same name, same street, nearby house number) and also shifts house numbers of true copies.

### 2.2 Solution Strategy
**Approach Type:** Blocking (multi-view retrieval) + stacked classifier (LightGBM) + constrained decision.
**Core Innovation:**
1. Normalization maps (native→Latin token dictionary, abbreviation maps) are **learned from training pairs** instead of hand-written, so no external data is needed and the method transfers to any country.
2. **Reverse (target→S1) retrieval** and **per-target competition features** exploit the many-to-one structure.
3. **Collective sibling-support features** separate true copies from planted lookalikes via copy-cluster consistency.
4. **Density-matched validation** simulates the test's higher distractor rate when choosing thresholds.

---

## 3. Candidate Generation (Blocking)
- **Partition:** by country label, taken from the data (open set; France is handled identically).
- **Views** (sparse TF-IDF, L2 cosine, top-K via multithreaded sparse matmul):
  1. name char_wb 3-grams (top-5)
  2. space-less name char 3-grams (top-5; handles, domains)
  3. address word unigrams incl. numbers (top-5)
  4. hybrid 0.5·name + 0.5·address (top-10)
  5. **reverse hybrid: for every target, its top-8 S1 records**
- **Candidate pairs generated:** 95.2M train (all folds, ≈43 per S1); **88.4M test (≈51 per S1)**.
- **How we ensured true matches were not lost:**
  - Measured on all 7.64M true train pairs. With 40/20/40/40/8 per view: pair recall 0.9905, entity full-recall 0.967, oracle macro F0.5 0.9972. The pruned budget keeps ≈0.989 recall (oracle 0.9965).
  - The reverse view alone has R@1 = 0.972.
  - Remaining misses are mostly empty-address targets whose generic name is shared by several S1s. They cannot be assigned safely, so they are deliberately left out.

---

## 4. Matching Model

**Features used** (87 in stage 1, +21 in stage 2):
- **Name:** exact TF-IDF cosines (char_wb 3-gram, space-less char 3-gram, word). RapidFuzz ratio / token-sort / token-set / partial / WRatio / Jaro-Winkler / normalized Levenshtein. Token Jaccard, IDF-weighted Jaccard and coverage, rarest shared token IDF. Core-token (data-learned generic tokens removed) Jaccard/equality, first-token match, initials/acronym, phonetic-key Jaccard, alias-part match, truncation-prefix, handle/domain prefix coverage.
- **Address:** word TF-IDF cosine, RapidFuzz ratios, IDF-weighted alphabetic overlap, number-set Jaccard, first-number equality, S1-number-missing, ≥3-digit number Jaccard, empty-address flags.
- **Other:**
  - Static multilingual embedding cosines for name, address and full record (potion-multilingual-128M, MIT, CPU).
  - Structural: script, source, alias/handle flags; name frequency among S1s and among targets (ambiguity).
  - **Competition:** rank, gap and margin of the pair among the S1's candidates AND among all S1s competing for the same target.
  - Retrieval ranks and scores per view.
- **Stage 2 (collective):** out-of-fold stage-1 probability p1. Similarity of the target to the S1's most confident other candidate. Probability-weighted address/name support from co-candidates. House-number agreement with the confident copy cluster. Same- vs cross-source sibling confirmation, isolation flag. p1 margins per target.

**Model type:** LightGBM binary (lr 0.08, 255 leaves, min_data 400, feature/bagging fraction 0.7, max_bin 127). 3-fold cross-fitting at both stages, and test predictions averaged over the fold models. The submitted file (sub_v1) uses the stage-1 ensemble trained at train density. The final configuration (`configs/final.json`) trains stage 1 on density-matched features (P-dense) and adds stage 2.
**Threshold selection method:**
- Threshold + many-to-one exclusivity (each target is kept only for its best-scoring S1), with τ chosen by exact macro-F0.5 on a held-out S1 fold.
- τ = 0.7 at train density; the optimum moves to 0.8 at test-like density.
- Expected-F0.5 per-entity set optimisation and an explicit "match-exists" (singleton) model were evaluated; they match but do not beat the tuned threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.98492 ± 0.00016 for the submitted stage-1 ensemble (validation fold of 220,683 S1; bootstrap SE). The stacked stage-2 model reaches **0.98703 ± 0.00016**.
  - No-ML rule baselines on the same candidates: 0.422 / 0.714 / 0.767.
  - Test-like density: 0.9834 (normal training) vs 0.9848 (density-matched training).
  - Unseen-country proxies (leave-one-country-out): India→US 0.974, US→India 0.945.
- **Common false positives (wrong merges):** planted sibling entities, i.e. same or near-same name on the same street with a nearby house number ("Amber Inc, 340 vs 345 Olympic Park Dr"). Also empty-address targets with a generic name that belong to a different entity. Stacking with sibling-support features raised pair precision from 0.9959 to 0.9973.
- **Common false negatives (missed matches):** empty-address targets whose name is shared by several S1s (49% of FN; the name alone cannot choose). True copies with shifted house numbers. Heavily truncated names.

---

## 6. Conclusion
The many-to-one structure of the data is the key: retrieving from the target side and scoring with competition features gives most of the accuracy. Collective copy-cluster features then remove planted lookalikes. Learned normalization maps make the system country-agnostic without any external resources. The largest remaining uncertainty is the unseen-country (France) shift; we address it with country-agnostic features and density-aware threshold selection.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/ber/`: library (normalization, learned maps, blocking, features, decision rules, exact metric).
- `modal_app/`: one file per stage; also runnable in-process.
- `run_pipeline.py`: Modal-free end-to-end runner; `configs/final.json`.
- `README.md`: exact commands; `requirements.txt`: pinned versions.

Entry points:
- `python run_pipeline.py --zip student_resource.zip --workdir <dir>`
- or the Modal commands in README (Option A).

### B. Additional Results
Full experiment table: MODEL_COMPARISON.md. Blocking recall@K per view: BLOCKING_STRATEGIES.md. Error taxonomy: ERROR_ANALYSIS.md. Data profile: DATA_ANALYSIS.md.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
