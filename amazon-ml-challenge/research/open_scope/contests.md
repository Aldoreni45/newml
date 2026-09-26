# Open-scope sweep: contests and challenge write-ups (blocking, candidate budget, post-processing, thresholds)

_Sweep date: 2026-09-25. Scope: SIGMOD Programming Contests 2020-2022 (and 2023/2024 for context), DI2KG 2020 finalist papers, Kaggle Foursquare Location Matching (2022), Kaggle Shopee Price Match (2021), per-entity F-score contests (ICDM 2015 Drawbridge, CIKM Cup 2016), Kaggle Quora Question Pairs, MWPD2020 (WDC product matching), SemTab 2020, Amazon KDD Cups 2022-2024 and past Amazon ML Challenges._

## How this was verified

- Every item below was read on its primary page. **Kaggle write-ups** were read in full from Kaggle's own discussion JSON endpoint (`/api/i/discussions.DiscussionsService/GetForumTopicById`). This is the same content the JS page renders. Earlier notes (`research/notes/contrastive_aug.md` #49, `research/notes/graph.md` #38) only had the Foursquare *blog* recap and state that the Kaggle pages "were not read". This sweep closes that gap and pins down the actual placings.
- **Kaggle evaluation metrics** were read from the competitions' Evaluation pages (`competitions.PageService/ListPages`).
- **SIGMOD contest posters** were downloaded from `dbgroup.ing.unimore.it/sigmod2{1,2}contest/posters/*.pdf`, and their text was extracted with PyMuPDF.
- Any number below was read on the page cited next to it. Items I could not open are marked **UNVERIFIED**.
- **Leak caveat (Foursquare).** About 67% of Foursquare test rows were exact copies of train rows ([leak thread](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335799)). So *public/private LB numbers and "train longer / overfit helped" claims in that competition are leak-confounded*. Local CV numbers and candidate-generation statistics (Max IoU) are not affected, and I lean on those.

### Our facts that decide what transfers (from `PROJECT_STATE.md`)
- Best fold-0 exact macro-F0.5 is **0.98703** (stage-2 stacked, thr 0.7 + exclusivity). The blocking oracle is **0.99645**. **The matching and decision gap (≈0.0094) is ~3× the blocking gap (≈0.0035).**
- Residual errors: (a) **FN on empty-address targets with shared names**, (b) **FP on planted sibling look-alikes** (same name, nearby house number, same street).
- **Test is denser.** The optimal threshold moves from 0.7 to 0.8, and mid-range probabilities become over-confident. The expected-F and π-hurdle rules did **not** beat a tuned threshold.
- The Modal spend limit is hit, so cheap, CPU-only, decision-layer ideas rank higher than anything needing GPU.

---

## 0. TL;DR: the tricks worth stealing

| # | Trick (source) | Evidence read | Our adaptation | P |
|---|---|---|---|---|
| 1 | **Density-conditioned / query-adaptive thresholds** (Shopee 58th dynamic thresholds; Drawbridge "AccessParameter"; Foursquare 4th "adapt thresholds to the sizes of the groups") | Drawbridge per-device-F0.5 pipeline: 0.876 → 0.88 with SSL + PP. 4th-place Foursquare calls size-adaptive thresholds "the key idea" of its post-processing | Fit τ(d) = f(per-S1 candidate density, top-k score mean, country) jointly on P-main **and** P-dense OOF. This directly targets our 0.7 → 0.8 threshold drift | **P0** |
| 2 | **Validate on a full-size candidate pool** (Foursquare 3rd; Shopee 3rd/4th; SIGMOD-22 RUPikachu) | RUPikachu: "despite achieving 0.9+ recall on the sample set, the final recall did not cross the 0.25 mark" (D2). Shopee 4th: some post-processing "excel on small sets but fail on larger sets" | Tune every decision/post-processing rule on P-dense (full test-like density), never on subsampled pools. Add a test-pool-size simulation for France | **P0** |
| 3 | **Neighbour-probability smoothing / graph convolution** (Foursquare 6th, 9th; 11th Dijkstra; 12th multi-hop) | 9th: LB 0.925 → 0.930 from "avg of neighbors predictions". 11th: 0.945 → 0.947 (Dijkstra) | Smooth p(t, s) over target–target edges (S2↔S3 copies of the same entity). This rescues empty-address targets whose copies have addresses (our FN class a) | **P1** |
| 4 | **Fill missing fields from nearest neighbours** (Foursquare 11th, 7th) | 11th: LB 0.931 → **0.940** from "fill na" (NaN texts filled from the 5 nearest points, for the BERT inputs; LB is leak-affected) | Impute an empty target address from its high-confidence target–target neighbours (same name + same S1 top candidate) before stage-2 features | **P1** |
| 5 | **Per-group sample weights for sample-wise F** (Shopee 2nd) | "Weighted sample with 1 / (label group size) ** 0.4" (the authors write "important to predict smaller label groups for micro-F1 score"; the competition metric is sample-wise mean F1). Foursquare 7th (team blog) used a related IoU-loss weight: positives 1/n_true, negatives 1/(n_true+1), mean-normalised | Weight training pairs by 1/(n_true(S1)+1)^α so the loss matches per-S1 macro F0.5. Recalibrate on unweighted OOF afterwards | **P1** |
| 6 | **Rank / percentile features robust to pool-size shift** (Shopee 5th; Shopee 2nd z-scored graph features; Foursquare 6th per-candidate rank features) | 5th: an extra XGB on "percentage rank features" because the test pool was larger than the fold | Add within-S1 and within-target percentile ranks and within-country z-scores. Train a second LightGBM on rank-only features and blend (density drift + France) | **P1** |
| 7 | **Feature "poisoning" to simulate extraction failure on unseen data** (SIGMOD-21 runner-up *panda*) | Avg F1 0.957 (runner-up). Zeros injected into the training feature matrix to mimic missing extractions | Randomly disable learned-map normalisation (translit dict, abbreviation maps) for a slice of training pairs. That mimics France, where our maps do not cover R./BD/CH/N°/bis/ter | **P1** |
| 8 | **Country-specific candidate mix** (Foursquare 2nd) | 128 candidates from two views "with country-specific optimized ratios", Max IoU 0.9895 | Tune per-view k per country. For France, choose it on the LOCO (US↔India) proxy and not a global constant | **P1** |
| 9 | **Confident-test pseudo-labelling to re-derive features** (Drawbridge 3rd, SSL) | Devices with top > 0.4 and second < 0.05 were fed back. Second-candidate < 0.1 covers 62% of devices at F0.5 > 0.99 | Self-train French abbreviation/translit maps from test pairs with p_top ≥ 0.95 and second-best ≤ 0.05 (planned E-8; same gate as §12 #11). Then recompute map features | **P1** |
| 10 | **Cascade: wide cheap retrieval → light GBDT filter at tiny threshold** (Foursquare 1st/4th/9th/13th, 2nd transformer blocker) | 9th: 420M candidates at Max IoU 0.994 → 3M at 0.987 (thr 0.005). 1st: thr 0.01 keeps ~10% | Re-admit the full 112/S1 union (recall 0.9905, vs 0.989 pruned) behind a 10-feature LightGBM filter. It would recover part of the 0.0007 oracle gap cheaply | P2 |
| 11 | **Neighbour-set merge + re-score** (Foursquare 1st stage 4) | CV 0.911 → 0.9166 | For target pairs whose candidate-S1 lists overlap > 50%, union their lists and score the new (t, s) pairs. This is recall for targets whose true S1 was missed | P2 |
| 12 | **Query expansion / DBA re-retrieval** (Shopee 1st INB, 5th WDBA; Foursquare 11th DBA/QE) | Shopee 1st: INB step, 0.776 → 0.784 (with extra text models) | Expand each S1's TF-IDF profile with the tokens of its confident targets (cross-script and alias copies), then re-query unmatched targets | P2 |
| 13 | **Set-level decision / cluster-size cap** (CIKM Cup 2016 1st; Shopee 3rd avg-cluster-size control) | CIKM: transitive inference capped at size 5 → F1 0.4155 → 0.4204 (pair-level F1, not per-entity) | Cap the per-S1 predicted set at the train 99th-percentile size. Also add a set-level accept/trim classifier on the predicted set | P2 |
| 14 | **Subsumption rule from mock labels** (SIGMOD-20 winner) | Label sets that are subsets of other matched labels are removed (EOS 7D vs EOS 7D Mark II) | A feature: is the S1 token set a strict subset of another candidate S1 that the target also contains? Targets sibling look-alikes | P2 |
| 15 | **Leak/overlap audit train↔test** (Foursquare) | 67% of test rows were duplicates of train. Winners gained 0.94 → 0.97+ | Measure exact-normalised overlap of test S1 and target strings with train. **Ask the user before using any overlap** (compliance) | P1 (audit only) |

---

## 1. ACM SIGMOD Programming Contest 2022: blocking at 1M scale under a fixed candidate budget

**Task (read from the WBSG poster and the SIGMOD Record report).** Two synthetic product datasets of about 1M rows each. D1 is notebooks with a **1,000,000-pair** budget, and D2 is multilingual Altosight products with a **2,000,000-pair** budget. The score is **average recall** over both, and ties are broken by runtime. The environment was 16 CPUs, 32 GB, no GPU, with a 35-minute limit. The organisers generated the data by "picking the first word from a randomly chosen tuple, the second one from another…" and made matches by "randomly shuffling words, deleting some words, or changing the letter case". That is the same *family* of synthetic noise as ours, although ours is richer. "60 teams from 10 countries in 2022 (with ≈2500 submissions)" (SIGMOD Record). Mannheim's news page says "Altogether 55 teams from all over the world participated". The sources disagree.
Sources: [finalists page](https://dbgroup.ing.unimore.it/sigmod22contest/leaders.shtml) · [SIGMOD Record report PDF](https://sigmodrecord.org/?smd_process_download=1&download_id=13459) · [archive summary](https://transactional.blog/sigmod-contest/2022) · [Mannheim news](https://www.uni-mannheim.de/dws/news/wbsg-wins-sigmod-programming-contest-2022/)

### 1.1 WBSG (1st): neural blocking with contrastive embeddings plus Jaccard re-rank
- **Technique.** XtremeDistil transformer, mean pooling, and a dense projection to **32-d**. Trained with supervised contrastive loss at **batch 1024**. Preprocessing: lowercase, stop-word removal, normalisation, and truncation to max sequence length (poster: D1 28, D2 32; the repo README says 28 and 24, so the sources disagree). **All unique preprocessed record texts are grouped first.** Records with identical text become pairs (similarity 1.0) at the head of the output. FAISS IVF+PQ index. **Re-rank = average of Jaccard and max approx cosine.** Then pairs are emitted from the most similar texts until the budget is full.
- **Evidence.** Avg recall **0.529** (D1 **0.713**, D2 **0.345**), runtime **1914.275 s**. The authors say the Jaccard re-rank "increased the generalization of the top-k similarity ranking", and that grouping by unique text cut indexing and retrieval time. Extra training data came from the WDC corpus (437,581 offers). That is **external data, which is forbidden for us**.
- **Adaptation.** (a) *Hybrid re-rank.* In our reverse target→S1 view, score = mean(dense cos, token Jaccard). WBSG found a pure embedding ranking over-trusts the encoder. We are already sparse-first, so this matters only if the e5 view is turned on. (b) **Unique-text grouping.** Before feature computation, collapse targets that are identical after normalisation, score once, and fan out. This is a pure compute saving, which matters now that Modal spend is capped. **P2.**
- Links: [poster](https://dbgroup.ing.unimore.it/sigmod22contest/posters/WBSG.pdf) · [code](https://github.com/abrinkmann/acm_sigmoid_2022_challenge) (no licence file; method reference only)

### 1.2 April (2nd, Rutgers): exact similarity joins, no heavy rules
- **Technique.** "around 10 bash commands to normalize the data": typo fixes, cross-language synonyms ("mémoire" → "memory"), and format unification ("USB 3" → "usb3"). **Near-duplicate records are merged when Jaccard > T1.** Then a Jaccard join (> T2) and an overlap join (> T3) build a pool, which is **ranked by weighted similarity** and cut to the budget.
- **Evidence.** D1 (T1 0.8, T2 0.57, T3 9) recall **0.743**. D2 (0.9, 0.57, 7) recall **0.297**. Avg **0.520**, runtime **1679 s**. Their stated hindsight: "Could use the training data ground-truth to boost the recall". They also note that the best thresholds differ per dataset.
- **Adaptation.** Supports a *per-country* (per-dataset) budget and threshold, not one global setting (see §0 #8). **P1** (merged into #8).
- Links: [poster](https://dbgroup.ing.unimore.it/sigmod22contest/posters/April.pdf) · [code](https://github.com/rutgers-db/SIGMOD2022-Programming-Contest-Public) (BSD-2-Clause)

### 1.3 QaisHousien (3rd, Haifa): sorted neighbourhood with an adaptive window
- **Technique.** spaCy tokens (stop words dropped, "Only nouns and pronouns are used", lowercased, lemmatised). Sorted-neighbourhood with Jaccard. **The window grows adaptively with the number of detected pairs** (D1), and a **"tolerance"** counts how many pairs may be discarded before moving on (D2).
- **Evidence.** Recall **0.726 / 0.301**, avg **0.514**.
- **Adaptation.** Weak for us, because TF-IDF kNN dominates sorted neighbourhood. The adaptive-window idea is the same "spend budget where hits keep coming" principle as §0 #10. **P3.**
- Link: [poster](https://dbgroup.ing.unimore.it/sigmod22contest/posters/QaisHousien.pdf)

### 1.4 SUSTech_DBGroup (finalist): regex key buckets plus per-bucket HNSW
- **Technique.** Regex feature extraction (e.g. `cpu_model [AaEe][0-9][\- ][0-9]{4}`). **Feature bucketing: high-confidence buckets emit all their pairs directly.** Elsewhere, BERT fine-tuned with triplet loss (128-d) and an **HNSW index per bucket**. Pairs where key features disagree (e.g. brand) are filtered out, then pairs are sorted by Euclidean distance until the output size is reached.
- **Evidence.** Poster text shows X2 recall **0.323**, time **1671 s**. The poster also shows a **0.692** that is most likely the X1 recall, but the text order is ambiguous, so treat that as UNVERIFIED.
- **Adaptation.** Supports a "hard-key veto before budget ranking". We should *not* veto on house numbers, because the generator shifts house numbers of true copies. **P3.**
- Link: [poster](https://dbgroup.ing.unimore.it/sigmod22contest/posters/SUSTech_DBGroup.pdf)

### 1.5 RUPikachu (finalist): hash keys, block-size cap, and the sample-vs-full warning
- **Technique.** Schema-aware hash keys: "loose keys (such as brand+model)" plus specific combinations. **"Extremely common patterns were filtered out by limiting the block size."** Survivors are sorted by Jaccard and overlap into the top 3M.
- **Evidence (the key lesson).** "despite achieving 0.9+ recall on the sample set, the final recall did not cross the 0.25 mark". The authors blame the "relatively small sample set X2" not representing the full D2.
- **Adaptation.** Our 0.996 reverse-view recall and 0.9905 union recall were measured on full train (7.64M true pairs), which is good. But **France and the denser test pool are out-of-sample in exactly this way.** Every blocking and decision knob must be validated at full pool size, with LOCO as the France proxy (see §0 #2). **P0 (process).**
- Link: [poster](https://dbgroup.ing.unimore.it/sigmod22contest/posters/RUPikachu.pdf)

---

## 2. ACM SIGMOD Programming Contest 2021: ER on small product sets (matching step)

**Task.** Notebook (538 instances, 100 entities), NotebookLarge (605 / 158) and Altosight (1,356 / 193), per the CyberPunk poster (the BoomBoomChicken poster lists 153 entities for NotebookLarge, so the sources disagree). Macro F-measure over the datasets (SIGMOD Record), with evaluation on hidden records (Z_i = D_i \ X_i, panda poster). 25-minute limit (CyberPunk poster).
Sources: [finalists page](https://dbgroup.ing.unimore.it/sigmod21contest/leaders.shtml) · [archive summary](https://transactional.blog/sigmod-contest/2021)

### 2.1 SUSTech_DBGroup "ExtER" (winner): complete, recycle and residual blocking
- **Technique.** Regex extraction plus **alias tables** (`{m620:620m, 620m:620m}`). *Complete blocking*: instances whose brand-specific **primary features** are all present go to the "solved set" and are blocked on the exact combination. *Recycle blocking*: other instances try feature combinations **in a fixed priority order** ((brand, model) before (brand, memory)) and join the first block that matches. *Residual blocking*: leftovers block only with instances that have **the same present-and-missing feature pattern**. All pairs inside a block are output.
- **Evidence.** P/R/F1: X2 **0.995/0.971/0.983**, X3 **0.991/0.980/0.986**, X4 **0.980/0.880/0.927**.
- **Adaptation.** "Residual blocking" is a disciplined answer to our **empty-address FN class**. A target with no address should only compete against S1 candidates in the same (name-key, missing-pattern) cell, and gets its own calibration. Concretely: add a `target_missing_pattern` categorical (which of name/street/number/city/postcode/state are empty) to the stage-2 model. Also fit a **separate threshold per missing-pattern** in the decision layer. **P1** (cheap: a feature plus a group-wise threshold).
- Link: [poster](https://dbgroup.ing.unimo.it/sigmod21contest/posters/SUSTech_DBGroup.pdf)

### 2.2 panda (runner-up, Georgia Tech): labelling functions as features, plus "data poisoning"
- **Technique.** Hand-written labelling functions (extract then compare, emitting +1/−1/0 on extraction failure) are used as **features** for an ML model. **"Data poisoning": "we randomly inject zeros to the feature matrix … to simulate the failure of information extraction. This adapts the training distribution to the test distribution."** They also trained a separate model per brand on Altosight.
- **Evidence.** F1 **0.972 / 0.992 / 0.907**, avg **0.957**.
- **Adaptation (France).** Our learned native→Latin dictionary and abbreviation maps were built on US and India. On France they will "fail to extract" (unknown R./BD/CH/N°/bis/ter, département/région swaps). Train stage-1 and stage-2 with **map dropout**: for a random 10-20% of training S1 groups, compute features from the *un-mapped* normalisation, so the model learns to lean on map-free features when maps are silent. Validate with LOCO (train US → test India, and the reverse). **P1.**
- Link: [poster](https://dbgroup.ing.unimo.it/sigmod21contest/posters/panda.pdf)

### 2.3 UKN (runner-up, Tsinghua): loose blocking, strict rules, XGBoost for the uncertain middle
- **Technique.** "A loose blocking method guarantees a high recall" (group by brand, or brand+size). Rule-based matching on extracted keys. **XGBoost only post-processes the "uncertain pairs left by rule-based method."** Cross-language synonyms (mémoire/memoria → memory).
- **Evidence.** F: **98.4 / 98.0 / 90.7**.
- **Adaptation.** Train a **specialist model on the uncertain band** (stage-2 p in [0.3, 0.9]), where our remaining errors concentrate. Train it on OOF pairs from that band only, with sibling-specific features. **P2.**
- Link: [poster](https://dbgroup.ing.unimo.it/sigmod21contest/posters/UKN.pdf)

### 2.4 CyberPunk2021 (4th, EPFL): clustering choice per dataset (TRANS vs ISO-CON)
- **Technique.** Matching, then clustering. Options were transitivity (TRANS), connecting isolated instances to the nearest cluster (ISO-CON), or connecting isolated pairs by score (ISO-ISO). The best choice differed by dataset.
- **Evidence.** Notebook (strict model rules + TRANS) F1 **94.5**, NotebookLarge (RF + ISO-CON) **99.0**, Altosight (RF + TRANS) **90.2**. Final **94.6**.
- **Adaptation.** Our analog of ISO-CON is **S1-empty rescue**: an S1 predicted empty whose best target has p ≥ τ_low, and that target is not claimed by a higher-p S1, gets it. Only 5.6% of S1 are truly singletons, so an empty prediction is usually a full miss. The expected-F layer should already cover this if calibrated, but it is mis-calibrated at test density. Grid-search τ_low on P-dense. **P2.**
- Link: [poster](https://dbgroup.ing.unimo.it/sigmod21contest/posters/CyberPunk2021.pdf)

### 2.5 BoomBoomChicken (finalist, Rutgers): Magellan RF, per-brand models
- **Evidence.** RF with 120 trees, or 50 trees per brand on dataset 3. F1 **0.950 / 0.982 / 0.874**, avg **0.935**. The poster notes RF "might get overfitting" with little data, and that data augmentation helped.
- **Adaptation.** None new. Per-segment models map to per-country models, which we cannot train for France. **P3.**
- Link: [poster](https://dbgroup.ing.unimo.it/sigmod21contest/posters/BoomBoomChicken.pdf)

---

## 3. SIGMOD 2020 (camera specs) and the DI2KG 2020 finalist papers

### 3.1 Blacher et al., "Fast Entity Resolution With Mock Labels and Sorted Integer Sets" (2020 winner)
- **Technique.** Build **"mock labels"** (canonical keyword sets per entity) and assign each record to a label by set containment. **"If we can match more than one label with the title, then we remove all labels that are subsets of other labels."** This separates *canon eos 7d* from *canon eos 7d mark ii*. When no full label matches, fall back to **simplified labels** with non-critical tokens removed. If the remaining labels disagree, output "ambiguous" (no match).
- **Evidence.** About 2,800 semi-automatic labels gave F **0.97**. Adding about 200 manual labels gave **0.99**. All top-5 teams hit 0.99, and runtime broke the tie.
- **Adaptation.** Our S1 table is literally a label set, and each target is assigned to ≤ 1 S1. Two features for the **sibling look-alike FP class**: (i) `s1_subsumed_by_other_cand`: the S1 token set is a strict subset of another candidate S1's token set, and both are contained in the target. (ii) `ambiguous_label`: ≥ 2 candidate S1s are fully contained in the target and neither subsumes the other. Use them as features, not hard rules, because "generic token injection" (Services/Center) could create false supersets. **P2.**
- Link: [CEUR Vol-2726 paper2](https://ceur-ws.org/Vol-2726/paper2.pdf)

### 3.2 Zecchini, Simonini, Bergamaschi, "Entity Resolution on Camera Records without Machine Learning" (2020 runner-up)
- **Evidence (a cautionary number).** DeepMatcher RNN reached F **98.26%** on the held-out labelled pairs of W. On the full dataset after token blocking it "could not go beyond a F-measure of 0.47" (recall 0.85, precision **0.32**). The abstract says "not reaching 0.49". Token blocking on the first 4 title words gave **3,914 blocks / 54,000,932 candidate pairs**, with P 0.28 and R 0.99 on W. A regex-plus-lists approach reached **0.99**.
- **Adaptation.** This confirms our practice of scoring with the **exact per-S1 metric over all S1s and all candidates** (fold 0, 220,683 S1). Never report pair-classifier AUC or accuracy on labelled pairs as progress. **P0 (process; already done).**
- Link: [CEUR Vol-2726 paper3](https://ceur-ws.org/Vol-2726/paper3.pdf)

### 3.3 Organisers' lessons (SIGMOD Record 52(2), 2023)
- All finalists "were optimized for the provided datasets". They put much weight on pre-processing and feature extraction with regexes, brand lists and **alias dictionaries**. Blocking "played a fundamental role concurring to determine the matches". The 2022 blocking entries were followed by "a pair/block cleaning and ranking step (based on intra-pair similarity) to comply with the submission structure".
- Link: [SIGMOD Record PDF](https://sigmodrecord.org/?smd_process_download=1&download_id=13459)

---

## 4. Kaggle Foursquare Location Matching (2022): the closest public analog

**Metric (read on the Evaluation page).** "mean Intersection over Union … calculated for each row … and the final score is their average". Places always self-match. This is structurally our per-S1 macro F0.5, with IoU instead of F0.5, and the self-match plays the role of our "empty prediction for singletons". About 1.1M train and about 600k test places, multilingual names and addresses plus lat/lon. **Leak:** about 67% of test rows were duplicates of train rows ([thread](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335799)). LB numbers below are leak-affected where noted.

### 4.1 1st place (write-up authors charmq, Takoi, pao; the team name is not given in the post), 4-stage cascade
- **Stage 1.** For each id, **100 candidates from each of two views**: lat/lon Euclidean, and name-embedding cosine (bert-base-multilingual-uncased, cuML kNN). A **LightGBM per view with only a few cheap features** (Jaro on name and category) keeps the **top 20 per view**, about 40 in total. **Max IoU 0.979.**
- **Stage 2.** About 120 features, including **"Character similarity statistics (maximum, minimum, average) using id as key. Also, the ratio of those statistics."** LightGBM CV **0.875**. **Threshold 0.01 cut candidates to about 10%.**
- **Stage 3.** CatBoost (CV 0.878). xlm-roberta-large (3 epochs, with about 70 GBDT features concatenated). mdeberta-v3-base with FGM + EMA (CV **0.907**). Ensemble CV **0.911**, threshold 0.5.
- **Stage 4.** "Compare the matches of two ids and **merge the matches of two ids if the common id exceeds 50% from either side**". The newly created pairs are re-scored with xlm-roberta-large at threshold 0.02. CV **0.9166**.
- Leak-free private score of the non-overfit model was **0.941**. Using the train/test overlap gave 0.977.
- **Adaptations.** (a) Our per-S1 and per-target competition features are the same idea as "statistics using id as key, and ratios", which confirms the design. (b) **Stage-4 neighbour-set merge** for recall: for targets t1, t2 whose top-candidate S1 lists overlap > 50%, and which are likely copies of each other (e.g. the same name key, one from S2 and one from S3), add s ∈ C(t2) \ C(t1) to C(t1) and score with stage 2. **P2.** (c) Keeping the top-20 per view with a light model is supervised meta-blocking, which `research/notes/blocking.md` already covers.
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336055)

### 4.2 2nd place (team: T0m, colum2131 and others)
- **Candidate generation.** 128 nearest ids by lat/lon, and 128 by name TF-IDF cosine. Then **"select a total of 128 from both with country-specific optimized ratios"**. **Max IoU 0.9895.**
- **Transformer candidate blocking.** Absolute features (lat/lon, **country embedding**, category embedding, xlm-roberta-base name embedding, TF-IDF-SVD) and relative features (edit distances, TF-IDF-SVD cosine). Single LB 0.918. **Threshold 0.005 + union-find → 4.1 candidates per id on average, Max IoU 0.986.**
- **GBDT part (colum2131).** Edit distances over 8 fields × 11 RapidFuzz/Jaro/Levenshtein variants, Jaccard/Dice/Simpson on split tokens, and TF-IDF cosines (word 1-gram, char 1-3). A BERT pair model whose input includes the country, plus a log-haversine numeric feature. LB 0.949 without the leak, 0.971 with it.
- **Adaptations.** **Country-specific view ratios** (§0 #8). Our budget is "rev<8 ∪ combo<10 ∪ name<5 ∪ cat<5 ∪ addr<5" for every country. Re-tune those five k's per country on train (US vs India). For France, pick the ratio that is best under LOCO **and** biased toward views that do not depend on learned maps (the char n-gram name view). **P1.**
- Links: [summary](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336062) · [candidate part](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336072) · [GBDT/BERT part](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336090)

### 4.3 3rd place (Psi): ArcFace retrieval, pair model, and *the* validation lesson
- **Technique.** Stage 1: ArcFace on xlm-roberta-large over record strings, with **lat/lon written into the string "split up every 3rd digit so that the tokenizers can work better"**. All-pairs similarity with a cut-off. Stage 2: a pair model on the stage-1 candidates, with both records pasted column by column (the author calls it a "bi-encoder", but the input is one joint sequence). The final prediction is a blend.
- **Validation (evidence).** "Retrieval/Recognition type problems are always heavily dependent on the list of candidates to choose from, so naturally having the possibility of more unknown candidates will deteriorate the score." So they held out a **test-sized block (600k) with its full candidate pool**. "Any random 100k subset scored nearly the same as the full 600k set (keeping all 600k candidates of course)."
- **Adaptations.** (a) This is our P-dense rule (§0 #2). The *queries* can be subsampled, but the *candidate pool* must be full size. (b) Splitting digits into groups is a cheap trick for numeric tokens (house numbers, postcodes) in any subword encoder view. **P3** (only if the e5 view or a cross-encoder is used).
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/338112)

### 4.4 4th place (Vincent Schuler, Theo Viel, ymatioun)
- **Technique.** Cleaning plus **unidecode**. Candidates from many cheap generators (near neighbours, name similarity, shared words, grouped-category equality, **cleaned phone equality, same address**, TF-IDF). **LGBM 1 with few features deletes pairs below threshold 0.007.** LGBM 2 uses 200+ features and 20 folds. **Post-processing: "the key idea was to adapt the thresholds to the sizes of the groups we were merging … merging 2 places has not the same impact (and probability) than merging 2 groups of 10 places."**
- **Negative results (useful).** "Tf-idf was not very useful: less than 3% of true matches added, and it added some false-positives that were hard to catch". "Translation of foreign languages was not a game-changer compared to unidecode (except pykakasi …for Japanese)".
- **Evidence.** 0.939 (15th) → 0.957 (4th) came from the leak, not from the method.
- **Adaptations.** (a) **Group-size-adaptive thresholds** feed §0 #1: the threshold for adding the k-th target to an S1 should depend on k and on how many other S1s claim that target. (b) A script-specific transliterator beat generic unidecode only for Japanese. For us, the analog is Indic scripts, where our *learned* dictionary is the "pykakasi". Keep it. **P1** (part of #1).
- Links: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810) · [code](https://github.com/TheoViel/kaggle_foursquare) (no licence file, so method reference only)

### 4.5 6th place (tomo20180402): "graph probability convolution"
- **Technique.** 28 candidates per id (**IoU 0.978**): kd-tree 6,500 neighbours, then a weighted sum of √distance, 1−Jaro-Winkler, normalised Levenshtein and 1−Simpson. **Rank features: "Create a rank of between 28 candidates per id for each feature."** Count encoding. CatBoost. **Post-processing:** update p(A,B) as a weighted average over A's confident neighbours C: AB′ = (AC·CB + AD·DB + AA·AB)/(AC + AD + AA), using only neighbours with prob ≥ 0.5. Then add nodes connected with ≥ 0.5 at adjacent nodes, and repeat twice.
- **Adaptation → §0 #3.** Our graph is S1–target with target–target (S2↔S3) edges. Define p′(t,s) = [p(t,s) + Σ_{t′} w(t,t′)·p(t′,s)] / [1 + Σ_{t′} w(t,t′)] over targets t′ with a strong target–target similarity w (for example the same normalised name and a compatible address, or a stage-1 target-pair model). Use p′ as a **stage-3 feature** rather than a replacement, so the model decides when neighbours are trustworthy (siblings!). **P1.**
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/348399)

### 4.6 7th place (nadare)
- **Technique.** Language-specific preprocessing (Sudachi/pykakasi for JP, segmentation for zh/th). **Missing addresses were filled "by connecting three neighborhoods of the distance".** Candidates: haversine 4 NN, 12 by a multiple regression of log-distance and embedding cosines, **ROUGE-1 precision/recall on words (4)**, **ROUGE-2 on characters (8)**, and name cosine (4). That is 32 in total, with maxIoU ≈ precision@32 = **0.9778** for retrieval only and 0.9935 after post-processing (both figures are from the team blog, read directly). LightGBM (num_leaves 2^12−1, `is_unbalance`). **Sample weights (team blog):** each pair is weighted by the IoU loss of getting only that pair wrong. Positives get 1 − (n−1)/n = 1/n and negatives get 1 − n/(n+1) = 1/(n+1), where n is the POI's true count, then mean-normalised. The blog's reason is that a false positive hurts the score more than a true negative helps (positives average about 0.8, negatives 1.0). **Post-processing: remove edges by betweenness centrality, union-find, then keep only pairs within graph distance ≤ 2.**
- **Adaptation.** Directional ROUGE precision/recall is the right shape for **truncation** and **generic-token injection** noise (A ⊂ B vs B ⊂ A). Check that `features.py` has *both* directions of token and char-n-gram containment, not only symmetric Jaccard. Treat the "missing field fill" as §0 #4. **P2** (feature check). The metric-derived sample weight is a candidate alternative to the Shopee 2nd α-exponent in §0 #5: weight a (t, s) pair by the per-S1 F0.5 loss of misclassifying it alone, given n_true(S1). It needs the same isotonic recalibration afterwards. **P1** (part of #5).
- Links: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335800) · [team blog (JP)](https://future-architect.github.io/articles/20220720a/)

### 4.7 8th place (kkanayj): 2-step fine-tuning with self-mined hard negatives
- **Evidence.** Distance kNN k=10 plus embedding kNN k=20 gives ideal IoU **0.986**. Second-stage fine-tuning on pairs mined by the first-stage model's own kNN raised ideal IoU (recall) **"from 0.97 to 0.986"**. The English MiniLM on unidecode/pykakasi-normalised text **beat multilingual models**. The authors' reason: there is too little data to learn multilingual invariance. The leak-confounded LB went 0.908 (550k ids) → 0.928 (700k) → 0.948 (1.1M).
- **Adaptation.** If GPU returns and we fine-tune the optional e5 view: do 2 rounds, where round 2 uses negatives from round 1's own top-k (the same S1-name siblings). Also try the **monolingual model on our transliterated text** against multilingual e5. **P3** (GPU-blocked).
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335928)

### 4.8 9th place (taksai): a very wide cascade
- **Evidence.** 400 candidates per id (350 lat/lon + 50 lat/lon + TF-IDF): **Max IoU 0.994, 420M candidates**. A light CatBoost with 40 features at **threshold 0.005** gives **Max IoU 0.987 with 3M candidates**. Target encoding of (name, name_match) value pairs. **Post-process: "calc the node predictions with avg of neighbors predictions", threshold 0.4.** LB CatBoost 0.925 → +pp **0.930** → +LGB 0.933 → +all data 0.943 → … 0.948.
- **Adaptations.** (a) §0 #10 cascade. (b) **Target encoding of token pairs** means learning, from train, how often a (target token, S1 token) pair co-occurs in true matches. That is the same machinery as our learned abbreviation maps, but it can be used as a feature for *all* token pairs (OOF-encoded). **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336415)

### 4.9 11th place (sakusaku): XGB per candidate rank and Dijkstra post-processing
- **Technique.** ArcMargin BERT with margin ramped 0.2 → 0.8, 4 encoders concatenated, **weighted DBA/QE**, 50 faiss candidates plus BallTree spatial candidates. **"We create XGB models for each rank of the candidates" (50 models).** Sample weights by the number of POI matches. **Dijkstra over edges weighted 1−p, with the graph made undirected by averaging forward and reverse weights**, and collect all nodes within a distance threshold. **Fill NaN texts with the nearest 5 points** (lat/lon neighbours, "for BERT models").
- **Evidence (timeline).** 0.922 → 0.928 → 0.931 → **0.940 (+fill na)** → 0.945 → **0.947 (+dijkstra)**. These are leak-affected, but the fill-NaN step is a local data fix.
- **Adaptation → §0 #4.** For each target with an empty address: find target neighbours (in S2 ∪ S3) with the same normalised name and a non-empty address, whose stage-1 top S1 is the *same* S1. Impute the address (or add "neighbour-address similarity to s" as a feature). Only impute when the neighbours agree, otherwise leave it empty. That keeps sibling traps out. **P1.**
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335924)

### 4.10 12th place (Matt Motoki and team): hop-dependent thresholds
- **Evidence.** 25 candidates per id with **max IoU 0.97675**, from 15 lat/lon + 15 name&lat/lon-concat + 10 category&lat/lon neighbours. Post-processing: match if a path exists over 1-hop edges with p > 0.5, 2-hop > **0.9**, 3-hop > 0.95, 4-hop > 0.998, 5-hop > 0.999.
- **Adaptation.** If we add target–target propagation (§0 #3), use **stricter thresholds for indirect evidence**. A target reached only via another target needs p ≥ 0.9 on both hops. **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336051)

### 4.11 13th place (213tubo): GNN node classification with an IoU loss
- **Technique.** 60 candidates (30 TF-IDF + 30 haversine). An LGBM filter keeps 3.7M pairs (**Max IoU 0.971**). DITTO-style xlm-r / mdeberta with country multitask (**+0.003**) and nlpaug swap/delete/split augmentation. **A 2-hop subgraph per query id raises Max IoU to 0.993.** A PNAConv GNN does node classification with a **"marker" node feature (1 = the query)**. **Loss = per-graph soft IoU + 0.1·BCE.**
- **Evidence.** LB 0.907 (20 candidates) → 0.924 (60 candidates, better models) → **0.946 (+GNN)**. CV 0.920.
- **Adaptation.** The per-S1 analog is a **listwise set model per S1 with a differentiable per-S1 F0.5 loss**. Input: the S1's candidate list (≤ 51), with stage-2 features and p. Output: per-candidate inclusion. This is the only thing that *learns* the decision instead of thresholding. Our expected-F rule is theoretically optimal only if p is calibrated, and it is not at test density. It can be a tiny MLP or DeepSets model trained on CPU. **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336124)

### 4.12 15th place (Shun_PI): multi-weight kNN, exact keys, and lazy group-average clustering
- **Technique.** kNN in L2 over the concatenation [lat/lon, name TF-IDF char_wb(3,4), category MDS] at **three weightings (3:10:1, 100:10:1, 3000:10:1)**. **Exact-match candidates on name, address, zip, phone (last 4 digits) and url (domain name).** 12 candidates per id (**about 7M, Max IoU ≈ 0.983**). DITTO-style mdeberta with **distance tokens [D0]-[D49] (percentile bins)** and a country token, and FGM eps 0.1. **Group-average clustering with a priority queue that lazily requests predictions for missing edges** gave **about +0.025**. Averaging **raw logits instead of probabilities gave +0.002**. Merging stops at 35% of test length.
- **Adaptations.** (a) **Exact keys** for web-domain and @handle names: normalise `www.`, the TLD and `@`, and add an exact-key blocker on the domain root. The noise model explicitly uses "web domains/@handles as names". **P2.** (b) Several weightings of our weighted-concat view (`blocking.py` "Weighted concat of L2-normalised views"), e.g. name-heavy, address-heavy and balanced, as separate views. **P2.** (c) When averaging model scores across evidence, average logits, not probabilities. **P3.**
- Link: [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335818)

### 4.13 Community "pitfalls" post (The Devastator), for completeness
- Split before pairing (GroupKFold by entity). Use the **same split for all first-stage models** that feed a second stage. Samples are compared unequal numbers of times. Our cross-fit already follows the first two.
- Link: [post](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/331170)

---

## 5. Kaggle Shopee: Price Match Guarantee (2021)

**Metric (read on the Evaluation page).** "mean F1 score … calculated in a sample-wise fashion". Posts always self-match, and groups are capped at 50. This is the same *shape* as our per-S1 macro F.

### 5.1 1st place (YoonSoo, limerobot): "From Embeddings to Matches"
- **Evidence (LB timeline).** Image-only 0.7 / text-only 0.64 → concat **0.724** → **min2 0.743** → normalise-then-concat 0.753 → **full data training 0.757** → union of comb/img/txt matches + tuned threshold **0.776** → **INB** + more text models **0.784** → multi-embedding INB stage 1 **0.793**.
- **Technique.** kNN k=51. **Min2**: since every item is guaranteed ≥ 1 other match, "we reject the second closest match *only if* the distance is over the min2-threshold, which we set high". **Iterative Neighbourhood Blending**: after thresholding, each embedding is replaced by the similarity-weighted sum of its neighbours' embeddings, re-normalised, then re-searched (3 stages). Of the 10 thresholds, only the stage-2 and stage-3 ones mattered at the end.
- **Adaptations.** (a) **"Min-k" prior**: 94.4% of our S1 have ≥ 1 match and a typical S1 has 2-6. The expected-F layer encodes this only through calibrated p and the match-exists model. A cheap extra rule to grid-search on P-dense: if exactly one target is kept and the second-best target has p ≥ τ₂ (high) and no higher-p S1 claims it, keep it. **P2.** (b) **INB for sparse views (§0 #12)**: after stage 1, build an expanded S1 profile = S1 TF-IDF vector + Σ p·(confident target vectors), then re-query the targets whose best score is still low. This carries cross-script and alias spellings learned from confident copies to their unmatched siblings. **P2.** (c) A final **refit on all training folds** (0.753 → 0.757 here, leak-free). **P1.**
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)

### 5.2 2nd place (lyakaap, tkm2261): LightGBM + GAT, and the group-size weighting
- **Technique and evidence.** Graph features: "Avg and std of top-K cosine similarities of each item (K=5, 10, 15, 30)" and "**Normalized avg to be mean=0 and std=1 — To handle the distribution difference between train and test**". PageRank gave 0.788 → 0.789, query expansion 0.789 → 0.790, and a GAT ensemble member 0.790 → **0.792**. **Recursive removal of the highest-betweenness edge** gave about +0.001. **"Sample weighting based on label group size — It is important to predict smaller label groups for micro-F1 score. Weighted sample with 1 / (label group size) ** 0.4."** For ensembling, subtract each model's own best threshold, then sum. The mutual-edge filter drops A–B unless B–A also exists.
- **Adaptations.** (a) **§0 #5 group-size weights.** In a pair-level logloss, an S1 with 6 true targets contributes 3× the positive gradient of an S1 with 2. The metric counts every S1 once. Try w = 1/(n_true(S1)+1)^0.4 (tune α ∈ {0.2, 0.4, 0.6}) on stage 2. **Recalibrate on unweighted OOF** (isotonic) afterwards, because weighting distorts probabilities and our decision layer consumes probabilities. **P1.** (b) **§0 #6 per-dataset z-scoring** of density-type features (top-k mean/std of stage-1 p per S1 and per target). This is a direct fix for "mid-range probabilities become over-confident at test density". **P1.** (c) **Threshold-subtracted ensembling** when we blend stage-1, stage-2 and a rank-feature model tuned at different thresholds. **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238022)

### 5.3 3rd place (Btbpanda): density features, and clustering to a target cluster size
- **Technique and evidence.** A 3M-pair training sample at a 4% duplicate rate. About 500 features, including **"density around both points (frequency of points on different radiuses)"** and "points ranks". When validating, "I was looking for candidates not only in validation fold, but in full train sample. This scheme reduces CV/LB gap". **Agglomerative clustering on GBM probabilities "until average cluster size become equal threshold (for train sample avg cluster size eq 2.8, for LB best size was 2.6)"**. Leftover singletons merge to their nearest cluster. 0.781 → **0.79**.
- **Adaptation → a sanity check on test.** Choose the test threshold so that **predicted matches per S1 fall into a band implied by train statistics, adjusted for the observed density** (a Saerens-EM prior estimate is in `research/notes/fmeasure.md`). The existing France submission sits at 3.45 matches per S1, and this gives a second, independent check on the 0.8 threshold. Do not use it as the only criterion, because the distractor rate may differ in test. **P1** (cheap check).
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238515)

### 5.4 4th place (Pascal Pfeiffer, Christof Henkel, Philipp Singer): validation with in-fold noise
- **Evidence.** They tracked three KPIs: OOF F1; OOF F1 "using infold embeddings as noise"; and the latter "weighting False Positives with a factor of 2". "Some post-processing methods excel on small sets but fail on larger sets (think about e.g. always picking rank2 prediction)". The post-processing list includes rank-2 reciprocal matching, a large rank-2/rank-3 gap → add rank 2, and re-matching unmatched rows.
- **Adaptation.** Our **P-dense regime is the in-fold-noise KPI**. Keep a third KPI: P-dense with FP×2 weighting as a pessimistic proxy for France. Adopt a decision rule only if it wins on all three. **P0 (process).**
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238295)

### 5.5 5th place (Ahmet Erdem and team): percentage-rank features and tiered agglomeration
- **Evidence.** "Since test set size was larger than our one fold size, some features were going to have different distribution on the test set due to higher possibility of False Positive matches. Therefore another XGB model is trained on **percentage rank features** and ensembled." "FP features" (closest-match distances into the train set) helped CV a lot but LB less, again blamed on size. **Tiered agglomeration**: merge clusters at p > **0.8**, attach unclustered items to a cluster at > **0.7**, and pair unclustered items together at > **0.3**.
- **Adaptations.** (a) **§0 #6**: a sibling LightGBM on per-S1 and per-target *percentile ranks* of every similarity feature, blended with the raw model. It is invariant to the train→test density change. **P1.** (b) **Tiered thresholds**: a high threshold to *add a target to an S1 that already has confident members*, a lower one to *give an empty S1 its first member*. That is the CyberPunk ISO-CON idea again (§2.4). **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238078)

### 5.6 6th place (kurupical): force-two and unit mismatch
- **Evidence.** "Force prediction one to two: If there is one prediction, we made it two … +0.01" (CV and LB). "Remove unmatch units": drop matches whose quantity units disagree (400gr vs 200gr).
- **Adaptation.** Force-two is the aggressive version of §5.1(a). Grid-search it on P-dense rather than assuming it transfers. **Do not** hard-veto on number mismatch (house numbers of true copies are shifted). Keep numbers as features. **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238010)

### 5.7 58th place (A. Demyanchuk): dynamic thresholds
- **Technique.** Per row, compute the mean of the best 6 similarities ("the majority of the training data has no more than 6 members in the group"). Bucket rows by the [0.3, 0.6, 0.9] quantiles of that mean, set each bucket's threshold to its quantile, and shrink by 0.99/0.95/0.9 ("the smaller the similarities the more restrictive").
- **Adaptation → §0 #1.** Our typical S1 has 2-6 targets, the same "6". Compute d(S1) = mean of top-6 stage-2 p (or of stage-1 p) and fit τ as a monotone function of d on P-main ∪ P-dense OOF. Because d is itself higher on test, τ(d) moves automatically where we now shift 0.7 → 0.8 by hand. **P0.**
- Link: [write-up](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238708)

---

## 6. Cross-device linking contests (6.1 uses a per-entity averaged F-measure, our metric family; 6.2 does **not**)

### 6.1 ICDM 2015 Drawbridge Cross-Device Connections (Kaggle): **per-device F0.5, averaged**
- **Metric evidence.** The 3rd-place paper: "The score is formed by averaging the individual F0.5 scores for each device in the test set". A forum clarification says every test device has ≥ 1 cookie, and predicting none scores 0 ([edge-case thread](https://www.kaggle.com/competitions/icdm-2015-drawbridge-cross-device-connections/discussion/14708)). Unlike ours, this contest had no "correctly empty" rows.
- **3rd place, Díaz-Morales (paper, arXiv 1510.01175).** Rule-based candidate selection kept **98.3%** of devices' true cookies in the candidate set (3,664,022 train pairs / 1,505,453 test pairs), with 67 features and bagged boosted trees. **Decision procedure (Table II):** Step 1, take the argmax cookie. If its score is below the Threshold, Step 2: **widen the candidate set** (every cookie sharing an IP) and take the argmax of that. Step 3: label it, together with all cookies of its handle. Step 4: **add every other candidate with ŷi > ŷk × AccessParameter**, where AccessParameter depends on how many cookies are already labelled and on handle properties. The Threshold and AccessParameter values were tuned by 10-fold CV. **Semi-supervised step:** test devices with **top > 0.4 and second < 0.05** are fed back to recompute features and retrain. For devices whose "second candidate scores less than 0.1 the average F0.5 score is higher than 0.99 and 62% of the devices satisfy that condition". **Scores (private):** selection only **0.5** → supervised **0.875** → +bagging **0.876** → +SSL+PP **0.88**, 3rd of 340 teams.
- **1st place (idle_speculation, forum summary).** (device, cookie, IP) triples. The most influential features were **ranks of each cookie within its device**. They added **OOF predictions of models "to predict whether a cookie was matched to any device"**. XGBoost `rank:pairwise` was grouped by device. For each device, "we took the drawbridge handle of the highest scoring cookie and submitted all the cookie_ids for that drawbridge handle."
- **Adaptations.**
  - **Relative-to-top inclusion (§0 #1).** For each S1: keep the top target if p ≥ τ_abs(d), and keep others if p_i ≥ α(k, claims)·p_top, where α depends on how many are already kept (k) and whether the target is contested. This is the AccessParameter recipe. It is 2-4 parameters, tuned on P-main and P-dense OOF on CPU. **P0/P1.**
  - **Recall fallback for weak tops (Step 2).** For S1s (and targets) whose best p is low, run a *wider* retrieval (bigger k, extra views) and re-score only those. This is on-demand candidate expansion, which is cheaper than a globally bigger budget. **P2.**
  - **Target "matchability" model (1st place).** An OOF model P(target is matched to any S1) built from target-only and best-candidate features (empty fields, name frequency, script, best score, margin), fed into stage 2. It is the target-side twin of our S1-side match-exists model and addresses the 26% distractor targets directly. **P1.**
  - **SSL gating rule (top > 0.4 and second < 0.05)** as the recipe for France pseudo-labels (§0 #9). **P1.**
- Links: [3rd-place paper](https://arxiv.org/abs/1510.01175) · [1st-place summary](https://www.kaggle.com/competitions/icdm-2015-drawbridge-cross-device-connections/discussion/16122)

### 6.2 CIKM Cup 2016 Track 1 (cross-device entity linking): 1st place, Phan, Tay, Pham (arXiv 1610.07119)
- **Metric caveat (audit).** The paper says "The evaluation metric used in the leaderboard ranking is the F1 measure", and contestants chose how many pairs to submit (the winners submitted 120,000). So this is a *pair-level* F1, not a per-entity average. The ideas below transfer as set-construction heuristics, not as metric-matched decision rules.
- **Evidence.** kNN candidates (k = **18** chosen from a recall curve). **Supervised inference:** merge two clusters only if the average score of all cross pairs, from a second classifier XGB2 (trained on cluster-merge-extended pairs), exceeds **α = 0.5**. **Unsupervised inference:** transitive closure with **cluster size capped at β = 5**, "since … most of the users are owning 3-5 devices". Final: the first 120,000 unique pairs of [supervised-extended (from the top 45,000), unsupervised-extended (from the top 80,000), raw sorted]. "Unsupervised inference increased our final F1 score from **0.4155 to 0.4204**, which was critical to win."
- **Adaptations.** (a) **Size cap from the known entity-size distribution.** Never let collective or transitive additions push an S1's predicted set beyond the train high quantile (our "typical 2-6"). **P2.** (b) **Set-level second classifier.** For each candidate addition to an S1's set, score (S1, current set ∪ {t}) with features over the whole set (min and mean pairwise target–target similarity inside the set, source mix S2/S3, name diversity). This catches sibling FPs that look fine pairwise but break set cohesion. **P2.**
- Link: [paper](https://arxiv.org/abs/1610.07119)

---

## 7. Kaggle Quora Question Pairs (2017): 1st place (Maximilien Baudry and team)
- **Evidence and technique.** "Structural features (i.e. from graph)": neighbour counts of each question, min/max, **intersections, unions**, shortest path when the main edge is cut, higher-order neighbour counts, connected-component size and **% of edges in train**. They also **weighted the graph with first-level model predictions** (raw predictions worked best). **Rescaling by "perimeter"**: train/test prior shift differed by strata defined by question occurrence counts (both once / one once / both more than once). An intermediate rescale gained about 0.001 log loss.
- **Adaptations.** (a) We already have collective features. The unexplored part is the **target–target graph**: for each (t, s), how many of t's target-neighbours also list s in their top-k, weighted by stage-1 p. That is the Quora "intersection" feature on the right graph. **P2.** (b) **Stratified prior correction**: calibrate p separately by strata (for example the S1 candidate-count bucket × country × target-missing-pattern) on P-dense. Test density shifts differ by stratum. **P2.**
- Link: [write-up](https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355)

---

## 8. MWPD2020 (ISWC Semantic Web Challenge): WDC product matching
- **Evidence.** Task-1 test set: 1,100 randomly chosen corner cases, plus 400 pairs by challenge type (**typos**, **dropped tokens**, **new products** with high or low similarity to known ones). Winner **PMap F1 86.05** (P 82.04, R 90.48). The top teams "vary only within 1% F1". Rhinobird and ISCAS-ICIP used **heuristic post-processing, e.g. "always predicting non-match if the brands of both offers do not match"**. All teams were shown per-challenge-type scores after Round 1. ISCAS-ICIP then improved on every challenge type in Round 2, most of all on *new products*.
- **Adaptation.** Make our error analysis **per noise type**. We know the generator's noise list (leetspeak, aliases, acronyms, cross-script, truncation, generic-token injection, state code/name, landmarks…). Tag validation pairs with detectable noise types (regex detectors: `a/k/a`, `dba`, `@`, domain TLD, all-caps acronym, non-Latin script, "Near …") and report macro-F0.5 per tag. Then fix the worst tag first. **P1** (cheap analysis).
- Link: [overview paper (CEUR Vol-2720 paper1)](https://ceur-ws.org/Vol-2720/paper1.pdf)

## 9. SemTab 2020: MTab4Wikidata (cell-entity annotation under misspellings)
- **Evidence.** Fuzzy entity search over about 150M names: "We first start with two edits; if the searcher does not have answers, the number of edits is gradually increased to a maximum of six edits", with "coverage of 99.89% on average". A "two cells search" (joint lookup of a cell with a related cell in the same row) is **preferred when it returns candidates**. Otherwise single-cell candidates are used.
- **Adaptation.** (a) **Adaptive-radius fallback** is the same idea as Drawbridge Step 2: widen only for weak queries. (b) **Joint-key lookup first**: a (name, city) or (name, postcode) combo view outranks name-only when it returns anything. Our "combo" view is this. Make the fallback explicit: use the name-only candidates only for targets whose combo view came back empty or weak. **P2.**
- Link: [paper (CEUR Vol-2775 paper9)](https://ceur-ws.org/Vol-2775/paper9.pdf)

## 10. Amazon contests: low transfer, listed for completeness
- **KDD Cup 2022 ESCI (multilingual query–product relevance).** The Task 1 (query-product ranking) winner (team "www", NetEase) reached NDCG **0.9043** with multilingual PLMs, "translation-based data augmentation", label smoothing, self-distillation, **AWP and FGM adversarial training**, pseudo-labelling and ensembling ([arXiv 2208.02958](https://arxiv.org/abs/2208.02958)). *Transfer:* only if we train a cross-encoder (GPU-blocked). **Translation augmentation is a compliance risk under our "no external augmentation" rule.** **P3.**
- **KDD Cup 2023 (multilingual session recommendation).** *Verified:* Task 2 is "next-product recommendations for underrepresented locales, specifically ES, IT, and FR", with pretraining on JP/UK/DE sessions ([Amazon-M2, arXiv 2307.09688](https://arxiv.org/abs/2307.09688)). The workshop page lists the Task 2 winners as MGTV-REC ("A Stacking and Transfer Learning with Diverse Similarity For Building Multilingual Session-based Recommendation Systems") and gpt_bot ([kddcup23.github.io](https://kddcup23.github.io/)). *Not verified:* the winners' actual methods, because the OpenReview PDFs return 403 and were not read. The earlier "candidate generation + GBDT ranking" description came only from search snippets and stays **UNVERIFIED**. **P3.**
- **KDD Cup 2024 (LLM shopping assistant).** The winning NVIDIA team fine-tuned **Qwen2-72B-Instruct** and won all 5 tracks ([arXiv 2408.04658](https://arxiv.org/abs/2408.04658)). That **violates our ≤ 8B rule**. **Not applicable.**
- **Past Amazon ML Challenges.** From secondary sources only (**UNVERIFIED**): 2023 product length prediction, 2024 entity-value extraction from product images, 2025 product price prediction. None is ER, so no transferable ER tricks. The Unstop page did not state problem statements.

## 11. Scanned, low transfer, or only partially verified
- **Alaska benchmark** (arXiv 2101.11259). Per the SIGMOD Record report, it supplied the SIGMOD 2020 camera data and the 2021 notebook data. The 2021 Altosight data came from the sponsor company, and the 2022 data was synthesised from the 2021 data. The abstract says "70k heterogeneous product specifications from 71 e-commerce websites". It is a dataset, not a method. **P3.** [abs](https://arxiv.org/abs/2101.11259)
- **SIGMOD 2023** (approximate kNN graph, k=100, 10^6 and 10^7 vectors, winner Jiayi Wang, Tsinghua) and **SIGMOD 2024** (hybrid vector search: 10M × 100-d vectors, 4M queries, k=100, categorical and timestamp filters). Task pages were read ([2023](https://transactional.blog/sigmod-contest/2023), [2024](https://transactional.blog/sigmod-contest/2024)). **Winning techniques are UNVERIFIED** (posters not read). This is relevant only if we move to dense retrieval with a per-country filter. **P3.**
- **OAEI instance-matching (SPIMBENCH)** and **DI2KG 2019/2020 challenge pages**: the DI2KG site did not resolve (DNS), and OAEI results were seen only in search snippets. **UNVERIFIED, not used.**

---

## 12. Consolidated action list for our pipeline

The list is ordered by expected value per unit of compute, given that the Modal spend limit is currently hit.

**P0: decision layer and validation discipline (CPU, existing OOF predictions)**
1. **τ(d) density-conditioned threshold** (Shopee 58th, Drawbridge AccessParameter, Foursquare 4th). Features for τ: mean of the top-6 p per S1, number of candidates with p ≥ 0.1, the S1's rank-1/rank-2 margin, and country. Fit a monotone mapping on the union of P-main and P-dense OOF. It must reproduce thr 0.7 on P-main and ≈ 0.8 on P-dense. Code goes in `src/ber/decide.py` as a new rule `density_threshold`.
2. **Relative-to-top rule**: keep t if p_t ≥ max(τ(d), α_k·p_top). Tune α_k for k = 2..6 on P-dense.
3. **Validation guardrails**: every rule must win on P-main, P-dense, and P-dense with FP×2 (Shopee 4th). Candidate pools stay full size (Foursquare 3rd, RUPikachu).

**P1: cheap features, weights and refits (one stage-2 retrain)**
4. **Target-missing-pattern categorical plus a per-pattern threshold** (SIGMOD-21 residual blocking), for the empty-address FN class.
5. **Target–target neighbour smoothing feature** p′(t,s) (Foursquare 6th/9th/11th), plus **address imputation from agreeing target neighbours** (Foursquare 11th: +0.009 LB from fill-NaN, applied to BERT inputs; the LB is leak-affected).
6. **Target matchability OOF model** as a stage-2 feature (Drawbridge 1st).
7. **Percentile-rank / z-scored feature twin model** blended with the raw model (Shopee 5th/2nd), for the density shift and France.
8. **Group-size sample weights** 1/(n_true+1)^α with post-hoc isotonic recalibration (Shopee 2nd). Also try the metric-derived single-error-loss weight (Foursquare 7th blog, §4.6).
9. **Map-dropout training** (panda "data poisoning"), validated with LOCO US↔India, for France.
10. **Per-country view budget ratios** (Foursquare 2nd). France gets the LOCO-optimal, map-independent-leaning mix.
11. **France pseudo-label self-training** with gate top ≥ 0.95 and second ≤ 0.05 (a stricter Drawbridge SSL gate) to learn French abbreviation maps. Planned as E-8.
12. **Final refit on all train folds**, with rounds scaled from the CV best iteration (Shopee 1st: +0.004 leak-free).
13. **Per-noise-type error report** (MWPD2020).
14. **Train↔test overlap audit.** Report it only. Using any overlap needs the user's explicit OK (compliance).

**P2: needs a new stage or more compute**
15. Cascade: full 112/S1 union behind a light LightGBM filter (Foursquare 1st/9th/13th).
16. Neighbour-set merge + re-score (Foursquare 1st stage 4). Weak-query fallback retrieval (Drawbridge Step 2, MTab adaptive edits).
17. S1-profile query expansion (INB/DBA) for cross-script and alias recall (Shopee 1st/5th, Foursquare 11th).
18. Listwise per-S1 set model with a soft-F0.5 loss (Foursquare 13th GNN idea, CPU-sized). Set-level cohesion classifier and size cap (CIKM 2016).
19. Subsumption and ambiguity features from mock labels (SIGMOD-20 winner). Directional ROUGE containment check (Foursquare 7th). Token-pair target encoding (Foursquare 9th). Domain/@handle exact keys and multi-weight concat views (Foursquare 15th).
20. Min-k / force-two and tiered thresholds (Shopee 1st/5th/6th, CyberPunk ISO-CON): grid-search on P-dense only.

**P3**
21. Contrastive or ArcFace encoder fine-tuning with self-mined negatives, a monolingual-on-transliterated baseline, digit-grouping for numbers, distance and percentile tokens in a cross-encoder, and FGM/AWP. All are GPU-blocked.
22. SIGMOD-23/24 ANN tricks (UNVERIFIED) and the KDD Cup items.

---

## 13. UNVERIFIED or discrepant items (do not cite as fact)
- SUSTech 2022 X1 recall "0.692": present in the poster text, but its attachment to X1 is inferred.
- WBSG max sequence length for D2: poster 32 vs README 24.
- SIGMOD 2022 team count: SIGMOD Record 60 vs Mannheim news 55. Both numbers were confirmed on their pages. The sources genuinely disagree.
- SIGMOD 2021 NotebookLarge entity count: CyberPunk poster 158 vs BoomBoomChicken poster 153.
- (Resolved by the audit.) The Foursquare 7th-place "32 candidates, maxIoU 0.9778" and the FP-penalising sample weights were read directly on the team's Japanese blog. The candidate breakdown (4+12+4+8+4) is in the Kaggle post.
- Every Foursquare LB number is leak-confounded (67% test rows duplicated from train). Foursquare CV and Max-IoU numbers are not.
- The Foursquare 1st-place team name ("re:waiwai" in the earlier draft) is not in the write-up. It was replaced with the write-up authors.
- KDD Cup 2023 winner *methods*, past Amazon ML Challenge problem statements, SIGMOD 2023/2024 winning techniques, and OAEI/DI2KG challenge details (the DI2KG site still does not resolve).

## 14. Sources (all opened during this sweep)
- SIGMOD 2022: https://dbgroup.ing.unimore.it/sigmod22contest/leaders.shtml ; posters: https://dbgroup.ing.unimore.it/sigmod22contest/posters/{WBSG,April,QaisHousien,SUSTech_DBGroup,RUPikachu}.pdf ; https://github.com/abrinkmann/acm_sigmoid_2022_challenge ; https://github.com/rutgers-db/SIGMOD2022-Programming-Contest-Public ; https://transactional.blog/sigmod-contest/2022
- SIGMOD 2021: https://dbgroup.ing.unimore.it/sigmod21contest/leaders.shtml ; posters: https://dbgroup.ing.unimo.it/sigmod21contest/posters/{SUSTech_DBGroup,UKN,panda,CyberPunk2021,BoomBoomChicken}.pdf ; https://transactional.blog/sigmod-contest/2021
- SIGMOD 2020 / DI2KG 2020: https://transactional.blog/sigmod-contest/2020 ; https://ceur-ws.org/Vol-2726/ (paper2, paper3)
- Lessons report: https://sigmodrecord.org/2023/07/06/experiences-and-lessons-learned-from-the-sigmod-entity-resolution-programming-contests/ (PDF: https://sigmodrecord.org/?smd_process_download=1&download_id=13459)
- Foursquare: evaluation page (Kaggle PageService) ; write-ups 336055, 336062, 336072, 336090, 338112, 335810, 348399, 335800, 335928, 336415, 335924, 336051, 336124, 335818, 331170 ; leak 335799 (all under https://www.kaggle.com/competitions/foursquare-location-matching/discussion/<id>) ; https://future-architect.github.io/articles/20220720a/
- Shopee: evaluation page (Kaggle PageService) ; write-ups 238136, 238022, 238515, 238295, 238078, 238010, 238708 (https://www.kaggle.com/competitions/shopee-product-matching/discussion/<id>)
- Drawbridge: https://arxiv.org/abs/1510.01175 ; https://www.kaggle.com/competitions/icdm-2015-drawbridge-cross-device-connections/discussion/16122 ; .../discussion/14708
- CIKM Cup 2016: https://arxiv.org/abs/1610.07119
- Quora: https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355
- MWPD2020: https://ceur-ws.org/Vol-2720/paper1.pdf
- SemTab 2020 MTab4Wikidata: https://ceur-ws.org/Vol-2775/paper9.pdf
- Amazon KDD Cups: https://arxiv.org/abs/2208.02958 ; https://arxiv.org/abs/2408.04658 ; https://arxiv.org/abs/2307.09688 ; https://kddcup23.github.io/
- SIGMOD 2022 team count (Mannheim): https://www.uni-mannheim.de/dws/news/wbsg-wins-sigmod-programming-contest-2022/
- Alaska: https://arxiv.org/abs/2101.11259 ; SIGMOD 2023/2024: https://transactional.blog/sigmod-contest/2023 , https://transactional.blog/sigmod-contest/2024

---

## Audit log

_Independent audit, 2026-09-25. The auditor assumed every item was hallucinated until it was re-read on its primary page. Nothing from the original sweep's cached files was reused. Everything was re-fetched into a fresh scratch folder._

**How it was checked**
- **Kaggle (28 pages).** Re-fetched from Kaggle's discussion JSON (`discussions.DiscussionsService/GetForumTopicById`) and page JSON (`competitions.PageService/ListPages`, competition ids 35476 and 24286). Each claimed number, quote, author and place was checked against the raw markdown. The pages were: 16 Foursquare topics (1st, 2nd ×3, 3rd, 4th, 6th, 7th, 8th, 9th, 11th, 12th, 13th, 15th, the pitfalls post, the leak thread), 7 Shopee write-ups, Drawbridge 16122 and 14708, Quora 34355, and the Foursquare and Shopee Evaluation pages.
- **Posters (10).** Downloaded all 10 SIGMOD 2021/2022 posters (both the `unimo.it` and `unimore.it` hosts resolve) and extracted their text with PyMuPDF. Also checked both finalists pages, the SIGMOD Record 52(2) PDF, the transactional.blog 2020-2024 pages, and the Mannheim news page.
- **Papers (7).** Read the PDFs of arXiv 1510.01175, 1610.07119, 2208.02958, 2408.04658, 2101.11259 and 2307.09688, and the CEUR Vol-2726 paper2/paper3, Vol-2720 paper1 and Vol-2775 paper9 PDFs.
- **Repos and blog.** Checked the GitHub API for the three linked repos (licences) and read the WBSG README. Read the future-architect 7th-place blog (Japanese) directly.

**Confirmed as written.** All Foursquare numbers were confirmed:
- 1st: 0.979 / 0.875 / thr 0.01 / 0.878 / 0.907 / 0.911 / 0.9166 / 0.941 vs 0.977.
- 2nd: 128+128, 0.9895, 0.918, thr 0.005, 4.1, 0.986, 8 fields × 11 measures, 0.949 / 0.971.
- 3rd: the 600k CV quote. 4th: thr 0.007, 20 folds, 200+ features, 0.939 → 0.957, and the negative results.
- 6th: 28 candidates, IoU 0.978, 6,500 kd-tree neighbours, and the convolution formula.
- 8th: 0.986 and 0.97 → 0.986, LB 0.908 / 0.928 / 0.948. 9th: 0.994 / 420M → 0.987 / 3M, and the LB chain.
- 11th: the timeline. 12th: 0.97675 and the hop thresholds. 13th: 0.971 / 0.993 / +0.003 / 0.907 → 0.924 → 0.946, CV 0.920.
- 15th: the weights, char_wb(3,4), ≈7M, ≈0.983, D0-D49, +0.025, +0.002, 35%. The leak is 67% (0.435/0.65).

All Shopee numbers were confirmed:
- 1st: the LB chain 0.7 → 0.793, min2, and INB. 2nd: 0.788 → 0.792, α 0.4, z-scoring, betweenness pruning.
- 3rd: 3M pairs at 4%, about 500 features, cluster size 2.8 / 2.6, 0.781 → 0.79. 4th: the three KPIs.
- 5th: percentage-rank features and 0.8 / 0.7 / 0.3. 6th: +0.01. 58th: the quantile/shrink rule.

Drawbridge (98.3%, 3,664,022 / 1,505,453 pairs, 67 features, the Table II steps, 0.4 / 0.05, 0.1 → 62% at > 0.99, 0.5 → 0.875 → 0.876 → 0.88, 340 teams) and the 1st-place summary were confirmed. So were the CIKM 2016 figures (k = 18, α = 0.5, β = 5, 45k / 80k / 120k, 0.4155 → 0.4204), the Quora perimeters and the ~0.001 gain, and the SIGMOD posters:
- 2022: WBSG 0.529 / 0.713 / 0.345 / 1914.275 s, 32-d, batch 1024, 437,581 WDC offers. April 0.743 / 0.297 / 0.520 / 1679 s with its T1-T3 values. Qais 0.726 / 0.301 / 0.514. SUSTech 0.323 / 1671 s. The RUPikachu quote and the 35-minute limit.
- 2021: SUSTech P/R/F1, panda 0.972 / 0.992 / 0.907 / 0.957 and the data-poisoning quote, UKN 98.4 / 98.0 / 90.7, CyberPunk 94.5 / 99.0 / 90.2 / 94.6, BoomBoomChicken 0.950 / 0.982 / 0.874 / 0.935.

Also confirmed: Blacher (2,800 labels → 0.97, +200 → 0.99), Zecchini (98.26%, 0.47 / 0.49, 3,914 blocks / 54,000,932 pairs, P 0.28 / R 0.99), MWPD (1,100 + 400 pairs, PMap 86.05 / 82.04 / 90.48, the brand heuristic, within 1%), MTab (2 → 6 edits, 99.89%, two-cells priority), KDD 2022 (www, 0.9043, AWP / FGM, etc.), and KDD 2024 (Qwen2-72B-Instruct). The internal project numbers (0.98703, 0.99645, 0.9905 vs ≈0.989, 3.45 matches per S1, 220,683 S1) match PROJECT_STATE.md, RESEARCH_LOG.md and FINAL_RECOMMENDATION.md.

**Fixed.** The last item is an addition rather than a fix.
1. **§6 header / §6.2 / §0 #13.** CIKM Cup 2016 was filed under "per-entity averaged F-measure". The paper states a plain leaderboard F1 over a contestant-chosen number of pairs. It is now relabelled as pair-level F1, with a caveat.
2. **§4.1.** The team name "re:waiwai" appears nowhere in the write-up. It was replaced with the write-up authors (charmq, Takoi, pao).
3. **§1.2.** "About 10 `sed` commands" became the poster's wording, "around 10 bash commands". The poster never mentions sed.
4. **§1.3.** "nouns only" became "nouns and pronouns", as the poster says.
5. **§0 #5.** The Shopee 2nd quote was paraphrased as "sample-wise F1". The authors wrote "micro-F1", so the wording now follows them.
6. **§0 #2.** The RUPikachu quote was paraphrased as ">0.9 … <0.25". It is now the exact poster sentence.
7. **§8.** "Teams improved on new products in round 2" is specific to ISCAS-ICIP in the paper, and the text now says so.
8. **§11.** Alaska is not the source of all SIGMOD 2021 data. It supplied the 2020 cameras and the 2021 notebooks, while Altosight data came from the sponsor. Corrected.
9. **§10.** KDD Cup 2022 now says the winner is the *Task 1* winner. KDD Cup 2024 now names the model (Qwen2-72B-*Instruct*) and says it won all 5 tracks.
10. **§10, KDD Cup 2023.** The Task 2 ES/IT/FR definition and the winner names are now verified (Amazon-M2 arXiv 2307.09688; kddcup23.github.io). The winner *methods* remain UNVERIFIED (OpenReview returns 403).
11. **§4.6 / §13.** The 7th-place blog figures (0.9778 retrieval-only, 0.9935 after post-processing) and the FP-penalising sample weights are now confirmed directly on the blog, not "through a summariser". The exact weight formula was added.
12. **§1.** The Mannheim "55 teams" figure had no URL. The URL was added and the quote confirmed. The SIGMOD Record "60 teams from 10 countries" was also confirmed.
13. **§0 #9 vs §12 #11.** The two sections gave different France pseudo-label gates ("margin ≥ 0.9" vs "second ≤ 0.05"). They now use the same gate.
14. **§0 #4, §4.9, §12 #5.** Added the qualifiers that the 11th-place fill-NaN was for the BERT text inputs and that its +0.009 is an LB (leak-affected) number.
15. **§2 header.** Added the NotebookLarge entity-count discrepancy (158 vs 153) and the source of the metric and hidden-set statements.
16. **Minor.** Repo licences were added: WBSG repo none, Rutgers BSD-2-Clause, TheoViel none (confirmed via the GitHub API). A note was added that the Foursquare 3rd-place "bi-encoder" takes a joint pasted sequence.
17. **New, verified adaptation.** The Foursquare 7th metric-derived weight is added as a candidate for §0 #5 and §12 #8.

**Still UNVERIFIED (unchanged, correctly flagged):**
- The SUSTech 2022 "0.692" attribution.
- The KDD Cup 2023 winner methods.
- The past Amazon ML Challenge problem statements.
- The SIGMOD 2023/2024 winning techniques.
- The OAEI and DI2KG pages. `di2kg.inf.uniroma3.it` still does not resolve.

**Removed:** nothing. Every cited primary page exists, and no fabricated item was found. The errors were paraphrase drift, one invented team name, and one metric misclassification.
