# Competition write-ups and GitHub implementations: research notes

Topic owner: research workflow, topic "competition intelligence + GitHub survey". Written 2026-09-25.

Scope: I looked for public winning solutions of competitions close to ours: Kaggle Foursquare Location Matching (2022), Kaggle Shopee Price Match Guarantee (2021), Kaggle Quora Question Pairs (2017), the WDC / MWPD2020 product-matching challenge, and Amazon ML Challenge 2023/2024/2025. I also read company-name-matching write-ups (ING, DNB, string_grouper, OpenSanctions) and surveyed the open-source ER stack. From each source I extracted methodology that transfers to the Amazon ML Challenge 2026 Business Entity Resolution task. **No code or data from any of these sources is to be copied into our pipeline.** Almost every Kaggle solution repo has *no license*, so all rights are reserved and we may read them for ideas only.

Verification rule:
- **Kaggle write-ups** were read in full through Kaggle's own discussion endpoint, the same content that renders at the listed `kaggle.com/competitions/.../discussion/<id>` URL. The competition metric text came from the competition's Evaluation page, via the same Kaggle API.
- **Papers** were confirmed on arXiv (API/abs), the Crossref DOI registry, vldb.org PDFs, ceur-ws.org, or ijpds.org.
- **GitHub metadata** (license SPDX, stars, last push) comes from the GitHub REST API (`gh api repos/...`) on 2026-09-25.
- **Model licenses** come from the Hugging Face model API (`cardData.license` plus `license:` tags) for `huggingface.co/<id>`.

Only numbers I actually read are reported. Anything not confirmed is marked UNVERIFIED. Nothing was run on the laptop except text extraction from fetched PDFs and JSON parsing.

Current project state, for context (PROJECT_STATE.md):
- Stage-2 stacked LightGBM reaches 0.98703 macro-F0.5 on fold 0. The blocking oracle is 0.99645.
- The residual errors are (a) FPs on sibling lookalike entities (same name, nearby house number) and (b) FNs on empty-address targets.
- Test is denser than train, so the optimal threshold moves from 0.7 to 0.8.

The recommendations below target those residuals.

---

## 0. TL;DR: what the winners did that we should copy (methodology only)

1. **Every top solution in Foursquare and Shopee used the same 4-stage template**, and all of them tracked an oracle ceiling at every stage:
   1. a union of heterogeneous kNN retrievers;
   2. a *cheap* GBDT pre-filter;
   3. a rich GBDT plus a cross-encoder;
   4. graph post-processing.

   The ceiling metric was "Max IoU" in Foursquare. Examples:
   - Foursquare 1st: candidate stage Max IoU 0.979.
   - Foursquare 2nd: 0.9895 after retrieval, then 0.986 after transformer blocking at 4.1 candidates per id.
   - Foursquare 13th (leak-free #2): 0.971 after LightGBM filtering, then 0.993 after 2-hop graph expansion.

   Our equivalent is the blocking-oracle macro-F0.5 (0.99645 now). The template says the next gains come from the stage-3 cross-encoder and stage-4 graph layers, not from more blocking.
2. **The metric in both closest competitions is a per-row mean, just like ours.** Foursquare scored mean IoU per row and Shopee scored mean F1 per row. The winners optimized per-query metrics directly:
   - Foursquare 13th trained a GNN on each query's 2-hop subgraph with a **per-graph IoU loss**, which added about +0.02 on CV and LB.
   - Shopee 2nd weighted training samples by **1/(group size)^0.4** to protect small groups under the per-row mean.
   - Shopee 1st used a **"min2" rule**: every posting was guaranteed at least one other match, so the 2nd-closest candidate was accepted unless it was very far. Our analogue is a learned "is this S1 a singleton?" decision, because ground truth usually has 2-6 copies and singletons score 1 only if we predict nothing.
3. **Per-query contextual features are the strongest GBDT features.**
   - Foursquare 1st: max/min/mean of each similarity over a query's candidates, plus ratios, about 120 features in total, CV 0.875.
   - Foursquare 13th: mean of top-k distances for k = 5, 10, 15.
   - Shopee 2nd: avg/std of top-K cosine similarities, **normalized to mean 0 / std 1 to absorb the train/test distribution shift**, plus PageRank.
   - Shopee 3rd: density around both points and rank features, about 500 features.

   This matches our own finding that `c_combo_tmargin` and `trank` dominate. **The Shopee-2nd normalization trick targets exactly our train→test density shift.**
4. **Cross-encoders on serialized pairs were in every top Foursquare and MWPD solution.**
   - Models: xlm-roberta-base/large and mdeberta-v3-base, all MIT.
   - Input format: `name1 [SEP] name2 [COL] addr1 [SEP] addr2 ...`, Ditto-style.
   - Numeric features (distance, top-k stats) were concatenated to the [CLS] vector (Foursquare 2nd, 13th).
   - Extras: pair-flip TTA, FGM + EMA training, and multitask learning with country as an auxiliary label (+0.003 for Foursquare 13th).
   - In MWPD2020 all system-paper teams fine-tuned transformers. PMap won with F1 86.05, against 72.73 for the DeepMatcher baseline.
5. **Graph post-processing is where the leak-free winners pulled ahead.**
   - Foursquare 13th: GNN node classification on 2-hop subgraphs lifted the public score from 0.924 to 0.946, and that team ranked #2 on the non-overlapping re-evaluation.
   - Foursquare 1st: merge two ids' match sets when they share >50% of ids, then re-score only the new pairs with the cross-encoder.
   - Foursquare 2nd: union-find after blocking.
   - Foursquare 4th: a **group-size-dependent threshold**.
   - Shopee 1st: iterative neighbourhood blending, i.e. query expansion over embeddings.
   - Shopee 2nd: keep only mutual edges, and recursively remove the highest-betweenness edge.
   - Shopee 3rd: agglomerative clustering until a target average cluster size is reached.
6. **Leakage and CV design decided the Foursquare leaderboard.**
   - A Kaggle dataset-version mix-up made train and test overlap. The leak-free re-run reshuffled the top 5: re:waiwai went from #1 to #4, and "Ri" from #10 to #1.
   - Quora's top scores were largely driven by graph "structural" features that exploited the sampling process.
   - Lessons: (a) audit test↔train record overlap before trusting any CV/LB gap; (b) our synthetic generator may have artifacts, so check every transductive or "structural" feature for stability across train, test and France; (c) build validation with a test-like candidate pool. Psi (Foursquare 3rd) points out that retrieval scores depend heavily on how many unknown candidates are in the pool.
7. **Transliterate first, then use a small model.** Foursquare 8th found that an English MiniLM on unidecode/pykakasi-normalized text beat multilingual encoders for fine-tuned embedding kNN, given limited data and GPU. Foursquare 4th found that translating foreign-language names was no better than unidecode. For our Indic cross-script names, a transliterate-then-Latin-model branch should be tried against (not instead of) a multilingual branch.
8. **Compliance traps found in this survey:**
   - `Qwen2-VL-7B-Instruct` (Apache-2.0), used by the claimed Amazon-ML-2024 winner, has **8,291,375,616 parameters** per its HF safetensors metadata, above the 8B cap. Always count parameters; do not trust the "7B" in a name.
   - DNB `name_matching` is MIT-labelled but vendors **GPL-3.0 Abydos** code in `distances/`.
   - libpostal and deepparse weights are trained on OpenStreetMap/OpenAddresses-derived data.
   - Zingg is AGPL-3.0.
   - DeepMatcher's default fastText vectors are CC BY-SA 3.0.

---

## 1. How the competition metrics compare to ours

| Competition | Unit scored | Metric (from Evaluation page) | Self in list? | Analogy to ours |
|---|---|---|---|---|
| **Ours (Amazon ML 2026 BER)** | each S1 entity | F0.5 per S1, macro-mean. Singleton = 1 iff empty prediction | no self | — |
| Foursquare Location Matching (2022) | each place id | mean IoU (Jaccard) per row | yes, self always included | Per-row set metric. Recall- and precision-symmetric (IoU), while ours weights precision 2x. |
| Shopee Price Match Guarantee (2021) | each posting | mean F1 per row; group sizes capped at 50 | yes, self included | Per-row F1 (β=1). Ours is β=0.5, so precision matters more. |
| Quora Question Pairs (2017) | each pair | Log Loss | n/a | Pairwise only. Lessons are about leakage and prior shift, not set decisions. |
| MWPD2020 Task 1 (WDC) | each pair | P/R/F1 on the positive class | n/a | Pairwise matcher quality only. |

Consequences:
- Foursquare and Shopee include the query itself in its own match list, so a "predict nothing" row still gets partial credit. **Ours gives no such credit.** A wrong match on a singleton S1 costs the full 1.0, which makes singleton detection (a per-S1 "abstain" classifier) more valuable for us than it was for them.
- With β=0.5, any tactic they used to *add* recall (min2, graph expansion, low re-scoring thresholds such as Foursquare 1st's 0.02) must be re-tuned toward precision. Keep the mechanism and move the thresholds up.

---

## 2. Kaggle Foursquare - Location Matching (2022)

Competition facts (Kaggle API):
- id 35476, 1,079 teams, deadline 2022-07-07.
- Metric: per-row mean IoU; places self-match.
- Records have name, address, city, state, zip, url, phone, categories and lat/lon, in many languages and countries.

**Leak and re-ranking.** Kaggle's post-mortem (discussion 338035, "Competition is Finalized - Investigation Update") says a platform error mixed dataset versions, so train and test rows overlapped. Kaggle re-ran submissions on the intended non-overlapping data:

| Team | New rank (no overlap) | Original rank | NoOverlap private | Overlap private |
|---|---|---|---|---|
| Ri | 1 | 10 | 0.932782 | 0.948152 |
| team merge master | 2 | 13 | 0.922937 | 0.946495 |
| Ethan & qyxs & hyd | 3 | 14 | 0.922847 | 0.945548 |
| re:waiwai | 4 | 1 | 0.920594 | 0.977683 |
| Psi | 5 | 3 | 0.919698 | 0.967847 |

The #13 team write-up ("13th place solution (GNN)", 213tubo) is the leak-free #2, so it is the most trustworthy recipe. "Ri" had no write-up in the forum list I retrieved (UNVERIFIED whether one exists elsewhere). The Foursquare developer blog ("Finding the right POI match") profiles winners in yet another order. I treat it as secondary.

### 2.1 re:waiwai (original #1; leak-free #4): discussion 336055

Four stages:
1. **Candidates.** For every id, 100 nearest by lat/lon distance plus 100 nearest by cosine of `bert-base-multilingual-uncased` name embeddings (kNN with cuML). A light LightGBM with only a few cheap features (Jaro on name and category) keeps the top 20 from each list, about 40 in total. Inference uses RAPIDS Forest Inference (FIL). Max IoU 0.979.
2. **Rich GBDT.** About 120 features:
   - Levenshtein and Jaro-Winkler;
   - **per-id statistics of each similarity (max, min, mean) plus ratios to them**;
   - Euclidean distance;
   - SVD of name embeddings.

   LightGBM reaches CV 0.875. A threshold of 0.01 cuts the candidates to about 10%.
3. **Models.** CatBoost (CV 0.878), xlm-roberta-large, and mdeberta-v3-base (FGM + EMA, about 90 extra numeric features, CV 0.907). The text input is name + categories + address + city + state. Blend weights: LGB 0.01, CatBoost 0.32, XLM-R 0.29, mDeBERTa 0.38. Blend CV 0.911.
4. **Post-processing.** If two ids' match sets overlap by more than 50% from either side, merge them. Newly created pairs are re-scored by xlm-roberta-large with threshold 0.02. CV 0.9166.

After that came the leak exploitation. They joined train and test on (name, lat, lon), which took the LB from 0.900 to 0.971. **Not transferable, and a warning sign.**

**Transfer to us:**
- Keep a two-tier GBDT. A few-feature pre-filter should cut our ~43-51 candidates per S1 before the expensive features and the cross-encoder, and we should measure the oracle drop each time.
- Add per-S1 statistics of *every* similarity: max, min, mean, the ratio to the S1 max, and the gap to the 2nd best. We have some of these (c_combo_tmargin); extend them to all fields.
- The "merge overlapping match sets, then re-score only the new pairs" step is a cheap recall repair. For us it only applies across targets (S1s never merge), so it becomes: add the within-source near-duplicates of accepted targets as new candidates and re-score them.

### 2.2 2nd place team (T0m, yabea, tkm2261, colum2131, rii): discussions 336062 / 336072 / 336090

- **Candidates.** 128 per id from lat/lon kNN plus name-TF-IDF kNN, **mixed with country-specific optimized ratios**. Max IoU 0.9895.
- **Transformer blocking.** A small network on absolute features (lat/lon, country and category embeddings, xlm-roberta-base name embedding, TF-IDF-SVD) and relative features (edit distances, TF-IDF cosine). Single-model LB 0.918. Threshold 0.005, then **union-find**, leaves an average of 4.1 candidates per id at Max IoU 0.986.
- **GBDT features.** Edit distances on each field: gesh, Levenshtein, Jaro, and the full **RapidFuzz suite** (ratio, partial_ratio, token_set_ratio, token_sort_ratio, partial_token_ratio, WRatio, QRatio). Also set similarities (Jaccard, Dice, Simpson) and TF-IDF cosine at word 1-gram and char 1-3-gram level on name, categories, address and all fields. The transformer probability is also a feature.
- **Cross-encoder.** xlm-roberta-base. The text is country + name + categories for both sides. log(haversine) is concatenated to [CLS] and fed to an MLP. Dropout is 0.
- **CV caveat.** They deliberately switched from GroupKFold by POI to GroupKFold by src_id *so that* the leak would show up in CV. Do the opposite: group folds by entity (S1 id), as our `VALIDATION_STRATEGY.md` already does.
- Score: LB 0.949 without the leak.

**Transfer:**
- The country-specific retriever mix is a good idea, but France is unseen, so the mix must be chosen per country **from unlabeled statistics** (e.g. share of non-Latin script, address completeness) or kept global.
- The RapidFuzz suite on every field is cheap on CPU with `rapidfuzz.process.cdist`/`cpdist`, which is multi-threaded.

### 2.3 Psi (Philipp Singer; original #3, leak-free #5): discussion 338112

- **Stage 1: ArcFace metric learning.** xlm-roberta-large is trained to classify the POI of each record, with all columns joined by separator tokens. Lat/lon are written into the string split every 3 digits so the tokenizer handles them. He reported ArcFace beat triplet-style losses once training was stabilized. All-pairs similarity plus a threshold gives the candidates.
- **Stage 2.** A second transformer reads both records' fields interleaved, trained on the in-sample stage-1 candidates, starting from all true positives and adding more false positives over time.
- **CV insight.** Retrieval quality depends on the *size of the candidate pool*. His valid setup held out about 600k records with unique POIs and kept the full pool. Random 100k subsets scored almost the same as the full 600k as long as all candidates stayed in the pool. He later tuned on public LB, which the leak biased.

**Transfer:**
- (a) Keep validation pools at test density. The P-dense variant already does this. Extend it: for every validation fold, include *all* S2/S3 targets of the split, not just those of the held-out S1s.
- (b) ArcFace over S1 ids means 2.2M classes. A full softmax head is expensive. Use in-batch contrastive losses, or partial-FC / sub-center variants.

### 2.4 13th place "GNN" (213tubo; leak-free #2): discussion 336124

This is the most relevant write-up for our residual errors.
- **Candidates.** 30 by TF-IDF cosine over "name, categories, address, city, state" as one sentence, plus 30 by haversine distance. 60 per id.
- **Filter.** LightGBM (missing flags, Jaro and Levenshtein on 8 fields, haversine, TF-IDF cosine, mean of the top-k haversine and TF-IDF for k = 5, 10, 15) cuts to 3.7M pairs. Max IoU 0.971. Inference with FIL.
- **Matcher.** Ditto-like xlm-roberta-base and mdeberta-v3-base on field-interleaved text, with numeric features. Training uses multitask (match + country, +0.003), nlpaug word swap/delete/split augmentation, and id-flip augmentation plus flip TTA. CV 0.920.
- **GNN post-processing.**
  - For each query id, build its **2-hop subgraph** over candidate edges (Max IoU rises to 0.993).
  - Node features: a "marker" flag for the query node plus missing-value flags.
  - Edge features: the mdeberta, xlm-r and lgbm predictions plus Jaro on several fields.
  - Model: 4 PNAConv layers (mean/min/max/std aggregators). Task: node classification ("does this node match the query?").
  - **Loss = per-graph soft IoU loss + 0.1·BCE.**
  - Public LB went 0.924 → 0.946. The same pipeline with LGBM stacking instead of the GNN scored 0.924.

**Transfer (P1/P2 for us):**
- A per-S1 subgraph GNN is the natural learned version of our `stage_collective.py` features. Its nodes are S1 plus its candidates plus the candidates' T-T neighbours. Its edges are S1-T probabilities and T-T similarities. Train it with a **soft per-S1 F0.5 loss**: 1 − (1.25·ΣpY)/(1.25·ΣpY + 0.25·Σ(1−p)Y + Σp(1−Y)), averaged over graphs.
- This targets sibling lookalikes directly. A sibling distractor forms its own tight copy group (high internal T-T similarity) with a slightly different house number. Message passing can learn that "group B is internally consistent but disagrees with group A on the number", which an independent pairwise model cannot see.
- Library: PyTorch Geometric (MIT). It runs on a T4.

### 2.5 4th place (Vincent Schuler, Youri Matiounine, Theo Viel): discussion 335810

- **Cleaning.** Special characters removed, **unidecode for non-Latin text**, lower-case. Groups of similar categories were built by hand and the typical match distance was computed per group.
- **Candidates.** Many cheap generators in union: near neighbours, name similarity, shared name words, category-group equality, phone equality, same address, and TF-IDF (they note it mattered **especially for airports and Indian places**).
- **Models.** A 5-fold LGBM on a few features prunes at threshold 0.007. A second LGBM with more than 200 features and 20 folds, trained off-Kaggle on the full 1.1M-row train set because of memory limits.
- **Post-processing.** The key idea was **thresholds that depend on the sizes of the groups being merged**.
- Scores: 0.939 without the leak, 0.957 with it.
- What did not help: TF-IDF overall (<3% of true matches added, and they were hard FPs), translation compared with unidecode, an offline reverse geocoder, and stacking XGB/CatBoost.

**Transfer:**
- Group-size-aware thresholds map to our copy-group-aware decision (graph.md §7.4). Accepting a whole copy group of size g costs g FPs when it is wrong, so the threshold should rise with g.
- The "unidecode beat translation" result supports a transliteration-first normalizer.

### 2.6 8th place "SBERT + LightGBM" (jsaikawa et al.): discussion 335928

- Candidates come from distance kNN (k=10) plus a fine-tuned SentenceBERT embedding kNN (k=20). The simulated ideal IoU is 0.986. The embedding kNN was also the most useful feature for the classifier.
- They deliberately used an **English model** (`sentence-transformers/all-MiniLM-L12-v2`, Apache-2.0, 33.4M params) on text normalized with pykakasi (Japanese) and unidecode (everything else). It beat multilingual models, which they attribute to too little data (about 1.1M records) and small batches on a single Colab GPU. Contrastive loss beat triplet and cosine-similarity losses, and they fine-tuned in two steps.

**Transfer:**
- For cross-script Indic names, run a *transliterated-Latin* embedding branch (indic_transliteration / anyascii / uroman, then a small Latin encoder fine-tuned on our train pairs). Compare it to a multilingual branch.
- With 2.2M S1 entities and multiple copies each, we have more metric-learning data than they did. The multilingual branch may win for us, so measure it.

### 2.7 Other Foursquare posts retrieved (titles/ids only, not used)

7th (335800), 9th (336415), 11th (335924), 12th (336051), 15th (335818), 21st (335857), 23rd arcmargin (335883), 6th (348399), and a community summary "Tricks used by the winning solutions" (336140).

---

## 3. Kaggle Shopee - Price Match Guarantee (2021)

Competition facts (Kaggle API):
- id 24286, 2,426 teams, deadline 2021-05-10.
- Evaluation: F1 computed **per row and averaged**. Posts always self-match. Group sizes capped at 50.
- Data: product image + title (Indonesian/English mix). The task is to find all postings of the same product.

### 3.1 1st place "From Embeddings to Matches" (Limerobot, YoonSoo): discussion 238136

- **Encoders.** Image: two eca_nfnet_l1. Text: xlm-roberta-large/base, two Indonesian BERTs, and bert-base-multilingual-uncased. All trained with **ArcFace**. The tuning tricks that mattered: margin warm-up (0.2 → 0.8-1.0), long warm-up, a higher learning rate for the cosine head, gradient clipping, and BatchNorm before L2-normalization. A class-size-adaptive margin (from the GLR2020 3rd-place report, arXiv 2010.05350) gave only a small gain.
- **Main lesson:** most of the gain came from *how matches are searched from embeddings*, not from better encoders. LB progression:
  - baseline 0.700;
  - concat embeddings 0.724;
  - **min2** 0.743;
  - normalize-then-concat 0.753;
  - **union of combined, image-only and text-only matches** 0.776;
  - INB + diverse text models 0.784;
  - use all three embedding types at INB stage 1 + joint threshold tuning 0.793.
- **min2.** Every post has at least 2 matches (itself plus one), so the 2nd-nearest neighbour is always kept unless it is beyond a high "min2" threshold.
- **INB (Iterative Neighbourhood Blending).** FAISS kNN (k=51), threshold, then replace each embedding with the similarity-weighted sum of its neighbourhood (query expansion + database-side augmentation), re-normalize and repeat. It is applied in 3 stages. In the end only 2 thresholds (stage 2 and stage 3) needed LB tuning.

**Transfer:**
- (a) **Union of matches from separate views** (name-only, address-only, combined), each with its own high-confidence threshold. That accepts pairs that one view strongly supports or that both views moderately support. We already union views in blocking. The idea here is to union *decisions*, with view-specific precision-safe thresholds.
- (b) **Neighbourhood blending / query expansion for recall of hard copies.** Blend an S1's embedding with its confidently matched targets' embeddings, then re-query. A cross-script or heavily corrupted copy that is far from S1 but close to other copies of the same entity becomes reachable. This is also a candidate fix for empty-address FNs, where the copy's name matches other accepted copies.
- (c) The min2 analogue for us: if the model is confident an S1 is *not* a singleton, lower the bar for its 2nd-best target. Otherwise stay strict.

### 3.2 2nd place "matching prediction by GAT & LGB" (lyakaap, tkm2261): discussion 238022

- **Stage 1.** Metric-learned similarities: image NFNet-F0/ViT with CurricularFace + SAM optimizer, and text Indonesian BERT / mBERT / paraphrase-XLM plus TF-IDF, plus a multimodal model.
- **Stage 2: meta-models.** LightGBM and a GAT that classify whether a pair is in the same group.
- **Graph features.** Avg and std of the top-K cosine similarities per item (K = 5, 10, 15, 30), **normalized to mean 0, std 1 to handle the train/test distribution difference**, plus PageRank.
- **GAT.** Uses only 4 similarity features and attends over the edges adjacent to the target edge.
- **Post-processing.**
  - Recursively remove the edge with the highest betweenness centrality, since true groups should be cliques and a bridging edge is suspicious. Gain about +0.001.
  - **Drop pairs without a mutual edge** (A→B kept only if B→A exists).
- **Training.** Sample weight 1/(group size)^0.4, which matters for the per-row F1.
- **Ensembling.** Subtract each model's own optimal threshold from its prediction, sum the results, and predict a match if the sum is > 0.
- **Engineering.** cuDF, cuPy, cuGraph and FIL. FIL cut GBDT inference from 40 min on CPU to 2 min on GPU. Each notebook cell was isolated as a subprocess to free memory.

**Transfer:**
- **Per-split z-normalization of density features** (top-K similarity stats) is a direct, cheap fix for our train→test density shift, where the threshold drifts from 0.7 to 0.8. Compute the stats per split, and per country within a split, so France gets its own normalization.
- Macro-aware sample weights: weight each training pair by 1/(n_true(S1))^α, α≈0.4 as a starting point, so every S1 contributes equally, as in the macro metric.
- Threshold-subtracted ensembling is a clean way to combine the stage-1/2 GBDT and a future cross-encoder when each has a different optimal threshold.

### 3.3 3rd place "Triplet loss, Boosting, Clustering" (Btbpanda): discussion 238515

- **Candidates.** Fixed-radius neighbours from many frozen embeddings: EfficientNet, Indonesian BERT, LaBSE, several TF-IDFs, CLIP. The candidates are unioned into a pair sample (about 3M pairs, 4% positive).
- **CatBoost features** (about 500): pairwise distances per embedding, **density around both points at several radii**, and ranks. This alone gives 0.76+ LB.
- **Fine-tuning.** Triplet-loss fine-tunes; ArcFace did worse for him. Validation searched candidates in the *full* train set, not only the validation fold, to reduce the CV/LB gap. Same lesson as Psi.
- **Clustering.** Agglomerative clustering over GBM probabilities until the **average cluster size equals a target** (2.8 on train, 2.6 best on LB). Leftover singletons are merged into their nearest cluster.

**Transfer:**
- Density-at-radius features: count of targets within similarity r of the S1, for several r.
- A "target average group size" is not directly usable, because our per-source caps (≤5 S2, ≤6 S3) are already known. The caps can act as an over-merge veto.

---

## 4. Kaggle Quora Question Pairs (2017): leakage lessons

Competition facts: id 6277, 3,295 teams, deadline 2017-06-06, metric Log Loss.

- **1st place (discussion 34355; Lam Dang, Guillaume Huard, Maximilien Baudry, Paul Todorov, Sebastien Conort).** Three feature families:
  - embedding features (Word2Vec, Doc2Vec/Sent2Vec, an ESIM encoding);
  - classical text features (LSI/LDA, TF-IDF char n-grams 1-8, edit distances, lengths);
  - **structural graph features** from the question co-occurrence graph over train+test: neighbour counts, intersections, higher-order neighbours, connected-component statistics, and graphs weighted by model predictions.

  Models: Siamese/attention networks (decomposable attention, ESIM), then GBDT/MLP stacking. They also **rescaled predictions for the train/test prior shift, separately per "perimeter"** (question-frequency strata).
- **Community review (discussion 34560).** A participant argues that most of the top scores came from these structural features, which exploit how Quora *sampled* the pairs. Even TF-IDF and embedding features partly capture sampling artifacts.

**Transfer:**
- (a) Our data is synthetic. Transductive "structural" features (copy-group sizes, how many S1s a target is near, name frequency) are legitimate because they are computed label-free from the provided data. They may still encode *generator* artifacts. Validate them on leave-one-country-out and check their distribution on the test split and on France before relying on them.
- (b) Prior-shift correction per stratum: recalibrate p on test per density stratum (e.g. by candidate count within a margin) instead of using one global threshold.

---

## 5. WDC / MWPD2020 product matching challenge

MWPD2020 overview (Zhang, Bizer, Peeters, Primpeli; CEUR-WS Vol-2720, paper1):
- Task 1: match product offers from different websites. Test set: 1,500 offer pairs in Computers & Accessories. Training came from the WDC product corpus (26M offers grouped into 16M clusters, per the overview text).
- Results (P / R / F1):

| Team | P | R | F1 |
|---|---|---|---|
| PMap | 82.04 | 90.48 | 86.05 |
| Rhinobird (Round 2) | 80.63 | 92.00 | 85.94 |
| ISCAS-ICIP (Round 2) | 85.77 | 84.95 | 85.36 |
| ASVinSpace | 86.20 | 82.10 | 84.10 |
| Baseline (DeepMatcher) | 70.89 | 74.67 | 72.73 |

- All teams with system papers fine-tuned transformers (BERT-large, RoBERTa-large/base, DistilRoBERTa, or Ditto/HierMatcher inside an ensemble).
- PMap used *only the title* and no extra data.
- Rhinobird and ISCAS-ICIP added **heuristic correction rules** (e.g. force non-match when brands or categories differ). Rhinobird also used SWA self-ensembling and focal loss.
- The top teams were within 1% F1 of each other. No team won every challenge type (seen vs new products, typos, dropped words).

Related WDC work (verified on arXiv/Crossref; details in §9):
- **R-SupCon** (Peeters & Bizer, WWW'22 Companion): supervised contrastive pre-training plus *source-aware sampling* reached F1 94.29 on Abt-Buy and 79.28 on Amazon-Google. Self-supervised contrastive pre-training *hurt* because of label noise.
- **WDC Products** (EDBT 2024): all matchers struggle with **unseen entities**, and contrastive learning is more data-efficient than cross-encoders.
- **Cross-Language Learning for Entity Matching** (WWW'22 Companion): adding English pairs to a small German training set improves transformer matchers, most of all in low-resource settings.

**Transfer:**
- Source-aware sampling matters for us. S2 and S3 have different noise profiles, and within-source copies share a corrupted base. Build contrastive batches so that positives come from *different* sources where possible.
- For France (unseen), the cross-language result says train/val pairs from US+India should transfer if normalization is shared. Keep the model language-agnostic, and verify with leave-one-country-out.
- Heuristic veto rules (brand ↔ house number/postcode) help precision. For us they should be *features*, because the generator also shifts house numbers of TRUE copies (PROJECT_STATE §4).

---

## 6. Past Amazon ML Challenge editions (different tasks; meta-lessons only)

All from participants' GitHub READMEs. The claims are **self-reported and not verified by Amazon**.

- **2023, product length prediction.** `pj-mathematician/Amazon-ML-Challenge-2023` ("winning solution notebook", team ART in Artificial Intelligence; no license).
  - Several sentence-transformers MiniLM/mpnet encoders embed title+bullets+description.
  - kNN top-10 (sentence-transformers `semantic_search`) and hnswlib ANN top-100 from a distilled multilingual model provide the neighbours' target values as features.
  - LightGBM (extra_trees, custom objective) sits on top.

  Lesson: *retrieval + GBDT on neighbour features, with a metric-aligned objective*. That is the same shape as our pipeline.
- **2024, entity-value extraction from images.** `KhadgaA/Amazon-ML-Challenge` ("winning code", Team NeuralNinjas; no license) fine-tuned **Qwen2-VL-7B-Instruct** with LLaMA-Factory. The README reports 0.679 after a 20k-sample fine-tune and **0.865 after a further fine-tune on 1,600 curated samples**.

  Lesson: a small, clean, curated fine-tuning set beat more noisy data. **Compliance note:** Qwen2-VL-7B-Instruct is Apache-2.0, but its HF safetensors total is 8,291,375,616 parameters, which is over 8B under our rule.
- **2025, Smart Product Pricing (SMAPE).** `NeelDevenShah/Amazon-ML-Challenge-2025` (rank 80 of about 23,000 teams, SMAPE 43.28; it quotes a winning score of 39.7, UNVERIFIED). The README restates the 2025 rules: **MIT/Apache-2.0 models ≤8B parameters, and no external price lookup**, which match our 2026 rules. So this rule set has been enforced before, and documentation of the approach counted in the final ranking.

**Transfer:**
- Keep a written, reproducible pipeline and document every pretrained weight with license and parameter count. Final rankings in 2025 considered documentation.
- Do not rely on "7B"-named models without counting parameters.

---

## 7. Company-name matching write-ups and tools (methodology)

- **ING, "Super Fast String Matching in Python"** (Chris van den Berg, 2017-10-14, bergvca.github.io).
  - Method: TF-IDF over **character 3-grams**, then cosine top-N via sparse matrix multiplication with a threshold (0.8 in the post).
  - Data: 663,000 company names from SEC EDGAR, about 45 minutes on a dual-core laptop.
  - This became `sparse_dot_topn` (ING) and `string_grouper`.
- **ING Entity Matching Model (EMM)** (`ing-bank/EntityMatchingModel`, MIT, Pandas + Spark).
  - Complementary indexers: TF-IDF cosine at word *and* char level (via sparse_dot_topn), plus **sorted-neighbourhood indexing (SNI)**.
  - A supervised classifier on string-based, **rank-based**, and **legal-entity-form** features.
  - An optional **aggregation layer** combines scores when several names belong to one account.

  Transfer: (a) add SNI on a normalized name key as a cheap extra blocker (it catches typos and suffix variations that share a prefix); (b) add a legal-form agreement feature (Pvt Ltd vs LLC vs SARL); (c) rank features in both directions (we have trank).
- **DNB name_matching** (`DeNederlandscheBank/name_matching`, MIT label).
  - Stage 1: TF-IDF char n-gram cosine top_n (default 50).
  - Stage 2: re-score with a configurable set of distance metrics ("bag", "typo", "refined_soundex", …), with optional legal-suffix removal and common-word handling.
  - Bundled data: `legal_names.csv`, `common_words.csv`.
  - **License caution:** the `distances/` folder contains files headed "This file is part of Abydos … GNU General Public License v3" (Abydos is GPL-3.0).
- **string_grouper** (`Bergvca/string_grouper`, MIT). TF-IDF cosine with a Rust sparse matmul backend (`sp_matmul_rs`), originally sparse_dot_topn. `group_similar_strings` deduplicates by grouping.
- **OpenSanctions matcher** (opensanctions.org/matcher):
  - Recommended algorithm `logic-v2`: rule-based with weights, e.g. `name_match` 1.00, country-mismatch penalty −0.20, date mismatch −0.15 to −0.25, and `nm_number_mismatch` (default 0.3) for **numbers that differ between names**.
  - Older `regression-v1` learned weights (e.g. name_match 0.91, country mismatch −0.23).
  - Transfer: add an explicit **name-number mismatch** feature. It is a strong precision cue for sibling entities ("Store 12" vs "Store 21", "Phase 2" vs "Phase 3").
- **nomenklatura** (MIT): an inverted-index blocker (`blocker.Index`), then candidate generation, then a **Resolver graph of positive/negative judgements** with connected components. Negative judgements block transitive merges (A≠B, B=C ⇒ no need to ask A=C).

  Transfer: our S1 is duplicate-free, so any two S1s are a standing negative judgement. Cluster/repair steps must never connect two S1s (graph.md option C).
- **rigour** (`opensanctions/rigour`, MIT, includes the old `fingerprints`).
  - `resources/names/org_types.yml` (3,568 lines) maps legal-form aliases to canonical forms, and covers *SARL*, *société par actions simplifiée* and *private limited company*.
  - The address-format data is derived from OpenCageData.
  - Compliance: hand-curated legal-form knowledge is a grey zone ("hand-written normalization"). Prefer mining abbreviation maps from our own data (already done in `aliases.py`). If rigour/cleanco lists are used as a seed for France, **document it**, and do not use the OpenCage-derived address formats.
- **cleanco** (`psolin/cleanco`, MIT). `termdata.py` holds legal-term lists per country, including France ("sarl") and India ("pvt. ltd.").

---

## 8. GitHub survey: ER / matching stack (GitHub API, 2026-09-25)

Legend. **Stage:** B = blocking/candidates, F = features, M = matching model, C = clustering/post-processing, E = eval/infra. **Offline?** = can run with no network after `pip install`. **Compliance** separates the software license (library) from bundled data and model weights.

| Repo | URL | License (SPDX) | Stars | Last push | Approach | Stage | Offline? | Compliance note |
|---|---|---|---|---|---|---|---|---|
| Splink | github.com/moj-analytical-services/splink | MIT | 2,429 | 2026-09-22 | Fellegi-Sunter with EM-estimated m/u; blocking rules; term-frequency adjustments; SQL backends (DuckDB, Spark, …); clustering with duplicate-free sources | B, F, M, C | yes | Library only; no bundled data. Good unsupervised baseline and feature generator. |
| dedupe | github.com/dedupeio/dedupe | MIT | 4,514 | 2025-07-29 | Active learning; learned blocking predicates; learnable string distances (Bilenko & Mooney lineage); hierarchical clustering | B, M, C | yes | Library only. Interactive labelling is not needed here since we have labels. |
| recordlinkage | github.com/J535D165/recordlinkage | BSD-3-Clause | 1,062 | 2024-02-21 | Indexers (block, sorted neighbourhood), comparators, ECM/LogReg/SVM classifiers | B, F, M | yes | Library only. Slow at 10M scale, pandas-bound. |
| RapidFuzz | github.com/rapidfuzz/RapidFuzz | MIT | 4,139 | 2026-09-12 | C++ edit-distance family: ratio, partial, token_set/sort, WRatio, QRatio, Jaro-Winkler, Levenshtein; multithreaded `cdist`/`cpdist` | F | yes | Library only. This is what the Foursquare 2nd GBDT used. |
| jellyfish | github.com/jamesturk/jellyfish | MIT | 2,233 | 2026-07-24 | Jaro/JW, Damerau, Hamming, Soundex, Metaphone, NYSIIS, Match Rating | F | yes | Library only. Phonetics are English-centric; use them on transliterated text. |
| FAISS | github.com/facebookresearch/faiss | MIT | 40,976 | 2026-09-25 | Dense ANN (IVF, PQ, HNSW, GPU brute force) | B | yes | Library only. Used by Shopee 1st (k=51 inner-product kNN). |
| hnswlib | github.com/nmslib/hnswlib | Apache-2.0 | 5,333 | 2026-09-15 | Header-only HNSW ANN | B | yes | Library only. Used by the Amazon-ML-2023 winner. |
| USearch | github.com/unum-cloud/USearch | Apache-2.0 | 4,319 | 2026-08-31 | HNSW with low memory; fp16/int8 | B | yes | Library only. |
| sparse_dot_topn | github.com/ing-bank/sparse_dot_topn | Apache-2.0 | 424 | 2026-09-14 | Sparse matmul with top-n and threshold, multithreaded | B | yes | Library only. The core of ING/DNB/string_grouper TF-IDF blocking. |
| string_grouper | github.com/Bergvca/string_grouper | MIT | 372 | 2026-07-26 | TF-IDF char n-grams + fast sparse cosine; grouping | B, C | yes | Library only. |
| name_matching (DNB) | github.com/DeNederlandscheBank/name_matching | MIT (repo label) | 169 | 2026-07-02 | TF-IDF top_n, then multi-metric re-scoring; legal suffix handling | B, F | yes | **Vendors GPL-3.0 Abydos code** (`distances/`). Bundles `legal_names.csv`/`common_words.csv`. Borrow ideas only. |
| EntityMatchingModel (ING) | github.com/ing-bank/EntityMatchingModel | MIT | 101 | 2026-05-18 | Word+char TF-IDF and SNI indexers; supervised rank/legal-form features; aggregation | B, F, M | yes | Library only. Has an example noise generator; don't use its data. |
| cleanco | github.com/psolin/cleanco | MIT | 360 | 2026-06-23 | Legal-suffix stripping/classification from per-country term lists | F | yes | **Bundled hand-curated term lists** (grey zone). Document if used. |
| rigour | github.com/opensanctions/rigour | MIT | 67 | 2026-09-22 | Name/org-type normalization (org_types.yml), text utilities | F | yes | Bundled curated lists. Address formats derived from OpenCageData, so avoid that part. |
| nomenklatura | github.com/opensanctions/nomenklatura | MIT | 267 | 2026-09-23 | Inverted-index blocking, xref, resolver graph with negative judgements | B, C | yes | Library only. FtM data model is overkill; ideas only. |
| yente | github.com/opensanctions/yente | MIT | 178 | 2026-09-23 | Matching API over sanctions data (search + scoring) | M | needs its index | Designed around OpenSanctions data, which would be external data. Do not use. |
| libpostal | github.com/openvenues/libpostal | MIT | 4,894 | 2026-05-13 | CRF address parser + expansion dictionaries | F | after a data download of several GB | **Model/data trained on OSM + OpenAddresses** (Senzing variant adds more). This is external-data derived: RISK, avoid. |
| pypostal | github.com/openvenues/pypostal | MIT | 880 | 2025-11-01 | Python bindings to libpostal | F | same | Same risk as libpostal. |
| usaddress | github.com/datamade/usaddress | MIT | 1,637 | 2025-08-07 | CRF parser for US addresses | F | yes | Ships a pretrained CRF (US-only). Low value for India/France. Pretrained-model use would need documenting. |
| deepparse | github.com/GRAAL-Research/deepparse | LGPL-3.0 | 354 | 2026-09-19 | Seq2seq multinational address parser (fastText/BPEmb + LSTM) | F | weights download | **Weights trained on deepparse-address-data, which was "generated using data from libpostal"** (OSM-derived). Weight license not stated in the repo (UNVERIFIED). RISK. |
| pyJedAI | github.com/AI-team-UoA/pyJedAI | Apache-2.0 | 101 | 2026-03-22 | Block building/cleaning, meta-blocking, similarity joins, embeddings NN, clustering | B, C, E | yes | Library only. Good reference implementations of meta-blocking and clustering. |
| py_entitymatching (Magellan) | github.com/anhaidgroup/py_entitymatching | BSD-3-Clause | 195 | 2024-05-29 | Blockers, auto features, ML matchers, debugging | B, F, M | yes | Library only. Stale, pandas-scale. |
| Ditto | github.com/megagonlabs/ditto | Apache-2.0 | 318 | 2024-04-17 | Serialized-pair LM fine-tuning; domain-knowledge injection (`--dk`), TF-IDF summarization, augmentation (`--da`) | M | yes (after weight download) | Code is Apache-2.0. Weights must be picked separately (e.g. xlm-roberta MIT). Default LM distilbert. |
| DeepMatcher | github.com/anhaidgroup/deepmatcher | BSD-3-Clause | 624 | 2024-06-18 | RNN/attention matchers over word embeddings | M | yes | **Default embeddings `fasttext.en.bin`**. fastText vectors are CC BY-SA 3.0, which is NOT compliant. Superseded by transformers anyway. |
| Sudowoodo | github.com/megagonlabs/sudowoodo | BSD-3-Clause | 19 | 2023-05-24 | Contrastive self-supervised representations for blocking + matching | B, M | yes | Code only. Choose compliant base weights. |
| Sparkly | github.com/anhaidgroup/sparkly | BSD-3-Clause | 19 | 2026-08-28 | Lucene top-k TF/IDF (BM25) blocking on Spark; auto attribute/tokenizer selection | B | yes (JVM+Spark) | Library only. Heavy JVM stack; reimplement the idea with bm25s/tantivy/sparse_dot_topn. |
| DeepBlocker | github.com/qcri/DeepBlocker | BSD-3-Clause | 30 | 2023-04-05 | Self-supervised DL blocking (autoencoder, hybrid) with top-k | B | yes | Code only. Uses fastText by default (UNVERIFIED in code), so check the weight license if used. |
| Zingg | github.com/zinggAI/zingg | **AGPL-3.0** | 1,251 | 2026-09-13 | Spark-based active-learning ER / MDM | B, M, C | yes | **AGPL copyleft**; Java/Spark. Avoid. |
| indic_transliteration | github.com/indic-transliteration/indic_transliteration_py | MIT | 212 | 2026-09-08 | Rule/scheme-based Indic ↔ Latin transliteration | F (normalize) | yes | Scheme tables only, no learned weights. OK. |
| anyascii | github.com/anyascii/anyascii | ISC | 424 | 2026-06-06 | Unicode → ASCII table transliteration | F | yes | Tables only. OK. |
| uroman | github.com/isi-nlp/uroman | NOASSERTION (MIT-style text + attribution clause) | 251 | 2024-07-26 | Universal romanizer | F | yes | OK with attribution. The clause requires acknowledging it in publications. |
| IndicXlit (AI4Bharat) | github.com/AI4Bharat/IndicXlit | MIT | 142 | 2023-10-13 | Neural transliteration models for 21 Indic languages | F | weights download | Pretrained weights under MIT per repo; they are *model weights*, so allowed if ≤8B and license verified on the weights' source. |
| polars | github.com/pola-rs/polars | MIT | 39,858 | 2026-09-25 | Rust DataFrame engine | E | yes | Library only. |
| DuckDB | github.com/duckdb/duckdb | MIT | 41,691 | 2026-09-24 | In-process OLAP SQL (Splink backend) | E | yes | Library only. |
| bm25s | github.com/xhluca/bm25s | MIT | 1,792 | 2026-09-18 | Fast BM25 with sparse matrices (numpy/numba) | B | yes | Library only. A Sparkly-style BM25 without the JVM. |
| tantivy | github.com/quickwit-oss/tantivy | MIT | 16,141 | 2026-09-24 | Rust Lucene-like full-text engine | B | yes | Library only. |
| entity-embed | github.com/vintasoftware/entity-embed | MIT | 162 | 2022-11-18 | Char-level embedding for ER blocking | B | yes | Stale. |
| PolyFuzz | github.com/MaartenGr/PolyFuzz | MIT | 803 | 2025-07-10 | Fuzzy matching (TF-IDF, edit distance, embeddings) + grouping | B, F | yes | Library only. |
| cuML (FIL) | github.com/NVIDIA/cuml | Apache-2.0 | 5,290 | 2026-09-24 | GPU kNN and **Forest Inference Library** for GBDT | B, M | yes | Library only. Used by Foursquare 1st/13th and Shopee 2nd for GBDT inference. |
| cuGraph | github.com/rapidsai/cugraph | Apache-2.0 | 2,237 | 2026-09-21 | GPU graph algorithms (CC, PageRank, betweenness) | C | yes | Library only. |
| LightGBM | github.com/lightgbm-org/LightGBM | MIT | 18,812 | 2026-09-25 | GBDT | M | yes | Library. Our own trained model, so it is compliant. |
| PyTorch Geometric | github.com/pyg-team/pytorch_geometric | MIT | 24,097 | 2026-09-01 | GNN layers (PNAConv, GATConv) | C | yes | Library only. |
| sentence-transformers | github.com/huggingface/sentence-transformers | Apache-2.0 | 19,120 | 2026-09-24 | Bi-/cross-encoder training (MNRL, contrastive) | B, M | yes | Library. Weight licenses are separate. |

Solution-code repos seen (all **no license**, so ideas only):
- `TheoViel/kaggle_foursquare` (14★, Foursquare 4th)
- `kiccho1101/kaggle-shopee-6th-place-solution` (27★)
- `pj-mathematician/Amazon-ML-Challenge-2023` (15★)
- `KhadgaA/Amazon-ML-Challenge` (26★)
- `arnav10goel/Amazon-ML-Challenge-24` (2★, rank 6)
- `NeelDevenShah/Amazon-ML-Challenge-2025` (8★)
- `aishik-rakshit/Amazon-ML-Challenge-2023-4th-Place` (3★)

---

## 9. Papers (verified)

| # | Paper | Authors | Venue / year | Verified via | Relevance to us |
|---|---|---|---|---|---|
| P1 | Deep Entity Matching with Pre-Trained Language Models (Ditto) | Y. Li, J. Li, Y. Suhara, A. Doan, W.-C. Tan | PVLDB 14(1), 2020 (DOI 10.14778/3421424.3421431) | arxiv.org/abs/2004.00584 | Serialized-pair cross-encoder. Abstract: up to 29% F1 over prior SOTA; **company-matching task with 789K and 412K records reaches F1 96.5%**. It is the template of the Foursquare 13th matcher. |
| P2 | Deep Learning for Entity Matching: A Design Space Exploration (DeepMatcher) | S. Mudgal, H. Li, T. Rekatsinas, A. Doan, Y. Park, G. Krishnan, R. Deep, et al. (first 7 per Crossref) | SIGMOD 2018, pp. 19-34 | Crossref DOI 10.1145/3183713.3196926 | MWPD baseline (F1 72.73). Default fastText weights are non-compliant. |
| P3 | Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation | R. Wang, Y. Li, J. Wang | ICDE 2023, pp. 1502-1515 | arxiv.org/abs/2207.04122 + Crossref 10.1109/ICDE55515.2023.00391 | Contrastive representations used for both blocking and matching. |
| P4 | Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching | D. Paulsen, Y. Govind, A. Doan | PVLDB 16(6):1507-1519, 2023 | vldb.org/pvldb/vol16/p1507-paulsen.pdf + Crossref 10.14778/3583140.3583163 | Top-k TF/IDF blocking beats 8 SOTA blockers. Sparkly Auto blocks 10M-tuple tables in under 100 min on 10 AWS nodes ($12.5), and 26M tuples in under 130 min on 30 nodes ($67.5). |
| P5 | Deep Learning for Blocking in Entity Matching: A Design Space Exploration (DeepBlocker) | S. Thirumuruganathan, H. Li, N. Tang, M. Ouzzani, Y. Govind, D. Paulsen, G. Fung, et al. (first 7 per Crossref) | PVLDB 14(11):2459-2472, 2021 | vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf + Crossref 10.14778/3476249.3476294 | DL blockers win on dirty/textual data. **DL ∪ non-DL blocking is best**, which backs the union-of-retrievers pattern. |
| P6 | Magellan: Toward Building Entity Matching Management Systems | P. Konda, S. Das, P. Suganthan G.C., A. Doan, A. Ardalan, J. Ballard, H. Li, et al. | PVLDB 9(12):1197-1208, 2016 | Crossref 10.14778/2994509.2994535 | Basis of py_entitymatching. The EM workflow (blocking debugging, feature auto-gen) is still the reference process. |
| P7 | Self-configured Entity Resolution with pyJedAI | V. Efthymiou, E. Ioannou, M. Karvounis, M. Koubarakis, J. Maciejewski, K. Nikoletos, G. Papadakis | IEEE BigData 2023, pp. 339-343 | Crossref 10.1109/BigData59044.2023.10386556 | pyJedAI's auto-configured pipelines. Reference implementations for meta-blocking and clustering. |
| P8 | Splink: Free software for probabilistic record linkage at scale | R. Linacre, S. Lindsay, T. Manassis, Z. Slade, T. Hepworth | Int. J. Population Data Science 7(3), 2022 | ijpds.org/index.php/ijpds/article/view/1794 (DOI 10.23889/ijpds.v7i3.1794) | EM-estimated Fellegi-Sunter. Term-frequency adjustments (rare-value agreement weighs more). |
| P9 | A Theory for Record Linkage | I. P. Fellegi, A. B. Sunter | JASA 64(328):1183-1210, 1969 | Crossref 10.1080/01621459.1969.10501049 | Foundational m/u likelihood-ratio model behind Splink/recordlinkage. |
| P10 | Adaptive duplicate detection using learnable string similarity measures | M. Bilenko, R. J. Mooney | KDD 2003, pp. 39-48 | Crossref 10.1145/956750.956759 | Learnable string distances; dedupe's lineage. |
| P11 | Blocking and Filtering Techniques for Entity Resolution: A Survey | G. Papadakis, D. Skoutas, E. Thanos, T. Palpanas | ACM Computing Surveys 53(2):1-42 (Crossref print year 2021) | Crossref 10.1145/3377455 | Taxonomy used to place our blockers (see blocking.md). |
| P12 | The WDC Training Dataset and Gold Standard for Large-Scale Product Matching | A. Primpeli, R. Peeters, C. Bizer | WWW'19 Companion, pp. 381-386 | Crossref 10.1145/3308560.3316609 | The WDC LSPM corpus behind MWPD2020. |
| P13 | MWPD2020: Semantic Web Challenge on Mining the Web of HTML-embedded Product Data | Z. Zhang, C. Bizer, R. Peeters, A. Primpeli | CEUR-WS Vol-2720 (ISWC 2020 challenge), 2020 | ceur-ws.org/Vol-2720/ and paper1.pdf (text read) | Challenge overview. PMap F1 86.05 vs DeepMatcher 72.73. Transformer ensembles + heuristic post-rules. |
| P14 | Supervised Contrastive Learning for Product Matching (R-SupCon) | R. Peeters, C. Bizer | WWW'22 Companion, pp. 248-251 | arxiv.org/abs/2202.02098 + Crossref 10.1145/3487553.3524254 | Source-aware sampling. Abt-Buy F1 94.29 (+3.24), Amazon-Google 79.28 (+3.7). Self-supervised pre-training hurt. |
| P15 | Cross-Language Learning for Entity Matching (Crossref title: "…for Product Matching") | R. Peeters, C. Bizer | WWW'22 Companion, pp. 236-238 | arxiv.org/abs/2110.03338 + Crossref 10.1145/3487553.3524234 | English pairs improve German matchers, especially in low-resource settings. Supports transfer to unseen France. |
| P16 | WDC Products: A Multi-Dimensional Entity Matching Benchmark | R. Peeters, R. C. Der, C. Bizer | EDBT 2024 | arxiv.org/abs/2301.09521 (comment: accepted at EDBT 2024) | All matchers struggle on unseen entities. Contrastive is more data-efficient than cross-encoders. |
| P17 | Billion-scale similarity search with GPUs | J. Johnson, M. Douze, H. Jégou | IEEE Trans. Big Data 7(3):535-547, 2021 | arxiv.org/abs/1702.08734 + Crossref 10.1109/TBDATA.2019.2921572 | FAISS GPU kNN (Shopee 1st INB). |
| P18 | The Faiss library | M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, et al. | arXiv 2024 | arxiv.org/abs/2401.08281 | Index selection reference. |
| P19 | Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs | Yu. A. Malkov, D. A. Yashunin | IEEE TPAMI 42(4):824-836, 2020 | arxiv.org/abs/1603.09320 + Crossref 10.1109/TPAMI.2018.2889473 | hnswlib/USearch ANN (Amazon-ML-2023 winner). |
| P20 | ArcFace: Additive Angular Margin Loss for Deep Face Recognition | J. Deng, J. Guo, J. Yang, N. Xue, I. Kotsia, S. Zafeiriou | TPAMI version (DOI 10.1109/TPAMI.2021.3087709 per arXiv); conference origin not re-verified here | arxiv.org/abs/1801.07698 | Metric-learning loss used by Shopee 1st and Foursquare 3rd (Psi). |
| P21 | Principal Neighbourhood Aggregation for Graph Nets (PNA) | G. Corso, L. Cavalleri, D. Beaini, P. Liò, P. Veličković | NeurIPS 2020 | arxiv.org/abs/2004.05718 | The GNN layer used by Foursquare 13th (leak-free #2). |
| P22 | Graph Attention Networks | P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio | ICLR 2018 | arxiv.org/abs/1710.10903 | Shopee 2nd edge-level GAT meta-model. |
| P23 | Fine-tuning CNN Image Retrieval with No Human Annotation (α-QE) | F. Radenović, G. Tolias, O. Chum | TPAMI 2018 (arXiv comment) | arxiv.org/abs/1711.02512 | Query-expansion family behind Shopee INB. Transfers to neighbourhood blending of S1 + accepted-target embeddings. |
| P24 | Re-ranking Person Re-identification with k-reciprocal Encoding | Z. Zhong, L. Zheng, D. Cao, S. Li | CVPR 2017 (arXiv comment) | arxiv.org/abs/1701.08398 | k-reciprocal (mutual-neighbour) re-ranking used by Shopee silver solutions. Same idea as our reverse-rank / mutual-edge filter. |
| P25 | Google Landmark Recognition 2020 Competition Third Place Solution | Q. Ha, B. Liu, F. Liu, P. Liao | arXiv tech report 2020 | arxiv.org/abs/2010.05350 | Class-size-adaptive ArcFace margin (cited by Shopee 1st). |
| P26 | Leveraging Subword Embeddings for Multinational Address Parsing (deepparse) | M. Yassine, D. Beauchemin, F. Laviolette, L. Lamontagne | IEEE CiSt 2020 (DOI 10.1109/CiSt49399.2021.9357170) | arxiv.org/abs/2006.16152 | About 99% parsing accuracy on training countries; zero-shot good on 33/41 countries. **Weights are libpostal/OSM-derived, so compliance RISK.** |
| P27 | Multinational Address Parsing: A Zero-Shot Evaluation | M. Yassine, D. Beauchemin, F. Laviolette, L. Lamontagne | iJIST (accepted, per arXiv comment), 2021/22 | arxiv.org/abs/2112.04008 | Zero-shot transfer of address parsers to unseen countries. Methodological analogue for France; do not use the weights. |
| P28 | Open benchmark for filtering techniques in entity resolution | F. Neuhof, M. Fisichella, G. Papadakis, K. Nikoletos, N. Augsten, W. Nejdl, M. Koubarakis | The VLDB Journal 33(5):1671-1696, 2024 | Crossref 10.1007/s00778-024-00868-7 (abstract not read) | Benchmark of filtering/blocking methods. Pointer only; no numbers used. |

Competition write-ups used as primary sources (Kaggle discussion pages, read in full on 2026-09-25):

| # | Write-up | Authors (Kaggle) | Date | URL |
|---|---|---|---|---|
| K1 | Foursquare 1st place solution | charmq, Takoi, pao (re:waiwai) | 2022-07-09 | kaggle.com/competitions/foursquare-location-matching/discussion/336055 |
| K2 | Foursquare 2nd place (Brief / T0m part / colum2131 part) | T0m, yabea, tkm2261, colum2131, rii | 2022-07-09 | …/discussion/336062, 336072, 336090 |
| K3 | Foursquare 3rd place solution | Psi | 2022-07-19 | …/discussion/338112 |
| K4 | Foursquare 4th place solution | Vincent Schuler, Youri Matiounine, Theo Viel | 2022-07-08 | …/discussion/335810 |
| K5 | Foursquare 8th place [SBERT + LightGBM] | jsaikawa, ockie1729, kkanayj, llanxh, bbbnodx | 2022-07-08 | …/discussion/335928 |
| K6 | Foursquare 13th place (GNN) | 213tubo | 2022-07-09 | …/discussion/336124 |
| K7 | Foursquare "Competition is Finalized - Investigation Update" | Addison Howard (Kaggle) | 2022-07-18 | …/discussion/338035 |
| K8 | Shopee 1st Place - From Embeddings to Matches | Limerobot, YoonSoo | 2021-05-11 | kaggle.com/competitions/shopee-product-matching/discussion/238136 |
| K9 | Shopee 2nd place (GAT & LGB) | lyakaap, tkm2261 | 2021-05-11 | …/discussion/238022 |
| K10 | Shopee 3rd Place (Triplet, Boosting, Clustering) | Btbpanda | 2021-05-12 | …/discussion/238515 |
| K11 | Quora 1st place solution | Lam Dang, Guillaume Huard, MaxBaudry, PaulTodo, Sebastien Conort | 2017-06-07 | kaggle.com/competitions/quora-question-pairs/discussion/34355 |
| K12 | Quora "To Kaggle and Quora: Review of the solutions" | JStr | 2017-06-11 | …/discussion/34560 |

---

## 10. Models seen in winning solutions: license check (HF API `cardData.license`, 2026-09-25)

| HF id | License | Params (HF safetensors total) | Compliant (MIT/Apache, ≤8B)? | Where it appeared / use for us |
|---|---|---|---|---|
| FacebookAI/xlm-roberta-large | mit | 561,192,082 | **yes** | Foursquare 1st, Shopee 1st, Psi. Cross-encoder / ArcFace encoder (A10G/L4 needed). |
| FacebookAI/xlm-roberta-base | mit | 278,885,778 | **yes** | Foursquare 2nd/13th. **Best first cross-encoder for us** (T4-trainable). |
| microsoft/mdeberta-v3-base | mit | not listed in safetensors (UNVERIFIED count) | **yes** (license) | Foursquare 1st (best single CV 0.907), 13th. |
| google-bert/bert-base-multilingual-uncased | apache-2.0 | 168,055,961 | **yes** | Foursquare 1st (candidate kNN, SVD features), Shopee 1st. |
| sentence-transformers/all-MiniLM-L12-v2 | apache-2.0 | 33,360,512 | **yes** | Foursquare 8th (English model on transliterated text). |
| sentence-transformers/LaBSE (= setu4993/LaBSE weights) | apache-2.0 | 470,927,360 | **yes** | Shopee 3rd. Cross-script bi-encoder candidate. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0 | 117,654,272 | **yes** | Small multilingual bi-encoder. |
| sentence-transformers/paraphrase-xlm-r-multilingual-v1 | apache-2.0 | 278,044,162 | **yes** | Shopee 2nd ("Paraphrase-XLM"). |
| google/muril-base-cased | apache-2.0 | not listed (UNVERIFIED) | **yes** (license) | Indic-specific BERT. Candidate for cross-script matching. |
| intfloat/multilingual-e5-small | mit | 117,654,272 | **yes** | Compact multilingual retriever. |
| BAAI/bge-m3 | mit | not listed (UNVERIFIED) | **yes** (license) | Multilingual dense+sparse retriever. |
| cahya/bert-base-indonesian-522M | mit | not listed | yes, but irrelevant language | Shopee 1st/3rd. |
| Qwen/Qwen2-VL-7B-Instruct | apache-2.0 | **8,291,375,616** | **NO: exceeds 8B** | Amazon-ML-2024 claimed winner. Cautionary example. |
| fastText cc/wiki vectors (fasttext.cc) | CC BY-SA 3.0 ("distributed under the Creative Commons Attribution-Share-Alike License 3.0") | n/a | **NO** | DeepMatcher default (`fasttext.en.bin`). Possibly DeepBlocker. |

---

## 11. Compliance findings (flag list)

1. **Model size:** "7B" in a name is not "≤8B". Qwen2-VL-7B-Instruct counts 8.29B, so record the parameter count of every weight from HF metadata in `research/MODEL_LICENSES.md`.
2. **External-data-derived artifacts:**
   - libpostal (OSM + OpenAddresses; Senzing variant adds customer-feedback data);
   - deepparse weights (from libpostal-generated data);
   - rigour address formats (from OpenCageData);
   - yente (OpenSanctions data).

   Avoid all four in the final pipeline.
3. **Hand-curated lists** (cleanco termdata, rigour org_types, DNB legal_names.csv) are a grey zone. They are not "lookups", but they are external knowledge. Our current learned alias maps (`aliases.py`) are the safer route. If a list is used as a seed (e.g. French legal forms, since France has no labels), declare it in the documentation as hand-written normalization knowledge.
4. **Software licenses** are separate from model licenses:
   - AGPL-3.0: Zingg.
   - GPL-3.0: Abydos, vendored in DNB name_matching.
   - LGPL-3.0: deepparse.

   None of these is a model-license violation, but avoid shipping copyleft code in `code/` without need. Everything else here is MIT, Apache-2.0, BSD-3 or ISC.
5. **Solution repos with no license** (all Kaggle/Amazon repos above): read for ideas, never copy code.
6. **Transductive features** (computed from unlabeled test S1/S2/S3) are provided data, not external. Document them as label-free (COMPLIANCE_CHECKLIST F3).

---

## 12. Adaptation plan: from the winners' template to our pipeline

Mapping (ours ↔ theirs):

| Winner component | Ours now | Gap / action |
|---|---|---|
| Union of kNN retrievers with a per-stage oracle | 5-view blocking, recall 0.9905, oracle 0.9972 → 0.99645 after pruning | Fine. Add SNI on a normalized name key and a transliterated-name TF-IDF view only if the oracle gain is ≥ 0.0005. |
| Cheap GBDT pre-filter (FIL) | Budget pruning to 43-51 per S1 | Consider a 2nd-tier prune to about 10-15 per S1 before any cross-encoder, to keep test cross-encoder pairs around 20-25M. |
| Rich GBDT with per-query stats (max/min/mean/ratio per id; top-k density) | 87 features + collective v1 (stage 2 reaches 0.98703) | Add per-S1 stats for *all* similarity fields, density-at-radius counts, **per-split (and per-country) z-normalized density features** (Shopee 2nd), name-number mismatch (OpenSanctions), and legal-form agreement (EMM). |
| Cross-encoder (xlm-r/mdeberta) with numeric features at [CLS] | none | P1: xlm-roberta-base (MIT, 279M). Input: `name_S1 [SEP] name_T [COL] addr_S1 [SEP] addr_T` (+ transliterated forms). Concatenate p1, trank and margin to [CLS]. Flip TTA. Train on hard pairs only (the stage-2 uncertainty band 0.2-0.95). |
| GNN per query with metric loss | stage_collective hand features | P2: PNA/GAT on the per-S1 subgraph (S1 + candidates + T-T edges). Soft per-S1 F0.5 loss + 0.1·BCE. |
| Macro-aware sample weights | uniform | P0: weight pairs by 1/n_true(S1)^0.4, and give singleton S1s' negatives weight 1 per S1. |
| Group-size-dependent thresholds / copy-group decisions | global thr 0.7-0.8 + exclusivity | P1: threshold that rises with copy-group size (Foursquare 4th). Group-level acceptance (graph.md §7.4). |
| Neighbourhood blending / 2-hop expansion for recall | reverse view + collective | P1: embedding QE (S1 ⊕ accepted targets), re-query, re-score only new pairs (Foursquare 1st post-process). Targets the empty-address FNs. |
| Mutual-edge filter | trank + exclusivity (≈0 gain) | Already covered. No action. |
| Prior-shift recalibration per stratum (Quora) | threshold moved to 0.8 on test | P1: per-density-stratum recalibration, estimated label-free on test (EM prior adjustment on the stage-2 probabilities) and validated on P-dense. |
| Leak / overlap audit | not done | P0: exact and near-exact overlap check of test records vs train records (normalized name+address). If overlap exists, CV will mislead. Do NOT exploit it without an explicit rules check. |

---

## 13. What we should do (priorities)

**P0 (now; cheap; directly supported by leak-free winners)**
1. **Overlap/leak audit, train vs test** (Foursquare, Quora). Hash normalized (name, address, country) for all S1/S2/S3 in train and test and report exact and near-duplicate overlap rates. If they are non-trivial, build a validation fold that mimics the test overlap rate. Treat exploitation as a rules question for the organisers, not a modelling trick.
2. **Per-split, per-country z-normalized density features** (Shopee 2nd): top-K mean/std of p1 and of the similarity scores per S1 and per target, plus density-at-radius counts. Their stated purpose was exactly the train/test distribution difference. Expected impact: stabilises the 0.7→0.8 threshold drift, and gives France (unseen) its own normalization.
3. **Macro-aware training weights** 1/n_true(S1)^α with α ∈ {0, 0.4, 0.7} (Shopee 2nd), measured on exact macro-F0.5.
4. **Per-S1 statistics for every similarity feature** (Foursquare 1st: max/min/mean per id + ratios) and a **name-number-mismatch** feature (OpenSanctions logic-v2 `nm_number_mismatch`). Both are cheap. The number feature targets the sibling-lookalike FPs.
5. **Test-density validation pools** (Psi, Shopee 3rd): keep all S2/S3 of the split in every validation pool. P-dense already approximates this, so make it the default selection metric.

**P1 (next; needs GPU, which is blocked on Modal spend right now, so use Kaggle/Colab once approved)**
6. **Cross-encoder re-scorer** on the uncertain band: xlm-roberta-base (MIT) with Ditto-style serialization, numeric features at [CLS], and flip TTA. Stack it with the GBDT via threshold-subtracted ensembling (Shopee 2nd). Train on T4/L4, with about 2-5M hard pairs. Expected: the largest single gain in both Foursquare and MWPD. In Foursquare 1st, going from LGBM alone (CV 0.875) to the blend with CatBoost + two transformers (CV 0.911) gained 0.036, and the best single transformer reached 0.907.
7. **Copy-group-aware / group-size-aware thresholds** (Foursquare 4th + graph.md §7.4).
8. **Neighbourhood-blending recall repair** (Shopee 1st INB, Foursquare 1st merge-and-rescore). Blend the S1 embedding with its accepted targets, re-query, and re-score only the new pairs at a precision-safe threshold. Targets the empty-address FNs.
9. **Transliterated-Latin encoder branch vs multilingual branch** for cross-script India (Foursquare 8th, 4th). Measure the oracle and F0.5 on the India subset.

**P2 (if time permits)**
10. **Per-S1 subgraph GNN** (PNAConv/GAT, PyG MIT) with a soft per-S1 F0.5 loss (Foursquare 13th: +0.022 public).
11. **Metric-learning retriever** (ArcFace/sub-center or in-batch MNRL over S1 entity ids, source-aware sampling from R-SupCon), if the blocking oracle becomes the binding limit.
12. **Splink/DuckDB Fellegi-Sunter model** as an unsupervised, interpretable baseline. Its m/u weights with TF adjustments can serve as extra features and as a documentation-friendly fallback for France.

**P3 (low value or rejected)**
13. Betweenness edge pruning (Shopee 2nd, +0.001) and agglomerative clustering to a target cluster size (Shopee 3rd). Our exclusivity constraint already covers most of this.
14. **Do not use:**
    - libpostal, pypostal or deepparse weights (external-data derived);
    - Zingg (AGPL);
    - DeepMatcher with fastText (CC BY-SA);
    - any >8B model;
    - code from unlicensed solution repos;
    - yente/OpenSanctions data.

---

## 14. Open items / unverified

- "Ri" (leak-free Foursquare #1) write-up: not found in the forum list retrieved, so UNVERIFIED whether one exists.
- mdeberta-v3-base, muril-base-cased and bge-m3 parameter counts: not in the HF safetensors metadata I read. Licenses are verified.
- DeepBlocker's default embedding (fastText) is assumed from the paper lineage and not confirmed in code.
- Amazon ML Challenge 2023/2024/2025 "winning" claims come from participants' READMEs (self-reported). The 2025 "winning SMAPE 39.7" is UNVERIFIED.
- The ArcFace conference venue (CVPR 2019) was not re-verified here. Only the TPAMI DOI on arXiv was.

## Audit log

- Kaggle content: forum topics via Kaggle's discussion endpoint (GetForumTopicById / GetTopicListByForumId), and competition metadata/evaluation text via the competition and page endpoints.
- MWPD2020, Sparkly and DeepBlocker PDFs: fetched and text-extracted locally (PyMuPDF). No model or heavy computation was run.
- DBLP API was behind a bot-check and was not used (no attempt to bypass). Crossref and arXiv APIs were used instead.
