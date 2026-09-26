# Industry sweep: business, merchant and place entity resolution at scale

Sweep date: 2026-09-25. Scope: engineering blogs, vendor methodology documents (methods only), patents, Kaggle write-ups and code, and industry papers on company, merchant and place matching. The focus is on practical tricks for generic names, chains and franchises, address normalization without gazetteers, threshold setting and cluster consistency.

Verification convention:
- **VERIFIED**: I read the claim on the primary page, PDF or code listed.
- **PARTIAL**: the main claim is verified, but some details come only from a secondary summary, and those details are marked.
- **UNVERIFIED**: I could not open the primary page (Kaggle write-ups render in JS, and some pages returned 403). Treat these as leads only.

All numbers below are copied from the source. None are estimates.

Pipeline context, taken from `FEATURE_CATALOG.md` so that the recommendations do not repeat existing work:
- "Core" tokens are currently defined by a document-frequency cutoff (DF ≤ 1% within the country).
- IDF-Jaccard, coverage and max-shared-IDF features already exist.
- `q_nm_freq_s1`, `t_nm_freq_s1` and `t_nm_freq_t` are raw global counts.
- Competition features (rank, gap and margin on the q-side and t-side) exist.
- Collective and sibling features are planned, not yet built.

---

## TL;DR: what industry does that we do not yet do

1. **Weight name tokens by context, not just global frequency** (Facebook's places dedup, D&B's "Uniqueness" component, Splink's term-frequency adjustments, Senzing's generic thresholds). How often a name appears *in the same city or postcode*, and whether a token is a "core" token or a "background" token, are separate signals from global IDF. We only have global DF and IDF.
2. **Treat addresses as values that can be generic too** (Senzing gives ADDRESS the frequency class FF, "few"; D&B has a "Density" component). An address shared by many S1 records (a mall, a business centre, "Near X" landmarks) is weak evidence. We have no address-frequency feature.
3. **Normalize count features across splits.** Foursquare's 4th-place solution scales count encodings by a `size_ratio` and rounds them on a log scale. Our raw counts will drift, because test S1 is 1.73M against 2.2M in train and France is 15% of test S1.
4. **Enforce cluster consistency after pairwise scoring** (7th and 4th place Foursquare post-processing, Splink bridges and density, Quantexa's overlinking model, Google EKG's node-to-cluster confidence). Our decision layer is per-S1, but it does not check whether the chosen targets agree *with each other*.
5. **Account for country shift.** Grab found that pooling countries made results *less* reliable. France is unseen in training, so validate with a leave-one-country-out split and prefer country-normalized features.
6. **Audit blocking recall on a name-similarity × address-similarity grid.** Overture found that blocking was "the largest bottleneck" and that it undermatched moderately similar names. An overall recall of 0.996 can hide weak cells, especially for France.

---

## Items

### 1. Facebook: "Deduplicating a Places Database" (Bohannon, Dalvi, Olteanu, Raghavan; WWW 2014). VERIFIED
- URL: https://research.facebook.com/publications/deduplicating-a-places-database/ (PDF read in full: https://research.facebook.com/file/353516043039579/deduplicating-a-places-database.pdf).
- Note: the one-core-per-name constraint is used only to *learn* C and B. At match time (their Section 5) the constraint is relaxed, and each token is core with probability α independently.
- **What it is.** An unsupervised language model for business names, deployed in production for Facebook's Checkins Places database. It uses only name and approximate location.
- **Key technique.**
  - Each name is modelled as one **core** word (drawn from a distribution C) plus k **background** words (drawn from a distribution B).
  - B, C and the per-token posterior z(w,n) are learned by EM. E-step: z(w,n) = [C(w)/B(w)] / Σ_{w2∈n} C(w2)/B(w2). The M-step is re-counting.
  - The constraint that each name has exactly one core token is what makes learning possible. Without it, both B and C collapse to document frequency.
  - **Spatial context.** The world is split into tiles, and each tile gets its own background model B[l] = λ·B_ml[l] + (1−λ)·B, learned jointly (their Algorithm 2). This captures the "Bleecker" effect: a word that is rare globally but frequent locally counts as background *in that tile*.
  - **Match score.** With token core probability c(w,p) = αC(w) / (αC(w) + (1−α)B[p.l](w)), the probability that two places share the same core is: (product over tokens only in p1 of (1−c)) × (product over tokens only in p2 of (1−c)) × (product over shared tokens of [c1·c2 + (1−c1)(1−c2)]).
  - This is extended by a dynamic program over edit operations (abbreviation "North West → NW", concatenation "North West → Northwest", typos), each with a probability π(e).
- **Evidence.**
  - Core-token classification beats an IDF-threshold classifier on both datasets. Peak F-measure is about 0.85 at α ≈ 0.5.
  - Accuracy at identifying the core in context peaks at about 91% (Places) and 87% (WikiMapia) at λ ≈ 0.9, against 80% and 75% with no local context (λ = 0).
  - Deduplication using names only reaches **90% recall at 90% precision** and beats Levenshtein and TF-IDF cosine.
  - Labelled sets: about 7K candidate pairs each, of which about 2K are duplicates.
- **Adaptation (P0).** This is the most directly relevant industry method for the "generic token injection" noise ("Services", "Center") and for moved legal suffixes.
  - Replace or augment the DF ≤ 1% core heuristic with EM-learned C and B per country, and let a tile = (country, state or city, or the postcode prefix when present).
  - Train it unsupervised on all S1+S2+S3 names, including the test split transductively. This is consistent with our existing policy of "label-free corpus statistics", and it gives France its own background model without any labels.
  - New features:
    - `n_samecore_p` (the eq. 3 score, computed on normalized, transliterated tokens);
    - `n_samecore_dp` (with our learned abbreviation map, concatenation and a JW ≥ 0.9 typo operator as edit ops);
    - the sum of core probability on unmatched tokens for each side (a better version of `n_core_missing` and `n_core_extra`).
  - Cost: EM is a few passes of sparse counting over about 20M names, which is cheap on CPU but should run on Modal, not locally.

### 2. Dun & Bradstreet match methodology (US patent 8768914B2, Direct 2.0 docs, "The Basics on Data Matching"). VERIFIED
- URLs:
  - https://patents.google.com/patent/US8768914B2
  - https://docs.dnb.com/direct/2.0/en-US/company/latest/getcleansematch/rest-API
  - https://www.dnb.com/content/dam/web/int/resources/pdf/DNB_The_Basics_On_Data_Matching.pdf (PDF read)
- **What it is.** The commercial reference-data matcher: input business record → DUNS reference file. This is the same many-to-one, "reference plus noisy inquiries" shape as our task.
- **Key techniques.**
  - **MatchGrade**, 11 components: Name, Street Number, Street Name, City, State, PO Box, Phone, Postal Code, **Density**, **Uniqueness**, SIC. Each component is graded A/B/F/Z, and the grade pattern is mapped to a **Confidence Code**. The docs say 1–10; the Direct 2.0 parameter is described as "4 (low) up to 10 (high)".
  - **Uniqueness** (patent): "If the city names match then count matching business names in city and score the number of matches based upon 100". If the cities do not match, count within the state instead.
  - **Density**: a "business weighted distance" = 100/D(log(A+B)+1), where D is the lat/lon distance and A and B are the business counts in the inquiry city and the candidate city.
  - **Zip score**: a decision tree. The first two digits must place both zip codes in the same state ("If not in the same state then zip score is zero"). Otherwise it uses the edit distance on the last four characters: 0 or 1 → 100, 2 → 80, 3 or more → 0.
  - Grades: the patent describes "A, B, F, Z grading" but does not define each letter in the text I read.
  - **Keys** (blocking): exact name, words, word pairs, shingles, **acronym**, soundex, lat/lon, phone. "Match candidates are limited for certain keys that return counts surpassing a predetermined threshold". This is a generic-key cap.
  - **Cleansing**: remove special characters; "A last word in the company name is removed if it is a standard company form"; uppercase; depluralize selected text; standardize selected words and phrases; use reference tables for "vanity city and vanity street names".
  - **Segmentation**: pairs are classified into "one of eleven distinct data segments by means of a decision tree", then "logistic modeling which uses data segments ... as predictors".
  - **Match Data Profile** (MDP) records *which* reference name variant matched: 00 business name, 01 registered name, 02 tradestyle (≈ dba), 03 CEO name, 04 additional executive, 05 former business name, 07 former CEO, 09 short name, 10 registered acronym (from the "Basics" PDF table).
  - **Multi-pass matching**: if the first pass fails, retry with alternative inputs.
  - Data-quality checklist: null tokens such as "NULL", "DO NOT USE" and "c/o"; postal codes of the wrong length.
  - Expected match rates range from "around 60%" to "95% or more" depending on input quality.
- **Adaptation.**
  - (P0) **Local uniqueness features.**
    - `q_core_n_city`: the number of S1 records in the same (country, city) whose core-name key equals q's. Back off to state when the city is missing, exactly as the patent does.
    - `t_core_n_city`: the same count for the target's name.
    - These go beyond our global `q_nm_freq_s1`. A chain name in one city with ten outlets should make the model lean on house number and street, and this is the explicit signal for that.
  - (P1) **Zip-score feature.** Require agreement of the postcode prefix, then take the edit distance on the remainder, as a single graded feature. For France the first two digits are the département. The noise includes "region ↔ département swaps", so a postcode-prefix agreement feature is robust to that swap.
  - (P1) **Generic-key cap in blocking.** For any exact-key blocker (name key, acronym key, domain key), drop keys whose posting list exceeds N, or back off to the compound key (key + postcode prefix).
  - (P2) **Segment-aware thresholds.** Define about 8–12 segments from field-presence patterns (target address empty, target script non-Latin, has alias, name is a domain or handle, is acronym-like). Fit a separate calibration and threshold per segment in the expected-F0.5 layer, or at least check calibration per segment.
  - (P2) **MDP-style indicator.** For alias records, record which side matched (primary or dba/aka) as a categorical feature, alongside the existing `n_alias_core_jacc`.

### 3. Senzing: principle-based ER (whitepapers, 2019 and 2020). VERIFIED
- URLs:
  - https://senzing.com/wp-content/uploads/Principle-Based-Entity-Resolution-092519.pdf
  - https://senzing.com/wp-content/uploads/Entity-Resolution-Processes-021320.pdf (both PDFs read)
  - https://senzing.com/docs/globalization/
- **What it is.** Commercial ER that needs no training. It uses about 35 general "principles" built on attribute behaviours.
- **Key techniques.**
  - Every attribute has a **frequency** class: F1 (one entity), FF (few; ADDRESS, PHONE and WEBSITE are FF), FM (many; DOB), FVM (very many; gender) and a special NAME class. Attributes also have **exclusivity** and **stability**.
  - **Generic values are detected at runtime.** A value used by too many entities becomes "generic" and is dropped from candidate selection. The example given is "a bank's toll-free phone number already associated with hundreds of people". "The final list of non-generic features is used to retrieve candidate entities".
  - **Entity-centric matching.** A record is compared with the whole resolved entity, not with single records.
  - **Real-time re-evaluation.** "As statistics evolve ... earlier entity resolution assertions are reevaluated".
  - **Graded comparator bands**: same, close, likely, plausible, unlikely, not the same.
  - Organizations: a "possible match" is the same address with different tax IDs, and "possibly related" is the same address and phone with different names. Shared address alone is **not** a match.
  - Globalization: cross-script name matching uses culture-aware libraries. For non-CJK addresses they recommend romanizing and keeping both the native and romanized forms.
- **Adaptation.**
  - (P1) **Generic address.** Add `q_addr_n_s1`: the number of S1 records sharing (street root + house number + postcode or city). Add `t_addr_n_t`: the same count over targets. An address shared by 5 or more S1 records (a business centre, a mall, a "Near X" landmark address) must not count as strong evidence. LightGBM can learn this interaction only if the feature exists.
  - (P1) **Entity-centric second pass.** After pass 1, form each S1's provisional entity: S1 plus targets with p > 0.8. Recompute features for every candidate t against that entity: the max and mean similarity to the entity's members, and agreement with the entity's consensus address and house number. Retrain the stacked model on out-of-fold pass-1 predictions. This is the "collective / support features" item in our plan, made concrete.
  - (P3) Keep both the native-script and transliterated forms as separate views. We already do this.

### 4. Splink (UK Ministry of Justice; used by ONS and others): term-frequency adjustments and graph metrics. VERIFIED
- URLs:
  - https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html
  - https://github.com/moj-analytical-services/splink/discussions/2022
  - https://moj-analytical-services.github.io/splink/topic_guides/evaluation/clusters/graph_metrics.html
  - https://realworlddatascience.net/applied-insights/case-studies/posts/2023/11/22/splink.html
- **What it is.** Open-source (MIT) Fellegi–Sunter linkage. The case study reports "over 100 million records" handled, "1 million records on a laptop in around a minute", and 7 million+ downloads.
- **Key techniques.**
  - **Term-frequency adjustment.** When two values match exactly, the match weight is adjusted by how common that value is.
    - For fuzzy (non-identical) matches, Splink uses the **greater of the two term frequencies**: "we err on the side of lowering the match score".
    - `tf_minimum_u_value` (example 0.001) floors the frequency so that rare misspellings do not get extreme weights.
    - `tf_adjustment_weight` (example 0.5) damps the adjustment for fuzzy levels.
  - **Token-level TF for company names** (Discussion #2022). Tokenize the name, attach each token's relative frequency (for example LIMITED = 0.2025, POSEIPORT = 6.1e-5), and **multiply the relative frequencies of the shared tokens**. Bucket the product into levels: < 1e-10, < 1e-8, < 1e-5, else.
  - **Graph metrics for cluster QA**: node degree; node centrality (a low value relative to peers suggests a false-positive link); **is_bridge** ("Bridges often signal false positives, particularly when connecting otherwise separate sub-clusters"); cluster size; cluster density. Clusters come from connected components at a threshold.
- **Adaptation.**
  - (P1) Add `n_shared_logtf = Σ log10(relfreq(w))` over shared tokens, using **per-(country, city) relative frequencies** with a global back-off (Splink's product combined with Facebook's spatial context).
  - For fuzzy token alignments, use the **max** of the two tokens' frequencies, and floor it at 1e-6 to follow the `tf_minimum_u_value` idea.
  - This differs from `n_max_shared_idf` in two ways: it accumulates over all shared tokens, and it is local.
  - (P1) Compute graph metrics on each S1's predicted cluster (see item 5 and the post-processing plan below).

### 5. Kaggle "Foursquare – Location Matching" (2022): top solutions. PARTIAL
- URLs:
  - 7th place, Future Architect blog (Japanese; VERIFIED): https://future-architect.github.io/articles/20220720a/
  - 4th place code (VERIFIED, code read via the GitHub API): https://github.com/TheoViel/kaggle_foursquare, files `src/pp.py` and `src/fe.py`
  - Foursquare recap blog (VERIFIED): https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/
  - 1st place write-up (re:waiwai, per the URL slug): https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution. **UNVERIFIED**: the page is JS-rendered and I could not read its content.
- **Task shape.** Records with name, address, city, state, zip, phone, category and lat/lon. Predict each record's set of matches. The metric is IoU. The Foursquare recap reports 22,050 submissions from 1,290 data scientists.
- **7th place (VERIFIED).**
  - Five retrievers:
    - haversine kNN (4 candidates);
    - a regression that combines distance with embedding cosine (12);
    - word-level name overlap (4);
    - character-level name overlap (8);
    - name-embedding cosine (4).
  - Total 32 candidates per record. Retrieval-only maxIoU was **0.9778**.
  - LightGBM with num_leaves 2^12, learning rate 0.1 and 2000–3000 iterations. Inference was about 100× faster with GPU ForestInference.
  - Features: gestalt, Levenshtein and Jaro-Winkler similarity, ROUGE-N and ROUGE-L on name and address, and k-means cluster distances.
  - **Post-processing**: union-find at a threshold, then **remove edges chosen by betweenness centrality**, then output pairs whose graph distance is ≤ 2. Retrieval plus post-processing reached maxIoU **0.9935**.
- **4th place (code, VERIFIED).**
  - Level 1 GBDT pre-filter, then a level 2 GBDT.
  - **Count encodings** of name, address, city, state, zip, phone and id. Each count is multiplied by `size_ratio`, capped at 10000, and **log-rounded**: exp(round(log(x+1), 1)) − 0.5. Features are the min and max across the pair.
  - **Local density**: the number of records within 2000, 1000 and 500 m, and the same count restricted to the same category.
  - **Cluster post-processing**:
    - grow clusters with a *different threshold once the cluster has more than 2 members* (`threshold_small` and `threshold_big`);
    - merge two clusters only if the best link exceeds `threshold_merge_max` (0.8 default), or the mean link exceeds `threshold_merge_avg` (0.5) with more than one link;
    - never merge above `max_size` = 300.
- **2nd, 3rd and other places.** From the Foursquare recap only; the exact ranks are inconsistent between sources, so treat attribution as UNVERIFIED.
  - Yuki Uehara used four stages: candidates from text and geography, a LightGBM filter, two transformer pair classifiers, and **GNN node-classification post-processing**, improving "from 0.907 to 0.946".
  - Philipp Singer used ArcFace metric learning for candidates plus a bi-encoder second stage.
  - re:waiwai used a 3-stage LightGBM with an XLM-R post-process; per the search snippet, "top 40 candidates per id" went to the stage-2 LightGBM.
- **Adaptation.**
  - (P0) Normalize our count features (`q_nm_freq_s1`, `t_nm_freq_s1`, `t_nm_freq_t` and the new locality counts). Either divide by the per-country corpus size (a rate) or multiply by `size_ratio` = train_size/test_size **per country**, then log-round to 0.1. This removes train/test drift: S1 is 2.2M against 1.73M, and France changes the country mix.
  - (P1) **Size-aware acceptance** in our expected-F0.5 layer. Once an S1 already has 2 or more accepted targets, raise or lower the marginal threshold according to validation. Singletons need an empty prediction, so the first acceptance is the riskiest.
  - (P1) **Graph pruning inside each S1 cluster.** Build the target–target similarity graph among targets assigned to one S1, weighting edges by name and address cosine. Drop a target that forms a **bridge** to an otherwise dense sub-cluster, or whose max similarity to its siblings is far below the cluster median. This is our many-to-one analogue of the 7th-place betweenness removal.
  - (P2) An L1 pre-filter LightGBM with 10–15 cheap features would let blocking return more candidates per target (for example top-16 instead of top-8 reverse neighbours) without the full feature cost.

### 6. Yelp: "Seeing Double" dedup (2015) and "Learning to Rank for Business Matching" (2014). VERIFIED
- URLs:
  - https://engineeringblog.yelp.com/2015/05/seeing-double-on-yelp.html
  - https://engineeringblog.yelp.com/2014/12/learning-to-rank-for-business-matching.html
- **Key techniques.**
  - Business Match runs Elasticsearch subqueries (name, location text, geo-distance, phone) and reranks them with a **pointwise learning-to-rank** model whose features are the per-component ES scores and the original rank. This raised F1 "from 91% to 95%".
  - The dedup classifier is a scikit-learn random forest. Its features include geo distance, synonym-aware field matching, edit distance and Jaccard. Two features are themselves classifiers:
    - (a) an NER model that flags person-based business names (lawyers, doctors);
    - (b) a word-alignment logistic regression that outputs a confidence score, **"the number of uncommon words that appeared in one name but not the other, and the number of uncommon words that appeared in both names"**.
  - Explicit false-positive classes: two businesses in **the same chain**, and a **sub-business** of another.
  - Operating point chosen by **F0.1**: 0.966 (99.1% precision, 27.7% recall) against a 0.915 baseline. More than 500,000 duplicates were merged.
- **Adaptation.**
  - (P2) Train a small word-alignment LR on train pairs: features over aligned token pairs, weighted by frequency class. Stack its probability plus the two "uncommon word" counts. Use the *core probability* from item 1 to define "uncommon" instead of DF.
  - (P3) The F-beta operating-point lesson is already built into our expected-F0.5 layer.

### 7. LinkedIn: Organization Entity Resolution (2022). VERIFIED
- URL: https://www.linkedin.com/blog/engineering/economic-graph/matching-external-companies-to-linkedin-s-economic-graph-at-scal
- **Key techniques.**
  - Clean the request: strip punctuation and stop words such as "Limited".
  - An inverted index over **word n-grams** of names plus **website keys "that go beyond mere domains"**, which handles aggregator URLs such as linkedin.com/company/x.
  - The L1 retrieval returns "hundreds, or even thousands" of candidates. String-similarity models pick the top 10 for the L2 ranker, a wide & deep model: wide = binary match features, string similarities and feature crosses; deep = learned embeddings of the string attributes.
  - About 1M training examples, labelled from manual overrides and crowdsourcing.
- **Adaptation.**
  - (P2) For targets whose name is a URL or @handle on an aggregator (facebook, instagram, linkedin, justdial and similar), use the **path segment** as the name key, not the domain. Add a blocking view keyed on it.
  - Check that our domain/handle extraction does not reduce "facebook.com/abcsweets" to "facebook". This is a cheap audit.

### 8. OpenSanctions logic-v2 matcher (2025) and fingerprints/rigour. VERIFIED
- URLs:
  - https://www.opensanctions.org/articles/2025-09-11-logic-v2/
  - https://www.opensanctions.org/docs/api/scoring/
  - https://www.opensanctions.org/matcher/
  - https://github.com/opensanctions/fingerprints (MIT; now merged into "rigour")
- **Key techniques.**
  - Tokens are tagged with **symbols**: organization types (LLC, OAO, "Joint Stock Company", "Aktiengesellschaft" are treated as equivalent), **generic business words** (Holding, Industries) as *soft* stop words, and numbers.
  - Fuzzy matching then runs strictly on the untagged, distinctive core. The worked example: "Sparkles Business Corporation" vs "Sporks Business Corporation" scores 89% with plain Levenshtein, but once the generic words are tagged it reduces to "Sparkles" vs "Sporks" at 50%.
  - Transliteration is applied **selectively**, "for language pairs where the romanization is reliable".
  - The default threshold is 0.7 (0.8–0.85 for low false-positive tolerance).
  - The docs admit the matcher is "particularly vulnerable to misspellings in the legal type parts of company names (e.g. `Lymited` vs. `Limited`)".
  - The fingerprints library normalizes legal forms across languages (Aktiengesellschaft → ag). Its lists come from OCCRP, ISO 20275 (GLEIF) and Wikipedia.
- **Adaptation.**
  - (P1) **Fuzzy legal-suffix and generic-token detection.** Our noise injects typos into *every* token, including suffixes. A typo'd "Pvt Ltd" such as "Pvt Lmited" is currently a rare, "core" token with high IDF, which inflates `n_core_extra`.
    - Map any token within edit distance ≤ 1 (or JW ≥ 0.92) of a **learned** generic token to that generic class *before* the core/IDF computation.
    - Learn the generic list from the data: the top-DF tokens and the EM background tokens from item 1.
  - (P1) **Soft weights, not removal.** Keep generic tokens with a small weight (for example the background probability from item 1) rather than deleting them. This preserves the signal in names that consist only of generic words ("City Services Center").
  - **Caution.** Using the GLEIF, OCCRP or Wikipedia legal-form lists would count as external data. Use the *method* only and learn the list from S1/S2/S3.

### 9. Overture Maps Foundation: places conflation (2026). VERIFIED
- URLs:
  - https://docs.overturemaps.org/guides/places/
  - https://overturemaps.org/blog/2026/inside-the-2026-overture-member-summit-the-road-ahead/
- **Key techniques.**
  - Four stages: history matches → blocking (spatial pre-filter) → pairwise gradient-boosted trees → clustering, with "promotion of the highest-match-count record".
  - Labels come from LLMs using an "evidence hierarchy (functional identity, phone, name uniqueness and distance, and supporting signals such as website, category, and brand)". "Two independent models label each pair; agreements are accepted automatically and disagreements go to human review".
  - Diagnosis: "Mapping the true distribution of matches by name similarity and pair distance" showed **undermatching**, "especially when records are more than 100 meters apart or have only moderately similar names", and that "the blocking stage was the largest bottleneck".
  - Confidence is "a relative filtering tool rather than a precise probability".
  - The summit recap reports that the pipeline with "open source LLMs and agent-driven prompt optimization" pushed "combined model accuracy past 98%".
- **Adaptation.**
  - (P0) **Blocking recall audit grid.** On a train validation fold, bin true pairs by (name similarity decile × address similarity decile) and by noise type (cross-script, acronym, handle/domain, truncation, alias, France-like abbreviations simulated on train). Report recall per cell.
    - The 0.996 sample recall is an average. The cells with moderate name similarity and moderate address similarity are where Overture found its losses.
  - (P2) **Two-model agreement for France pseudo-labels.** On test France, take pairs where LightGBM and an independent scorer (a fine-tuned e5 cross- or bi-encoder) *both* give p > 0.95 or both give p < 0.05. Use them to fit a France-specific recalibration of the decision layer.
    - This is transductive and uses no external data, but check the rules before using any test-derived labels.

### 10. Facebook: "Place Deduplication with Embeddings" (Yang, Hoang, Mikolov, Han; WWW 2019). VERIFIED
- URL: https://arxiv.org/abs/1910.04861 (PDF read)
- **Key techniques.**
  - fastText name embeddings, plus address embeddings. Address "is especially useful in differentiating branches of the same stores". About two-thirds of their places have an address.
  - Coordinates and categories are **smoothed on a place graph** (grid-bin and category edges) rather than concatenated; concatenation hurt.
  - Supervised metric learning uses a **pairwise contrastive loss**, which beat triplet loss, plus:
    - **batch-wise hard sampling** (keep pairs whose distance is worse than the batch average scaled by a slack β);
    - **source-oriented attentive weighting** of noisy label sources;
    - **clustering-based label denoising**, a DEC-style KL loss that uses the transitivity of duplicates.
- **Evidence.** Pairwise accuracy: PEHAD 0.9279 against GBDT 0.7828, RF 0.7786, SVM 0.7699 and LR 0.7514 on their setup. The paper reports "over 18% relative improvement on GBDT". Their GBDT used different inputs, so this is not a like-for-like comparison with our LightGBM.
- **Adaptation (P2, GPU on Modal).** If we fine-tune multilingual-e5 (MIT) for a dense view:
  - use the pairwise contrastive loss with in-batch hard-pair selection;
  - weight S2-derived and S3-derived pairs separately (their "sources");
  - embed name and address separately, since address is what separates branches.

### 11. Grab: "Spatial Entity Resolution between Restaurant Locations and Transportation Destinations in Southeast Asia" (Gao and Widdows; GISTAM 2020). VERIFIED
- URL: https://arxiv.org/abs/2401.08537 (PDF read)
- **Key techniques.** Geohash blocking plus rule pre-screening: pairs cluster within 200 m, with a name Levenshtein-ratio cutoff of 0.4. Features: great-circle distance, Levenshtein and **Jaro** similarity on the name, and Levenshtein on the street. Separate models per country (RF, AdaBoost, GBDT, XGBoost).
- **Evidence.**
  - Test accuracy is at least 93% in every country.
  - Feature importance order is the same in every country: Jaro(name) > Levenshtein(name) > geo distance > Levenshtein(street).
  - Matchable-restaurant rate: about 37% for Indonesia, 53% Malaysia, 51% Philippines and 23% Singapore.
  - **Pooling the countries into one training set made results "less reliable"**. The one exception was a small recall gain for Malaysia, which they attribute to its similarity to Indonesian addresses.
- **Adaptation (P0, validation design).** France is unseen, so estimate the shift cost now.
  - (a) Train on US and evaluate on India, and train on India and evaluate on US, with the same features.
  - (b) Identify features whose importance or calibration flips across countries. Raw counts, address-number position and postcode shape are the likely ones.
  - (c) Make those features country-relative (rank or percentile within country) or drop them.
  - (d) Check the expected-F0.5 layer's calibration on the held-out country. The match-exists model is the most shift-sensitive part, because the distractor rate may differ in France.

### 12. libpostal v1.1 near-dupe hashing and lieu (OpenVenues; MIT). VERIFIED
- URLs:
  - https://github.com/openvenues/libpostal/releases
  - https://github.com/openvenues/lieu
- **Key techniques.**
  - **Address root expansions** "remove tokens that are ignorable such as 'Ave', 'Pl', 'Road'", so "West 125th St" ≈ "W 125". Whitespace is stripped in street roots, so "Sea Grape Ln" = "Seagrape Ln".
  - A modified **double metaphone** for Latin-script names.
  - **Acronym guessing** when a name has 2 or more tokens ("Brooklyn Academy of Music" → "BAM").
  - Name quad-grams.
  - Geo qualifiers: a 6-character geohash plus its **8 neighbours**, falling back to postcode or city when there are no coordinates.
  - Component-wise dedupe classes: NON, POSSIBLE_NEEDS_REVIEW, LIKELY, EXACT.
  - lieu scores names with Soft-TFIDF / Soft-Information-Gain: "likely dupe" at ≥ 0.9, "needs review" at 0.7.
- **Adaptation.**
  - (P1) **Street-root view.** Build an address string with street-type tokens removed and spaces collapsed inside street names. The set of type tokens is learned from data: the top-DF address tokens plus our learned abbreviation map. Add the cosine or Jaccard on this view as `a_root_*` features and as one more blocking view.
    - This makes "R. de la Paix" ≈ "Rue de la Paix" ≈ "BD de la Paix" match on the root without a French abbreviation list, which matters because France has no training labels.
  - (P1) **Soft-TFIDF name similarity.** Align tokens with JW ≥ 0.9 (or our phonetic key) and weight them by IDF or core probability. Our IDF-Jaccard gives zero credit when the rare token has a typo ("Laxmii" vs "Laxmi"), and Soft-TFIDF fixes exactly that.
  - (P2) A **postcode-neighbour qualifier** for the handful of blockers keyed on postcode: numeric ±1 on the last digits, the analogue of the 8 geohash neighbours.

### 13. addok-fr (Etalab / BAN French geocoder plugin; MIT). VERIFIED
- URLs:
  - https://github.com/addok/addok-fr
  - https://raw.githubusercontent.com/addok/addok-fr/main/addok_fr/utils.py
- **Key technique.** A French-specific `phonemicize` processor: an ordered list of regex rewrites (38 rules as listed in utils.py). Examples:
  - ph→f;
  - c before e/i/y → s, and other c → k;
  - qu → k;
  - g before e/i/y → j;
  - s between vowels → z;
  - silent final s, t, d and g dropped;
  - "ngt" → "n" (vingt → vin);
  - eau/oeu folding;
  - doubled letters collapsed.
  The plugin also ships a synonyms file for street types (rue, avenue, boulevard, …).
- **Adaptation (P1, France).**
  - Reimplement the rewrite rules as our own French phonetic key. The rules are code, not data. Apply the key to name and street tokens of FR-country records and add a `n_phon_fr_jacc` and `a_phon_fr_jacc` view.
  - This complements the existing `n_phon_jacc`, which is presumably built for English or Indic. French typo and diacritic noise (é/è/ê, silent endings) should collapse under it.
  - Do **not** import the synonyms file (it is external data). Learn street-type abbreviations transductively from high-confidence France test pairs, or rely on the street-root view in item 12.

### 14. Zingg (open-source ER, AGPL, so method only). VERIFIED
- URLs:
  - https://www.zingg.ai/post/entity-resolution-at-scale-part-3-blocking
  - https://docs.zingg.ai/latest/improving-accuracy/stopwordsremoval
- **Key techniques.**
  - Blocking is **learned from labelled matches**: it derives rules from which field values the matching pairs share. "Typical Zingg comparisons are 0.05–1% of the possible problem space".
  - A `verifyBlocking` phase reports block-size distributions, including the top 10% of records by block size.
  - **Stopword recommendation** "extracts 10% of the high-frequency unique words" of a field.
- **Adaptation (P3).** We already learn generic tokens from DF. A block-size audit (top 1% of posting lists per blocker) is worth adding to the recall audit in item 9.

### 15. Quantexa: Entity Quality Score and overlinking detection. PARTIAL
- URL: https://www.quantexa.com/blog/er-accelerate/ (VERIFIED: "Entity Quality Score" is a "machine learning model ... used to detect overlinking"). The community pages on the overlinking tool were blocked, so feature details are UNVERIFIED.
- **Adaptation (P1).** Train a **cluster-level overlinking classifier** on out-of-fold predictions for train S1s.
  - Target: "does the predicted target set of this S1 contain at least one false positive?"
  - Features:
    - cluster size;
    - min, mean and max of the target–S1 probabilities;
    - min sibling–sibling similarity;
    - density;
    - number of bridges;
    - number of distinct core-name keys and distinct house numbers among members;
    - share of S2 vs S3.
  - If the classifier fires, re-run the expected-F0.5 subset selection with the weakest member removed. This directly targets the precision half of per-S1 F0.5.

### 16. Google Cloud Enterprise Knowledge Graph: entity reconciliation confidence. VERIFIED
- URL: https://docs.cloud.google.com/enterprise-knowledge-graph/docs/confidence-score
- **Key technique.**
  - Confidence is **per node** (does the node belong to its cluster?), not per pair. It falls with distance from the cluster's members and falls further "if there are other clusters close to an entity".
  - It is adjusted by **cluster density**: dense clusters lower the confidence at a fixed distance.
  - Scores are bucketed to 0.1, and the docs say "do not depend on the exact confidence values".
- **Adaptation (P2).**
  - Add `t_density_norm_margin` = (sim(t, S1_best) − sim(t, S1_second)) / (the typical intra-cluster similarity of S1_best's current members).
  - Our `*_tmargin` features have no density normalization. A 0.05 margin means more in a loose cluster than in a tight one.

### 17. Amazon Science: Neural LSH for entity blocking (Wang, Kong, Tao, Borthwick, et al.; arXiv 2401.18064). VERIFIED (abstract)
- URLs:
  - https://arxiv.org/abs/2401.18064
  - PDF hosted at https://cdn.amazon.science/5c/4e/5d1dd29d45c4a8f0cb40603a5199/neural-locality-sensitive-hashing-for-entity-blocking.pdf
- **Key technique.** Trains deep networks as LSH hash functions for a task-specific similarity, with an LSH-based loss on top of a pre-trained LM. The abstract claims "significant performance improvements".
- **Adaptation (P3).** Only relevant if dense blocking becomes the bottleneck. Our sparse multi-view blocking already reaches 0.996 in a sample.

### 18. Amazon Science: "Pretraining and finetuning language models on geospatial networks for accurate address matching" (Maheshwary, Paul, Sohoney; EMNLP 2024). VERIFIED (abstract page)
- URL: https://www.amazon.science/publications/pretraining-and-finetuning-language-models-on-geospatial-networks-for-accurate-address-matching
- **Key technique.** Treat the addresses as an **address graph**, use geospatial proximity and weak supervision to build contextual training pairs, and train self-supervised. Reported "24.49% improvement in recall while maintaining 95% precision", and the model was deployed.
- **Adaptation (P3).** We have no coordinates, but (postcode, street root) co-membership gives a weak address graph. It could generate self-supervised pairs for fine-tuning the e5 address view. Check the rule on "augmentation": this uses only competition data.

### 19. AWS Entity Resolution: methodology docs. VERIFIED
- URLs:
  - https://aws.amazon.com/blogs/industries/resolve-imperfect-data-with-advanced-rule-based-fuzzy-matching-in-aws-entity-resolution/
  - https://aws.amazon.com/about-aws/whats-new/2026/09/entity-resolution-record-confidence/
- **Key techniques.**
  - Rule-based fuzzy matching: Levenshtein, Cosine and Soundex inside boolean rules, for example `Cosine(Name,0.6)` and `Levenshtein(Phone,5)`.
  - The ML workflow moved (September 2026) from a **group-level** confidence, where every record in a match group had the same score, to **record-level** confidence. The stated reason is that group-level scores made it "impossible to distinguish a near-certain match from a borderline one". It allows "differentiated thresholds".
- **Adaptation (P3).** This confirms our design choice: per-target probabilities feed a per-S1 subset decision. Do not collapse to one group-level score.

### 20. Plaid: transaction merchant parsing. VERIFIED
- URL: https://plaid.com/blog/how-plaid-parses-transaction-data/
- **Key technique.** A **rule-based light fuzzy path first** for clear formats, then a BERT-like MLM plus a BiLSTM NER for hard strings. Reports 95% correct identification of merchant and location "when present".
- **Adaptation (P3).** This is an efficiency pattern only: route exact or near-exact normalized-key matches through a fast path and spend GPU scorers on the ambiguous remainder.

### 21. SafeGraph patent US10877947B2: "Deduplication of metadata for places". VERIFIED
- URL: https://patents.google.com/patent/US10877947B2
- **Key technique.** Remove category-like stop words from place names; assign **brand labels** heuristically; block by geohash prefix, city or zip proximity; score with LSTM similarity over "diff vectors" (embedding distances, Levenshtein variants, geohash string distance).
- **Adaptation (P3).** "Brand label" = our high-`q_nm_freq_s1` core key. It is already covered once the locality counts from item 2 exist.

### 22. Foursquare Places: technical guide part 2. VERIFIED
- URL: https://foursquare.com/resources/blog/products/technical-guide-to-foursquare-places-part-2-how-does-foursquare-get-location-data-right/
- **Key techniques.** Match against existing POIs using key attributes as indexing criteria. A similarity model trained on multi-country, multi-language human labels. Clustering. Attribute values are picked by **"weighted mode"** (consensus voting within a cluster). Explicit chain and franchise IDs.
- **Adaptation (P2).** When building the entity-centric second pass in item 3, derive the entity's consensus house number, postcode and core name by weighted mode over the S1 and its confident targets. Then compare each candidate against the consensus as well as against S1. S1 itself can carry noise, such as a missing component.

### 23. Refinitiv/LSEG PermID Record Matching. UNVERIFIED (search summary only)
- URL: https://developers.lseg.com/en/api-catalog/open-perm-id/permid-record-matching-restful-api (not opened)
- Match levels as summarized: Excellent ≥ 0.9, Good ≥ 0.6, Possible ≥ 0.1. Low relevance (P3).

### 24. Moody's / Bureau van Dijk Orbis batch search. UNVERIFIED
- Per a library guide snippet: A–E confidence classes, with "A" matches auto-selected. No method detail was found. P3.

### 25. Tamr. VERIFIED (docs), low detail
- URL: https://cloud.docs.tamr.com/docs/understanding-clustering-rules
- Model-based clustering first, then **deterministic clustering rules** that force records together or apart, then curation. The docs give no algorithm detail.
- Adaptation (P3): a hard "cannot-link" rule layer (for example, conflicting house numbers *and* conflicting postcode prefixes → never accept) placed after the model. Use it only if validation shows the model occasionally accepts such pairs.

### Searched but not found or not usable
- **Uber**: no merchant or place dedup post found. One search result credited an embedding place-dedup pipeline to Uber, but the text was the Facebook WWW 2019 paper (item 10).
- **Walmart Global Tech**: an ER framework blog post exists, but it returned 403. UNVERIFIED.
- **Swiggy, Zomato, Flipkart, Paytm**: no public merchant or restaurant dedup posts found.
- **Airbnb, Stripe, Square**: no relevant public posts found.
- **Babel Street (Rosette)**: only marketing-level pages were found.
- **ONS**: Splink adoption was confirmed, but no business-register matching method document was found.

---

## Consolidated action list

| # | Action | Priority | Source items | Where |
|---|---|---|---|---|
| 1 | Country-normalized or size-scaled, log-rounded count features. Audit every raw count feature for train→test drift. | P0 | 5 (4th place), 11 | `features.py`, section F |
| 2 | Local uniqueness: `q_core_n_city` and `t_core_n_city` (S1 records with the same core key in the same city, backing off to state), plus generic-address counts `q_addr_n_s1` and `t_addr_n_t` | P0 | 2 (D&B Uniqueness/Density), 3 (Senzing FF / generic) | new section F2 |
| 3 | EM core/background name model with per-(country, city/state) background. Features: same-core probability (eq. 3), the DP version with learned edit ops, and core mass of unmatched tokens | P0 | 1 | new module, runs on Modal |
| 4 | Leave-one-country-out validation (US↔India) to find shift-fragile features and check decision-layer calibration before France | P0 | 11 | `VALIDATION_STRATEGY.md` |
| 5 | Blocking recall audit on a name-sim × address-sim grid and per noise type, with block-size (posting-list) audit | P0 | 9, 14 | `stage_block.py` diagnostics |
| 6 | Soft-TFIDF name similarity (JW ≥ 0.9 alignment, IDF or core-prob weights), plus Σ log local token frequency with "max TF on fuzzy" and a 1e-6 floor | P1 | 4, 12 | section E |
| 7 | Fuzzy generic and legal-suffix folding before core/IDF computation. Use soft weights, not deletion | P1 | 8 | normalization |
| 8 | Street-root address view (learned street-type tokens removed, spaces collapsed) as a feature and a blocking view. Postcode-prefix-gated zip score | P1 | 12, 2 | normalization, section E, blocking |
| 9 | French phonetic key (addok-fr rules reimplemented) for FR records' names and streets | P1 | 13 | normalization |
| 10 | Cluster post-processing: size-aware thresholds, bridge and low-centrality pruning inside each S1's target set, cluster-level overlinking classifier | P1 | 5, 4, 15 | decision layer |
| 11 | Entity-centric second pass: features against the S1 + confident siblings consensus (weighted mode) | P1 | 3, 22 | stacked stage 2 |
| 12 | Learn a postcode-prefix ↔ state/region/département name map from S1 co-occurrence only (a learned mini-gazetteer, no external data), to neutralize "region ↔ département" and "state code vs name" swaps | P1 | 2 (zip), our noise list | normalization |
| 13 | Density-normalized margin feature (node-vs-competing-cluster confidence) | P2 | 16 | section G |
| 14 | Segment-wise calibration or thresholds (field-presence segments) | P2 | 2 | decision layer |
| 15 | Aggregator-URL path-as-name handling audit | P2 | 7 | normalization |
| 16 | e5 fine-tune with pairwise contrastive loss, hard sampling and separate name/address towers | P2 | 10 | GPU on Modal |
| 17 | Two-model-agreement pseudo-labels for France recalibration (only if the rules allow it) | P2 | 9 | decision layer |
| 18 | L1 cheap pre-filter GBDT to allow more candidates per target | P2 | 5 | pipeline |
| 19 | Word-alignment LR stacked feature with "uncommon word" counts | P2 | 6 | section E |
| 20 | Neural LSH blocking; geospatial-graph address pretraining; rule fast-path | P3 | 17, 18, 20 | optional |

Compliance notes:
- Everything above is a *method*.
- Do not import curated lists: GLEIF/OCCRP legal forms, the addok synonyms file, or libpostal's dictionaries and trained models. libpostal's parser is trained on external OpenStreetMap data, so using the library itself would bring in external data.
- Learn every list from S1/S2/S3.
- Transductive, label-free statistics on test (EM, frequencies) match our existing policy of "label-free corpus statistics". Pseudo-labels from test predictions (action 17) need a rules check first.
