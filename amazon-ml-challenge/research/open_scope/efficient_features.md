# Open-scope sweep: efficient large-scale string similarity and feature engineering (efficient_features)

Date: 2026-09-25. Scope covered:
- RapidFuzz internals and bit-parallel / SIMD / GPU edit distance;
- soft token-set similarities beyond plain Monge-Elkan and SoftTFIDF (generalized Monge-Elkan, soft cardinality, fuzzy-token Jaccard);
- learned string similarity (logistic regression over sparse pair features, CRF edit distance, alignment-learned alias tables);
- char-level Siamese models for company names and for cross-script names;
- MinHash/LSH at the 10M scale for grouping near-duplicate targets;
- numeric-token handling in addresses;
- GBDT feature-set studies in EM (AutoML-EM and the EDBT AutoML study);
- LightGBM/CatBoost training tricks and ranking-style objectives (per-query softmax, query cross-entropy, lambdarank, XE-NDCG).

Rules followed. Every item was opened on its primary page (paper PDF or HTML, official docs, repo README or patent text) unless it is marked **UNVERIFIED**. Numbers are only those read on those pages. Figures read off plots are labelled "read from figure". Research informs **method only**: none of the external training corpora mentioned below (GeoNames, gene dictionaries, UK AddressBase, WDC) may enter our models.

What this sweep does not repeat. `notes/classical.md` already covers Cohen, Ravikumar & Fienberg 2003 (SoftTFIDF best, and Jaro-Winkler about 1/10 the cost of Monge-Elkan), plain Monge-Elkan and SoftTFIDF definitions, Christen 2006, MARLIN/Bilenko, Moreau et al. 2008, the RapidFuzz library table, and Mudgal et al. 2018 (DeepMatcher design space: 5.4 h vs 1.5 min, 87.9 vs 88.8 F1 on structured data). `open_scope/contests.md` covers rank and percentile features. `notes/blocking.md` covers datasketch MinHash LSH blocking. This file goes deeper on **how to compute features cheaply at 10M scale** and on **families and objectives not yet in the notes**.

Project context taken into account (from `FEATURE_CATALOG.md` and `ERROR_ANALYSIS.md`, 2026-09-25):
- Current CV is about 0.987 macro F0.5.
- The binary LightGBM is dominated by the many-to-one competition features.
- Cross-script, alias, acronym and handle cases are "essentially solved".
- The remaining FPs are **planted sibling distractors with nearby house numbers**.
- The remaining FNs are **undecidable ambiguity** (empty address plus a shared name).
- The Modal spend limit is hit, so CPU-cheap ideas rank higher.

The priorities below reflect that context.

---

## 0. TL;DR

1. **Compute cost is the binding constraint, and RapidFuzz already does the hard part.** It uses Hyyrö's bit-parallel Levenshtein, Damerau and LCS (References page) and SIMD kernels for short strings, but **inside `cdist`** (changelog 2.11.0: "add SIMD implementation … to improve performance for short strings in cdist"). The cheap wins are on our side, not in a new engine:
   - (a) dedupe identical normalized strings and **unique token pairs** before scoring;
   - (b) return `dtype=np.uint8` scores, because LightGBM's `max_bin` defaults to 255, so nothing is lost;
   - (c) pass `score_cutoff` where a floor is acceptable (3.6.0 banded LCS/Indel runs in `O((score_cutoff/64)*|s2|)`);
   - (d) benchmark `cpdist` against per-target grouped `cdist`, because only the latter is documented to hit the SIMD path.

   **P0 (efficiency bundle).**
2. **Compute one token×token similarity matrix per pair and derive six features from it.** The features: Monge-Elkan in both directions, generalized Monge-Elkan with m=2, fuzzy-token Jaccard at δ ∈ {0.8, 0.9}, soft cardinality with IDF weights, and SoftTFIDF. The evidence:
   - Jimenez et al. 2009: all curves peak at m > 1 on 12 name-matching sets.
   - Jimenez et al. 2010: soft cardinality SC3 averaged IAP 0.776 vs SoftTFIDF 0.760, and on the company-name set "Business" 0.776 vs 0.704.
   - Wang, Li & Feng 2011: fuzzy-Jaccard found 1,520 pairs at 93% precision vs 405 at 94% for Jaccard.

   With a global cache of unique token pairs, the JW computations drop from "per pair" to "per unique token pair". **P2** accuracy (typos and reorder are largely solved already), but near-free once the cache exists.
3. **Number features should separate "typo-like" from "nearby building".** Two sources back this:
   - A Salesforce patent (US 9,026,552 B2, company-location matching) penalizes "numeric differences as opposed to non-numeric differences". Its zip matcher models proximity ("93256 … similar to 93252") and transposition ("93256 and 39256") as **separate** tolerances. It learns digit-blurred alias rules ("md-dddd") counted by the **number of distinct supporting pairs**, not the raw sum.
   - Linacre (2025) scores addresses with **token rarity recomputed inside the candidate set** and a **penalty for tokens missing from this candidate that appear in other candidates**.

   Together these give concrete features aimed at our #1 FP source:
   - `num_digit_edit` versus `num_absdiff`;
   - "t's number matches another candidate but not this one";
   - number support among the targets retrieving the same S1.

   **P1.**
4. **Ranking objectives: use them as an extra stacked view, not as a replacement.** CatBoost ships `QuerySoftMax` (per-group softmax) and `QueryCrossEntropy(α) = (1−α)·LogLoss + α·LogLoss_group` with α default 0.95, on CPU and GPU. LightGBM ships `lambdarank` and `rank_xendcg` ("faster than and achieves the similar performance as lambdarank"). Our structure fits a **per-target softmax with an explicit "no-match" row** (each target has ≤ 1 true S1, and 26% have none). It produces exclusivity-aware probabilities that the expected-F0.5 layer can consume. The cheapest test needs no new objective: a `p_share = p(q,t)/Σ_q' p(q',t)` feature from OOF binary probabilities. **P1 for `p_share`, P2 for the softmax model.** Per-S1 lambdarank is **not** recommended as the main scorer, because F0.5 needs calibrated probabilities, not orderings.
5. **Learned string similarity from sparse pair identities (Tsuruoka et al. 2007) is the cheapest "learned metric".**
   - Method: logistic regression over *which* bigrams and tokens are shared or differ, plus prefix/suffix, acronym-subsequence and same-number flags.
   - Result: it beat SoftTFIDF at rank 1 (69.3% vs 62.5%) on BioCreAtIvE.
   - Adaptation: a hashed sparse logistic regression over (shared token), (unaligned token pair) and (number-conflict) identities, used as one OOF stacked feature. It learns which substitutions are benign ("Svcs"↔"Services") and which are fatal (number, "North"↔"South").

   **P2.**
6. **Char-level Siamese models work, but the classical baseline is strong.**
   - Basile et al. (ESWA 2024), company names: Jaro-Winkler alone reached 0.957 balanced accuracy on one test set. The Siamese LSTM reached 0.976 only with the large training size. The RF baseline plateaued before 400 labels, while the Siamese needed about 2,000.
   - Gan et al. (ICASSP 2017): a char-CNN beat BoC by about 17 points R@1 (98.90 vs 82.09).

   Given the spend limit and ≤ 5% cross-script errors: **P3**, except as a CPU-trainable tiny char-CNN if the budget reopens.
7. **Near-duplicate target grouping at 10M with MinHash is cheap and gives sibling-cluster evidence.** The toolbox:
   - one-permutation hashing with optimal densification (hashes in O(d+k) instead of O(dk));
   - weighted MinHash (Ioffe; datasketch), so generic tokens do not glue clusters;
   - union-find (text-dedup, Apache-2.0);
   - GPU `nvtext::minhash` in cuDF.

   Our use: cluster S2∪S3 targets with a **numeric guard**, then add cluster-level features (cluster size, fraction of the cluster whose top-1 S1 is q, number agreement with the cluster majority). A tight cluster that agrees on "345" is a sibling entity, not noise around "340". **P2.**
8. **GBDT training cost.** LightGBM quantized gradients (`use_quantized_grad`, `num_grad_quant_bins` default 4). The NeurIPS 2022 paper finds 2-3 bits enough "without hurting any performance", with "up to 2× speedup". **P1** as a switch-on for the next training run, compared on OOF F0.5.

---

## 1. Edit distance at scale: RapidFuzz internals, bit-parallel and SIMD kernels, alternative engines

### 1.1 Myers (1999), "A Fast Bit-Vector Algorithm for Approximate String Matching Based on Dynamic Programming"
- **What.** Journal of the ACM 46(3):395-415, May 1999. PDF read (page 1).
- **Technique.** It computes a bit representation of the *relocatable* DP matrix, so it runs in O(nm/w) time independent of k. The block version gives O(kn/w) expected time for arbitrary m.
- **Evidence.** The abstract says the algorithm is "found to be more efficient than the previous results for many choices of k and small m", and that the block version "yields a code that is either superior or competitive with all existing algorithms except for some filtration algorithms that are superior when k/m is sufficiently small".
- **Adaptation.** Nothing to implement. This is why a Levenshtein on strings of 64 characters or fewer costs O(n) word operations in RapidFuzz. Names and address lines are almost always ≤ 64 characters after normalization. So the cost per pair is dominated by **Python/array overhead, not DP**. Optimize the data movement (§1.6).

### 1.2 Hyyrö (2003) and Hyyrö (2004): the kernels RapidFuzz cites
- **What.** The RapidFuzz References page (read) lists three references:
  - [Hyyro03] "A bit-vector algorithm for computing Levenshtein and Damerau edit distances", Nordic J. of Computing, 2003;
  - [Hyy04] "Bit-parallel LCS-length computation revisited", AWOCA 2004;
  - [WF74] Wagner & Fischer.
- **Technique.** A Myers-style bit-vector for Levenshtein, plus Damerau (transpositions) and bit-parallel LCS. LCS underlies `fuzz.ratio`/`Indel` and `LCSseq`.
- **Adaptation.** Everything we use (ratio, token_sort, token_set, partial, JW, Levenshtein, OSA) already runs on these kernels. One cheap new feature is exposed by the LCS kernel (Tsuruoka's acronym feature, §3.1): `n_subseq_cov = LCSseq.similarity(short, long) / len(short)` on space-less names. It equals 1 exactly when the shorter string is a subsequence of the longer, which covers acronyms, dropped vowels, truncation and handles. **P2** (the acronym and handle cases are already solved per ERROR_ANALYSIS, so the gain is likely small).

### 1.3 Hyyrö, Fredriksson & Navarro (JEA 2005), "Increased Bit-Parallelism for Approximate and Multiple String Matching"
- **What.** ACM Journal of Experimental Algorithmics. The conference version is WEA'04, LNCS 3059. PDF read (pp. 1-2).
- **Technique.** When the pattern is much shorter than the machine word, "w − m bits in the computer words get unused". The paper packs **several short patterns into one word**: for r patterns of length m ≤ w/2 the time is O(⌈rm/w⌉n) instead of O(rn). It also extends the idea to "one-against-all computation of edit distance and longest common subsequences".
- **Evidence.** The abstract reports "significant speedups over the best existing alternatives especially on short patterns and moderate number of differences allowed".
- **Adaptation.** This is the principle behind RapidFuzz's SIMD `cdist` path for short strings (§1.4): **one query against many short choices** fills the vector lanes. Our workload is pairwise (cpdist). The lanes fill naturally only if we **group by one side**, for example one target against its ≤ 8 reverse-retrieved S1s, or one S1 against its ≤ 51 candidates. See §1.6 for the benchmark to run.

### 1.4 RapidFuzz changelog and `process` docs: what is actually vectorized
- **What.** CHANGELOG.rst on GitHub and the `rapidfuzz.process` docs (read).
- **Evidence (quoted entries).**
  - 2.11.0: "add SIMD implementation for `fuzz.ratio`/`fuzz.QRatio`/`Levenshtein`/`Indel`/`LCSseq`/`OSA` to improve performance for short strings in cdist".
  - 3.2.0: "build x86 with sse2/avx2 runtime detection".
  - 3.4.0: "add simd implementation for Jaro and Jaro Winkler".
  - 3.5.0: "improve performance of simd implementation for `LCS` / `Indel` / `Jaro` / `JaroWinkler`".
  - 3.6.0: "add banded implementation of LCS / Indel. This improves the runtime from `O((|s1|/64) * |s2|)` to `O((score_cutoff/64) * |s2|)`".
  - 3.8.0: "added `process.cpdist`".
  - 3.10.1: fixed incorrect SIMD Levenshtein/OSA results on 32-bit targets.
  - 3.13.0: "support for any DTypeLike as dtype in `cdist` and `cpdist`".
  - 3.14.2: "enable free threading".

  The docs list `score_cutoff`, `score_hint` ("Used to select a faster implementation"), `dtype` and `workers`. Workers are "only available for scorers using the RapidFuzz C-API", which release the GIL.
- **What is not documented.** Whether `cpdist` uses the multi-string SIMD path. Treat it as **unknown and benchmark it** before relying on it.
- **Adaptation (P0).**
  1. **`dtype=np.uint8`** for every 0-100 ratio feature. LightGBM `max_bin` defaults to 255 (Parameters page), and a 0-100 integer score has at most 101 distinct values, so binning loses nothing.
     - Own arithmetic: 90M pairs × 20 fuzz features is 1.8 GB as uint8 vs 14.4 GB as float64.
     - This also cuts parquet I/O on Modal.
  2. **`score_cutoff`** on the slow scorers (partial_ratio, WRatio, token_set on long address strings). Scores below the cutoff return 0. For a GBDT feature, a floor at, say, 40 is harmless because everything below 40 is "no match" for these views. Keep **no cutoff** on the main name ratio and JW, which the decision layer leans on.
  3. **Benchmark before switching** (1M pairs, on Modal CPU, not locally):
     - (a) `cpdist(q_strs, t_strs, workers=-1)`;
     - (b) the same pairs sorted by target and scored with one `cdist([t], cand_list)` per target;
     - (c) `cdist` over **unique** strings only, followed by a gather.

     Pick the fastest per scorer.

### 1.5 Alternative engines (method notes, no switch recommended)
- **edlib** (Šošić & Šikić, Bioinformatics 2017, DOI 10.1093/bioinformatics/btw753; GitHub README read, MIT).
  - Myers bit-vector with banding and alignment-path recovery; NW/HW/SHW modes.
  - The README says "characters spanning multiple chars/bytes are not supported". It needs a manual mapping for alphabets of 256 or fewer, and it "works best for large, similar sequences".
  - **Not useful for us.** Our strings are short and Unicode.
- **StringZilla** (GitHub README read; "Apache 2.0 or the Three-clause BSD license").
  - `stringzillas.LevenshteinDistances` does batch Levenshtein with UTF-8 codepoint support, on multi-core CPU and CUDA.
  - `Fingerprints` builds rolling-hash MinHash and Count-Min sketches in one pass.
  - README benchmark, "Levenshtein distances, ≅ 100 byte DNA, one core (MCUPS)": `rapidfuzz.process.cdist` 4,970-18,370, StringZilla.C 15,680-22,844, StringZilla.Py 14,130-21,930 (ranges across processors). That is DNA at 100 bytes, **not** our short multilingual strings.
  - **P3.** Only worth a look if Levenshtein becomes the bottleneck on GPU.
- **cuDF nvtext** (NVIDIA docs read).
  - `edit_distance(input, targets)` is **row-wise**: "`output[i]` is the edit distance between `input[i]` and `targets[i]`", with nulls treated as empty strings. That is exactly a GPU `cpdist` for Levenshtein.
  - `minhash`, `minhash64`, `minhash_ngrams` and `minhash64_ngrams` use MurmurHash3 with the permutation `((hv*a + b) % mp) & max_hash`, where mp = 2^61−1.
  - A row-wise character-ngram `jaccard_index` also exists. That came from a search snippet of its doc page, which was not opened: **UNVERIFIED detail**.
  - **P3** for features, since GPU spend is capped. **P2** for §5 (MinHash of 10M targets) if a GPU hour is available.

### 1.6 Our engineering recipe (own design, grounded in §1.1-1.4)
- **U1. Unique-string dedup.** Before any scorer, map every normalized name and address to an integer id with `pd.factorize`. Score unique (id_q, id_t) pairs, then gather.
  - Many S2/S3 copies normalize to identical strings (exact copies after case, space and diacritic folding). The dedup ratio is **unmeasured**, so measure it on one country first.
  - contests.md already lists this as a P2 compute saving (WBSG 1st place). Here it moves to **P0**, because Modal is capped.
- **U2. Unique token-pair cache** for all token-level soft measures (§2.4).
- **U3.** uint8 outputs and `workers=-1`, as above.
- **U4. Chunking.** Chunks of 5-10M pairs, written to parquet as uint8/float16. This is already the plan in classical.md; the only change is the dtype.

---

## 2. Soft token-set similarities beyond plain Monge-Elkan and SoftTFIDF

### 2.1 Jimenez, Becerra, Gelbukh et al. (CICLing 2009), "Generalized Mongue-Elkan Method for Approximate Text String Comparison"
- **What.** LNCS 5449. The author PDF was read, pp. 561-568.
- **Technique.** Monge-Elkan averages, over tokens of A, the best `sim'` against tokens of B. The paper replaces the arithmetic mean with a **generalized (power) mean**:
  `sim_ME_m(A,B) = ( (1/|A|) Σ_i ( max_j sim'(a_i, b_j) )^m )^(1/m)`.
  With m > 1, well-matched tokens are promoted. The paper also notes that ME is asymmetric, and that ME "approximates the solution to the optimal assignment problem" in O(|A|·|B|).
- **Setup.** 12 name-matching sets: Birds ×4, Business, Game-Demos, Parks, Restaurants, UCD-people, Animals, Hotels and Census. Internal measures: bigrams (Dice, padded), normalized edit distance and Jaro. m ∈ {0.00001, 0.5, 1, 1.5, 2, 5, 10}. Metrics: interpolated average precision (IAP) and F1, weighted by dataset size.
- **Evidence.**
  - Quoted: "values of m below 1 obtain lower performance than the baseline in both metrics. All curves reach their maximum performance at values of m above 1".
  - Read from figure: bigrams and edit distance peak around m ≈ 1.5-2. Jaro keeps improving up to larger m on F1.
  - Worked example: "Lenovo inc." vs "Lenovo corp." scores 0.5833 with standard ME and 0.7168 with m=2.
- **Adaptation.** Add `n_gme2_qt` and `n_gme2_tq` (m=2, JW inner) next to plain ME. Both come from the same token matrix (§2.4). Also add `n_me_min = min(ME_qt, ME_tq)` as a symmetric version. **P2.**

### 2.2 Jimenez, Gonzalez & Gelbukh (SPIRE 2010), "Text Comparison Using Soft Cardinality"
- **What.** LNCS 6393. The author PDF was read, pp. 297-302.
- **Technique.** The soft cardinality of a token set S under an affinity α (for example bigram similarity between tokens) is `|S|'_α = Σ_i w_i / Σ_j α(s_i, s_j)^p` with p ≥ 0. Near-duplicate tokens *inside* a set count less than 1 each. Soft |A∪B| is computed on the union, soft |A∩B| = |A|' + |B|' − |A∪B|', and any resemblance coefficient (Jaccard, cosine) then applies. Importance weights `w_i` (for example IDF) are decoupled from the affinity.
- **Evidence** (Table 1, IAP; same 12 sets as §2.1):

| variant | Avg IAP | Avg F1 |
|---|---|---|
| bigrams | 0.678 | 0.717 |
| cosine | 0.715 | 0.779 |
| SC1 (p=1, bigram affinity) | 0.740 | 0.776 |
| SC2 (p=2) | 0.764 | 0.809 |
| **SC3 (p=2, w=idf)** | **0.776** | **0.827** |
| SoftTFIDF (STI) | 0.760 | 0.808 |

  On **Business** (company names), IAP was SC3 0.776, SC2 0.732 and STI 0.704. The authors note that SC1 and SC2 are "static" (they use only the pair) and still match STI, which needs corpus statistics.
- **Adaptation.** Add `n_scard_cos` (SC3 with the country-level IDF we already compute, p=2, JW or bigram-Dice affinity) and `n_scard_jacc`. **Why it fits our noise:** injected generic tokens and duplicated tokens ("Services Services", "SC" plus "S.C.") inflate crisp cardinality but not soft cardinality. **P2.**

### 2.3 Wang, Li & Feng (ICDE 2011), "Fast-Join: An Efficient Method for Fuzzy Token Matching based String Similarity Join"
- **What.** The PDF was read (pp. 1-3, 9-10).
- **Technique.**
  - **Fuzzy overlap:** build a bipartite graph between the tokens of s1 and s2, with edge weight = normalized edit similarity, keeping edges ≥ δ. The overlap is the **maximum-weight matching**.
  - Then `FJACCARD_δ = |T1 ∩̃ T2| / (|T1| + |T2| − |T1 ∩̃ T2|)`, and similarly Fuzzy-Dice and Fuzzy-Cosine.
  - δ=1 reduces to plain Jaccard. A single-token pair with δ=0 reduces to edit similarity.
  - Unlike Chaudhuri's AGES/ME, the measure is **symmetric**, and each token is used at most once. So "wnba nba" vs "nba" is not a perfect match.
- **Evidence.**
  - Table II, 100k query-log strings, δ=0.8, τ=0.8: Jaccard returned 405 pairs at 94% precision; Fuzzy-Jaccard returned 1,520 pairs at 93%.
  - At τ=0.7, Fuzzy-Jaccard had 84% precision on 2,698 pairs.
  - Against hybrids at δ=0.8, τ=0.8: GES returned 486 pairs (97%) and AGES 25,017 pairs, with precision "only 6%". The authors conclude that fuzzy-Jaccard "has nearly the same recall with AGES, but achieves much higher precision".
  - With δ=0.75, edit similarity reached only 27% precision vs 90% for fuzzy-Jaccard (τ=0.8).
- **Adaptation.** Add `n_fjacc_80` and `n_fjacc_90`, using JW or normalized Levenshtein, with **greedy** max-weight matching (sort edges descending, take an edge if both endpoints are free). With ≤ 8 tokens, greedy matching is a close approximation of optimal matching. That is our assumption, not a claim from the paper. The one-to-one constraint is exactly what ME lacks: ME lets "Amber Amber Inc" fully cover "Amber Inc". **P2.**
- **Reference implementation.** py_stringmatching `GeneralizedJaccard` (Magellan's library; docs read) is "a softened version of the Jaccard measure", with default `sim_func` = Jaro and `threshold` 0.5. The docs do not say whether the pairing is greedy or optimal. The license is BSD according to the search result; the LICENSE file was not opened. Use it only as a test oracle, not in the pipeline.

### 2.4 The "one matrix, many features" recipe (own design; untested)
This is how to get ME×2, GME2×2, FJacc×2, SoftCard×2 and SoftTFIDF for about the cost of one JW pass over **unique token pairs**:

```
# 1. vocab: per-country token -> int id (already have token lists for IDF)
# 2. For every candidate pair p=(q,t): enumerate cross pairs (a,b), a in toks(q), b in toks(t), a!=b
#    key = min(a,b)<<32 | max(a,b);  keys = np.unique(all_keys)          # dedupe across ALL pairs
# 3. sims = rapidfuzz.process.cpdist(tok_str[keys>>32], tok_str[keys&0xffffffff],
#                                    scorer=JaroWinkler.normalized_similarity,
#                                    dtype=np.float32, workers=-1)
# 4. numba kernel over pairs (CSR token arrays), sims looked up by np.searchsorted(keys, key):
#       M = |A|x|B| matrix (1.0 on equal ids)
#       me_ab = mean(max over rows), me_ba = mean(max over cols)
#       gme2_ab = sqrt(mean(rowmax**2)),  gme2_ba = ...
#       greedy matching on edges >= delta -> fuzzy overlap -> fjacc_delta for delta in (0.8, 0.9)
#       softcard(S) = sum_i idf_i / sum_j M_S[i,j]**2  on A, B, A∪B -> sc_jacc, sc_cos
#       softtfidf(theta=0.9) with the same idf
```

- **Why the cache matters.** A cost of "per pair × |A|·|B|" JW calls becomes "per unique token pair". Name vocabularies are heavy-tailed, so the unique-pair count should be far smaller than the cross-pair count. **Unmeasured:** log both counts on a 1M-pair sample.
- **Where the features apply.** Names (core tokens and all tokens) and the **alphabetic part of the address**. Numbers are handled separately (§6); never pass them through JW.

---

## 3. Learned string similarity

### 3.1 Tsuruoka, McNaught, Tsujii & Ananiadou (Bioinformatics 23(20):2768-2774, 2007), "Learning string similarity measures for gene/protein name dictionary look-up using logistic regression"
- **What.** The OUP full text was read.
- **Technique.** Logistic regression over **sparse identity features** of a string pair:
  1. common character bigrams (as individual features), plus bigram similarity;
  2. prefix/suffix, "Up to three characters are extracted from the beginning of each string, and the combination of them are used as features";
  3. "a binary feature which indicates whether the strings contain the same number or not";
  4. acronym, "whether all the characters in the shorter string are included in the longer string in the same order";
  5. common tokens (the intersection as features);
  6. differing tokens (the symmetric difference as features);
  7. SoftTFIDF value.

  Training pairs are both synonymous and non-synonymous. Candidates are filtered by bigram similarity ≥ 0.7 or the subsequence condition. The paper "discarded three quarters of the non-synonymous samples by random sampling".
- **Evidence.**
  - BioCreAtIvE: "69.3% versus 62.5% at rank 1 and 76.8% versus 73.8% at rank 2" (LR vs SoftTFIDF).
  - Feature ablation, rank-1 recall drops across six dictionaries: character bigrams −4.1, −3.7, −2.2, −1.9, −1.7, −7.1; prefix/suffix −3.6, −3.0, −0.4, −1.0, −3.8, −3.2.
  - Caveat: the model is "highly tuned to the training dictionary" and generalized poorly across species. For us, that means **train per country** and watch France, which is unseen.
- **Adaptation (P2).** A **hashed sparse logistic regression** "token-substitution model", trained on our blocked train pairs and used as one OOF feature `lr_sub`. Features:
  - hash("shared:"+tok) for core tokens;
  - hash("diff:"+sorted(a,b)) for **aligned unequal token pairs** (alignment from the greedy matching in §2.4);
  - hash("drop_q:"+tok) and hash("drop_t:"+tok);
  - hash("numconf:"+role);
  - the acronym-subsequence flag.

  It learns the *identity* of benign vs fatal differences, which the GBDT cannot, since it sees only aggregate scores.
  - Tooling: `sklearn.linear_model.SGDClassifier(loss='log_loss')` over a `HashingVectorizer`-style CSR, CPU only.
  - France has no labels. Features built on *normalized* tokens transfer partly; the France-specific substitutions (R./BD/CH, N°) must come from the rule and abbreviation layer.

### 3.2 McCallum, Bellare & Pereira (UAI 2005), "A Conditional Random Field for Discriminatively-trained Finite-state String Edit Distance"
- **What.** arXiv 1207.1406. The abstract was read.
- **Technique.** A finite-state CRF over edit sequences. It is trained on **both positive and negative** pairs, which generative pair-HMMs (Ristad-Yianilos) are not. It allows "complex, arbitrary actions and features of the input strings". The edit alignment is latent.
- **Evidence.** The abstract says "positive experimental results on several data sets". No numbers were read.
- **Adaptation.** **P3.** It is the principled form of §3.1 and §3.3, but training is heavy and our typo noise is already handled. It is listed so we know that the cheap sparse-LR route is the approximation of this model.

### 3.3 Jagota / salesforce.com, US Patent 9,026,552 B2 (filed 2011, granted 2015-05-05), "System and method for linking contact records to company locations"
- **What.** A company-location matcher (Jigsaw). The patent PDF was read: pp. 1, 3-8, 15-20, 23-26.
- **Techniques (quoted or paraphrased from the text).**
  - **Field-type-specific fuzzy matchers.** The field scoring logic tolerates "some spelling errors, penalizing for numeric differences as opposed to non-numeric differences".
  - **Zip matcher tolerances**, each modelled separately:
    - proximity: "93256 is considered similar to 93252";
    - transposition: "93256 and 39256";
    - cross-country formatting: "1-12345, 112345, and I 12345 are all considered pairwise likely matches";
    - country patterns: "US zips 12345-6789 and 12345 are considered moderately high-scoring matches".
  - **Street prefix alignment.** A DP alignment of tokens ("words or numbers"). Unalignable or low-scoring tail pairs are discarded, for example the pair (220, 235) is dropped because "the suite numbers are different". The remaining unequal aligned pairs (Dr, Drive) and (Ste, Suite) are accumulated as **alias candidates**.
  - **Learned knowledge table.** The confidence is `tanh(n/c)`, where n is the number of occurrences and c ≥ 1 is tuned. Aliases are **country-scoped**; the example is Bombay→Mumbai for INDIA.
  - **Digit blurring to learn general rules.** All digits become 'd', so "md-2023"→"2023" becomes "md-dddd"→"dddd". The rule's count is "the number of different non-blurred alias→normalization pairs that formed this blurred pair, instead of the sum of the counts". Reason: "one high-scoring alias→normalization pair leads to a high confidence blurred rule" otherwise.
  - **Record-level rule.** Reject if the country mismatches. Reject if the street is "not very high". Accept if the city or zip strongly matches, or if the "aggregated field-level match scores are sufficiently large".
- **Adaptation.**
  - **(a) Learned abbreviation map (P1, cheap).**
    - Build the map from **aligned token pairs of true train pairs** (greedy alignment from §2.4), not only from co-occurrence.
    - Score each rule by the number of **distinct supporting S1 entities**, not the raw count. This stops one heavily-copied S1 from minting a bogus rule.
    - Add **digit-blurred rules** (`#ddd`→`ddd`, `No.ddd`→`ddd`, `ddd-A`→`dddA`, `ddd bis`→`dddB`) learned per country.
  - **(b) Separate number-proximity and number-typo features** (§6). **P1.**
  - **(c) Country-scoped aliases.** We already condition on country, because the labels agree on 100% of true pairs.

---

## 4. Char-level neural Siamese models for names

### 4.1 Gan, Singh, Joshi, He, Chen, Gao & Deng (ICASSP 2017), "Character-level Deep Conflation for Business Data Analytics"
- **What.** arXiv 1702.02640. The PDF was read, pp. 2-4.
- **Technique.** A shared char-level encoder (BiLSTM, or a CNN with filter widths {2,3,4} and 100 maps each, max-pooled) with cosine relevance. Training is a **softmax over 1 positive and J=50 random negatives**, `P(D+|Q) = exp(γR)/Σ exp(γR)`, with γ=10.
- **Evidence.**
  - Data: 10,000 manually annotated query/target pairs from a proprietary business dataset.
  - 10-fold CV, correct→misspelled, R@1: BoC 82.09, LSTM 86.66, **CNN 98.90**. In the reverse direction: 83.56, 87.63 and 99.25.
  - The authors' hypothesis: local n-gram order (CNN) matters more than global order (LSTM) for names.
  - CNN training took "around 45 minutes" for 20 passes on a K40.
- **Adaptation.** If a neural name view is ever added, use a **char-CNN, not an LSTM**, trained with **in-batch softmax over our blocked candidate lists** (hard negatives, rather than random ones). **P3** under the spend cap.

### 4.2 Basile, Crupi, Grasso, Mercanti, Regoli, Scarsi, Yang & Cosentini (arXiv 2303.05391; Expert Systems with Applications 238C, 2024), "Disambiguation of Company names via Deep Recurrent Networks"
- **What.** The arXiv HTML was read.
- **Technique.**
  - Char one-hot input (63 symbols, padded to 300) → LSTM(16) Siamese.
  - Distance features (L1, L2, L∞, cosine, |diff|) → an MLP.
  - Active learning (least-confident) for labelling.
- **Evidence** (balanced accuracy):
  - "RO" test set, large training: Levenshtein stump 0.641, **Jaro-Winkler stump 0.956**, RF on 9 string features 0.967, Siamese 0.976.
  - "JO" set, large: JW 0.687, RF 0.72, Siamese 0.903.
  - Small training: the Siamese (0.892) *loses* to JW (0.957) on RO.
  - Active learning: the Siamese needed about 2,000 labels to saturate; the RF "plateaued before 400".
- **Reading for us.**
  - A well-featured GBDT is a strong baseline for company names.
  - A char model adds value only on the hard slice (JO-like pairs) and only with thousands of labels.
  - We have millions of labels, but our hard slice is ambiguity and siblings, not surface form.
  - **P3.**

### 4.3 Gadd (arXiv 2601.06932, v4 2026-03-29), "Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching"
- **What.** The abstract page was read.
- **Technique.** Teacher-student distillation. The teacher learns from articulatory features of IPA transcriptions. The **char-level student** maps names from 20 scripts to a 128-d space, with no language identification or phonetic resources needed at inference.
- **Evidence.**
  - "Trained on 32.7 million triplet samples drawn from 67 million toponyms (GeoNames, Wikidata, Getty)".
  - Student on MEHDIE: Recall@1 **85.2%**, MRR **90.8%**. "An ablation using raw articulatory features alone yields only 45.0% MRR".
  - **External training data**: method only.
- **Adaptation.** Distill **our own** teacher, the rule transliteration plus the learned native→Latin dictionary, into a tiny char-CNN student trained on train cross-script pairs only. The student would help on unseen native tokens that the dictionary misses. ERROR_ANALYSIS puts cross-script at ≤ 5% of errors (FP 160, FN 441 on fold 0), so this is **P3**.

---

## 5. MinHash/LSH for near-duplicate target grouping at 10M

### 5.1 Li, Owen & Zhang (NIPS 2012), "One Permutation Hashing"
- **What.** arXiv 1208.1259; the abstract was read.
- **Technique.** Permute once, split the columns into k bins, and keep the minimum per bin. This replaces k independent permutations. Classic minwise hashing "requires applying (e.g.,) k=200 to 500 permutations".
- **Evidence.** Preprocessing cost drops to 1/k. Accuracy is "similar (or even better)" than k-permutation. On news20 OPH "noticeably outperforms the k-permutation scheme when k is not too small".

### 5.2 Shrivastava (ICML 2017, PMLR 70), "Optimal Densification for Fast and Accurate Minwise Hashing"
- **What.** arXiv 1703.04664; the abstract was read.
- **Technique and evidence.** Densification fills the empty OPH bins so that the LSH property holds, at a cost of O(d + k) instead of O(dk). The paper's scheme is "variance-optimal" and yields "the same variance and collision probability as minwise hashing". Earlier densification had "unnecessarily high" variance, "especially when the data is sparse".
- **Why it matters.** Our records are *very* sparse: d is about 10-40 shingles per record. That is exactly the regime where naive densification hurts.

### 5.3 Weighted MinHash (Ioffe), as implemented in datasketch
- **What.** The datasketch `WeightedMinHashGenerator` docs were read.
- **Technique.** It estimates **weighted Jaccard** on weighted vectors without expanding by weight. The docs say "Weighted MinHash is created by Sergey Ioffe, and its performance does not depend on the weights". The default `sample_size` is 256, and the result can be indexed with MinHashLSH or LSHForest.
- **Adaptation.** Use IDF-weighted shingle or token vectors, so that generic tokens ("Services", "Center", legal forms) and common address words cannot by themselves put two different businesses in one bucket.

### 5.4 text-dedup (GitHub ChenghaoMou/text-dedup, Apache-2.0)
- **What.** The README was read.
- **Technique.** MinHash + MinHashLSH, followed by **union-find** clustering. Example config: `num_perm = 240`, `threshold = 0.7`, `ngram_size = 5`, 64- or 128-bit hashes.
- **Evidence (README benchmarks).** On pinecone/core-2020-05-10, MinHash scored Macro F1 0.9518 and accuracy 0.9277 in 11.09 s. On NEWS-COPY it scored ARI 0.7293 in 3.01 s.

### 5.5 Our target-grouping design (P2; own design, grounded in §5.1-5.4 and §6.2)
- **Goal.** Cluster the S2∪S3 targets (10M train or test) into near-duplicate groups, so that:
  - (i) planted **sibling entities** show up as their own tight clusters;
  - (ii) the match-exists model sees whether a distractor's copies cluster **away** from every S1.
- **Signature.** Char 3-grams of normalized name core plus street tokens, with **numbers excluded from the signature**. Numbers are then used as a **guard edge filter**: two targets join only if their house numbers are equal **or** one side's number is missing.
  - Without the guard, "Amber Inc 340" and "Amber Inc 345" merge.
  - With it, sibling clusters stay apart.
  - True copies with number noise (104 vs 103) may split into singletons. That is acceptable, because the features below are soft.
- **Hashing.** OPH with optimal densification, or cuDF `minhash64_ngrams` on GPU; k=128.
  - Banding (own arithmetic, the standard S-curve threshold ≈ (1/b)^(1/r)):
    - b=32, r=4 gives about 0.42;
    - b=16, r=8 gives about 0.71.
  - Use b=16, r=8, followed by an **exact verification** of each candidate edge with RapidFuzz on the pair, then union-find.
  - Memory (own arithmetic): 10M × 128 × uint32 = 5.1 GB of signatures, or stream per country and never hold them all.
- **Features.**
  - `tc_size`: cluster size;
  - `tc_frac_top1_q`: fraction of cluster members whose top-1 S1 is q;
  - `tc_num_major`: the cluster's majority house number;
  - `tc_num_eq_q`: whether it equals q's number;
  - `tc_src_mix`: S2/S3 mix;
  - `tc_max_p1`: the max stage-1 p(q, member).
- **Relation to the existing collective features.** The existing sibling-support stage looks only **inside q's candidate list**. These clusters are global, so they also see copies that retrieved a *different* S1.
- **Cheaper first step** (no LSH): compute the same statistics with clusters defined as "targets sharing the same top-1 S1 and the same house number". If that already captures the gain, skip LSH.

---

## 6. Numeric tokens in addresses

### 6.1 Evidence collected
- **Salesforce patent (§3.3).** Numeric differences are penalized more than text differences. Proximity and transposition tolerances are modelled separately. Suite-number mismatches are dropped from alignment (they are treated as tail, not as evidence). Digit blurring is used for format rules.
- **Linacre, "Building Accurate Address Matching Systems" (blog, 2025-06-29).** Read. The author builds Splink and uk_address_matcher. Quoted or paraphrased:
  - Tokens are weighted by rarity, and "within candidate sets, token frequencies are recalculated to measure discriminative power among specific candidates rather than globally".
  - There is a "penalty … for missing tokens that appear in other candidates".
  - Discriminating tokens among neighbouring addresses are found "by working back to front and eliminating common tokens".
  - Confidence uses the "score differential between top and second-best candidates". The companion library outputs this as a `distinguishability` column.
  - The blog gives no accuracy numbers.
- **uk_address_matcher (GitHub, MIT per README).** Multi-stage: exact matches first, then Splink probabilistic matching. Output includes `match_reason`, `match_weight` and `distinguishability`. The README claims "Match 100,000 addresses in ~30 seconds" for small areas.
  - PR #502, "missingness-aware sub-premise feature derivation", is from a **search snippet only (UNVERIFIED)**. Per the snippet, it keeps "a marker token, an explicitly known role, and a sub-premise identifier" and avoids "promoting an ambiguous leading number to a confidently typed flat".
- **Tsuruoka 2007 (§3.1).** A binary "same number" feature, because "Numbers in the names often convey important information".
- **ONS Working Paper 17** (Address Index; page read). A CRF parser with 17 feature categories, including digit presence (all, some, none). Elasticsearch candidates, a bespoke score and a score ratio. Top-match rate on baseline sets 97.50% (193,681), wrong 1.34% (2,659), not found 0.03% (51).
- **JointMatcher** (Ye et al., Knowledge-Based Systems 251, 2022). **UNVERIFIED.** The ScienceDirect page returned 403. The claims come from search snippets: a "numerically-aware encoder" that emphasizes number-containing segments, plus a relevance-aware encoder. No numbers were read.

### 6.2 Concrete feature upgrades (P1 unless noted; check against the v2 "house-number geometry" features before coding)
ERROR_ANALYSIS 2026-09-25 sets the target. FPs are sibling distractors with nearby numbers ("Amber Inc 340 vs 345", "Quex 1927 vs 1931"). True copies also carry number noise ("Downtown Deli 104 vs 103").
1. **Typo-likeness vs offset-likeness** of the house-number pair (both present):
   - `hn_absdiff` = |int(a) − int(b)| (capped);
   - `hn_reldiff`;
   - `hn_dl_edit` = Damerau-Levenshtein on the digit strings (RapidFuzz `DamerauLevenshtein`);
   - `hn_same_len`;
   - `hn_same_prefix_len` (common leading digits).

   A digit typo or transposition gives edit 1 with arbitrary |Δ| (1927→1972). A nearby-building sibling gives small |Δ| with edit ≥ 1 and possibly 2 digits changed (1927→1931). Only the *joint* distribution separates them, and the GBDT can learn it if both are present. This follows the patent's separate proximity and transposition tolerances.
2. **Candidate-set distinguishing numbers** (Linacre "missing tokens that appear in other candidates"). For pair (q, t):
   - `hn_eq_other_cand` = t's house number equals the number of *another* S1 in t's candidate list but not q's number. This should be a strong negative.
   - `hn_unique_in_cands` = q's number is unique among t's candidates and equals t's number. This should be a strong positive.

   Both are computed on the existing candidate graph, O(total candidates).
3. **Number support among q's claimants.** Among all targets whose candidate list contains q:
   - `hn_support_q` = share with number == q's;
   - `hn_support_t` = share with number == t's.

   If t's number has its own support cluster (≥ 2 other targets), t is probably a copy of a sibling entity. This extends the collective stage and overlaps with §5.5.
4. **Missingness-aware triplet** instead of one Jaccard: `hn_state` ∈ {both-equal, both-conflict, q-missing, t-missing, both-missing}, as a categorical feature (uk_address_matcher PR #502 idea, UNVERIFIED). "Missing" must never be encoded like "conflict".
5. **Number roles**, per country: house number, unit/suite, postcode (US 5-digit, India 6-digit PIN, France 5-digit CP) and other.
   - Compare within role only. Suite mismatches are down-weighted, as in the patent's discarded tail pair (220, 235).
   - France: "12 bis/ter" → house number "12B"/"12T"; "N°12" → "12".
   - This is the "Learned per-country house-number position and postcode shape features" item in FEATURE_CATALOG §H. The patent's digit-blurred alias rules (§3.3a) are the learning mechanism.
6. **Name-side numbers** ("Pipefitters Local 836", "7-Eleven", "3M"): a separate `n_num_state` with the same 5 states. Local and chapter numbers are the only disambiguator in the FN ambiguity bucket, so a conflict there is fatal. **P1.**
7. Numbers must stay excluded from leetspeak folding and JW (already the rule in classical.md).

---

## 7. GBDT for EM: what feature-set studies say, and training cost

### 7.1 Wang, Zheng, Wang & Pei (ICDE 2021), "Automating Entity Matching Model Development" (AutoML-EM)
- **What.** The author PDF was read, pp. 1-9.
- **Findings.**
  - **Magellan's rule-based feature generation** assigns similarity functions by attribute type and average length (Table I; for example "Long String (>10 words)" gets only Cosine-space and Jaccard-3gram). AutoML-EM instead applies **all 16 string similarity functions** regardless of length (Table II) and lets auto-sklearn select.
  - Figure 9: the feature count rose (for example Abt-Buy 15→72), and F1 rose on every dataset (Abt-Buy 48.1→59.2, iTunes-Amazon 88.1→96.3).
  - Takeaway, quoted: "It is recommended to use all similarity functions (rather than manually select them based on string length) to generate feature vectors."
  - **Tuning sensitivity** on Abt-Buy: max_features swung F1 by **10.08%**, the number of selected features by **13.99%**, and RobustScaler q_min by 1.17%.
  - End-to-end: average F1 went from 78.1 (Magellan) to 83.9 (AutoML-EM), +5.8, with an RF-only model space. Examples: Amazon-Google 49.1→66.4, Walmart-Amazon 71.9→78.5.
  - It was competitive with DeepMatcher on structured data but below it on textual data (Abt-Buy 58.1 vs 62.8, Amazon-Google 63.8 vs 69.3).
  - An RF-only model space "converged faster". For example, Abt-Buy needed 1200 s vs 6000 s for 62% F1.
- **Adaptation.**
  - **(a)** Our length-agnostic feature set is already the right design. Do **group ablations**, not single-feature ones: drop §D fuzz, §E token, §G competition and the new §2.4 soft family one group at a time, scored on **per-S1 F0.5 OOF**, not AUC. The 13.99% swing shows feature-set choices are first-order. **P2.**
  - **(b)** Prune groups that do not move F0.5 to cut Modal feature cost.

### 7.2 Paganelli, Del Buono, Guerra, Pevarello & Vincini (EDBT 2021, short paper), "Automated Machine Learning for Entity Matching Tasks"
- **What.** The OpenProceedings PDF was read, pp. 325-330.
- **Findings.**
  - Out-of-the-box AutoML (AutoSklearn, AutoGluon, H2O) on raw EM pairs averaged F1 of "48.66% for AutoGluon, to 51.4% for H2OAutoML, and 52.33% for AutoSklearn". They are "not competitive" with DeepMatcher.
  - Adding an "EM adapter" (transformer-embedding pair encoding) raised average F1 by 24.96%, 28.02% and 23.6%.
  - Within a 2% tolerance, adapted AutoML matched or beat DeepMatcher on 9 of 12 datasets at 1 h, and 11 of 12 at 6 h.
- **Reading for us.** The **pair representation** dominates the model search. This confirms we should spend on features and objectives, not on AutoML or hyper-search. **Informational.**

### 7.3 Shi, Ke, Chen, Zheng & Liu (NeurIPS 2022), "Quantized Training of Gradient Boosting Decision Trees", and LightGBM parameters
- **What.** arXiv 2207.09682 (abstract read) and the LightGBM Parameters page (read).
- **Evidence.** "the necessary precisions of gradients without hurting any performance can be quite low, e.g., 2 or 3 bits". There is "up to 2× speedup … compared with SOTA GBDT systems". The code went into LightGBM.
  - Params: `use_quantized_grad` (default false) and `num_grad_quant_bins` (default 4).
  - Also: `max_bin` 255, `min_data_in_bin` 3, `is_unbalance` and `scale_pos_weight` for class balance, and `linear_tree`.
- **Adaptation (P1).** Turn on `use_quantized_grad=true` for the next full-train run with ~90 features over tens of millions of pairs. Compare OOF F0.5 and wall-time against the current run. It is a single flag. **Unverified:** interaction with custom objectives (§8.4), so test separately.

---

## 8. Ranking-style objectives for many-to-one EM

### 8.1 CatBoost ranking losses (official docs, read; CatBoost LICENSE is Apache-2.0, read)
| loss | definition (quoted or paraphrased) | defaults | where |
|---|---|---|---|
| **QueryCrossEntropy(α)** | `(1 − α)·LogLoss + α·LogLoss_group` | α = **0.95** | CPU & GPU |
| **QuerySoftMax** | `−Σ_group Σ_i w_i t_i log( w_i e^{β a_i} / Σ_j w_j e^{β a_j} ) / Σ w_i t_i` | β = 1 ("input scale coefficient") | CPU & GPU |
| YetiRank / YetiRankPairwise | ranking-metric approximation. Modes Classic/DCG/NDCG/MRR/ERR/MAP ("Non-Classic modes are supported only on CPU") | permutations 10, decay 0.85 | GPU: Classic only |
| PairLogit | pairwise logistic loss over (winner, loser) pairs within a group. "The object weights are not used … The weights of object pairs are used instead" | max_pairs = all | CPU & GPU |
| LambdaMart | directly optimizes NDCG/DCG/MRR/ERR/MAP | sigma 1.0, norm True | CPU only |

A search-result snippet of the CatBoost "Common parameters" page says that for QueryCrossEntropy, YetiRankPairwise and PairLogitPairwise, if bagging_temperature is not set, Bernoulli bootstrap is used with subsample 0.5. The page itself was not opened: **UNVERIFIED detail**.

### 8.2 LightGBM ranking (Parameters page, read; ffineis blog, read)
- **Parameters.**
  - `lambdarank`: labels must be smaller than the length of `label_gain`.
  - `rank_xendcg`: "rank_xendcg is faster than and achieves the similar performance as lambdarank".
  - `lambdarank_truncation_level`: default 30.
  - `lambdarank_norm`: default true, "normalize the lambdas for different queries, and improve the performance for unbalanced data".
  - `label_gain`: default 0,1,3,7,…
  - `eval_at`: default 1,2,3,4,5.
- **Internals (Fineis blog).** The lambdarank gradient loops pairs up to the truncation level, weighted by |ΔNDCG|. Quoted: "the `lambdarank` LightGBM objective is at its core just a manipulation of the standard binary classification objective".

### 8.3 Bruch (WWW 2021), "An Alternative Cross Entropy Loss for Learning-to-Rank" (XE-NDCG)
- **What.** arXiv 1911.09798; the abstract was read.
- **Evidence.** A cross-entropy listwise loss that is "a convex bound on NDCG" and "consistent with NDCG". Implemented with GBDTs, it shows "superiority … over existing algorithms in quality and robustness". This is `rank_xendcg` in LightGBM.

### 8.4 Adaptation to our structure (own analysis)
**The structure.** Each target t matches **≤ 1** S1 (many-to-one), and 26% match none. The metric is per-S1 F0.5 macro-averaged. So there are two natural "queries":
- **Per-target group (exclusivity).** Candidates are t's retrieved S1s (the reverse top-8 plus forward hits). There is at most one positive. This is a **softmax** problem with a "none" option.
- **Per-S1 group (the metric's unit).** Candidates are the targets in q's list, and the metric needs a *set*, not a ranking. Lambdarank or xendcg would give orderings. Our expected-F0.5 layer needs **calibrated marginal probabilities**, so a ranking objective per S1 is at best a feature.

**Plan (cheapest first).**
1. **`p_share` feature, P1, no new model.** From stage-1 OOF binary probabilities, compute `p_share(q,t) = p(q,t) / (Σ_{q'∈C(t)} p(q',t) + ε)`, plus `p_none_hat(t) = 1 − max(Σ p, 1)`-style variants. Stack into stage 2.
   - It encodes soft exclusivity, whereas the existing `tmargin`/`trank` are hard ranks and gaps.
   - ERROR_ANALYSIS says hard exclusivity adds about 0, so expect a small gain, if any. The test costs one column.
2. **Per-target softmax with a null row, P2.** For each target t, add one synthetic row "t→NONE":
   - label 1 iff t has no true S1 (distractor);
   - pair features NaN;
   - `is_null=1`;
   - target-only features copied (script, source, best retrieval score, |C(t)|, cluster features from §5.5).

   Train either:
   - (a) **CatBoost `QuerySoftMax`** with `group_id = t`; or
   - (b) a **LightGBM custom objective**. The Python API accepts a callable `objective(preds, train_data) -> (grad, hess)` in `params`, and receives **raw** scores (lightgbm.train docs, read).

   ```python
   # untested sketch: rows sorted by target; starts = group start offsets, sizes = group sizes
   def group_softmax(preds, data):
       y = data.get_label()
       m = np.maximum.reduceat(preds, starts); z = preds - np.repeat(m, sizes)
       e = np.exp(z); s = np.add.reduceat(e, starts); p = e / np.repeat(s, sizes)
       return p - y, np.maximum(p * (1 - p), 1e-6)
   ```

   Output per candidate: P(t→q) with Σ_q P + P(NONE) = 1. That is exactly the exclusivity-aware marginal the expected-F0.5 layer wants. Blend with, or stack on, the binary p. Recalibrate with isotonic regression on OOF.
   - Groups whose true S1 was not retrieved (blocking miss, about 0.4%) should be **dropped** from training, not relabelled NONE.
3. **CatBoost `QueryCrossEntropy` (α=0.95), P2.** A single model that keeps a pointwise LogLoss term (calibration) while adding the group term. Use it as a second stage-1 model, grouped by target, for the blend.
4. **Per-S1 `rank_xendcg`, P3.** Only as an extra stacked feature (within-S1 relevance), never as the decision input.

**Kaggle usage (UNVERIFIED).** In the H&M 6th place write-up (the Kaggle page body did not render), LightGBM lambdarank, LightGBM xendcg, XGBoost lambdarank and CatBoost ranker were blended, per a search snippet. OTTO 15th and 20th used LGBMRanker together with a CatBoost classifier and a CatBoost YetiRank ranker, also per a search snippet. Both are recommender tasks, and no ranker-vs-binary comparison numbers were read.

---

## 9. Prioritized action list

| # | action | priority | why / evidence |
|---|---|---|---|
| A1 | Unique-string and unique-token-pair dedup before all string scorers. `dtype=np.uint8` for 0-100 scores. `workers=-1`. `score_cutoff` floors only on slow secondary scorers | **P0** | Modal spend cap. RapidFuzz 3.13 dtype support, 3.6 banded LCS with score_cutoff. LightGBM max_bin 255 makes uint8 lossless |
| A2 | Micro-benchmark `cpdist` vs per-target grouped `cdist` vs unique-string `cdist` + gather (1M pairs, Modal CPU) | **P0** | SIMD is documented only for `cdist` on short strings (2.11.0, 3.4.0, 3.5.0). The multi-pattern packing principle is from Hyyrö-Fredriksson-Navarro 2005 |
| A3 | House-number typo-vs-offset features (`hn_absdiff`, `hn_dl_edit`, `hn_same_len`, `hn_same_prefix_len`) and candidate-set distinguishing numbers (`hn_eq_other_cand`, `hn_unique_in_cands`) | **P1** | #1 FP source is sibling distractors. Patent US 9,026,552 models proximity and transposition separately and penalizes numeric differences. Linacre's within-candidate rarity and missing-token penalty |
| A4 | Name-side number state (`n_num_state`, 5 states) and missingness-aware address number state (`hn_state`) | **P1** | FN bucket "Pipefitters Local 836". Tsuruoka's "same number" flag. uk_address_matcher missingness (UNVERIFIED) |
| A5 | `p_share` soft-exclusivity feature from OOF stage-1 probabilities | **P1** | Cheapest test of per-target softmax logic. Many-to-one structure |
| A6 | `use_quantized_grad=true` on the next full LightGBM run | **P1** | Shi et al. 2022: 2-3 bits suffice, up to 2× speedup |
| A7 | Learned abbreviation-map upgrade: alignment-derived pairs, distinct-entity support counting, digit-blurred rules per country (FR: bis/ter, N°) | **P1** | Patent US 9,026,552: tanh(n/c) confidence, blurred rules counted by distinct pairs |
| A8 | Number-support features among q's claimants (`hn_support_q`, `hn_support_t`) | **P1** | Sibling clusters. Extends the collective stage |
| A9 | One-matrix soft token features: ME×2, GME m=2 ×2, fuzzy-Jaccard δ∈{0.8,0.9}, soft cardinality (IDF, p=2), SoftTFIDF θ=0.9, via the unique token-pair cache + numba | **P2** | Jimenez 2009 (peaks at m>1), Jimenez 2010 (SC3 0.776 vs STI 0.760 IAP; Business 0.776 vs 0.704), Fast-Join (1,520 pairs at 93% vs 405 at 94%) |
| A10 | Per-target softmax with a NONE row (CatBoost QuerySoftMax or LightGBM custom objective), blended or stacked with the binary model, isotonic-recalibrated | **P2** | CatBoost docs. Exclusivity plus 26% distractors |
| A11 | CatBoost QueryCrossEntropy (α=0.95) as a second stage-1 model grouped by target | **P2** | CatBoost docs (CPU & GPU) |
| A12 | Global target clustering (OPH/densified or weighted MinHash, b=16 r=8, numeric guard, union-find) with cluster features. First try the no-LSH proxy "same top-1 S1 and same number" | **P2** | Li-Owen-Zhang 2012, Shrivastava 2017, Ioffe WMH (datasketch), text-dedup union-find |
| A13 | Hashed sparse LR "token-substitution" OOF feature | **P2** | Tsuruoka 2007: 69.3 vs 62.5 R@1 over SoftTFIDF. Bigram and prefix features matter most in the ablation |
| A14 | Group ablation of feature families on per-S1 F0.5 OOF; prune groups that do not pay for their Modal cost | **P2** | AutoML-EM: feature selection swung F1 by 13.99% on Abt-Buy; all-functions generation beat Magellan's rules |
| A15 | `n_subseq_cov` (LCSseq coverage of the shorter string) | **P2** | Tsuruoka acronym-subsequence feature; Hyyrö 2004 bit-parallel LCS makes it cheap |
| A16 | Per-S1 `rank_xendcg` model as a stacked feature only | **P3** | Bruch 2021; LightGBM docs (xendcg faster than lambdarank) |
| A17 | Char-CNN Siamese name view (in-batch softmax over blocked candidates) or a distilled cross-script student | **P3** | Gan 2017 (CNN 98.90 vs BoC 82.09 R@1), Basile 2024 (JW baseline 0.956; Siamese needs thousands of labels), Symphonym 2026. Spend cap; cross-script ≤ 5% of errors |
| A18 | GPU engines (cuDF `edit_distance`/`minhash`, StringZilla) | **P3** | Only if a GPU hour is available. RapidFuzz is sufficient on CPU for short strings |
| A19 | CRF edit distance (McCallum 2005) | **P3** | Principled, but heavy. A13 is the cheap approximation |

---

## 10. Source ledger

| # | item | venue / type | URL | verification |
|---|---|---|---|---|
| 1 | RapidFuzz References | docs | https://rapidfuzz.github.io/RapidFuzz/References.html | page read |
| 2 | RapidFuzz CHANGELOG | GitHub | https://github.com/rapidfuzz/RapidFuzz/blob/main/CHANGELOG.rst | page read |
| 3 | RapidFuzz process (cdist/cpdist) | docs | https://rapidfuzz.github.io/RapidFuzz/Usage/process.html | page read |
| 4 | Myers 1999, bit-vector approximate matching | JACM 46(3):395-415 | https://dl.acm.org/doi/10.1145/316542.316550 (PDF: https://www.gersteinlab.org/courses/452/09-spring/pdf/Myers.pdf) | PDF p.1 read (ACM page 403) |
| 5 | Hyyrö, Fredriksson, Navarro, Increased bit-parallelism | ACM JEA 2005 (WEA'04) | https://users.dcc.uchile.cl/~gnavarro/ps/jea06.pdf | PDF pp.1-2 read |
| 6 | edlib | GitHub (MIT); Bioinformatics 2017 | https://github.com/Martinsos/edlib | README read (paper not opened) |
| 7 | StringZilla | GitHub (Apache-2.0 / BSD-3) | https://github.com/ashvardanian/stringzilla | README read |
| 8 | cuDF nvtext minhash / edit_distance | NVIDIA docs | https://docs.nvidia.com/cudf/26.08/libcudf/api_docs/nvtext_minhash/index.html ; https://docs.nvidia.com/cudf/latest/libcudf/api_docs/nvtext_edit_distance/index.html | pages read (jaccard page UNVERIFIED; cuDF license not re-checked) |
| 9 | Jimenez et al., Generalized Monge-Elkan | CICLing 2009, LNCS 5449 | https://www.gelbukh.com/CV/Publications/2009/Generalized%20Mongue-Elkan%20Method%20for%20Approximate%20Text%20String.pdf | PDF pp.561-568 read |
| 10 | Jimenez, Gonzalez, Gelbukh, Soft cardinality | SPIRE 2010, LNCS 6393 | http://www.gelbukh.com/CV/Publications/2010/Text%20Comparison%20Using%20Soft%20Cardinality.pdf | PDF pp.297-302 read |
| 11 | Wang, Li, Feng, Fast-Join (fuzzy-token similarity) | ICDE 2011 | https://www2.cs.sfu.ca/~jnwang/papers/icde2011-fastjoin.pdf | PDF pp.1-3, 9-10 read |
| 12 | py_stringmatching GeneralizedJaccard | docs | https://anhaidgroup.github.io/py_stringmatching/v0.4.x/GeneralizedJaccard.html | page read (license from search result only) |
| 13 | Tsuruoka et al., LR string similarity | Bioinformatics 23(20):2768-2774, 2007 | https://academic.oup.com/bioinformatics/article/23/20/2768/229308 | full text read (two passes; only exact quoted numbers used) |
| 14 | McCallum, Bellare, Pereira, CRF string edit distance | UAI 2005 | https://arxiv.org/abs/1207.1406 | abstract read |
| 15 | Jagota / salesforce.com, US 9,026,552 B2 | patent (2015) | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9026552 | pp.1, 3-8, 15-20, 23-26 read |
| 16 | Gan et al., Character-level Deep Conflation | ICASSP 2017 | https://arxiv.org/abs/1702.02640 | PDF pp.2-4 read |
| 17 | Basile et al., Company-name disambiguation via Siamese LSTM | arXiv 2023; ESWA 238C (2024) | https://arxiv.org/abs/2303.05391 | HTML read |
| 18 | Gadd, Symphonym | arXiv 2601.06932 (2026) | https://arxiv.org/abs/2601.06932 | abstract read |
| 19 | Li, Owen, Zhang, One Permutation Hashing | NIPS 2012 | https://arxiv.org/abs/1208.1259 | abstract read |
| 20 | Shrivastava, Optimal Densification | ICML 2017 | https://arxiv.org/abs/1703.04664 | abstract read |
| 21 | datasketch WeightedMinHash (Ioffe) | docs | https://ekzhu.com/datasketch/weightedminhash.html | page read |
| 22 | text-dedup | GitHub (Apache-2.0) | https://github.com/ChenghaoMou/text-dedup | README read |
| 23 | Linacre, Building Accurate Address Matching Systems | blog, 2025-06-29 | https://www.robinlinacre.com/address_matching/ | page read |
| 24 | uk_address_matcher | GitHub (MIT) | https://github.com/moj-analytical-services/uk_address_matcher | README read; PR #502 (https://github.com/moj-analytical-services/uk_address_matcher/pull/502) **UNVERIFIED** (snippet) |
| 25 | ONS Working Paper 17, Address Index | ONS methodology | https://www.ons.gov.uk/methodology/methodologicalpublications/generalmethodology/onsworkingpaperseries/onsworkingpaperseriesno17usingdatasciencefortheaddressmatchingservice | page read |
| 26 | JointMatcher (numerically-aware EM) | KBS 251 (2022) | https://www.sciencedirect.com/science/article/abs/pii/S0950705122005044 | **UNVERIFIED** (403; snippets only) |
| 27 | Wang, Zheng, Wang, Pei, AutoML-EM | ICDE 2021 | https://www.cs.sfu.ca/~jnwang/papers/icde2021-automl-er.pdf | PDF pp.1-9 read |
| 28 | Paganelli et al., AutoML for EM | EDBT 2021 | https://openproceedings.org/2021/conf/edbt/p260.pdf | PDF pp.325-330 read |
| 29 | Shi et al., Quantized GBDT training | NeurIPS 2022 | https://arxiv.org/abs/2207.09682 | abstract read |
| 30 | LightGBM Parameters | docs | https://lightgbm.readthedocs.io/en/latest/Parameters.html | page read |
| 31 | LightGBM train (custom objective) | docs | https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html | page read |
| 32 | Fineis, inner workings of LightGBM lambdarank | blog 2021 | https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html | page read |
| 33 | CatBoost ranking objectives | docs | https://catboost.ai/docs/en/concepts/loss-functions-ranking | page read; LICENSE (Apache-2.0) https://github.com/catboost/catboost/blob/master/LICENSE read |
| 34 | Bruch, XE-NDCG | WWW 2021 | https://arxiv.org/abs/1911.09798 | abstract read |
| 35 | H&M 6th place (ranker blend) | Kaggle write-up | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/hard2rec-6th-place-solution | **UNVERIFIED** (body not rendered) |
| 36 | OTTO 20th place (LGBMRanker + CatBoost) | Kaggle write-up | https://www.kaggle.com/competitions/otto-recommender-system/writeups/kicchotto-20th-place-solution | **UNVERIFIED** (not opened) |

Cross-references in this repo: `notes/classical.md` (Cohen 2003, ME/SoftTFIDF basics, Mudgal 2018), `notes/blocking.md` (datasketch LSH), `open_scope/contests.md` (rank features, unique-text grouping, group-size weights), `FEATURE_CATALOG.md` §E/§G/§H, `ERROR_ANALYSIS.md` (2026-09-25 findings log).
