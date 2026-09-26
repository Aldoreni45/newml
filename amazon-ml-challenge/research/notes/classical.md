# Classical record linkage, string similarity and phonetics: research notes

Scope: Fellegi-Sunter (FS) and probabilistic record linkage (EM estimation of m/u, Splink, fastLink), deterministic and rule-based ER, approximate string matching (Levenshtein, Damerau, Jaro-Winkler, q-gram/Jaccard/overlap, Monge-Elkan, SoftTFIDF, learnable metrics), phonetic codes (Soundex, Metaphone, Double Metaphone, NYSIIS, Beider-Morse, French and Indic variants), benchmark comparisons, and when classical methods beat neural ones.

Written 2026-09-25. Every paper below was checked against a primary page or the DOI registry (Crossref API record for the DOI). The column "verified via" says which. Numbers are quoted only where I read them on the source (PDF text extracted and read, or the landing page). Items I could not verify are listed in section 13. **An independent audit on 2026-09-25 re-checked every paper, repo and model. Corrections are applied in place and listed in the "Audit log" at the end.**

Related sibling notes (so this file does not repeat them): `blocking.md` (candidate generation at scale), `name_address.md` (romanization and address parsing detail), `fmeasure.md` (expected-F0.5 decision rule and calibration), `neural_em.md`, `models.md`. Measured data facts come from `../../DATA_ANALYSIS.md`. All compute described here is meant for cloud machines (Modal, Kaggle, Colab, Lightning; CPU boxes with 64+ cores). Nothing is to be run on the laptop.

---

## 0. TL;DR

1. **Use FS as a scoring layer and as a feature, not as the whole system.** Splink (MIT) or our own FS implementation turns the comparison vector into an interpretable log-likelihood ratio. Its u-probabilities and term frequencies can be re-estimated **without labels on the test data for each country**, France included. This is the cheapest domain-adaptation tool available for an unseen country. Feed the FS match weight into the supervised GBDT as a stacked feature. Do not use it as the final classifier on its own: the conditional-independence assumption fails for name tokens, and Splink's README says it is "not designed for linking a single column containing a 'bag of words'".
2. **The strongest evidence from benchmark comparisons is that no single string measure wins.** Cohen, Ravikumar & Fienberg (2003) found SoftTFIDF (TF-IDF with Jaro-Winkler soft token matches, θ=0.9) best overall. Jaro-Winkler came close to Monge-Elkan at about 1/10 of the cost. Pure token measures failed on the misspelling-heavy census data. Christen (2006) found pattern-matching measures clearly better than phonetic codes. His recommendation is explicit: phonetic encoding followed by exact comparison of codes should not be used. So the plan is a **vector of complementary measures learned by a GBDT** (the Magellan and MAMBA pattern). Phonetic keys are used only as blocking keys and as one feature among many.
3. **Missing from FEATURE_CATALOG.md and worth adding**, all cheap on CPU:
   - SoftTFIDF(JW, θ=0.9) on names;
   - Monge-Elkan(JW) in both directions;
   - q-gram **overlap coefficient** (Christen 2006 found it better than Dice or Jaccard; it handles truncation);
   - Jaro-Winkler on the core-token string (Winkler's prefix bonus suits truncated heads);
   - Damerau/OSA distance on the rarest token;
   - FS-style **multi-level number agreement** (exact, same after stripping leading zeros or suffixes, one-digit edit, conflict, missing);
   - TF-adjusted rare-token match weight (Splink's u replaced by the token's frequency);
   - Indic phonetic-skeleton agreement after romanization;
   - the FS match weight itself.
4. **Where classical methods beat neural ones in this challenge:**
   - (a) **Scale and cost.** Scoring 35-90M candidate pairs with RapidFuzz `cpdist(workers=-1)` on a 64-core box is minutes of CPU work. Mudgal et al. (2018) measured deep-learning training at 5.4 h against 1.5 min for Magellan on average, for equal accuracy on structured data (87.9% vs 88.8% F1).
   - (b) **Numbers and exact identifiers** (house numbers, ZIP/PIN/code postal, domains). Ditto's own authors say their LMs "might not capture well EM tasks with a lot of numeric data". In their company case study they injected domain knowledge with a simple rule-based recognizer that tags the first number string in the address and the last 4 phone digits (not manual tagging; corrected in audit). In our true pairs, 77-84% share a number (DATA_ANALYSIS §7).
   - (c) **Calibration and interpretability.** An FS weight is a log Bayes factor. GBDT plus isotonic calibration is simple. Guo et al. (2017) showed modern neural nets are poorly calibrated.
   - (d) **Robustness under distribution shift.** Character-level measures behave the same on French text after accent folding. PLM matchers learn spurious correlations and lose robustness off-distribution (Akbarian Rastaghi et al., CIKM 2022).
   - Neural methods win on textual or dirty records and on semantic variation (Mudgal: +3.0 to 22.0 F1 on textual EM) and for blocking recall with SentenceBERT (Zeakis et al. 2023). The **cross-script Indian names** (name token-set p10 ≈ 11 without transliteration) are the one place where classical methods need a learned bridge: rule-based romanization, a phonetic skeleton, plus a token dictionary mined from the train pairs.
5. **The metric turns calibrated classical scores into decisions.** Each target belongs to at most one S1: target multiplicity is 1 for all 7.64M train pairs. So apply the many-to-one argmax first, then choose per S1 the set that maximizes expected F0.5 (see `fmeasure.md`). Worked example: with n=4 true matches and 3 sure true positives, a 4th candidate is worth adding only if p > 0.75. This is why good calibration (item 4c) matters more than a slightly better AUC.

---

## 1. How this topic maps to the challenge (facts from DATA_ANALYSIS.md)

| Measured fact | Classical implication |
|---|---|
| S1 names are highly non-unique (64% of India S1 names distinct, 74% of US); France is built from a small vocabulary plus city names | Name agreement alone has low likelihood ratio for frequent names. **FS term-frequency adjustment** (u = frequency of the agreed value) is the principled fix. Address comparisons carry most of the evidence. |
| Target multiplicity = 1 (no target linked to two S1s); country agreement = 100% | Many-to-one assignment constraint; country as a lossless partition (open set: whatever labels appear). |
| True pairs share a number in 77-84% of cases; number noise includes leading zeros, suffixes ("5935-D"), "#"/"No"/"H.NO" prefixes, digit typos | Multi-level **numeric comparison levels** (exact / normalized-equal / 1-edit / conflict / missing), where "conflict" has a strongly negative FS weight. This is classical territory. |
| Postcodes rare (US ZIP5 ~11%, India PIN6 <2%, France ~0.5% in targets) | A postcode is a near-identifier when present, and "missing" must be a neutral level (FS handles missing as its own level; fastLink models it explicitly). |
| Cross-script India names; name token-set p10 ≈ 11 (S2 India) and 14 (S3 India) vs S1 without transliteration | Classical string measures need a romanization step first. After it, phonetic skeleton plus JW/q-gram features apply. |
| Noise: case, doubled spaces, junk prefixes, bracketed tokens, leetspeak (0/o, l/I), injected diacritics, reordered tokens, suffix moved to front, truncation, aliases (dba/aka/fka), domains/handles, acronyms | Most of these are **invertible by deterministic normalization** (section 11.1). Token reordering calls for set-based or Monge-Elkan measures. Truncation calls for overlap coefficient and the JW prefix bonus. Leetspeak calls for confusable-character folding (names only, never numbers). Transpositions call for Damerau/OSA. |
| Within-source copies share the same corrupted base | Target-target similarity carries signal. A classical "collective" step (cluster near-duplicate targets, then link clusters) is cheap. |
| Test has ~24% more targets per S1 than train (probably more distractors) | FS priors (λ) and thresholds must be re-estimated per test country. EM on unlabeled test data does exactly this. |

---

## 2. Fellegi-Sunter and probabilistic record linkage

### 2.1 The model
- **Fellegi & Sunter (1969)**, JASA 64(328):1183-1210. For a record pair with comparison vector γ, rank pairs by R = P(γ|M)/P(γ|U) and use two thresholds (link / possible link / non-link) that are optimal for given error bounds. Under conditional independence, R factorizes into per-field m_k = P(agree_k|M) and u_k = P(agree_k|U). Log2 of the factor is the "match weight".
- Cohen et al. (2003) show that FS token agreement, under a frequency assumption, reduces to an **IDF weight** (the incremental log-odds for agreeing on word w is −log P_D(w)). They build "SFS" (simplified FS) with a penalty k·log P_D(w) for disagreements, k=0.5. So FS and TF-IDF are the same idea from two communities. This justifies IDF-weighted token features as FS-consistent.
- **Jaro (1989)**, JASA 84(406):414-420: EM estimation of m/u for the 1985 Tampa census linkage, and the Jaro string comparator. **Winkler (1990)** (ERIC ED325505; ERIC lists it only as an 8-page non-journal report. The usual citation, Proc. ASA Section on Survey Research Methods pp.354-359, comes from secondary indexes and was not confirmed on a primary page): the Jaro-Winkler prefix-bonus comparator, used in production for the 1990 US census Post Enumeration Survey, plus how to adjust FS weights for partial agreement. fastLink quotes Winkler's recommended three JW agreement levels with thresholds 0.88 and 0.94.

### 2.2 EM for m/u, missing data and scale: fastLink
- **Enamorado, Fifield & Imai (2019)**, APSR 113(2):353-371 (PDF read).
  - Canonical FS with EM, discrete agreement levels, and **missing data handled as MAR within the likelihood** (missing values are not treated as disagreements).
  - Key scaling trick: since the match probability is identical within an agreement pattern, **the count of pairs per agreement pattern is a sufficient statistic**, computed with a hashing scheme.
  - Numbers read in the paper:
    - 150,000 × 150,000 records merged in under 6 h on one core and under 3 h on 8 cores;
    - the Python `recordlinkage` package took more than 24 h for 20,000 × 20,000;
    - applied to two nationwide voter files of over 160 million records each.
  - JW thresholds used vary by application (0.94; 0.85/0.92; 0.88/0.92).
  - The paper notes that when overlap between files is small, error rates grow, so **effective blocking is essential when the expected number of matches is relatively small**.
- **Implication for us:** the pattern-count trick means EM on hundreds of millions of candidate pairs is cheap. Discretize each comparison into 3-6 levels, hash patterns, run EM on counts. Refit **per country on test** (France) without labels.

### 2.3 Splink (the modern implementation)
- **Linacre, Lindsay, Manassis, Slade & Hepworth (2022)**, IJPDS 7(3), doi 10.23889/ijpds.v7i3.1794 (landing page read). Builds on fastLink's EM for FS, runs on Spark and other SQL backends. The paper reports 2 million downloads at the time.
- From the Splink README (read 2026-09-25): "Capable of linking a million records on a laptop in around a minute". It runs on DuckDB or Spark "for 100+ million records"; "No training data is required". Two caveats matter for us:
  - Splink "performs best with input data containing **multiple** columns that are **not highly correlated**".
  - It "is not designed for linking a single column containing a 'bag of words'", with the example of a single company-name column.
  - So Splink is usable here **only after address parsing** into house number, street, city, state and postcode, alongside name comparisons.
- Parameter estimation (Splink docs, training rationale page):
  - u by random sampling ("if you take two random records, they will almost certainly not be a match");
  - m by EM within blocking rules. Columns used in the EM blocking rule get no estimates in that session, hence round-robin EM passes with different blocking rules;
  - λ (prior match probability) from user deterministic rules plus an assumed recall.
  - Fixing u and λ "anchor[s] the EM training process", giving few convergence problems.
- **Term-frequency adjustments** (Splink docs): replace u for an exact agreement by the value's frequency. This "doesn't depend on m, and therefore does not have to be estimated by the EM algorithm". Available on fuzzy levels with `tf_adjustment_weight`; `tf_minimum_u_value` guards against over-weighting rare misspellings. **This addresses non-unique S1 names and France's repetitive names directly.**
- Blocking diagnostics: `splink.blocking_analysis.count_comparisons_from_blocking_rule`, `cumulative_comparisons_to_be_scored_from_blocking_rules_chart/_data`, `n_largest_blocks`. The audit checked these names against the v4.0.17 tag, where they are correct. On master (5.0.0.dev5) the module exports only `count_comparisons_from_blocking_rules`, `chart_comparisons_from_blocking_rules` and `n_largest_blocks`, so pin the version.

### 2.4 Bayesian and bipartite variants (background; lower priority)
- **Sadinle (2017)** JASA 112(518):600-612: Bayesian estimation of **bipartite** (1-1) matchings. **Steorts, Hall & Fienberg (2016)** JASA 111(516):1660-1672: graphical record linkage plus dedup across files. **Binette & Steorts (2022)** Science Advances 8(12) eabi8021: survey "(Almost) all of entity resolution".
- These are principled, but MCMC does not scale to 1.7M × 10M without heavy blocking. Our structure is **many-to-one**, not 1-1 (each target has at most one S1, S1 has 0-11 targets). Borrow the constraint idea, not the samplers.
- **Bai, Binette & Reiter (2025)** Statistics and Computing 35(5), arXiv 2311.13923: point estimates of the linkage structure that **maximize expected F-score** from posterior or match probabilities in the bipartite setting. Conceptually the same as our per-S1 expected-F0.5 decoder (see `fmeasure.md`).

### 2.5 Calibration of FS scores
- **Belin & Rubin (1995)** JASA 90(430):694-707: a mixture model on FS weights to calibrate false-match rates. **Larsen & Rubin (2001)** JASA 96(453):32-41: mixture models with dependence between fields plus iterative clerical review.
- Takeaway: raw FS posteriors are miscalibrated when fields are dependent (name tokens, city-state-postcode). Calibrate on held-out labels (isotonic per country), or use FS weights as GBDT inputs and calibrate the GBDT.
- **Hand & Christen (2018)** Statistics and Computing 28(3):539-547 criticize using F-measure to compare record linkage algorithms: the relative weight of precision and recall depends on the number of predicted matches. Our metric is fixed by the organizers. What we take from the paper: F-beta rewards differ across entities of different match counts, so the decoder must be per-entity.

---

## 3. Deterministic and rule-based ER

- **Hernández & Stolfo (1995)** SIGMOD pp.127-138: merge/purge with the **sorted neighbourhood** method plus an equational theory (declarative rules), with multi-pass sorting keys and transitive closure. **McCallum, Nigam & Ungar (2000)** KDD pp.169-178: canopies with a cheap distance, then an expensive distance within canopies.
- **Singh et al. (2017)** PVLDB 11(2):189-202: synthesizing interpretable EM rules (Boolean formulas over similarity predicates) from examples.
- **Chaudhuri, Ganjam, Ganti & Motwani (2003)** SIGMOD pp.313-324 (PDF read). Their framing matches our task closely: **clean reference relation plus dirty incoming tuples**.
  - Their reference table was a Customer[name, city, state, zip] relation of **about 1.7 million tuples**, the same order as our 1.73M test S1.
  - Fuzzy-match similarity (fms): IDF-weighted token transformation cost, which fixes the failure of edit distance on high-frequency tokens.
  - Error-Tolerant Index: min-hash signatures of token q-grams (q=4 in their experiments), giving candidates 2-3 orders of magnitude faster than naive search.
  - Their error model (spelling errors, abbreviation replacement, missing values, truncation, token merge, token transposition) is essentially our noise catalogue.
- **Cuffe & Goldschlag (2018)**, US Census Bureau CES WP 18-46 (working paper, **not peer-reviewed**; PDF read). **MAMBA** links **business names plus addresses** to the Census Business Register:
  - standardization of abbreviations and suffixes;
  - blocking passes (EIN, ZIP5, city, ZIP3, state, then a residual non-geographic pass); for the state and non-geographic passes a candidate must share at least one word's soundex code;
  - a 3-gram name-similarity prefilter at 0.3 (EIN, ZIP5, city, ZIP3 blocks) or 0.4 (state and non-geographic blocks) that lets them "remove 99.5% of potential matches in any given block";
  - comparators derived from FEBRL (Christen 2008);
  - 13 string comparators fed to a random forest (32 trees, max 15 features, depth 10) with cross-validated precision "over 95%".
  - This is essentially our GBDT-over-comparators design, on the same entity type.
- **Wasi & Flaaen (2015)** Stata Journal 15(3):672-697: preprocessing utilities for business name and address standardization before linking. Their `stnd_compname` parses a company name into 5 components: official name, **DBA name, FKA name, business entity type** and attention name. Their `reclink2` generalizes reclink to **many-to-one** matching (author PDF read in audit). Both ideas map directly onto our "X dba Y" aliases, moved legal suffixes and many-to-one structure.

**Role in our pipeline.** Add a deterministic "tier 0" of normalized exact matches: core name plus house number plus street token, within country. It serves three purposes:
- it gives high-precision anchors;
- it provides Splink's λ-estimation rules and seeds for learning test-side dictionaries (France region↔département, section 11.1);
- it provides a sanity baseline.

Rules must never be the final decision alone. Christen (2006) shows threshold choice is fragile ("Even small changes of the threshold can result in dramatic drops in matching quality").

---

## 4. Approximate string matching: measures and what they fix

### 4.1 The measures
- **Edit-based.**
  - Levenshtein (unit costs).
  - **Damerau (1964)** CACM 7(3):171-176 adds transpositions. The OSA variant is in RapidFuzz.
  - Affine-gap / Smith-Waterman: **Monge & Elkan (1996)** KDD.
  - **Navarro (2001)** ACM CSUR 33(1):31-88 is the reference survey (filters, q-gram lower bounds, bit-parallelism).
  - All are O(|s|·|t|) except bit-parallel implementations (RapidFuzz uses them).
- **Jaro / Jaro-Winkler.** Common characters within a window plus transpositions, O(|s|+|t|). The Winkler variant boosts a common prefix of up to 4 characters. Intended for short strings such as personal names (Cohen et al. 2003). Christen (2006) Rec. 7: the Winkler prefix modification "can be used with all techniques to improve matching quality".
- **q-gram / set measures.** Jaccard, Dice, cosine, and the **overlap coefficient** |A∩B|/min(|A|,|B|). Christen Rec. 6: overlap "seems to achieve better matching results (compared to using the Dice coefficient or Jaccard similarity)". Overlap equals 1 when one string is a truncation of the other, which suits our "Deleon", "Allied" truncations.
- **Token TF-IDF cosine** (Cohen 2000, TOIS 18(3):288-321, WHIRL): order-invariant and weighted by rarity.
- **Hybrid.**
  - **Monge-Elkan**: mean over tokens of s of the max secondary similarity to tokens of t. It is asymmetric, so compute both directions.
  - **SoftTFIDF** (Cohen et al. 2003): TF-IDF over the token pairs whose secondary JW similarity exceeds θ (0.9), weighted by that similarity. It combines rarity weighting with typo tolerance inside tokens.
  - **Moreau, Yvon & Cappé (2008)** COLING pp.593-600 (PDF read) propose a generic model mixing measures, with a SoftTFIDF-based combination using a trigram sub-measure that "performs better than all existing similarity metrics on two corpora". One of those corpora is an annotated **French** corpus.
- **Learnable.** **Bilenko & Mooney (2003)** KDD pp.39-48 (MARLIN): learned edit-distance costs per field plus an SVM over vector-space features, combined in two layers. Improvements are over fixed metrics.

### 4.2 Benchmark comparisons (what the evidence says)
- **Cohen, Ravikumar & Fienberg (2003)**, IIWeb 2003 pp.73-78 (venue and pages from the first author's CMU publication page; PDF read):
  - Among token measures, TF-IDF was generally best.
  - Among edit measures, Monge-Elkan was best on average, but Jaro and Jaro-Winkler were "close in average performance" and took "about 1/10 the time".
  - Among hybrids, **SoftTFIDF** (JW secondary, θ=0.9) was best overall. On the two clustering tasks it scored MaxF1/AvgPrec 0.89/0.91 (UVA) and 0.85/0.914 (CoraATDV), ahead of TFIDF (0.79/0.84, 0.84/0.907).
  - A learned SVM combination of the measures slightly outperformed SoftTFIDF, "particularly at extreme recall levels".
  - Token methods "perform poorly on the census-like dataset, which contains many misspellings".
  - Their knowledge-free token blocker found 93.3-100% (avg 98.9%) of correct pairs, and the 4-gram blocker averaged 99.0%.
  - Timing on these data: SoftTFIDF about 13 s, JW about 20 s, TFIDF 1.2 s.
- **Christen (2006)**, ICDMW pp.290-294, tech report TR-CS-06-02 read. He ran 123 tests on four name datasets and found "no single best technique". Pattern matching "clearly outperform[s] phonetic encoding techniques". Simple Phonex beat Phonix and Double-Metaphone. Recommendations:
  - (3) do not use phonetic encoding with exact code comparison;
  - (4) Jaro and Winkler work well for parsed given names and surnames, as do uni- and bigrams;
  - (5) longest common substring for unparsed names with swapped words;
  - (6) overlap coefficient;
  - (8) thresholds are fragile without labels;
  - (9) prefer linear-time measures (q-grams, Jaro, Winkler) and prefilter with bag distance.
- **Mudgal et al. (2018)** SIGMOD pp.19-34 (PDF read), structured data. Magellan (random forest and similar over automatically generated similarity features) vs the best DL model:
  - BeerAdvo-RateBeer 78.8 vs 72.7;
  - Walmart-Amazon1 71.9 vs 67.6;
  - iTunes-Amazon1 91.2 vs 88.5;
  - DL wins on Amazon-Google (69.3 vs 49.1).
  - On the **Company** textual dataset, Magellan 79.8 vs Hybrid DL 92.7. Audit correction: Company is **not** short company names. The paper says it "tries to match company homepages and Wikipedia pages describing companies", i.e. one long free-text attribute. The result therefore says little about our short business-name strings.

### 4.3 Noise type to measure mapping (for our data)

| Noise (DATA_ANALYSIS §5) | Measures that handle it | Measures that fail |
|---|---|---|
| typos (insert/delete/replace) | Levenshtein ratio, JW, char q-gram cosine/overlap, SoftTFIDF | exact tokens, phonetic-exact |
| swaps ("Madr") | Damerau/OSA, JW (counts transpositions) | plain Levenshtein counts 2 edits |
| token reorder, suffix moved to front | token-set/sort ratio, TF-IDF cosine, Monge-Elkan, SoftTFIDF, q-gram sets | Levenshtein on raw string, JW on raw string |
| generic token injection/deletion (LLC, Services, The) | IDF-weighted measures (TF-IDF, SoftTFIDF, fms), core-token Jaccard | unweighted Jaccard, Levenshtein |
| truncation | overlap coefficient, JW prefix bonus, prefix measures, coverage of the shorter side | Jaccard, cosine |
| leetspeak (0/o, l/I), injected diacritics | fold confusables and accents before measuring (names only) | any measure on raw text |
| abbreviations (Rd/Road, Pvt/Private) | canonicalize with a learned map, then any measure; SoftTFIDF partly | raw measures |
| acronyms ("SC") | initials feature (already in catalog) | all string measures |
| domains/handles | space-less q-gram cosine, prefix coverage (in catalog) | token measures |
| cross-script | romanize, then phonetic skeleton plus JW/q-gram; or a learned token dictionary | everything without romanization (p10 ≈ 11) |
| number typos | 1-edit numeric level, digit-string Levenshtein | exact-only |

---

## 5. Phonetic codes

| Code | Origin (verification) | Design | Fit for our data |
|---|---|---|---|
| Soundex | Russell, US patent 1,261,167, filed 1917-10-25, granted 1918-04-02 (Google Patents read) | first letter plus 3 digits from consonant classes | English-centric. Useful only as a coarse blocking key for Latin names. MAMBA used a soundex block for residuals. |
| Metaphone / Double Metaphone | Philips; Double Metaphone in C/C++ Users Journal 18(6):38-43, June 2000 (trade magazine; see §13, only partially verified) | English-plus-some-foreign rules; primary and alternate codes | English-centric. Christen (2006) found Double-Metaphone worse than the simpler Phonex. Blocking key at most. |
| NYSIIS | Taft 1970, NY State Identification and Intelligence System (not verified from a primary page, §13) | rule-based, keeps vowel position info | English surnames. Low value here. |
| Beider-Morse (BMPM) | Beider & Morse, Avotaynu, Summer 2008 (stevemorse.org page read) | guesses the language, applies language-specific phonetic rules, returns a *set* of codes. Languages include **French**, English, German, Spanish, Russian, Polish, Romanian, Hungarian, Hebrew. Claimed to give far fewer false hits than Soundex or Daitch-Mokotoff. | The only classical code with French rules. Available in Apache Commons Codec (Apache-2.0, Java). French names in test are mostly generic words plus cities, so the value is modest. Blocking key for French names only. |
| French algorithms (Phonex, Soundex2, FONEM, Sonnex, Phonetic) | implemented in `talisman` (MIT, JS) `src/phonetics/french/` (repo listing read) | French orthography rules | Reimplement the rules in Python if needed. Low priority. |
| IndicSoundex | Thottingal (libindic; LGPL; README read). Maps Indic letters to shared codes, so `compare('ಬೆಂಗಳೂರು','बॆंगळूरु')` returns 2 (same sound across Kannada and Devanagari). | works because the Unicode Indic blocks are laid out in parallel (ISCII order) | Good idea for **Indic↔Indic** comparison. For Indic↔Latin we need romanization first. The LGPL license is for a library; it is not a model, and the method is easy to reimplement under our own code. |
| Hindi Soundex | Anand, Mahajan, Verma & Singh (2019), LNEE "Advances in Data Sciences, Security and Applications" pp.285-293 (existence verified on Crossref; abstract not read) | Soundex adapted to Devanagari | Existence only. Evidence quality unknown. |
| Universal romanization | **uroman**, Hermjakob, May & Knight, ACL 2018 demos pp.13-18 (ACL Anthology page read). The paper says romanization "enables the application of string-similarity metrics to texts from different scripts without the need and complexity of an intermediate phonetic representation". | Unicode tables plus m-to-n rules; also converts native-script digits to 0-9 | **Core tool for the cross-script bridge.** MIT-style license with an attribution clause (LICENSE.txt read); PyPI classifier says Apache. Record both in the README. Python version 1.3.1.1. Its changelog notes improved final-schwa deletion for Indian languages (repo page). |

**The design conclusion for phonetics (P0):**
- Do not decide on phonetic-code equality (Christen 2006, Rec. 3).
- Use codes (a) as extra **blocking keys** for rare name tokens and (b) as **features**: Jaccard of per-token keys (already `n_phon_jacc`), plus JW between the concatenated skeletons.
- For India, build a custom **Indic phonetic skeleton** on the romanized string: collapse aspiration (bh→b, th→t, dh→d, kh→k, gh→g, ph→f/p), vowel length (aa→a, ii/ee→i, uu/oo→u), w/v, z/j, x→ksh, ksh/sh/s, doubled letters, final schwa, anusvara to n/m. For Tamil, also collapse voiced/unvoiced (k/g, t/d, p/b, c/j), since Tamil script does not mark voicing.
- Validate every rule on train pairs: keep a rule only if key agreement rises on true pairs more than on random same-country pairs.
- `name_address.md` has the romanization detail. This file only fixes the principle.

---

## 6. Scaling classical similarity (short; details in `blocking.md`)

- **Gravano et al. (2001)** VLDB pp.491-500 (PDF read): approximate string joins inside a DBMS via positional q-grams, with count, position and length filters that are exact for edit-distance thresholds.
- **Bayardo, Ma & Srikant (2007)** WWW pp.131-140 (AllPairs prefix filter) and **Xiao, Wang, Lin & Yu (2008)** WWW pp.131-140 (PPJoin).
- **Mann, Augsten & Bouros (2016)** PVLDB 9(9):636-647, doi 10.14778/2947618.2947620 (PDF read). "The key technique is the prefix filter"; AllPairs "is still a relevant competitor"; complex filters (PPJoin+, AdaptJoin) "are rarely competitive", because efficient verification is fast.
- **Jiang, Li, Feng & Li (2014)** PVLDB 7(8):625-636: experimental evaluation of string similarity joins.
- **Christen (2012)** TKDE 24(9):1537-1555: survey of indexing and blocking for record linkage. **Papadakis et al. (2021; online 2020)** ACM CSUR 53(2):1-42: blocking and filtering survey.
- **Practical recipe on a 64-core box.**
  - Char-3gram TF-IDF with sparse top-k (`sparse_dot_topn`, Apache-2.0, or the existing `stage_block.py`) within country for names and addresses.
  - Union with rare-token and phonetic-key inverted indexes, and house-number plus street-token keys.
  - Then compute all features with RapidFuzz `process.cpdist(..., workers=-1)`, which scores corresponding pairs elementwise in C++. The docs say "Supply -1 to use all available CPU cores".

---

## 7. Where do classical methods beat neural methods? (explicit answer)

| Dimension | Classical advantage (evidence) | Caveat / where neural wins |
|---|---|---|
| **Scale and cost** | Mudgal et al. 2018: on structured EM, DL is "competitive with Magellan ... (87.9% vs 88.8% average F1), but require[s] far longer training time (5.4h vs 1.5m on average)". Chaudhuri et al. 2003 fuzzy-matched against a ~1.7M-tuple reference 2-3 orders of magnitude faster than naive. Splink: about 1M records per minute on a laptop. fastLink: pattern hashing makes EM cost proportional to the number of distinct agreement patterns. Ditto's company case study (789K × 412K records, 10.65M candidates): 6.49 h end-to-end without advanced blocking, 1.69 h with SentenceBERT blocking. | Ditto reached F1 96.53 there. A cross-encoder is affordable on a **re-ranked, ambiguous subset**, not on all 35-90M pairs. |
| **Exact identifiers and numbers** | Ditto (Li et al. 2020) authors: LMs "might not capture well EM tasks with a lot of numeric data"; "the street number and the phone number are both useful signals", so they implemented a simple recognizer that tags the first number string in the address. FS levels encode exact / near / conflict / missing numbers directly, with explicit likelihood ratios and TF weighting. | Neural models can take numeric features as side inputs (hybrid), which is our GBDT anyway. |
| **Calibration** | FS weights are log Bayes factors with an explicit prior. Miscalibration under dependence is known and fixable (Belin & Rubin 1995; Larsen & Rubin 2001). GBDT plus isotonic is standard. Guo et al. 2017: "modern neural networks ... are poorly calibrated" (temperature scaling helps). | Temperature scaling fixes much of the neural gap in-distribution, but not under shift. |
| **OOD / unseen country (France)** | Character measures after accent folding do not care about language. FS **EM and TF statistics can be refit on unlabeled test data per country** (transductive, label-free, allowed by C-F3 in COMPLIANCE_CHECKLIST). Akbarian Rastaghi et al. 2022 (CIKM, pp.3786-3790): PLM EM models "exhibit tendency to learn spurious correlations", have robustness problems from training imbalance, and "data augmentation alone is not sufficient". Tu et al. 2022 (SIGMOD pp.443-457) needed dedicated domain adaptation (DADER) to move deep ER to new domains. | Multilingual encoders do carry prior knowledge of French. Useful as an extra feature, not as the only signal. |
| **Benchmarks overstate DL** | Papadakis, Kirielle, Christen & Palpanas, ICDE 2024 pp.3435-3448 (arXiv 2307.01231, conclusions read): "Most of the popular datasets used as benchmarks for DL-based matchers involve almost linearly separable candidate pairs, or are perfectly solved by most existing matching algorithms". | Our data has hard parts (cross-script, aliases) that are not linearly separable in string-feature space. That is where to spend GPU. |
| **Interpretability and debugging** | Per-field weights (Splink waterfall), rule audits, deterministic tiers. | n/a |
| **Where neural clearly wins** | Mudgal: DL beats Magellan by 3.0-22.0% F1 on textual EM; Company dataset 92.7 vs 79.8 (Company = company homepages vs Wikipedia pages, long text, not short names). Zeakis et al. 2023 (PVLDB 16(9):2225-2238, conclusions read): SentenceBERT models "consistently outperform" other embeddings in blocking and unsupervised matching. Cross-script transliteration of unseen names: IndicXlit (MIT, ~11M params) exists, but its README top-1 on Aksharantar **named entities** (Indic→En) is only 12.87-60.77% per language, so outputs need fuzzy matching anyway. | Use neural for (1) semantic/cross-script blocking recall and (2) re-scoring the uncertain band. |

**Bottom line:** for this dataset (short names, structured addresses, synthetic character-level noise, many numbers, 10M targets, an unseen but Latin-script country), a classical comparison vector plus GBDT is the backbone. Its main gaps are cross-script names and aliases. Neural components should be judged by how much they add on top of it, measured per country and per noise type.

---

## 8. Papers (verified)

| # | Paper | Venue, year | URL | Verified via | Use here |
|---|---|---|---|---|---|
| 1 | Fellegi & Sunter, A Theory for Record Linkage | JASA 64(328):1183-1210, 1969 | https://doi.org/10.1080/01621459.1969.10501049 | Crossref record (tandfonline 403) | FS model, match weights, thresholds |
| 2 | Jaro, Advances in Record-Linkage Methodology ... 1985 Census of Tampa | JASA 84(406):414-420, 1989 | https://doi.org/10.1080/01621459.1989.10478785 | Crossref | EM for m/u; Jaro comparator |
| 3 | Winkler, String Comparator Metrics and Enhanced Decision Rules in the FS Model | 1990, ERIC ED325505 (non-journal report, 8 pp.). The venue "Proc. ASA Section on Survey Research Methods pp.354-359" comes from secondary indexes only, so it is **UNVERIFIED** | https://eric.ed.gov/?id=ED325505 | ERIC record read (re-read in audit) | JW, partial-agreement weights |
| 4 | Monge & Elkan, The Field Matching Problem | KDD 1996 | https://aaai.org/papers/KDD96-044-the-field-matching-problem-algorithms-and-applications/ | AAAI page read | Monge-Elkan, affine gaps |
| 5 | Cohen, Ravikumar & Fienberg, A Comparison of String Distance Metrics for Name-Matching Tasks | IIWeb (IJCAI-03 workshop), 2003, pp.73-78 | https://www.cs.cmu.edu/~wcohen/postscript/ijcai-ws-2003.pdf | PDF read (6 pp.); venue and pages confirmed on the author's page https://www.cs.cmu.edu/~wcohen/pubs.html | SoftTFIDF best; JW about 10x faster than ME |
| 6 | Christen, A Comparison of Personal Name Matching: Techniques and Practical Issues | ICDM Workshops 2006, pp.290-294 | https://doi.org/10.1109/ICDMW.2006.2 | Crossref + TR-CS-06-02 PDF read | phonetic codes worse; overlap coef.; recommendations |
| 7 | Christen, Data Matching (book) | Springer 2012 | https://doi.org/10.1007/978-3-642-31164-2 | Crossref | reference text |
| 8 | Christen, A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication | IEEE TKDE 24(9):1537-1555, 2012 | https://doi.org/10.1109/TKDE.2011.127 | Crossref | blocking survey |
| 9 | Chaudhuri, Ganjam, Ganti & Motwani, Robust and Efficient Fuzzy Match for Online Data Cleaning | SIGMOD 2003, pp.313-324 | https://doi.org/10.1145/872757.872796 | Crossref + MSR PDF read | reference-table matching, fms, ETI; 1.7M-tuple reference |
| 10 | Gravano et al., Approximate String Joins in a Database (Almost) for Free | VLDB 2001, pp.491-500 | https://www.vldb.org/conf/2001/P491.pdf | PDF read | q-gram filters |
| 11 | Damerau, A Technique for Computer Detection and Correction of Spelling Errors | CACM 7(3):171-176, 1964 | https://doi.org/10.1145/363958.363994 | Crossref | transpositions |
| 12 | Navarro, A Guided Tour to Approximate String Matching | ACM CSUR 33(1):31-88, 2001 | https://doi.org/10.1145/375360.375365 | Crossref | survey of edit-distance algorithms |
| 13 | Bilenko & Mooney, Adaptive Duplicate Detection Using Learnable String Similarity Measures | KDD 2003, pp.39-48 | https://doi.org/10.1145/956750.956759 | Crossref | learned edit costs (MARLIN) |
| 14 | Hernández & Stolfo, The Merge/Purge Problem for Large Databases | SIGMOD 1995, pp.127-138 | https://doi.org/10.1145/223784.223807 | Crossref | sorted neighbourhood, rules |
| 15 | McCallum, Nigam & Ungar, Efficient Clustering of High-Dimensional Data Sets with Application to Reference Matching | KDD 2000, pp.169-178 | https://doi.org/10.1145/347090.347123 | Crossref | canopies |
| 16 | Bayardo, Ma & Srikant, Scaling Up All Pairs Similarity Search | WWW 2007, pp.131-140 | https://doi.org/10.1145/1242572.1242591 | Crossref | prefix filter |
| 17 | Xiao, Wang, Lin & Yu, Efficient Similarity Joins for Near Duplicate Detection | WWW 2008, pp.131-140 | https://doi.org/10.1145/1367497.1367516 | Crossref | PPJoin |
| 18 | Mann, Augsten & Bouros, An Empirical Evaluation of Set Similarity Join Techniques | PVLDB 9(9):636-647, 2016 | https://doi.org/10.14778/2947618.2947620 | Crossref + PDF read | prefix filter is what matters |
| 19 | Jiang, Li, Feng & Li, String Similarity Joins: An Experimental Evaluation | PVLDB 7(8):625-636, 2014 | https://doi.org/10.14778/2732296.2732299 | Crossref | join algorithm choice |
| 20 | Enamorado, Fifield & Imai, Using a Probabilistic Model to Assist Merging of Large-Scale Administrative Records (fastLink) | APSR 113(2):353-371, 2019 | https://doi.org/10.1017/S0003055418000783 | Crossref + PDF read | EM, missing data, pattern hashing |
| 21 | Linacre et al., Splink: Free Software for Probabilistic Record Linkage at Scale | IJPDS 7(3), 2022 | https://doi.org/10.23889/ijpds.v7i3.1794 | IJPDS page read + Crossref | modern FS at scale |
| 22 | Belin & Rubin, A Method for Calibrating False-Match Rates in Record Linkage | JASA 90(430):694-707, 1995 | https://doi.org/10.1080/01621459.1995.10476563 | Crossref | FS calibration |
| 23 | Larsen & Rubin, Iterative Automated Record Linkage Using Mixture Models | JASA 96(453):32-41, 2001 | https://doi.org/10.1198/016214501750332956 | Crossref | dependence between fields |
| 24 | Sadinle, Bayesian Estimation of Bipartite Matchings for Record Linkage | JASA 112(518):600-612, 2017 | https://doi.org/10.1080/01621459.2016.1148612 | Crossref | 1-1 constraint (contrast: ours is many-to-one) |
| 25 | Steorts, Hall & Fienberg, A Bayesian Approach to Graphical Record Linkage and Deduplication | JASA 111(516):1660-1672, 2016 | https://doi.org/10.1080/01621459.2015.1105807 | Crossref | joint link plus dedup |
| 26 | Hand & Christen, A Note on Using the F-measure for Evaluating Record Linkage Algorithms | Stat. Comput. 28(3):539-547, 2018 | https://doi.org/10.1007/s11222-017-9746-6 | Crossref | metric caveats |
| 27 | Bai, Binette & Reiter, Optimal F-score Matching for Bipartite Record Linkage | Stat. Comput. 35(5), 2025 (arXiv 2311.13923, whose title is "Optimal F-score **Clustering** for Bipartite Record Linkage") | https://doi.org/10.1007/s11222-025-10701-y | Crossref + arXiv abs read | expected-F decoding |
| 28 | Binette & Steorts, (Almost) All of Entity Resolution | Science Advances 8(12) eabi8021, 2022 | https://doi.org/10.1126/sciadv.abi8021 | Crossref | survey |
| 29 | Mudgal et al., Deep Learning for Entity Matching: A Design Space Exploration | SIGMOD 2018, pp.19-34 | https://doi.org/10.1145/3183713.3196926 | Crossref + PDF read | classical parity on structured data (its "Company" dataset is long-text homepages vs Wikipedia, not short names) |
| 30 | Konda et al., Magellan: Toward Building Entity Matching Management Systems | PVLDB 9(12):1197-1208, 2016 | https://doi.org/10.14778/2994509.2994535 | Crossref | similarity-feature plus RF pattern |
| 31 | Li, Li, Suhara, Doan & Tan, Deep Entity Matching with Pre-Trained Language Models (Ditto) | PVLDB 14(1):50-60, 2020 | https://doi.org/10.14778/3421424.3421431 | Crossref + arXiv PDF read | numbers weakness; company case study |
| 32 | Papadakis, Skoutas, Thanos & Palpanas, Blocking and Filtering Techniques for Entity Resolution: A Survey | ACM CSUR 53(2):1-42, 2021 (online March 2020) | https://doi.org/10.1145/3377455 | Crossref | blocking survey |
| 33 | Papadakis, Kirielle, Christen & Palpanas, A Critical Re-evaluation of Record Linkage Benchmarks for Learning-Based Matching Algorithms (arXiv title: "... Benchmark Datasets for (Deep) Learning-Based Matching Algorithms") | ICDE 2024, pp.3435-3448 | https://doi.org/10.1109/ICDE60146.2024.00265 | Crossref + arXiv 2307.01231 PDF read | most benchmarks nearly linear |
| 34 | Zeakis, Papadakis, Skoutas & Koubarakis, Pre-Trained Embeddings for Entity Resolution: An Experimental Analysis | PVLDB 16(9):2225-2238, 2023 | https://doi.org/10.14778/3598581.3598594 | Crossref + VLDB PDF read | SentenceBERT best for blocking |
| 35 | Akbarian Rastaghi, Kamalloo & Rafiei, Probing the Robustness of Pre-trained Language Models for Entity Matching | CIKM 2022, pp.3786-3790 | https://doi.org/10.1145/3511808.3557673 | Crossref + authors' repo README | PLM robustness issues |
| 36 | Tu et al., Domain Adaptation for Deep Entity Resolution | SIGMOD 2022, pp.443-457 | https://doi.org/10.1145/3514221.3517870 | Crossref (abstract via search only) | shift needs DA for deep ER |
| 37 | Singh et al., Synthesizing Entity Matching Rules by Examples | PVLDB 11(2):189-202, 2017 | https://doi.org/10.14778/3149193.3149199 | Crossref | interpretable rules |
| 38 | Cohen, Data Integration Using Similarity Joins and a Word-Based Information Representation Language (WHIRL) | ACM TOIS 18(3):288-321, 2000 | https://doi.org/10.1145/352595.352598 | Crossref | TF-IDF similarity joins |
| 39 | Moreau, Yvon & Cappé, Robust Similarity Measures for Named Entities Matching | COLING 2008, pp.593-600 | https://aclanthology.org/C08-1075/ | ACL Anthology PDF read | SoftTFIDF-based mixtures; French corpus |
| 40 | Hermjakob, May & Knight, Out-of-the-box Universal Romanization Tool uroman | ACL 2018 demos, pp.13-18 | https://aclanthology.org/P18-4003/ | ACL Anthology page read | cross-script bridge |
| 41 | Madhani et al., Aksharantar: Open Indic-language Transliteration Datasets and Models ... (IndicXlit) | arXiv 2205.03018 (extended version of EMNLP Findings 2023 paper) | https://arxiv.org/abs/2205.03018 | arXiv abs read | neural transliteration baseline |
| 42 | Wasi & Flaaen, Record Linkage Using Stata: Preprocessing, Linking, and Reviewing Utilities | Stata Journal 15(3):672-697, 2015 | https://doi.org/10.1177/1536867X1501500304 | Crossref + author PDF read in audit | business-name parsing into official/DBA/FKA/entity-type; many-to-one `reclink2` |
| 43 | Cuffe & Goldschlag, Squeezing More Out of Your Data: Business Record Linkage with Python (MAMBA) | US Census CES WP 18-46, 2018 (**working paper, not peer-reviewed**) | https://www2.census.gov/ces/wp/2018/CES-WP-18-46.pdf | PDF read | business name+address RF over 13 comparators |
| 44 | Christen & Vatsalan, Flexible and Extensible Generation and Corruption of Personal Data | CIKM 2013, pp.1165-1168 | https://doi.org/10.1145/2505515.2507815 | Crossref | synthetic corruption models (our data is synthetic-noised) |
| 45 | Christen, Febrl: an open source data cleaning, deduplication and record linkage system with a GUI | KDD 2008, pp.1065-1068 | https://doi.org/10.1145/1401890.1402020 | Crossref | classical toolkit; comparators used by MAMBA |
| 46 | Guo, Pleiss, Sun & Weinberger, On Calibration of Modern Neural Networks | ICML 2017 (PMLR 70) | https://proceedings.mlr.press/v70/guo17a.html | PMLR page read | neural miscalibration |
| 47 | Dembczyński, Waegeman, Cheng & Hüllermeier, An Exact Algorithm for F-Measure Maximization | NIPS 24, 2011 | https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html | NeurIPS page read | exact F decoding (see fmeasure.md) |
| 48 | Anand, Mahajan, Verma & Singh, Soundex Algorithm for Hindi Language Names | LNEE (Advances in Data Sciences, Security and Applications), online Dec 2019, print 2020, pp.285-293 | https://doi.org/10.1007/978-981-15-0372-6_22 | Crossref only (abstract not read) | Indic phonetic precedent |
| 49 | Gadd, Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching | arXiv 2601.06932, 2026 (v4) | https://arxiv.org/abs/2601.06932 | arXiv abs read | **idea only**: trained on GeoNames/Wikidata/Getty toponyms, which is external gazetteer data and a compliance RISK. Reports R@1 85.2%, MRR 90.8% on the MEHDIE benchmark. |
| 50 | Russell, Index (Soundex), US patent 1,261,167 | 1918 | https://patents.google.com/patent/US1261167A/en | Google Patents page read | Soundex origin |
| 51 | Beider & Morse, Beider-Morse Phonetic Matching: An Alternative to Soundex with Fewer False Hits | Avotaynu, Summer 2008 (genealogy journal) | https://stevemorse.org/phonetics/bmpm.htm | authors' page read | multi-language phonetics incl. French |

---

## 9. Libraries and repos (GitHub API or repo files read 2026-09-25; library licenses are **software** licenses, separate from model licenses)

| Library | URL | License | Stars | Last push | Compliance / notes | Use |
|---|---|---|---|---|---|---|
| RapidFuzz | https://github.com/rapidfuzz/RapidFuzz | MIT | 4,139 | 2026-09-12 | safe | **P0** all string features. `process.cpdist` for elementwise pair scoring, `workers=-1`. Metrics: Levenshtein, DamerauLevenshtein, OSA, Jaro, JaroWinkler, Indel, LCSseq, Hamming, Prefix, Postfix. |
| Splink | https://github.com/moj-analytical-services/splink | MIT | 2,429 | 2026-09-22 | safe; PyPI 4.0.17 (released 2026-09-03). **Pin `splink==4.0.17`**: the master branch is already 5.0.0.dev5 and renames the blocking-analysis API | **P1** FS/EM with TF adjustments, DuckDB/Spark |
| fastLink (R) | https://github.com/kosukeimai/fastLink | GPL (>= 3) (DESCRIPTION file) | 292 | 2026-02-28 | GPL software, R | reference implementation only |
| dedupe | https://github.com/dedupeio/dedupe | MIT | 4,514 | 2025-07-29 | safe | `Gazetteer` class = messy vs canonical reference with `n_matches`; learned blocking predicates. Its `RecordLink` default is one-to-one, which is **wrong for us**. |
| recordlinkage | https://github.com/J535D165/recordlinkage | BSD-3-Clause | 1,062 | 2024-02-21 | safe; slower-moving | ECMClassifier (unsupervised FS EM), comparators; pandas-bound, will not scale naively |
| jellyfish | https://github.com/jamesturk/jellyfish (source moved to codeberg.org/jpt/jellyfish) | MIT | 2,233 | 2026-07-24 | safe | Soundex, Metaphone, NYSIIS, MRA; JW, Damerau (no Double Metaphone) |
| metaphone (PyPI) | https://pypi.org/project/Metaphone/ (repo https://github.com/oubiwann/metaphone) | BSD (PyPI); BSD-3-Clause (GitHub) | 86 | 2024-02-28 | safe | Double Metaphone in Python |
| Apache Commons Codec | https://github.com/apache/commons-codec | Apache-2.0 | 490 | 2026-09-24 | safe (Java) | Beider-Morse, Double Metaphone, NYSIIS reference |
| talisman (JS) | https://github.com/Yomguithereal/talisman | MIT | 732 | 2024-06-23 | safe | French phonetic algorithms (fonem, phonex, sonnex, soundex2), ports only |
| abydos | https://github.com/chrislit/abydos | GPL-3.0 | 194 | 2022-11-10 | GPL, stale | huge catalogue of phonetic/distance algorithms; use as reference only |
| textdistance | https://github.com/life4/textdistance | MIT | 3,542 | 2025-04-18 | safe | pure-Python reference (slow); Monge-Elkan, bag, etc. |
| Levenshtein (python) | https://github.com/rapidfuzz/Levenshtein | GPL-2.0-or-later (LICENSE read) | 397 | 2026-09-12 | GPL; avoid, RapidFuzz covers it | none |
| string_grouper | https://github.com/Bergvca/string_grouper | MIT | 372 | 2026-07-26 | safe | TF-IDF char-ngram top-n company-name matching |
| sparse_dot_topn | https://github.com/ing-bank/sparse_dot_topn | Apache-2.0 | 424 | 2026-09-14 | safe | sparse matrix multiply with top-n, multithreaded; blocking |
| PolyFuzz | https://github.com/MaartenGr/PolyFuzz | MIT | 803 | 2025-07-10 | safe | quick baselines |
| name_matching (DNB) | https://github.com/DeNederlandscheBank/name_matching | MIT (GitHub) | 169 | 2026-07-02 | safe | company-name matching with multiple metrics; ideas for feature list |
| cleanco | https://github.com/psolin/cleanco | MIT | 360 | 2026-06-23 | **LOW risk**: ships a hand-curated, country-keyed legal-form list (France: sarl..., India: pvt. ltd. ...). A term list, not a dataset, but prefer terms learned from our data (C6). | cross-check of learned legal-form lists |
| uroman | https://github.com/isi-nlp/uroman | MIT-style + attribution clause (LICENSE.txt); PyPI classifier says Apache | 251 | 2024-07-26 | safe with attribution; ships Unicode-derived romanization tables (character tables, not an entity dataset) | **P0** romanization |
| indic_transliteration | https://github.com/indic-transliteration/indic_transliteration_py | MIT | 212 | 2026-09-08 | safe | ISO/ITRANS/HK scheme conversion for Brahmic scripts |
| indic_nlp_library | https://github.com/anoopkunchukuttan/indic_nlp_library | MIT | 648 | 2024-06-07 | safe. Some modules need the separate `indic_nlp_resources` repo. Audit: that repo's README says "released under the MIT License", but GitHub detects no LICENSE file. It ships morph-analyzer and transliteration **models**, so treat it as a pretrained-model dependency and record it. Unicode-offset script conversion should not need it. | Indic script conversion via Unicode offsets |
| aksharamukha | https://github.com/virtualvinodh/aksharamukha | AGPL-3.0 (README "released under GNU AGPL 3.0"; PyPI classifier AGPLv3). Note: the repo root ships a `gpl-3.0.txt` and GitHub detects no license | 220 | 2025-03-25 | **copyleft (AGPL/GPL), avoid** | none |
| libindic soundex | https://github.com/libindic/soundex | LGPL-3.0 (GitHub) / LGPL-2.1+ (PyPI) | 59 | 2019-02-09 | LGPL, stale; reimplement the idea | Indic↔Indic phonetic compare |
| anyascii | https://github.com/anyascii/anyascii | ISC | 424 | 2026-06-06 | safe | fallback transliteration |
| Unidecode | https://github.com/avian2/unidecode | GPL-2.0-or-later (README: "version 2 of the License, or (at your option) any later version"; setup classifier GPLv2+; GitHub sidebar shows GPL-2.0) | 609 | 2026-01-05 | GPL; prefer anyascii | none |
| libpostal | https://github.com/openvenues/libpostal | MIT (code) | 4,894 | 2026-05-13 | **compliance RISK**: parser trained on OpenStreetMap (ODbL) plus OpenAddresses (over 1 billion addresses per README); ships place data | do **not** use |
| usaddress | https://github.com/datamade/usaddress | MIT | 1,637 | 2025-08-07 | **compliance RISK**: loads a CRF model (`usaddr.crfsuite`) trained with parserator on the repo's `training/` data. That folder includes `openaddress_us_ia_linn.xml`, `synthetic_osm_data_xml.xml` and `synthetic_clean_osm_data.xml`, so the model embeds OpenAddresses/OSM-derived external data. Same risk class as libpostal. | avoid; write a rule-based typer |

---

## 10. Models

Only one model sits squarely in this topic's scope. Encoders and rerankers are in `models.md`.

| HF id | License (card) | Compliant | Params | Notes |
|---|---|---|---|---|
| ai4bharat/IndicXlit | `mit` (card metadata `license: mit`, tag `license:mit`; the card README contains only that front matter; https://huggingface.co/ai4bharat/IndicXlit. Code repo https://github.com/AI4Bharat/IndicXlit: MIT, 142 stars, last push 2023-10-13) | Yes (MIT, far below 8B) | ~11M (GitHub README: "a total of 11M parameters") | Transformer transliteration Roman↔native for 21 Indic languages, trained on Aksharantar (26M pairs). Pretrained weights are allowed. **Do not re-train on Aksharantar**, since that would be external data augmentation. README top-1 on Aksharantar **named entities**, in the Indic→En direction we would use, ranges from 12.87 (kas) to 60.77 (kan), e.g. Hindi 59.22, Tamil 35.46, Telugu 57.57, Bengali 52.54. En→Indic ranges from 20.23 to 58.87. So exact-string equality after transliteration is unreliable. Use it only to produce a romanized variant, then fuzzy/phonetic compare. Rule-based uroman or indic_transliteration is deterministic and should come first. |

---

## 11. Concrete adaptation to this challenge

### 11.1 Deterministic inverse-noise normalization (all rules validated on train pairs; maps learned from data where possible)

| Noise | Rule | Source of the map |
|---|---|---|
| case, doubled spaces, junk prefixes (`--`,`<<`,`##`,`***`), brackets `[Inc]` `(Ltd)` | NFKC, casefold, strip non-alphanumeric runs at ends, unbracket, collapse spaces | generic |
| injected diacritics | NFKD then drop combining marks for Latin-script tokens only (never for Indic scripts, where they are vowel signs) | generic |
| leetspeak `0/o`, `1/l/I`, `5/s` | fold confusables **inside alphabetic tokens of names only**. Never touch address numbers. Keep both raw and folded features. | generic, validated |
| "null", "NULL", "<NULL>", "N/A" | drop token; mark field missing when empty | data |
| legal forms anywhere in the name ("LLC Moncada ...") | move to a separate `legal` field, position-independent; compare as its own FS field (agree / one missing / conflict) | **learn** from train pairs: tokens frequently added or deleted between true pairs. Cross-check with cleanco. For France (unseen), take high-frequency trailing tokens of test S1 names (SARL, SAS, ...), label-free. |
| aliases `X dba/aka/fka Y` | split into name variants; score = max over variants (catalog has `n_alias_core_jacc`) | generic |
| domains/handles | strip scheme, www, TLD, `@`; compare space-less forms | generic |
| abbreviations (Rd/Road, Pvt/Private, R./Rue, BD, CH, AV) | canonical token map | **learn** aligned token pairs from train true pairs (US, India). For France, mine from high-confidence tier-0 test links (label-free co-occurrence). |
| US state code↔name; Indian states incl. native script ("महाराष्ट्र") | canonical state id | **learn** from train pairs. Native-script state strings co-occur with Latin state names in linked pairs. |
| France region↔département swaps | canonical region id | Hand-coding a 101-département table is a small **gazetteer, so flag it as a compliance risk**. Preferred: mine département→region co-occurrence from high-confidence test links (same name + street + number), label-free and transductive. |
| number noise (leading zeros, "5935-D", "#", "No", "H.NO", "Door No") | extract digit runs, strip prefixes and leading zeros, keep a suffix flag | generic |
| native-script digits | map to 0-9 (uroman does this) | generic |
| cross-script names | romanize (uroman / indic_transliteration), Indic phonetic skeleton, plus a token dictionary mined from train pairs (e.g. मार्केटिंग↔marketing, प्राइवेट↔private) | train data only |

### 11.2 FS / Splink configuration sketch (P1; names are indicative, check against the Splink 4 API)
- **Inputs:** S1, S2, S3 as three tables, `link_type="link_only"` (optionally `link_and_dedupe` to exploit target-target copies). Partition by country label; fit **one model per country, including France on test**.
- **Comparisons** (each level gets m/u by EM):
  - `name_core`: exact (TF-adjusted) / JW ≥ 0.94 / JW ≥ 0.88 / SoftTFIDF ≥ τ / phonetic-skeleton equal / else.
  - `name_legal`: agree / missing / conflict.
  - `house_number`: exact / normalized-equal / 1-edit / conflict / missing (null level).
  - `street`: exact (TF) / JW ≥ 0.9 / token-overlap ≥ 0.5 / else.
  - `city`: exact / JW ≥ 0.9 / else.
  - `state_canonical`: agree / conflict / missing.
  - `postcode`: exact / first-3 agree / conflict / missing.
  - City, state and postcode are correlated. Either merge them into one "locality" comparison or accept a biased weight that the GBDT will re-weight.
- **Training:**
  - `estimate_u_using_random_sampling` with a large max_pairs (cheap on a 64-core box);
  - λ from deterministic rules (tier-0) with an assumed recall;
  - EM round-robin with blocking on (a) house_number + state, (b) name_core first rare token, (c) postcode.
- **Output use:** `match_weight` as a GBDT feature, and as a standalone baseline for per-country ablations (especially France).
- **Why:** it is unsupervised, so it adapts to France's distribution without labels. It also gives a sanity check on the GBDT: a large disagreement between the FS weight and the GBDT score flags a likely calibration shift.

### 11.3 Features to add to FEATURE_CATALOG.md (classical, CPU)
1. `n_softtfidf_jw90`: SoftTFIDF with JW secondary, θ=0.9, IDF computed per country on S1 ∪ targets (label-free).
2. `n_me_jw_qt`, `n_me_jw_tq`: Monge-Elkan(JW) in both directions on core tokens.
3. `n_q3_overlap`, `a_q3_overlap`: char-3gram overlap coefficient (truncation-robust).
4. `n_core_jw`: JW on the space-joined sorted core tokens. `n_rare_osa`: OSA similarity of the rarest token pair.
5. `n_tf_u`: log(1 / frequency) of the agreed normalized core name in S1 (Splink-style TF adjustment); `n_rare_agree_weight`: sum over shared tokens of −log2 df.
6. `a_num_level`: FS-style level {exact, norm-equal, 1-edit, conflict, missing}, plus `a_num_conflict_idf` (a conflicting number on an otherwise identical street is strong evidence of a different branch).
7. `n_skel_jw`: JW between Indic phonetic skeletons of the romanized names. `n_translit_dict_cov`: share of target tokens mapped by the train-mined dictionary.
8. `fs_weight`: Splink/FS match weight (per-country EM, refit on test).
9. `n_bmpm_jacc`: Beider-Morse code-set Jaccard for French names (P2, only if validation on a pseudo-French split shows gain).

### 11.4 Decision layer
- Many-to-one first: for each target, keep only its best S1 (plus a margin feature; the catalog's `*_tmargin` already encodes this).
- Then per S1, run the expected-F0.5 decoder from `fmeasure.md` on calibrated probabilities.
- Worked illustration: n=4 true matches, 3 sure TPs. Omitting the 4th gives F = 3.75/4.0 = 0.9375. Adding it gives F = 1.0 if right, or 3.75/5.0 = 0.75 if wrong. Adding is worth it only if 0.75 + 0.25p > 0.9375, i.e. **p > 0.75**.
- Singletons (5.58% in train) score 1 only if predicted empty. Use a calibrated "has any match" model per country.

### 11.5 Compute plan (cloud only)
- Normalization and romanization: embarrassingly parallel over records (Modal CPU, 64 cores, multiprocessing).
- Blocking: sparse TF-IDF top-k (sparse_dot_topn) per country.
- Features: RapidFuzz `cpdist(workers=-1)` over pair arrays in chunks of 5-10M pairs, written as parquet.
- FS/EM: Splink on DuckDB on the same box, or our own pattern-count EM (fastLink trick) in numpy.
- GBDT on GPU (T4/L4) or CPU.
- Nothing runs on the laptop.

---

## 12. What we should do

**P0 (do now; biggest expected gain per hour)**
1. Implement the inverse-noise normalization of 11.1 with **data-learned** maps (abbreviations, legal forms, states, the cross-script token dictionary) from train pairs. For each rule, measure on train how much it moves positive-pair similarity vs random same-country pairs. Expected: fewer false negatives on abbreviation, legal-form, state and cross-script noise. Evidence: Chaudhuri 2003 error model; MAMBA standardization; Christen 2006 Rec. 1-2.
2. Add the classical features in 11.3 (items 1-7) to the GBDT and ablate by country and by noise type. Expected: gains on truncation, reorder and typo-in-rare-token cases. Evidence: Cohen 2003 (SoftTFIDF best, learned combination best overall); Christen 2006 (overlap coefficient, Winkler prefix).
3. Numeric comparison levels with an explicit **conflict** state for house numbers and postcodes. Expected: fewer false positives among same-name S1s (names are 26-36% non-unique in S1). Evidence: Ditto's numeric weakness; FS level design; DATA_ANALYSIS §7.
4. Cross-script bridge (uroman or indic_transliteration romanization, Indic skeleton, train-mined token dictionary) for names, as features and extra blocking keys. Expected: large recall gain on India targets with native-script names (27.9% of India S2 names are non-ASCII). Evidence: uroman paper; DATA_ANALYSIS §7 (p10 ≈ 11 without it).
5. Many-to-one argmax plus per-S1 expected-F0.5 decoding on calibrated scores (with `fmeasure.md`).

**P1**
6. FS/Splink per-country model with TF adjustments, refit on unlabeled test per country (France), used as the `fs_weight` feature and as an ablation baseline. Expected: better France robustness and a calibration cross-check. Evidence: Splink/fastLink EM; Mudgal (parity of classical on structured data); Akbarian Rastaghi (PLM shift fragility).
7. Tier-0 deterministic anchors, then label-free mining of test-side dictionaries for France (département↔region, abbreviations R./BD/CH/AV, legal forms) from high-confidence links. Expected: France recall without external data. Evidence: C6 in COMPLIANCE_CHECKLIST; Splink λ-from-rules practice.
8. Collective step: cluster near-duplicate targets within and across S2/S3 (they share the same corrupted base) and add cluster-support features (the catalog's planned H). Expected: recall on heavily corrupted copies. Evidence: DATA_ANALYSIS §5; Steorts 2016 (joint link plus dedup) conceptually.
9. Leave-one-country-out validation (US→India, India→US) of every classical feature, as a proxy for France.

**P2**
10. Beider-Morse or French phonetic keys (from Apache Commons Codec rules or talisman ports) as a France-only blocking key and feature. Keep only if the pseudo-OOD ablation helps.
11. Learned edit costs (MARLIN-style) or a learned confusable-substitution table from train alignments (RapidFuzz editops over true pairs). Expected: small gains on systematic typos.
12. Isotonic calibration per country, with a check that FS weights and GBDT probabilities agree on held-out data.

**P3 (ideas; mostly do not ship)**
13. Symphonym-style phonetic embeddings: **not compliant** as-is (trained on GeoNames/Wikidata/Getty toponyms, i.e. external gazetteers). Idea only.
14. libpostal / usaddress parsers: **compliance risk** (embedded models trained on external address data). Use rule-based component typing instead.
15. Bayesian bipartite samplers (Sadinle, Steorts): principled, but they do not fit our many-to-one structure or the scale.

---

## 13. Not verified / dropped
- **Double Metaphone** (Philips 2000, C/C++ Users Journal 18(6):38-43): metadata seen in search results pointing to the ACM DL entry 10.5555/349124.349132 and a mirror of the article; I did not fetch a primary page (ACM DL blocks automated fetches). Status: PARTIALLY VERIFIED (trade magazine, not peer-reviewed).
- **NYSIIS** (Taft 1970, NYSIIS Special Report): UNVERIFIED primary source. Cited only as the algorithm implemented in jellyfish.
- **Levenshtein (1966)**, Soviet Physics Doklady: foundational, not verified from a primary page. Not listed in the table.
- DBLP and ACM DL pages were blocked for automated fetching (bot-protection challenge, not bypassed). DOIs were verified through the Crossref API instead.
- Udupa & Kumar (EMNLP 2010) hashing for multilingual name search: my guessed ACL Anthology ID was wrong and the paper was not found; dropped.
- Anand et al. 2019 (Hindi Soundex): existence verified on Crossref; content not read. Do not rely on its claims.
- WebFetch summaries of PDFs were unreliable twice (the Mann 2016 summary was invented by the fetch summarizer). All numbers above come from text extracted from the PDFs and read directly.
- (Audit) **Winkler 1990 venue**: ERIC ED325505 confirms the title, author, year, abstract (1990 census PES, extension of Jaro, partial-agreement weights) and 8 pages. The "ASA Proceedings, Section on Survey Research Methods, pp.354-359" venue appears only in secondary indexes (Semantic Scholar, BibSonomy). Status: venue UNVERIFIED; content verified.
- (Audit) **Double Metaphone**: still PARTIALLY VERIFIED. Search results show an ACM DL record (10.5555/349124.349132), but ACM DL returned 403 on fetch, and 10.5555 is not a Crossref DOI.
- (Audit) **Zeakis et al. 2023**: quotes were re-read from the arXiv version (2304.12329v1), which has the same title plus "[Experiment, Analysis & Benchmark]". The quoted conclusions match. The PVLDB PDF itself was not re-fetched.

---

## Audit log

This was an independent citation and license audit, done on 2026-09-25. It treated every item as possibly hallucinated until a primary source confirmed it. It covered 80 items: 51 papers in §8, 26 libraries in §9, 1 model in §10, plus Double Metaphone and NYSIIS from §13. Nothing was run locally except HTTP metadata fetches and PDF text extraction; no models were run.

### Method (primary sources used)
- **DOIs (39)**: Crossref REST API `api.crossref.org/works/<doi>`. Checked title, subtitle, authors, container, volume, issue, pages, and print and online dates. **All 39 DOIs resolve, and title, authors and venue match the notes.**
- **arXiv (6)**: export.arxiv.org API. Checked 2205.03018, 2311.13923, 2307.01231, 2004.00584, 2304.12329 and 2601.06932 for title, authors, comment and abstract.
- **PDF full text** (text extracted and grepped for each quoted number or phrase):
  - Cohen 2003 (CMU)
  - Christen TR-CS-06-02 (ANU)
  - Chaudhuri 2003 (Microsoft Research)
  - Gravano 2001 (VLDB)
  - Mann 2016 (VLDB)
  - fastLink (Harvard)
  - Mudgal 2018 (UW-Madison)
  - Ditto (arXiv)
  - Papadakis 2023 (arXiv)
  - Zeakis 2023 (arXiv)
  - Moreau 2008 (ACL Anthology)
  - uroman (ACL Anthology)
  - MAMBA (census.gov)
  - Wasi & Flaaen (author PDF)
- **Landing pages**:
  - ERIC ED325505
  - AAAI KDD-96 page
  - PMLR v70 (Guo)
  - NeurIPS 2011 (Dembczynski)
  - Google Patents US1261167
  - stevemorse.org BMPM
  - IJPDS Splink page
  - Imperial Spiral (Hand & Christen abstract)
  - Cohen's CMU publication list
- **GitHub**: `gh api repos/<owner>/<repo>` gave SPDX license, stars and `pushed_at` for all 26 repos plus IndicXlit and indic_nlp_resources. LICENSE, README, DESCRIPTION, pyproject and setup.cfg files were read wherever the SPDX id was missing or ambiguous: fastLink, python-Levenshtein, uroman, aksharamukha, libindic soundex, Unidecode, jellyfish, cleanco, dedupe `api.py`, usaddress, libpostal, Splink docs and API, and RapidFuzz `process_py.py`.
- **PyPI JSON**: license and classifier fields for aksharamukha, libindic-soundex, Metaphone, splink, uroman, jellyfish, rapidfuzz and Levenshtein.
- **Hugging Face API**: `huggingface.co/api/models/ai4bharat/IndicXlit` and the raw card README.

### Confirmed exactly as written (selection of load-bearing numbers)
- **Cohen 2003**: SoftTFIDF θ=0.9. UVA MaxF1/AvgPrec 0.89/0.91; CoraATDV 0.85/0.914; TFIDF 0.79/0.84 and 0.84/0.907. Jaro variants take "about 1/10 the time" of Monge-Elkan. SoftTFIDF about 13 s, JW about 20 s, TFIDF 1.2 s. Blockers 93.3-100% (avg 98.9%; 4-gram 99.0%). Census "perform poorly" quote. SVM combination better "particularly at extreme recall levels". SFS k=0.5 and the -log P_D(w) IDF derivation.
- **Christen 2006 TR**: 123 tests on four datasets. "Pattern matching clearly outperform phonetic encoding." Phonex beats Phonix and Double-Metaphone. Recommendations 3-9 are quoted correctly, including "Even small changes of the threshold...".
- **fastLink**: 150,000 x 150,000 in under 6 h on 1 core and under 3 h on 8 cores. Python recordlinkage took more than 24 h for 20,000 x 20,000. Voter files of over 160 million records each. "the number of pairs per agreement pattern becomes our sufficient statistic". MAR missing-data assumption. JW thresholds 0.88/0.94 (Winkler), 0.85/0.92 and 0.92/0.88. The blocking caveat when "the expected number of matches is relatively small".
- **Mudgal 2018**:
  - structured 87.9 vs 88.8 average F1, 5.4 h vs 1.5 min;
  - textual +3.0-22.0% F1;
  - BeerAdvo 72.7 vs 78.8; iTunes-Amazon1 88.5 vs 91.2; Walmart-Amazon1 67.6 vs 71.9; Amazon-Google 69.3 vs 49.1; Company 92.7 vs 79.8.
- **Ditto**: numeric-data limitation quote. 789,409 x 412,418 records, 10,652,249 candidates. F1 96.53. 6.49 h vs 1.69 h.
- **Chaudhuri 2003**: Customer[name, city, state, zip] reference of "about 1.7 million tuples". The paper also says "approximately 2 million" once in the results section, an internal inconsistency. q=4. "2 to 3 orders of magnitude faster than the naive algorithm". Error-model table: spelling, abbreviation, missing, truncation, token merge, transposition.
- **Other quotes**: Mann 2016 prefix-filter and "rarely competitive" quotes. Papadakis ICDE 2024 "almost linearly separable" quote. Zeakis "SentenceBERT models consistently outperform". Moreau "better than all existing similarity metrics on two corpora", plus a French-speaking media corpus. The uroman abstract quote. Guo 2017 quotes. Splink README and docs quotes (1M records in about a minute, 100+ million, bag-of-words caveat, TF adjustment "does not have to be estimated by the EM algorithm", random-sampling u, "anchor"). The IJPDS "downloaded 2 million times" line. RapidFuzz `cpdist` "Supply -1 to use all available CPU cores".
- **Symphonym**: R@1 85.2%, MRR 90.8% on MEHDIE, 20 writing systems, trained on GeoNames, Wikidata and Getty TGN. The compliance-risk flag stands.
- **IndicXlit**: HF card `license: mit` and card README empty apart from front matter. GitHub README "~11M" and "a total of 11M parameters". Indic-to-En named-entity top-1 from 12.87 (kas) to 60.77 (kan); hin 59.22, tam 35.46, tel 57.57, ben 52.54. En-to-Indic named-entity range 20.23-58.87. **Compliant (MIT, far below 8B).**
- **Licenses exactly as stated**: RapidFuzz MIT; Splink MIT; fastLink "GPL(>= 3)" (DESCRIPTION); dedupe MIT; recordlinkage BSD-3-Clause; jellyfish MIT (no Double Metaphone; source moved to codeberg); Apache Commons Codec Apache-2.0; talisman MIT (french/: fonem, phonetic, phonex, sonnex, soundex, soundex2); abydos GPL-3.0; textdistance MIT; python-Levenshtein GPL-2.0-or-later (LICENSE text; GitHub shows NOASSERTION); string_grouper MIT; sparse_dot_topn Apache-2.0; PolyFuzz MIT; name_matching MIT; cleanco MIT (termdata has "France" with sarl and "India" with pvt. ltd.); uroman MIT-style plus attribution clause (LICENSE.txt), with PyPI classifier Apache and version 1.3.1.1; indic_transliteration MIT; indic_nlp_library MIT; libindic soundex LGPL-3.0 on GitHub and LGPL-2.1+ in setup.cfg/PyPI (the compare example returns 2); anyascii ISC; libpostal MIT code plus OSM/OpenAddresses training data ("over 1 billion addresses"). Stars and last-push dates match within 1 star.
- dedupe `RecordLink.join` defaults to `constraint="one-to-one"`. "many-to-one" is also available, and `Gazetteer.search(..., n_matches=...)` exists.

### Changed (what and why)

| # | Item | Problem found | Fix applied |
|---|---|---|---|
| 1 | Mudgal et al. 2018, "Company" dataset (§4.2, §7, §8 row 29) | Described as "company names without informative attributes". The paper says Company "tries to match company homepages and Wikipedia pages describing companies": one long free-text attribute. The misdescription overstated how well the result transfers to our short names. | Corrected in all three places and flagged the weak transfer. |
| 2 | Ditto (§0 TL;DR 4b, §7) | "had to tag street numbers by hand" / "hand-tagged". The paper says they "implemented a simple recognizer that tags the first number string in the addr attribute and the last 4 digits of the phone". | Wording corrected. |
| 3 | Winkler 1990 (§2.1, §8 row 3) | Notes put the venue as ASA Proc. Survey Research Methods, "ERIC record read". The ERIC record lists only a non-journal 8-page report. The ASA venue and pp.354-359 appear only in secondary indexes. | Venue marked UNVERIFIED; content kept (it is on ERIC). |
| 4 | MAMBA, Cuffe & Goldschlag 2018 (§3) | "3-gram prefilter at 0.3" is incomplete: 0.3 applies to the EIN, ZIP5, city and ZIP3 blocks, and 0.4 to the state and non-geographic blocks. The soundex restriction applies to the state and non-geographic passes, not only the residual. An EIN block was omitted. | Corrected; added the FEBRL-derived comparators (stated in the paper). |
| 5 | Papadakis, Skoutas, Thanos & Palpanas, CSUR 53(2) (§6, §8 row 32) | Year given as 2020. Crossref shows print 2021-03-31 and online 2020-03-20; pages 1-42. | Year set to 2021 (online 2020), pages added. |
| 6 | Bai, Binette & Reiter 2025 (§8 row 27) | The arXiv 2311.13923 title differs ("Optimal F-score Clustering for Bipartite Record Linkage"). | Noted. |
| 7 | Anand et al. 2019 (§8 row 48) | Crossref: online 2019-12-03, print 2020. | Both dates given. |
| 8 | Unidecode (§9) | License given as "GPL-2.0". README and classifier say GPL v2 **or later**. | Set to GPL-2.0-or-later. Still copyleft; the verdict (avoid) is unchanged. |
| 9 | usaddress (§9) | Risk basis given as "its own labelled US addresses". The repo `training/` folder includes `openaddress_us_ia_linn.xml` and OSM-derived synthetic files, and the model is built from these. | Risk statement strengthened: it embeds OpenAddresses/OSM-derived data. |
| 10 | aksharamukha (§9) | README and PyPI say AGPL-3.0, but the repo root ships `gpl-3.0.txt` and GitHub detects no license. | Discrepancy recorded. Still copyleft, avoid. |
| 11 | Splink (§2.3, §9) | The blocking-analysis function names are correct for release v4.0.17, but master (5.0.0.dev5) renamed them. | Added a version-pin note (`splink==4.0.17`). |
| 12 | RapidFuzz and Splink stars (§9) | 4,138 to 4,139 and 2,428 to 2,429 (drift). | Updated. |
| 13 | indic_nlp_library (§9) | The note said "check before use". indic_nlp_resources has MIT in its README but no LICENSE file on GitHub, and it ships models. | Details recorded. |
| 14 | metaphone (§9) | GitHub metadata missing. | Added repo URL, BSD-3-Clause, 86 stars, last push 2024-02-28. |
| 15 | Cohen 2003 (§4.2, §8 row 5) | Pages missing; venue not tied to a primary page. | Added IIWeb 2003 pp.73-78 from the author's CMU publication list. |
| 16 | Mann 2016 (§6, §8 row 18) | Pages missing. | Added pp.636-647 (Crossref). |
| 17 | Wasi & Flaaen 2015 (§3, §8 row 42) | Description was Crossref-only. | Author PDF read: `stnd_compname` splits official/DBA/FKA/entity-type and `reclink2` does many-to-one matching. Added, since both are directly relevant. |
| 18 | §1 data table | "p10 ≈ 11 for India targets" holds for S2 India; S3 India is 14 (DATA_ANALYSIS §7). | Clarified. (Out of citation scope; spot-check only. 27.9%, 5.58%, 7.64M, 64%/74% and 77-84% were also found in DATA_ANALYSIS.md.) |
| 19 | IndicXlit (§10) | License URL and code-repo metadata missing. | Added the HF URL, the exact card field and the GitHub repo (MIT, 142 stars, last push 2023-10-13). |

### Still not fully verified (kept, clearly marked)
- **Winkler 1990**: the ASA proceedings venue (see #3).
- **Double Metaphone** (Philips 2000): the ACM DL page returns 403 and 10.5555 is not a Crossref DOI. Partially verified.
- **NYSIIS** (Taft 1970): no primary page found. UNVERIFIED; it is cited only as an algorithm implemented in jellyfish.
- **Crossref-only items**: content paraphrases for Hernández & Stolfo, McCallum et al., Belin & Rubin, Larsen & Rubin, Sadinle, Steorts et al., Singh et al., Christen & Vatsalan, Febrl, Magellan, Bayardo, Xiao, Jiang, Navarro, Damerau, Christen 2012 book and TKDE survey, and Anand et al. For these, bibliographic metadata is verified but the abstract text was not read in this audit. They are background citations, and no numbers are quoted from them.
- **Tu et al. 2022 (DADER)**: an abstract via the Semantic Scholar API confirms the framing (domain adaptation for deep ER without or with few target labels).

### Removed
- Nothing. No listed paper, repo or model turned out to be fabricated.
