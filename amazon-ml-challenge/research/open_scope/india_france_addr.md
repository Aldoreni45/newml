# Open-scope sweep: India and France address/name normalization, transliteration, and legal forms

Sweep key: `india_france_addr`. Date: 2026-09-25.
Scope: Indian address normalization, parsing and matching (Flipkart, Myntra, Meesho, Amazon IN, Delhivery,
Swiggy); PIN structure; landmarks; romanization variation of Indian names; French address normalization (BAN
ecosystem, La Poste/AFNOR, CNIG); region vs département; French legal forms.
**Method only.** Nothing here should be shipped as a lookup table at inference unless it is re-derived from
competition data (see the compliance note in section 0).

How each item was checked: primary page fetched (arXiv abs, ACL Anthology, Amazon Science, Google Patents,
GitHub API or raw source, official .gouv.fr or insee.fr page). For PDFs I extracted the text locally with PyMuPDF
and read the relevant section. Numbers are quoted only where I read them. Items I could not open are marked
**UNVERIFIED**.

Overlap with `research/notes/name_address.md`: that file already covers Aksharantar/IndicXlit, Dakshina, uroman,
RomanSetu, MuRIL, deepparse/libpostal, Rustogi 2018 (landmarks), Mangalgi 2020 (address classification), Swiggy
POI mining, AFNOR NF Z10-011, and PIN/ZIP/FR postcode structure. Below I add **new sources**. For three overlapping
papers I add **method details not captured there** (Mangalgi's spell-variant rule, IndicXlit per-language
named-entity accuracy, and the Hunterian/schwa angle).

---

## 0. TL;DR (what changes in our pipeline)

1. **Make canonicalization maps country-scoped. P0.** France reuses tokens with other meanings: `st` = *saint*
   (US: street), `rd` = *route départementale* (US: road), `ch/che/chem` = *chemin*, `r` = *rue*, `n` / `no` / `N°`
   = number marker. This is all in the MIT-licensed addok-fr synonym file (222 mappings). A global map learned on
   US+IN train would mis-canonicalize FR test records. Key the learned maps on `(country, field)` and bootstrap FR
   maps from **test pseudo-pairs**.
2. **French house-number grammar. P0.** Glue number+suffix (`12 bis` → `12bis`), then fold
   `bis/ter/quater/quinquies/sexies` → `b/t/q/c/s` (addok-france, MIT). Strip the number marker `N°/no`, strip
   leading zeros, and compare the base number and the suffix separately. The CNIG standard says suffixes are
   `bis, ter, quater, quinquies` **or** letters A/B/C, so `12 bis` vs `12 B` is a *soft* match, not a conflict.
3. **Remove French postal-routing tokens before address similarity. P0.** Strip `CEDEX n`, `BP n`, `CS n`,
   `TSA n` and `CIDEX`. When CEDEX is present, compare the postcode **only on its 2-digit département prefix**
   (addok-france's clean patterns do exactly this). Also map `s/` → *sur*, `s/s` → *sous*, drop a leading
   `lieu-dit`, and drop `2e étage`.
4. **Region↔département swaps: learn the reconciliation transductively. P0.** France has 13 metropolitan
   regions over 96 metropolitan départements (fr.wikipedia). The noise swaps them. Without external lookup, build
   `P(admin_token | postcode[:2])` from *unlabeled test* S1∪S2∪S3. Two admin tokens are "compatible" if their
   postcode-prefix distributions overlap. Features: `admin_compat`, `admin_same_level`.
5. **Postcode is a soft, hierarchical signal, never a hard gate. P0.** Flipkart (COLING 2025 Industry): when
   pincodes were randomly altered, the per-pincode production model fell from 64.3% to 46.7% (<100 m). The
   pincode-free model stayed at 78.9%. Meesho GeoIndia (EMNLP 2024 Industry): treating the whole PIN as one token
   raised mean error from 0.6 km to 2.6 km (Nagpur). Splitting it hierarchically fixed that. So: add a
   `pc_common_prefix_len` feature (0-6 IN, 0-5 US/FR). Keep text-only blocking views so recall does not depend on
   the postcode ("number noise" can corrupt postcodes).
6. **Transductive spell-variant clustering for Indian locality tokens. P0.** Myntra's rule (KDD-workshop 2020):
   `Ta` is a variant of `Tb` iff `count(Ta) < count(Tb)` **and** `Lev(Ta,Tb) < 3` **and**
   `Metaphone(Ta) == Metaphone(Tb)`, for tokens longer than 6 characters. Either test alone fails: *Bommasandra*
   vs *Dommasandra* are one edit apart but are different places, and *Mathkur* vs *Mathikere* share a Metaphone
   code but are different places. Run it per country on test tokens and replace each variant with its most
   frequent "leader". The same paper and GeoIndia also do **frequency-based compound splitting** (`hsrlayout` →
   `hsr layout`, `NewDelhi` → `New Delhi`).
7. **Neural Indic→Latin romanization alone rarely gives the exact spelling. Use top-k plus phonetic keys. P1.**
   IndicXlit's README gives Indic→En top-1 on the *named-entity* test set for our seven scripts: hin 59.22, ben
   52.54, guj 50.56, kan 60.77, mal 45.23, tam 35.46, tel 57.57. About half of romanized names will not match
   exactly. Take the max similarity over beams (top-4/5) and over our learned dictionary and rule romanizers,
   following the ensemble idea in Google Maps' Indic transliteration. Add a Hunterian-style key, which drops vowel
   length and does not tell retroflex from dental. That is how official Indian place names are romanized.
8. **Acronyms written in Indic script. P1.** Google Maps built a dedicated acronym module for this case. For us:
   a native-script token that spells out Latin *letter names* (e.g., an SBI-style acronym written as syllables)
   should map to the Latin initials. Learn the letter-name→letter table from train cross-script positives whose
   Latin side is an all-caps token of 5 letters or fewer.
9. **French name cleanup. P0.** Follow the annuaire-entreprises analyzer (MIT, the official French company search):
   remove elisions (`l' d' qu' n' s' j' c' m' t' jusqu' lorsqu' puisqu' quoiqu'`) **before** deleting apostrophes,
   and delete dots so `S.A.R.L.` → `SARL`. Also ICU folding, French stopwords and minimal French stemming.
10. **French legal forms: group them into families so they never produce a false conflict. P0.** EURL = SARL
    unipersonnelle and SASU = SAS unipersonnelle (Bpifrance). INSEE merged SASU into code 5710 in 2020. So set
    `lf_conflict = 0` inside {SARL, EURL} and inside {SAS, SASU}. SA, SNC, SCI (société civile), SCOP and EI are
    distinct. Generic company words (`Société/Sté/Soc`, `Établissements/Ets`, `Compagnie/Cie`) get low weight.
    Learn them from IDF, not a list.

---

## 1. India: address normalization, parsing, matching, geocoding

### 1.1 GeoIndia: A Seq2Seq Geocoding Approach for Indian Addresses (Meesho). VERIFIED
- Singhal, Aditya, Todwal, Jain, Mukherjee. *EMNLP 2024 Industry Track*, pp. 395-407.
  https://aclanthology.org/2024.emnlp-industry.29/ (PDF text read locally).
- **What:** Hierarchical H3-cell prediction as a seq2seq task. Models: Flan-T5-base (220M) and QLoRA Llama-3-8B.
  About 29 models, one per state. Abstract: "more than an 50% reduction in mean distance error and more than a 85%
  reduction in 99th percentile distance error compared to Google Maps" in multiple states.
- **Preprocessing (read in §4.1.2):** lowercase and trim punctuation; "merged ordinal indicators with numbers"
  (`1 st Avenue` → `1st Avenue`); **"removed sequences over six digits to avoid confusion with pincodes"**
  (phone numbers); **probabilistic camel-case splitting** by observed frequency (`NewDelhi` → `New Delhi`);
  **redundant-phrase reduction** that keeps the first occurrence (`JP Nagar, Bangalore, JP Nagar` → `JP Nagar,
  Bangalore`); and a custom-trained tokenizer.
- **PIN finding (App. D.2):** a whole-PIN special token "increased the mean distance error from 0.6 km to 2.6 km
  in Nagpur". Splitting as `4400` + `14` fixed it: "each digit encodes progressively smaller geographical
  regions".
- **Adaptation:**
  - (a) Add phone-number removal (digit runs of 7+ that are not the country's postcode length) and ordinal glue to
    the IN/US normalizer. P0.
  - (b) Add a duplicate-phrase collapse, since injected tokens and aliases can repeat phrases. P1.
  - (c) Replace the binary PIN features with `pc_common_prefix_len` plus the prefix-3/prefix-2 flags. P0.
  - (d) Per-state models suggest **country-conditional features**. Train one LightGBM with country interaction
    features rather than per-country models, because France has no training labels. P2.

### 1.2 Geo-Spatially Informed Models for Geocoding Unstructured Addresses (Flipkart). VERIFIED
- Singh (IIT Bombay, Flipkart intern), Devanapalli, Bellala, Goel. *COLING 2025 Industry Track*, pp. 236-242.
  https://aclanthology.org/2025.coling-industry.19/ (PDF text read locally).
- **What:** The production system is fastText, **one model per pincode**, over H3 resolution-10 cells. The new
  system is an LLM-backbone geocoder that does not take the pincode as input. The abstract reports +20% over SOTA
  and +54% over a commercial system in drift accuracy within 100 m, and 8% fewer potential incorrect hub
  assignments. Data is split by delivery count (train <15, val 15-20, test >20 deliveries).
- **Key evidence (Table 2, synthetic wrong pincodes):**
  - Production <100 m: 64.3% (actual PIN) → **46.7%** (incorrect PIN).
  - Pincode-free model: 78.9% in both cases.
  - "users frequently provide incorrect pincodes".
- **Adaptation:** our noise includes "number noise", which can corrupt postcodes. (a) Blocking must include
  non-postcode views; we have them, so verify the recall of pairs whose postcodes disagree. (b) In LightGBM, add
  `pc_disagree × name_sim_high` and `pc_disagree × street_sim_high` interactions so a corrupted postcode cannot
  veto a strong match. (c) Cheap check: among train positives, measure the rate of postcode disagreement per
  country and per source. P0.

### 1.3 Learning Geolocations for Cold-Start and Hard-to-Resolve Addresses via Deep Metric Learning (Amazon IN/UAE). VERIFIED
- Kothari (Govind), Sohoney. *EMNLP 2022 Industry Track*, pp. 322-331.
  https://aclanthology.org/2022.emnlp-industry.33/ (PDF text read locally).
- **What:** RoBERTa (6 layers) pre-trained from scratch on addresses with a 52K BPE vocabulary, then fine-tuned
  with a triplet loss (margin 5, Euclidean). Weak labels come from delivery scans. **Multi-resolution triplets:**
  positives from the same H3 cell at L ∈ {11, 10, 9}, negatives from the parent's 1-skip ring. They also add
  **city-level negatives because "addresses in close vicinity tend to differ only in the header part"**. 37M
  triplets for IN, 7M for UAE. Preprocessing is minimal (collapse repeated spaces and punctuation).
- **Results (abstract):** delivery defects −22% (IN), −55% (UAE); p50 distance −43% (IN), −90% (UAE).
- **Adaptation:**
  - (a) If we fine-tune e5 (optional path), sample hard negatives at several granularities: same postcode, same
    postcode prefix-3/prefix-2, same city. These are the siblings a macro per-S1 F0.5 punishes. P1.
  - (b) For LightGBM, the "header vs tail" insight means computing similarity **separately on the head tokens**
    (house/building/number) and the **tail tokens** (locality/city/state/postcode), plus the interaction
    `tail_equal ∧ head_diff`. That is the classic sibling false positive. P0 if not already present.

### 1.4 Amazon address-matching line (building-level same/not-same). VERIFIED (abstract level)
- Maheshwary, Sohoney. *Learning Geolocation by Accurately Matching Customer Addresses via Graph based Active
  Learning*. WWW '23 Companion. https://www.amazon.science/publications/learning-geolocation-by-accurately-matching-customer-addresses-via-graph-based-active-learning
  - "9.3% absolute improvement in recall on average across IN and UAE while preserving 95% precision". A/B:
    delivery precision +7.84%, defects −12.32%. Combines text, delivery history and graph theory to pick
    informative and diverse pairs to label.
- Paul, Maheshwary, Sohoney. *Accurate customer address matching via weak supervision for geocode learning*.
  ACM SIGSPATIAL 2024. https://www.amazon.science/publications/accurate-customer-address-matching-via-weak-supervision-for-geocode-learning
  - Transformer encoder in place of manual features. "15.57% improvement in recall at 95% precision" across 4
    geographies. A/B delivery precision +8.09%, defects −11.78%.
- **Adaptation:** both report **recall at a fixed high precision**, which suits F0.5. The graph idea transfers as
  **training-pair selection**: sample LightGBM negatives from the *candidate graph*, i.e. pairs that share a block
  neighbor or sit in the same connected component at a loose threshold, instead of random ones. This mirrors
  "informative and diverse pairs". P2. (The weak labels from delivery scans do not apply; we have labels.)

### 1.5 Deep Contextual Embeddings for Address Classification in E-commerce (Myntra): spell-variant rule. VERIFIED
- Mangalgi, Kumar, Ravindra Babu T. AI for Fashion Supply Chain workshop (KDD), Aug 2020. arXiv 2007.03020
  (PDF text read locally). Also listed in `notes/name_address.md` #37; the **preprocessing details below are new**.
- **Basic cleaning:** remove special characters, lowercase, **remove numbers longer than six digits** (phones),
  append the pincode after cleaning.
- **Probabilistic splitting (§4.2):** split a compound token if the parts' corpus frequencies beat the compound's
  (`hsrlayout` → `hsr` + `layout`).
- **Spell correction (§4.3):** leader clustering, O(n). `Ta` is a variant of `Tb` iff `Count(Ta) < Count(Tb)` &
  `Levenshtein(Ta,Tb) < 3` & `Metaphone(Ta) == Metaphone(Tb)`. Only tokens with length > 6 are candidates. Counter
  examples: Bommasandra/Dommasandra (edit distance 1, different phonetics) and Mathkur/Mathikere (same Metaphone,
  edit distance too large).
- **Bigram separation (§4.4):** handles a missing space combined with a typo.
- **Adaptation (P0, cheap, CPU):** run it transductively on **test** tokens, per country and per field (city,
  locality). Use our Indic phonetic key instead of Metaphone for IN and a French phonetic key for FR (see 3.8).
  Add a feature `canon_token_jaccard` on leader-mapped tokens. Record the leader map, since it is derived only from
  competition text.

### 1.6 Delhivery patents: hierarchical blocking plus a similarity threshold for "same address". VERIFIED (Google Patents)
- **US 11,381,928 B2**, *System and method for assigning a unique identification to an address*. Delhivery;
  inventors Kabir Rustogi and Rahul Kumar; priority 2020-03-16, published 2022-07-05.
  https://patents.google.com/patent/US11381928B2/en
  - The claims describe: parse into **hierarchical addressing blocks** (state, city, locality, sub-locality,
    street, building, house number, postal code); choose a reference block such as sub-locality; take the
    candidates that share it; compare the **lower-hierarchy portions** with "string metrics, fuzzy searching, or
    phonetic analysis"; assign the same ID when the similarity is above a threshold.
- **US 11,343,638 B2**, *System and method for validating accuracy of geographic location for an address*.
  Rustogi, Kumar, Soni; filed 2020-10-07, granted 2022-05-24. https://patents.google.com/patent/US11343638B2/en
  - The address hierarchy is stored as a DAG of parent→child entities.
- **Adaptation:** (a) This is our blocking-then-compare, with *hierarchy-conditioned* comparison. It supports the
  head/tail split in 1.3. (b) Build a **locality→city→state DAG from S1** by co-occurrence, without external
  data. Feature `hier_consistent` = the other side's city/state is a known parent of this side's locality. That
  catches the "missing components" and "state code vs name" noise without penalizing them. P1.
- Background (**UNVERIFIED**; the Medium post returned 403): Rustogi's post "Learning to Decode Unstructured
  Indian Addresses" (https://medium.com/@kabirrustogi/learning-to-decode-unstructured-indian-addresses-c80ffcda2e84).
  According to search snippets only, graphical models learn locality names and their "alternative spellings"
  unsupervised from millions of addresses, and AddFix v3 resolves the locality for >90% of shipments with a
  median precision of 200 m.

### 1.7 Swiggy: Address Location Correction System for Q-commerce. PARTIALLY VERIFIED
- Bytasandram (Reddy), Sadu, Ganesan, Mathew. AIMLSystems 2022 (industry track).
  https://dl.acm.org/doi/10.1145/3564121.3564800. I read the search-result abstract; the Swiggy Bytes blog fetch
  failed.
- **Method (per abstract/snippets):** a self-supervised classifier for "is this GPS location accurate". It builds
  training data (i) by perturbing locations with Gaussian noise and (ii) **by swapping pairs of addresses**.
- **Adaptation:** "swap" negatives are a hard-negative recipe for us too. Swap the address between two S1 records
  in the same city with different entities to make "name matches, address wrong" training pairs, and the reverse.
  This teaches the model not to over-trust one field. It only uses our own train data. P2.

### 1.8 Indian address parser with IndicBERTv2-CRF (open source). VERIFIED (README)
- https://github.com/howdoiusekeyboard/indian-address-parser (MIT). 15 entity types, including house number,
  floor, block, sector, **gali**, colony, area, subarea, **khasra**, pincode, city and state. It includes a
  Devanagari→Latin "HindiTransliterator" and an abbreviation-expanding normalizer. Reports 94.3% F1 on **150**
  balanced test samples, which is a tiny evaluation.
- **Adaptation:** P3. We do not need a parser. The *entity inventory* is a useful checklist of IN head/tail tokens
  (gali, khasra, sector, block, phase, colony, nagar) to verify that our learned stop/role tokens cover them.

### 1.9 Other Indian items (context, lower value)
- Gupta, Gupta, Garg, Garg. *Improvement in Semantic Address Matching using NLP*. INCET 2021; arXiv 2404.11691.
  https://arxiv.org/abs/2404.11691. OCR → BM25 candidates → BERT refinement. Generic. P3.
- Bhattacharya, Sathya, Rustogi, Raskar. *Economic Impact of Discoverability of Localities and Addresses in
  India*. arXiv 1802.04625 (2018). https://arxiv.org/abs/1802.04625. Poor addressing "costs India $10-14B
  annually". Motivation only.
- **UNVERIFIED (403):** *GeoIndia V2: A Unified Graph and Language Model for Context-Aware Geocoding*, CIKM 2025,
  https://dl.acm.org/doi/10.1145/3746252.3761512. Title and venue come from search results only.

### 1.10 Indian state codes are unstable. Learn them, don't hardcode them. VERIFIED
- ISO 3166-2:IN (Wikipedia, change log): on **2023-11-23**, IN-OR → IN-OD (Odisha), IN-CT → IN-CG
  (Chhattisgarh), **IN-TG → IN-TS (Telangana)**, and IN-UT → IN-UK (Uttarakhand).
  https://en.wikipedia.org/wiki/ISO_3166-2:IN. Per search snippets (not verified on a primary source), Telangana
  vehicle plates moved the other way (TS → TG). One state can therefore appear under two conflicting 2-letter
  codes.
- **Adaptation (P0, trivial):** learn the `state_code ↔ state_name` many-to-one map from train positives by
  co-occurrence, the same way as `notes/name_address.md` §5.3. Allow several codes per state. Never use a
  hardcoded ISO table.

---

## 2. Indian name transliteration variation

### 2.1 IndicXlit per-language Indic→En accuracy on named entities. VERIFIED (README raw)
- https://github.com/AI4Bharat/IndicXlit (MIT, about 11M params, 21 languages). Top-1 **Indic→En, Aksharantar
  named-entity** test: asm 37.86, ben 52.54, guj 50.56, hin 59.22, kan 60.77, mal 45.23, mar 56.76, ori 47.68,
  pan 48.00, tam 35.46, tel 57.57, urd 23.14.
- **Implication:** for our seven scripts, top-1 exact romanization is right only about 35-61% of the time. Use
  (a) top-k beams (4-5) and take the max similarity against the Latin side, and (b) compare through
  phonetic/consonant-skeleton keys (`notes/name_address.md` §5.2), never by exact equality. P1, offline GPU
  cache per unique token.

### 2.2 Google Maps: Improving Indian Language Transliterations (ensemble plus acronym module). VERIFIED (blog)
- Cibu Johny, Saumya Dalal, Google Research blog, 2021-01-22.
  https://research.google/blog/improving-indian-language-transliterations-in-google-maps/
- **Method:** an **ensemble** of a finite-state model, an LSTM trained partly on Dakshina, and a specialized
  **acronym transliteration module** (the blog's examples are KD and NIT). It uses romanization dictionaries
  "tailored for place names, proper names, or common words", weights candidates by their frequency in a large text
  sample, and adds deterministic ISO 15919 romanization as a candidate. The weights are tuned on small POI dev
  sets. The direction is Latin→native. Reported improvement factors (as read from the page): Hindi 3.2x/1.8x,
  Bengali 19x/3.3x, Tamil 19x/3.6x, Kannada 24x/2.3x.
- **Adaptation:**
  - (a) Keep **several romanizers**: the learned train dictionary, rule ISO 15919/Hunterian, and optionally
    IndicXlit top-k. The feature is the max similarity over candidates, plus which romanizer won (categorical). P1.
  - (b) **Acronym path:** a native-script token that is a sequence of Latin letter names maps to initials. Learn
    the letter-name table from train cross-script positives where the Latin token is uppercase with 5 letters or
    fewer. The noise includes both acronyms ("SC") and cross-script names, so they will co-occur. P1.

### 2.3 Hunterian transliteration: how official Indian place names are romanized. VERIFIED (Wikipedia)
- https://en.wikipedia.org/wiki/Hunterian_transliteration. It is "the 'national system of romanization in
  India'", used for geographic names. Retroflex and dental consonants are **not** distinguished (द and ड both →
  `d`). Aspirates are written with `h` (kh, gh). Vowel length is marked with a macron, which is dropped in
  practice (ASCII).
- **Adaptation:** a "Hunterian ASCII" key (no length, no retroflex/dental split, keep `h` aspirates) sits between
  our `key_strict` and `key_loose`. Add it as one more candidate romanization for city and state tokens. P1,
  cheap.

### 2.4 Schwa deletion: the main systematic gap between rule romanization and real spellings. VERIFIED (abstract)
- Arora, Gessler, Schneider. *Supervised Grapheme-to-Phoneme Conversion of Orthographic Schwas in Hindi and
  Punjabi*. ACL 2020. https://arxiv.org/abs/2004.10353. It is a statistical classifier from orthography alone that
  beats previous approaches. The code at https://github.com/aryamanarora/schwa-deletion has **no license**, so
  use the idea only.
- **Adaptation:** rule romanizers output the inherent `a` (e.g. *kamala* for the name usually written *kamla*).
  For Devanagari, Gujarati and Bengali tokens not in our learned dictionary, generate two candidates: full schwa,
  and word-final plus medial schwa deleted (V C a C V → V C C V). Take the max similarity. Tamil, Telugu, Kannada
  and Malayalam need no deletion rule (their romanizations keep the vowel). P1.

### 2.5 Other transliteration sources (context)
- Kunchukuttan, Jain, Kejriwal. *A Large-scale Evaluation of Neural Machine Transliteration for Indic
  Languages*. EACL 2021, pp. 3469-3475. https://aclanthology.org/2021.eacl-main.303/. 600K mined word pairs;
  analyzes direction, language family and word origin. Background only.
- Azam et al. *Beyond Specialization: Benchmarking LLMs for Transliteration of Indian Languages*. arXiv
  2505.19851 (2025). https://arxiv.org/abs/2505.19851. GPT-family models generally beat IndicXlit, including
  under noise. **Not usable here** (closed models or >8B). Take-away: an open ≤8B LLM is not an obvious win over
  IndicXlit.
- Gharami et al. *Modeling Romanized Hindi and Bengali*. BigCyber workshop, arXiv 2511.22769 (2025).
  https://arxiv.org/abs/2511.22769. About 1.8M Hindi and 1M Bengali pairs built specifically for **spelling and
  pronunciation diversity**. It confirms that romanization is one-to-many. External data, so method only.
- Merhav, Ash. *Design Challenges in Named Entity Transliteration*. COLING 2018. arXiv 1808.02563. Compares WFST,
  RNN and Transformer for names. Background.
- IndicSoundex (Santhosh Thottingal, 2009): https://thottingal.in/blog/2009/07/26/indicsoundex/. It maps letters
  of **all Indic scripts** into about 20 shared phonetic groups (short and long vowels merged, halant ignored,
  consonant families grouped). Example: Malayalam and Hindi forms of *Santosh* get the same code body. The
  library `libindic/soundex` is **LGPL-3.0**, so reimplement the idea rather than ship it.
  **Adaptation:** a script-independent key compares **native vs native in different scripts** (e.g., S2 Tamil vs
  S3 Telugu) *without* romanizing. Our romanize-then-compare path adds two sources of error there. P2.

---

## 3. France: address normalization and matching

### 3.1 addok-france (BAN geocoder plugin): query cleaning and house-number grammar. VERIFIED (source code)
- https://github.com/addok/addok-france (MIT; `addok_france/utils.py` read). addok is the geocoder behind BAN
  (https://github.com/addok/addok, MIT, 386 stars).
- **Street types:** 67 regexes, for example `av(enue)?`, `b(oulevar|l?v?)?d`, `che?(m(in)?)?`, `r(ue)?`,
  `imp(asse)?`, `f(aubour|b|bour)?g`, `pl(ace)?`, `r(ou)?te`, `r[ée]s(idence)?`, `rd?pt`, `squ?(are)?`,
  `t(erra)?sse`, `trav(erse)?`.
- **Suffixes:** `ORDINAL_REGEX = "bis|ter|quater|quinquies|sexies|[a-z]"`, with
  `FOLD = {bis:b, ter:t, quater:q, quinquies:c, sexies:s}`.
  - `glue_ordinal`: `6 bis` → `6bis`.
  - `fold_ordinal`: `3bis` → `3b`.
  - `flag_housenumber`: `\b\d{1,4}[a-z]?\b` is a house number if it is the first token or precedes a street type.
  - `remove_leading_zeros`.
- **clean_query patterns (verbatim intent):**
  - space out 5-digit postcodes;
  - drop `(b.p.|cs|tsa|cidex) [n°] <digits>`;
  - when CEDEX is present, keep only the first 2 postcode digits (`([\d]{2})[\d]{3}(.*)c(e|é)dex ?[\d]*` → `\1\2`);
  - drop `cedex n`;
  - drop `Ne étage`;
  - `s/` → *sur*, `s/s` → *sous*;
  - drop a leading `lieu(x)-dit(s)`.
- **extract_address:** a regex that pulls `<number>[suffix] <type> ... [postcode]` out of polluted strings (e.g.,
  a company name and building prefix before the street).
- **Adaptation (P0):** port these ideas, not the code, into the FR branch of our normalizer: number+suffix
  canonical form, routing-token removal, CEDEX→dept-prefix comparison. Features:
  - `hn_base_equal`;
  - `hn_suffix_equal` (b = bis = B);
  - `hn_suffix_one_missing`;
  - `street_type_equal_after_canon`.
  The extract-address idea also helps for "component reordering": locate the `number type name` span wherever it
  sits.

### 3.2 addok-fr: French synonyms and a light French phonemicizer. VERIFIED (source code)
- https://github.com/addok/addok-fr (MIT). `resources/synonyms.txt` has **222 mappings**, including:
  - `r => rue`, `av, ave, aven => avenue`, `bd,bld,bou,bould,bvd,boul => boulevard`, `ch,che,chem => chemin`;
  - `fg, fbg, fbourg => faubourg`, `imp => impasse`;
  - **`st => saint`, `ste => sainte`, `rd => route departementale`, `rn => route nationale`**;
  - `zi/za/zac`, `hlm`;
  - French ordinals (`1er, 1e, ier => premier`, `2e, 2eme, iie => deuxieme`, ...) and roman numerals.
- `utils.py` has about 37 ordered regex rules for a French phonemicizer:
  - nasal clusters `mp/nd/nt → n`;
  - `g` before e/i/y → `j`;
  - `c` before a non-front vowel → `k`;
  - intervocalic `s → z`, `qu → k`, `ph → f`;
  - silent `h`, final `s/t/d/g/e` dropped;
  - `oeu → eu`, `ei/ae → e`;
  - collapse doubled letters.
- **Adaptation:**
  - (a) **Country-scoped canonicalization (P0).** `st`, `rd`, `ch`, `r` mean different things in FR vs US.
    Canonical maps must be keyed by country. Since FR has no train labels, **learn FR maps from test
    pseudo-pairs** (`notes/name_address.md` §5.3 step 5). Use this list only as a *validation checklist*, e.g.
    report which of its families our miner rediscovered. Shipping the list as-is is a compliance grey area (see
    section 0).
  - (b) **French phonetic key (P2):** reimplement a subset of these rules as `fr_phon_key` for FR name, street and
    city tokens, as a secondary similarity and blocking key. It helps with typos and injected diacritics.
    Alternatives with MIT JS implementations are in Talisman (3.8).

### 3.3 normadresse (BAN): how official French street labels get abbreviated. VERIFIED (source code; NO LICENSE)
- https://github.com/BaseAdresseNationale/normadresse (**no license**, so method only). `normadresse.csv` has
  **352 regex rules in 9 stages**. `abrev(lib, maxi=32)`:
  1. Fold to upper-case ASCII.
  2. If the label is ≤ 32 characters, **do not abbreviate**.
  3. Otherwise abbreviate in stages, stopping as soon as the label fits:
     - (1) leading street type (`AVENUE→av`, `BOULEVARD→bd`, `IMPASSE→imp`, `ALLEE→all`, `PLACE→pl`,
       `ROUTE→rte`, `ZONE INDUSTRIELLE→zi`);
     - (2) military, religious and civil titles (`GENERAL→gal`, `MARECHAL→mal`, `DOCTEUR→dr`, `PRESIDENT→pdt`,
       `NOTRE DAME→nd`);
     - (4) general words (`SAINT/SAINTE` are handled in stage 6 → `ST/STE`, `BIS→B`, `TER→T`, `QUATER→Q`,
       `SOCIETE→SOC`, `ETABLISSEMENTS→ETS`, `COMPAGNIE→CIE`, `SERVICE→SCE`);
     - (5) about 120 less common street types, **mostly truncated to 4 letters** (`CHEMIN→CHEM`, `FAUBOURG→FAUB`,
       `HAMEAU→HAME`, `RESIDENCE→RESI`, `LOTISSEMENT→LOT`);
     - (3) **given names reduced to initials** unless preceded by SAINT (`JEAN MARIE → j m`).
  - Example from the README: `BOULEVARD DU MARECHAL JEAN MARIE DE LATTRE DE TASSIGNY` → `bd mal j m DE LATTRE DE
    TASSIGNY`.
  - A quirk in the data: the stage-4 row `QUINQUIES → CAB` looks like a typo; addok-france folds quinquies → `c`.
- **Adaptation:**
  - (a) Official French abbreviation is **length-triggered** and follows a **4-letter truncation** pattern for
    rare types. Add a generic feature `token_prefix_match`: token a is a prefix of token b, len(a) ≥ 2, same slot.
    It catches `chem/chemin`, `faub/faubourg` and `resi/residence` without any list. P0.
  - (b) **Initials:** add a Monge-Elkan variant where a 1-letter token matches any token starting with that
    letter (`j m de lattre` vs `jean marie de lattre`). P1.
  - (c) The noise spec says French CH = chemin, while the official form is CHEM (normadresse, La Poste) and
    addok-fr accepts `ch/che/chem`. So the learned map must be **many-to-one**.

### 3.4 CNIG "Standard Adresse" V1 (2024-10-28): data model rules. VERIFIED (PDF text)
- https://cnig.gouv.fr/IMG/pdf/cnig-standard_adresse-v1-v20241028.pdf
- `indiceRepetition`:
  - "bis, ter, quater, quinquies (en minuscules)" or "A, B, C…. (en majuscules)";
  - do not mix the two systems within one street;
  - "Ne pas abréger les indices" (a recommendation, which the noise violates);
  - avoid suffixes with **metric numbering** (numéro = distance in metres from the start of the street, so numbers
    can be large).
- **Postcode (§ zone postale):** "Un code postal est couramment associé à plusieurs communes", and some communes
  have several postcodes. The postcode "n'a pas de relation directe avec le découpage administratif".
- **Adaptation:**
  - (a) `postcode_equal ∧ city_diff` is **not** strong negative evidence in FR; learn it from data. P1.
  - (b) Do not treat 3-4 digit house numbers as noise, and keep the house-number (≤4 digits) vs postcode
    (5 digits) disambiguation. P1.
  - (c) Map `bis ↔ b` and `ter ↔ t` after folding. Careful: the letter system (A, B, C) is a separate,
    positional scheme and does not equal bis/ter. So `12 bis` vs `12 B` or `12 ter` vs `12 C` is only a *soft*
    match (a feature value, not an equality). P0 (part of 3.1).

### 3.5 La Poste abbreviation guidance. PARTIALLY VERIFIED (WebFetch summary)
- https://www.laposte.fr/envoyer/abreviation-adresses-postales. As read: "l'abréviation reste une exception"
  (abbreviate only when space requires). Listed forms include AV, BD, CHEM, IMP, ALL, PL, RTE; `St`/`Ste` without
  a dot; Bât., Imm., Rés., App., B.P.; cardinal points N/S/E/O. The fetch summary said RUE has no abbreviation,
  which conflicts with the `R.` in our noise spec and with `rue, r` in the official annuaire-entreprises synonyms.
  I did not open the raw page to settle this.
- AFNOR XP Z10-011 / NF Z10-011 details (6 lines × 38 characters) remain **UNVERIFIED on a primary source**, as
  in `name_address.md`.

### 3.6 Region vs département facts. VERIFIED
- fr.wikipedia *Région française*: 18 regions since **2016-01-01**, 13 metropolitan and 5 overseas. The
  metropolitan regions are Auvergne-Rhône-Alpes, Bourgogne-Franche-Comté, Bretagne, Centre-Val de Loire, Corse,
  Grand Est, Hauts-de-France, Île-de-France, Normandie, Nouvelle-Aquitaine, Occitanie, Pays de la Loire and
  Provence-Alpes-Côte d'Azur. The metropolitan regions went from 22 to 13. There are 96 metropolitan départements.
  https://fr.wikipedia.org/wiki/R%C3%A9gion_fran%C3%A7aise
- CNIG: the first postcode digits correspond to the département.
- **Adaptation (P0, transductive, no external data):**
  1. From all FR test records (unlabeled), extract admin tokens: n-grams in the state/region slot or trailing
     address tokens that are not the city. Record the postcode prefix-2 of each record.
  2. For each admin string `a`, build the distribution `D_a` over prefix-2 values. A **département** name
     concentrates on 1 prefix. A **region** name spreads over 2-13 prefixes. This separates the two levels without
     labels.
  3. `admin_compat(a, b)` = 1 if `argmax D_a ∈ supp(D_b)` or vice versa. `admin_level_diff` = one is dept-like and
     the other region-like. Then the swap noise becomes a *neutral* event instead of a mismatch.
  4. Hyphen and space variants (`Hauts-de-France` / `Hauts de France`) and multi-word names need token-set
     matching. Watch for acronyms (e.g. "PACA", "IDF"; my assumption, check in the data) and let the acronym key
     handle them.

### 3.7 INSEE Code officiel géographique: articles are stored separately. VERIFIED
- https://www.insee.fr/fr/information/2114773. Commune names have `NCC` (uppercase label), `NCCENR` (rich
  typography) and **`TNCC`**, the article type: 0 none (consonant), 1 none (vowel), 2 LE, 3 LA, 4 LES, 5 L',
  6 AUX, 7 LAS, 8 LOS (per the example `TNCC = 8`, "LOS MASOS"). Datasets therefore contain `LE HAVRE`, `HAVRE`
  and `HAVRE (LE)`.
- **Adaptation (P0, cheap):** a FR city key that removes a leading or parenthesized trailing article
  (`le|la|les|l'|aux|las|los`), maps `st/ste` ↔ `saint/sainte`, turns hyphens into spaces and folds diacritics.
  Compare this key and the plain city string, and take the max.

### 3.8 French phonetic algorithms. VERIFIED (docs)
- Talisman (MIT, JS) French phonetics: https://yomguithereal.github.io/talisman/phonetics/french
  - **FONEM** (Bouchard, Brard, Lavoie 1981; Saguenay family names; `Beaulac → BOLAK`);
  - **Phonex** (Frédéric Brouard; `Henri → H1RI`);
  - **Sonnex** (Frédéric Bisson; `ontologie → 3toloji`);
  - French Soundex and **Soundex2** (`Florentin → FLRN`);
  - Bergé's "phonetic".
- `chrislit/abydos` has these too but is **GPL-3.0**; avoid it.
- **Adaptation:** P2. The noise is typos plus diacritics rather than phonetic respelling, so gains are likely
  small. Use it as a cheap FR-only secondary key if the error analysis shows FR name misses.

### 3.9 annuaire-entreprises (official French company search): analyzer design. VERIFIED (source code)
- https://github.com/annuaire-entreprises-data-gouv-fr/search-infra (MIT),
  `workflows/data_pipelines/elasticsearch/mapping_index.py`.
- **Analyzer order:**
  1. **char filters:**
     - elision removal (regex `\b(l|m|t|qu|n|s|j|d|c|jusqu|lorsqu|puisqu|quoiqu)('|’|`|‘)` → "");
     - then delete `' ‘ ` ’ , .`.
     A code comment explains why the order matters: `L'EDUCATION` → `education`, not `leducation`. Deleting dots
     turns `R.P.S.` into `RPS`.
  2. `icu_tokenizer`.
  3. **token filters:** lowercase → French stop words → `icu_folding` → **synonyms (expand=True)**
     `avenue,av | boulevard,bd | rue,r | route,rte | impasse,imp | allée,all | place,pl | cours,crs | chemin,chem |
     appartement,app,apt | bâtiment,bât,b | résidence,résid,res | lieu-dit` → asciifolding → `minimal_french`
     stemmer.
- Indexed name fields: `nom_complet`, `denomination`, **`sigle`** (acronym), **`enseigne_1..3`** (trade/shop
  names) and `nom_commercial`. French companies officially carry several names, like the "X a/k/a Y" and
  "Z dba W" alias noise.
- **Adaptation:**
  - (a) Put the elision and dot rules into the FR (and generic) name normalizer. Note that dot removal must happen
    **after** domain detection (`acme.fr`). P0.
  - (b) Treat our alias-split variants like `sigle/enseigne`: score = max over variant pairs, which we already
    plan. P0, already in `name_address.md` §5.5.

### 3.10 FOPPA: French company linkage to SIRENE on name plus address. VERIFIED (PDF text)
- Potin, Labatut, Morand, Largeron. *FOPPA: An Open Database of French Public Procurement Award Notices From
  2010-2020*. Scientific Data 2023; arXiv 2305.18317. https://arxiv.org/abs/2305.18317
- **Observed noise:** "diacritics and punctuation signs are not used consistently, acronyms are used
  non-systematically"; name fields polluted with building numbers and internal departments; PO box in the city
  field; "SIRENE typically proposes several alternative and/or former names".
- **Method:**
  1. Normalize (remove punctuation, collapse spaces, uppercase, drop text in parentheses, remove PO words).
  2. Filter candidates by **date/activity and département**.
  3. Filter by approximate name comparison.
  4. Score location.
  5. Take the best. Unresolved agents are clustered with Dedupe.
- **Results:** full SIRET recovered for 66.6% of unique buyers and 72.4% of winners (known-SIRET ground truth).
  On 500 manually labeled agents: 71.2% and 62.8% full, 80.8% and 73.2% full+partial.
- **Adaptation:** (a) evidence that **département-level blocking works for French company data**; make sure one of
  our blocking views is `(country=FR, postcode[:2]) × name char-grams`. P1. (b) **Drop parenthesized text** from
  names as a variant, not a replacement. P1. (Their Hexaposte zipcode filling is external; not allowed.)

### 3.11 French legal forms. VERIFIED (INSEE, Bpifrance, GLEIF)
- **INSEE catégories juridiques** (https://www.insee.fr/fr/information/2028129): 3 levels with 9 / 37 / 260
  positions. The last update I saw was 2022-09-01.
  - The older XML enumeration (https://xml.insee.fr/schema/cj-enum.html) lists 5498 "SARL unipersonnelle",
    5710 "SAS", 5720 "SAS associé unique", 5202 "Société en nom collectif", 6540 "Société civile immobilière",
    9220 "Association déclarée".
  - Per the INSEE page summary and search snippets, 5498 and 5720 were suppressed in 2020 (SASU folded into 5710).
    That is **partially verified**; I did not read the change note itself.
- **Bpifrance Création** (https://bpifrance-creation.fr/encyclopedie/structures-juridiques/choix-du-statut-generalites/structures-juridiques-comparaison):
  - It compares EI, EURL, SARL, SASU, SAS, SA, SNC, SCOP and Association.
  - EURL = "SARL unipersonnelle"; SASU = SAS with one associate.
  - SA needs "2 associés minimum dans les sociétés non cotées", 7 for listed companies.
- **GLEIF ISO 20275 ELF code list** v1.6 (Feb 2026): "more than 3,600 entity legal forms across more than 200
  jurisdictions". Each entry has local and transliterated names **and abbreviations**.
  https://www.gleif.org/en/about-lei/code-lists/iso-20275-entity-legal-forms-code-list. External data, so use it
  only to *sanity-check* the legal-form tokens our miner finds. Do not ship it.
- **Adaptation (P0):**
  - Legal-form **families**: {SARL, EURL, S.A.R.L., "SARL unipersonnelle"}, {SAS, SASU, S.A.S.}, {SA},
    {SNC}, {SCI, SC, "société civile ..."}, {SCOP}, {EI, EIRL, micro/auto-entrepreneur: not a company}. Then
    `lf_family_equal`, `lf_family_conflict` and `lf_missing_one_side`. EURL vs SARL counts as *equal family*.
  - Legal forms appear as **prefix** (`SARL DUPONT`) as well as suffix. Detect them at any position; the noise
    also "moves" suffixes.
  - Generic company words (`société/sté/soc`, `établissements/ets`, `compagnie/cie`, `& fils`, `frères`) get low
    IDF automatically. The normadresse stage-4 list confirms the official short forms `SOC`, `ETS`, `CIE`.
  - France has no training labels, so **mine FR legal tokens transductively**: tokens at the name boundary with
    high document frequency across FR S1 and low IDF, mapped through dot/space-insensitive matching
    (`s.a.r.l` = `sarl`).

---

## 4. Consolidated action list

| # | Action | Priority | Evidence |
|---|---|---|---|
| 1 | Key all canonicalization maps by country; bootstrap FR maps from test pseudo-pairs (st=saint, rd=route départementale, ch/che/chem=chemin, r=rue) | P0 | addok-fr synonyms (3.2); normadresse (3.3) |
| 2 | FR house-number grammar: glue+fold bis/ter/quater/quinquies/sexies → b/t/q/c/s, strip N°/no and leading zeros; features hn_base_equal / hn_suffix_equal / suffix_one_missing | P0 | addok-france (3.1); CNIG (3.4) |
| 3 | Strip CEDEX/BP/CS/TSA/CIDEX tokens; when CEDEX is present compare postcode prefix-2 only; s/ → sur, lieu-dit prefix | P0 | addok-france clean patterns (3.1) |
| 4 | Transductive region↔département reconciliation via P(admin \| postcode[:2]) from unlabeled test; features admin_compat, admin_level_diff | P0 | regions/depts (3.6); CNIG |
| 5 | Postcode as soft hierarchical feature (common-prefix length); keep text-only blocking; add pc_disagree × strong-name interactions; measure postcode disagreement rate on train positives | P0 | Flipkart COLING'25 (1.2); GeoIndia (1.1) |
| 6 | Transductive spell-variant leader clustering (count, Lev<3, same phonetic key, len>6) per country and field; frequency-based compound splitting; camel-case split; duplicate-phrase collapse; drop 7+ digit runs (phones) | P0 | Mangalgi 2020 (1.5); GeoIndia (1.1) |
| 7 | Head/tail split of the address: separate similarity for house/building/number vs locality/city/state/pc; interaction tail_equal ∧ head_diff (sibling FP) | P0 | Kothari & Sohoney (1.3); Delhivery UAID patent (1.6) |
| 8 | FR name normalizer: elision removal before apostrophe removal; dot removal (S.A.R.L. → SARL) after domain detection; drop parenthesized text as a variant | P0 | annuaire-entreprises (3.9); FOPPA (3.10) |
| 9 | Legal-form families (EURL≈SARL, SASU≈SAS; SA, SNC, SCI, SCOP, EI distinct), detected at prefix or suffix; FR legal tokens mined transductively | P0 | INSEE, Bpifrance (3.11) |
| 10 | Learn state code↔name many-to-one from train co-occurrence (IN codes churned in 2023: TG→TS, OR→OD, CT→CG, UT→UK) | P0 | ISO 3166-2:IN (1.10) |
| 11 | FR city key: strip articles (le/la/les/l'/aux, leading or parenthesized), st/ste ↔ saint/sainte, hyphen → space | P0 | INSEE COG TNCC (3.7) |
| 12 | Generic `token_prefix_match` (len ≥ 2, same slot) and initials-aware Monge-Elkan | P0/P1 | normadresse 4-letter truncation and initials (3.3) |
| 13 | Romanization ensemble: learned dictionary + ISO15919/Hunterian key + IndicXlit top-k; max similarity; never exact equality | P1 | IndicXlit README NE accuracy 35-61% (2.1); Google Maps (2.2); Hunterian (2.3) |
| 14 | Schwa-deletion variant for Indo-Aryan scripts (Devanagari/Gujarati/Bengali) OOV tokens | P1 | Arora et al. ACL 2020 (2.4) |
| 15 | Indic-script acronym path (letter names → initials), table learned from train | P1 | Google Maps acronym module (2.2) |
| 16 | Locality→city→state DAG from S1 co-occurrence; hier_consistent feature | P1 | Delhivery patents (1.6) |
| 17 | FR département-prefix blocking view (FR, pc[:2]) × name char-grams | P1 | FOPPA (3.10) |
| 18 | Multi-granularity hard negatives (same pc / pc3 / city) for any e5 fine-tune; "address swap" negatives for LightGBM | P1/P2 | Kothari & Sohoney (1.3); Swiggy (1.7) |
| 19 | Script-independent Indic phonetic key (IndicSoundex idea, reimplemented) for native-vs-native different-script pairs | P2 | IndicSoundex (2.5) |
| 20 | FR phonetic key (addok-fr phonemicize subset / Phonex / Sonnex) as a secondary feature | P2 | addok-fr (3.2); Talisman (3.8) |
| 21 | Candidate-graph-based negative sampling for training | P2 | Amazon WWW'23 (1.4) |

## 5. Cheap checks to run on cloud (no heavy compute)
1. Train positives: the rate of postcode disagreement per country and per source (S2 vs S3), and the rate of
   disagreement on the first 3 digits. This decides how soft #5 must be.
2. Test FR: the top-200 most frequent tokens in the street-type slot and the admin slot. Check that the
   pseudo-pair miner recovers `r/rue`, `bd/boulevard`, `ch/chemin`, `av/avenue`, `imp/impasse`, `st/saint`, and
   bis/ter.
3. Test FR: the distribution of distinct postcode prefix-2 values per admin string. Confirm the dept (1 prefix)
   vs region (2-13 prefixes) separation.
4. Train IN cross-script positives: the exact-match rate of our romanizer vs skeleton-key match vs IndicXlit
   top-1/top-5 on unique tokens. This decides whether the #13 GPU cache is worth it.
5. Tokens where `count(Ta) < count(Tb)`, `Lev < 3` and keys are equal: sample 100 pairs and eyeball the precision
   of the leader map.

## 6. Compliance notes
- **Licensed and usable for method (and code if needed):** addok, addok-france, addok-fr, annuaire-entreprises
  search-infra, Talisman, IndicXlit, indian-address-parser (all MIT).
- **Idea only:**
  - normadresse: no license.
  - schwa-deletion repo: no license.
  - libindic soundex: LGPL-3.0.
  - abydos: GPL-3.0.
- **External data, never shipped:** synonym lists (addok-fr, La Poste, normadresse CSV), GLEIF ELF, INSEE
  nomenclatures, and region/département tables. Hand-coding a few dozen well-known abbreviations is a grey zone,
  so the recommended route is to *learn* them from competition data (train pairs for US/IN, test pseudo-pairs for
  FR) and use these lists only to audit what the miner found.

## Sources (all opened unless marked)
- https://aclanthology.org/2024.emnlp-industry.29/
- https://aclanthology.org/2025.coling-industry.19/
- https://aclanthology.org/2022.emnlp-industry.33/
- https://www.amazon.science/publications/learning-geolocation-by-accurately-matching-customer-addresses-via-graph-based-active-learning
- https://www.amazon.science/publications/accurate-customer-address-matching-via-weak-supervision-for-geocode-learning
- https://arxiv.org/abs/2007.03020
- https://patents.google.com/patent/US11381928B2/en
- https://patents.google.com/patent/US11343638B2/en
- https://dl.acm.org/doi/10.1145/3564121.3564800 (abstract via search only)
- https://github.com/howdoiusekeyboard/indian-address-parser
- https://arxiv.org/abs/2404.11691
- https://arxiv.org/abs/1802.04625
- https://en.wikipedia.org/wiki/ISO_3166-2:IN
- https://github.com/AI4Bharat/IndicXlit
- https://research.google/blog/improving-indian-language-transliterations-in-google-maps/
- https://en.wikipedia.org/wiki/Hunterian_transliteration
- https://arxiv.org/abs/2004.10353
- https://aclanthology.org/2021.eacl-main.303/
- https://arxiv.org/abs/2505.19851
- https://arxiv.org/abs/2511.22769
- https://arxiv.org/abs/1808.02563
- https://thottingal.in/blog/2009/07/26/indicsoundex/
- https://github.com/addok/addok-france
- https://github.com/addok/addok-fr
- https://github.com/addok/addok
- https://github.com/BaseAdresseNationale/normadresse
- https://cnig.gouv.fr/IMG/pdf/cnig-standard_adresse-v1-v20241028.pdf
- https://www.laposte.fr/envoyer/abreviation-adresses-postales
- https://fr.wikipedia.org/wiki/R%C3%A9gion_fran%C3%A7aise
- https://www.insee.fr/fr/information/2114773
- https://yomguithereal.github.io/talisman/phonetics/french
- https://github.com/annuaire-entreprises-data-gouv-fr/search-infra
- https://arxiv.org/abs/2305.18317
- https://www.insee.fr/fr/information/2028129
- https://xml.insee.fr/schema/cj-enum.html
- https://bpifrance-creation.fr/encyclopedie/structures-juridiques/choix-du-statut-generalites/structures-juridiques-comparaison
- https://www.gleif.org/en/about-lei/code-lists/iso-20275-entity-legal-forms-code-list
- UNVERIFIED (not opened): https://medium.com/@kabirrustogi/learning-to-decode-unstructured-indian-addresses-c80ffcda2e84 ; https://dl.acm.org/doi/10.1145/3746252.3761512
