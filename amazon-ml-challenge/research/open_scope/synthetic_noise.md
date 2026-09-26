# Open-scope sweep: synthetic data generation and corruption (synthetic_noise)

Date: 2026-09-25. Scope: how record-linkage test data is usually generated and corrupted (GeCo / FEBRL / Gecko / pseudopeople / TDGen / DaPo+ / SOG / LANCE / EMBench / libpostal / LLM-generated ER sets / benchmark difficulty work), read with two goals:
(a) reverse-engineer how the challenge data was probably generated, so that each corruption can be inverted by a feature or a normalization;
(b) build synthetic French-like training pairs **from the challenge data only** (no external data).

Rules followed: every item below was opened on its primary page (paper PDF/HTML, official docs, source code or README) unless it is marked **UNVERIFIED**. Numbers are only those read on those pages. Research informs method only. No external data enters a model (see section 7).

Related project docs: `DATA_ANALYSIS.md` (noise catalogue), `FEATURE_CATALOG.md`, `ERROR_ANALYSIS.md`, `VALIDATION_STRATEGY.md` (P-loco, P-dense), `EXPERIMENT_PLAN.md` (E-8 self-trained maps, E-15 synthetic augmentation).

---

## 0. TL;DR

1. **The name corruptor is country-agnostic; the address corruptor is per country.** Pattern rates in `experiments/reports/profile.json` for France targets match US/India targets on name noise: all-lower 0.065 vs 0.060 (US), S2 all-upper 0.208 vs 0.204 (US), domain/handle 0.035 vs 0.036, bracket 0.10 vs 0.095 (India), S3 dba 0.010 vs 0.009–0.013, and the same 2-letter acronym names (CC, PC, SC) top the lists. France address noise is different: null tokens 0.000 (US 0.039), `#` junk 0.000 (0.026–0.031), landmarks 0.000 (India 0.115), PO box/unit ~0. Name features and thresholds should transfer to France. The address side is where France shifts, and the LOCO proxy (US↔India) does not measure it.
2. **These generators all have the same structure**, and our data fits it: clean originals → k duplicates per original (GeCo: uniform/poisson/zipf, `max_num_dup_per_rec`) → per-attribute corruption probability → a weighted choice of corruption function → a position function that places edits in the middle or end of the string (GeCo `position_mod_normal`: mean len/2+1, sd len/4). Extra pieces: households/siblings (FEBRL 2009), copy propagation (DaPo+) and source-specific distractor entities (FakER). Each piece has a trace in our data (section 2).
3. **House-number noise is an arithmetic offset, not a keyboard typo.** In the local samples, true copies with a different number differ by small offsets (2774→2771, 1405→1413) or by digit insertion/deletion and leading zeros (514→00514). The FP "siblings" sit on small offsets too, but much more often: |Δ|∈[1,5] in 25/106 FP pairs vs 23/1,198 true-copy comparisons (sample only, section 8). Offset magnitude is a useful feature. Keyboard adjacency is not (hypothesis refuted).
4. **The "planted sibling" distractors match FEBRL's family/household generator** (copy an original, change a few fields, keep the address). Test has ~24% more targets per S1 than train. If the number of true matches per S1 is unchanged (3.46 train; our sub_v1 predicts 3.45 for France), the implied distractor share is ~40% (test non-France) and ~37% (France), against 26% in train. The France simulator must reproduce this density.
5. **Synthetic French pairs are feasible with challenge data only.** Mine the French style (street-type abbreviations, département↔région, number formats, legal-form variants, accent processes) from unlabeled test France targets, via anchor pairs and a city join. Take the country-agnostic name-noise rates from train pairs. Generate hierarchically: a source base variant, then copies, then distractors and siblings. **Gate it with a LOCO-sim experiment** (simulate India from India S1 + India-mined style, train US-real + India-sim, score on real India) before using it for France. LLM-generated pairs are not recommended: Steiner et al. found them often mislabeled, and GPT-4o-mini fine-tuning lost 6.42 F1.

---

## 1. The generic generator template (what the literature builds)

| Generator stage | Literature parameterisation (verified) | Trace in our data |
|---|---|---|
| Originals | Frequency look-up tables or attribute functions, with dependencies between up to 3 attributes (GeCo); build from real data profile (DaPo+) | S1 names built from invented brands ("Kelonylalum", "Miraquojax") or small vocabularies ("Lille Club SARL", "{City} {Noun} {LegalForm}") |
| Duplicates per original | `max_num_dup_per_rec`, `num_dup_dist` ∈ {uniform, poisson, zipf} (GeCo). Zipf uses θ=0.5; Poisson mean = 1 + mod/org | S2 copies 0–5 and S3 copies 0–6 per S1, mode 1 each; total 0–11 (DATA_ANALYSIS §3). Looks like independent per-source draws with caps |
| Which attributes are corrupted | `attr_corr_prob` per attribute, `num_mod_per_rec`, `max_num_mod_per_attr` (GeCo). Row → column → cell with an "error score limit" per cell (TDGen). `cell_probability` + `token_probability` (pseudopeople defaults 0.01 / 0.1) | Per-source style differences (S2 uppercase USPS addresses, S3 full state names) = per-source probabilities |
| Which corruption | Weighted list per attribute, e.g. `'last_name': [(0.2, edit), (0.5, misspell-lookup), (0.3, phonetic)]` (GeCo Fig. 4) | Typos, leetspeak, diacritics, reorder, suffix moves, generic tokens, truncation, aliases, handles, acronyms, cross-script |
| Where in the string | `position_mod_normal`: Gaussian, mean len/2+1, sd len/4, "errors … more likely … towards the middle and end … but not at the beginning" (GeCo code) | Prefix-preserving typos ("Madr", "SACRAEMNTO") |
| Token-level partial application | pseudopeople `token_probability`: each token corrupted independently | Partial transliteration ("Urban हेल्थकेयर प्राइवेट लिमिटेड") |
| Copy propagation | DaPo+ event-based error model: errors "from outdated values and copying processes" | Within-source copies share one corrupted base (DATA_ANALYSIS §5, "B/80+802+803") |
| Households / siblings | FEBRL 2009: copy an original, change given name, age, gender (and rarely the address) to make family/household records | Planted siblings: same name, same street, nearby house number (ERROR_ANALYSIS 2026-09-25) |
| Distractors | FakER: 150 shared + 150 source-specific entities per source; "source-specific entities act as natural distractors" | ~26% of train targets (more in test) match no S1 |
| Increasing noise per copy | Splink synthetic data: probability of corruption rises as more duplicates are generated (`start_prob_corrupt` … `end_prob_corrupt` comment) | Copies of one entity differ in noise level (a clean copy plus heavily corrupted ones) |
| Address formatting | libpostal training: templates, random abbreviation (`abbreviate_prob=0.3`, `separate_prob=0.2`, `add_period_hyphen_prob=0.3`), component dropout in dependency order, state full-name vs abbreviation probability | USPS abbreviations, state code↔name, missing components, reordering. France: R./BD/CH/AV, région↔département |

---

## 2. Reverse-engineered hypothesis for this challenge's generator

Everything in this section is **inference** from our measurements plus the template above. Each claim ends with a cheap test.

### 2.1 Entities and distractors
- Clean S1 originals come from per-country templates (brand generator + generic words + legal forms; address formatter per country). France S1 is repetitive (non-unique names; only 3 régions and about 15 cities appear in `samples_test.txt`).
- Distractor entities are generated the same way and have copies in S2/S3, but no S1 row (the FakER design).
- Siblings are copies of a real entity with a small house-number offset and/or a small name edit (FEBRL household analog). FP samples: "Amber Inc 340 vs 345 Olympic Park Dr", "XQ vs XQX Legacy Holdings, 3004 Lucinda St".
- *Test*: in train GT, take distractor targets (matched to no S1) whose best S1 by core-name + street equality exists. Measure their house-number |Δ| distribution and name edit distance. If it is concentrated at |Δ|∈[1,20] with equal core names, the sibling generator is confirmed and its parameters are known. These distractors are also the negative pool for the France simulator.

### 2.2 Copies per source
- S2 and S3 copy counts per S1 look like independent capped draws (S2 max 5, S3 max 6). *Test*: compare P(S2=0 ∧ S3=0) = 5.58% singletons with P(S2=0)·P(S3=0). Fit uniform/poisson/zipf per source. Use the fitted caps as a hard per-source limit in the decision layer (never predict more than 5 S2 or 6 S3 targets for one S1).

### 2.3 Source style profiles
- S2: uppercase addresses (US 0.90, India 0.24, France 0.29), USPS abbreviations, double spaces. S3: never all-upper addresses, "Unit"/"PO Box", full state names, the only source with "dba" names (S2 dba rate 0.000). These are per-source configurations of one engine (GeCo `attr_corr_prob` per source; FakER-style asymmetric noise).

### 2.4 Name corruptor: shared across countries (evidence)
Target-name pattern rates from `profile.json` (test files; train is nearly identical):

| pattern | S2 US | S2 India | S2 France | S3 US | S3 India | S3 France |
|---|---|---|---|---|---|---|
| all_lower | 0.060 | 0.038 | 0.065 | 0.063 | 0.046 | 0.066 |
| all_upper | 0.204 | 0.141 | 0.208 | 0.032 | 0.025 | 0.056 |
| double_space | 0.113 | 0.097 | 0.091 | 0.110 | 0.103 | 0.079 |
| has_domain | 0.036 | 0.027 | 0.035 | 0.035 | 0.030 | 0.034 |
| has_paren | 0.062 | 0.095 | 0.102 | 0.065 | 0.101 | 0.100 |
| has_dba | 0.000 | 0.000 | 0.000 | 0.013 | 0.009 | 0.010 |
| leading_junk | 0.023 | 0.015 | 0.009 | 0.023 | 0.016 | 0.009 |
| trailing_junk | 0.018 | 0.083 | 0.003 | 0.016 | 0.046 | 0.003 |
| digit | 0.062 | 0.029 | 0.011 | 0.061 | 0.032 | 0.011 |
| legal form (own country) | 0.512 | 0.534 | 0.542 | 0.505 | 0.610 | 0.534 |

Reading: case, double-space, bracket, domain, dba and acronym processes run at the same rates in France. The junk and digit rates are lower in France. So leetspeak digits and junk prefixes are rarer there, or were never applied (the leading/trailing junk may partly be country-specific, e.g. India trailing tokens).

### 2.5 Address corruptor: per country (evidence)

| pattern | S2 US | S2 India | S2 France | S3 US | S3 India | S3 France |
|---|---|---|---|---|---|---|
| empty | 0.029 | 0.023 | 0.031 | 0.028 | 0.025 | 0.029 |
| null_token | 0.039 | 0.029 | **0.000** | 0.037 | 0.028 | **0.000** |
| hash_junk | 0.026 | 0.031 | **0.000** | 0.026 | 0.030 | **0.000** |
| near_landmark | 0.000 | 0.115 | 0.000 | 0.000 | 0.087 | 0.000 |
| unit | 0.001 | 0.186 | 0.001 | 0.081 | 0.149 | 0.001 |
| po_box | 0.017 | 0.000 | 0.000 | 0.016 | 0.000 | 0.000 |
| double_space | 0.028 | 0.013 | 0.001 | 0.002 | 0.000 | 0.000 |
| has_paren | 0.000 | 0.047 | 0.017 | 0.000 | 0.041 | 0.016 |

Reading: France has its own address module. Visible in samples: "N°37", "#69", "(13)", "019" (leading zero), "85BIS", "R."/"R"/"AV"/"AVE"/"BLVD"/"BD"/"CH", uppercase, région↔département, accent injection "DÉS", "Àmis". Note **"AVE" and "BLVD" (US forms) on French street types**. So the generator's abbreviation step mixes US-style forms with French ones, and part of our US-learned abbreviation map transfers.

### 2.6 Train→test drift
- Empty target address rate: train US S2 0.037 → test US S2 0.029 (and 0.035 → 0.028 for S3). Test targets per S1: 4.677 (train) → 5.794 (test non-France) → 5.531 (France). The generator configuration moved between splits, so base rates learned on train (P(match | empty address), and so on) should be re-checked on test-like density (P-dense).

### 2.7 Number noise (true copies)
- Mechanisms seen on true pairs: leading zeros (514→00514), digit deletion/truncation (27194→2718, 6912→691), small arithmetic offsets (2774→2771, 1405→1413, 1201→1185), rare full replacements. The keyboard-typo corruptor (GeCo row/column neighbours) does **not** explain the offsets. SOG calls this "value perturbation".

---

## 3. Corruption → inversion matrix

"Have" = already in FEATURE_CATALOG / normalization. Priorities follow the notes' P0–P3 scale.

| Corruption | Literature analog | Have | Gap / new inversion | France note | Pri |
|---|---|---|---|---|---|
| Char typos (ins/del/sub/transpose) | GeCo CorruptValueEdit + `position_mod_normal` | JW, Levenshtein, char n-gram cosines | **Token-prefix agreement**: share of core tokens whose first 2–3 chars match. GeCo edits avoid the string start, so prefixes are reliable. Verify first on train pairs (edit-position histogram) | same engine | P2 |
| Keyboard / OCR / leet confusions | GeCo keyboard matrix, OCR table (5↔s, m↔rn, cl↔d, 0↔o, l↔1), nlpaug KeyboardAug/OcrAug | leet repair | **Learned channel cost**: substitution probabilities from aligned train pairs (Brill–Moore style substring edits) → weighted edit distance / channel log-likelihood feature | France digit rate low → minor | P2 |
| Phonetic variants | GeCo/FEBRL ~350 positional rules (ph↔f, END e→@) | phonetic keys | none needed | — | — |
| Injected / removed diacritics | TDGen resolve/revert umlauts | NFKD folding (assumed) | Count of accent-only token differences as a feature (accent-only diff = strong match cue) | France S1 **has** real accents, targets inject/remove → fold both sides | P2 |
| Token reorder, suffix moved | EMBench "word permutations", LANCE vt3 shuffle | token-set, core equality | none | "Fil-SAS", "E.U.R.L." → split hyphen-attached and dotted legal forms before core-token extraction | **P0** (cheap) |
| Legal suffix add/drop | GeCo categorical swap | core tokens (DF ≤1%) | Make sure French forms (SARL, SAS, SASU, SCI, EURL, EI, SA, Cie, "& Fils") fall out as generic. DF-based, so automatic if DF is computed per country on test | learned per country | P1 (check) |
| Generic token injection | — | core tokens | Mine French generic tokens (possibly "Maison", "Groupe") from anchors: tokens present in target but absent in S1 at high rate | — | P1 |
| Truncation | TDGen "Keep First Token", "Drop Last Character(s)", "Keep Longest Token" | `n_trunc_prefix`, coverage | Add "t core ⊂ q core at any position" (keep-longest-token case, not prefix) if coverage features miss it | — | P3 |
| Aliases a/k/a, f/k/a, dba | EMBench "aliases" | alias split | none. Pure alias records ("Nexpyra") are address-only: keep conservative | — | — |
| Domains / handles | — | cat view, `n_dom_prefix_cov` | France domains carry legal forms ("festivalgroupeeurl.com") → strip learned legal-form suffixes from the domain core before prefix coverage | — | P1 |
| Acronyms | — | `n_initials` | none | same top acronym names in France | — |
| Cross-script (Indic) | pseudopeople token-level; ISCII parallel Unicode layout | learned dictionary | **Offset-normalize** all Brahmic scripts to Devanagari before dictionary lookup (Indic NLP Library method). Pools statistics for rare scripts (Oriya 0.4%, Gurmukhi 0.3%). Errors here are already ≤5% → low gain | n/a | P3 |
| Address abbreviations | libpostal abbreviate(); TDGen "Abbreviate" (first letter + dot) | learned maps | **Mine France maps** (E-8): Rue→R/R./RUE; Boulevard→BD/BLVD; Avenue→AV/AVE; Chemin→CH/CHE; also "first letter + dot" generic rule | US forms also appear on French types | P0/P1 |
| State code↔name; région↔département | libpostal state full-name probability | learned state maps | **Admin canonicalizer for France**: map département→région, mined by city join (section 4, A3). Or drop admin tokens from address similarity when both sides carry a known admin token | 3 régions, ~4+ départements in samples | **P0** |
| Component reorder / dropout | libpostal dependency-ordered dropout; Deepparse incomplete addresses | order-free sims | Any learned component tagger must be trained with dropout (Deepparse: −20–40% accuracy on incomplete addresses when not trained on them) | — | P3 |
| Number format noise | TDGen "Digit To Dot", "Drop Nondigits" | `#`, "No", "Door No" handling | Add N°, "(13)", attached "85BIS", "12 C", "QUATER", leading zeros → canonical (num, suffix) pair | France-specific | **P0** (cheap) |
| Number value perturbation / siblings | FEBRL household; SOG "value perturbation" | `a_num_*`, v2 geometry (coded) | **|Δ| buckets** {0, 1–5, 6–20, 21–50, >50, missing}, relative Δ, same-length flag, digit-edit class (ins/del, leading-zero, substitution). Sample LR for |Δ|∈[1,5]: FP 24% vs true 1.9% | same | **P0** if not in v2 |
| Within-source copies share base | DaPo+ copying model | collective / sibling features | **Consensus-of-copies**: majority-vote token set over a target cluster (same source, and cross-source S2∧S3). Compare S1 to the consensus. Tokens agreed by S2 and S3 copies are almost surely original, which down-weights per-copy noise | same | P1 |
| Null tokens, landmarks, PO box | pseudopeople "fake name"-style fillers | null removal, landmark handling | none for France (rates 0) | — | — |
| Copies-per-source caps | GeCo `max_num_dup_per_rec` | expected-F layer | Per-source cap (S2 ≤5, S3 ≤6) + fitted count prior in the match-exists / expected-F0.5 layer | assume same caps | P2 |

---

## 4. FR-sim: synthetic French-like training pairs from challenge data only

Goal: (i) a **France validation set** with labels, used to calibrate the threshold / π model under the unseen-country shift (LOCO showed OOD models are over-confident and the best threshold rises); (ii) optional low-weight training rows. Everything below uses only test France S1/S2/S3 records and train pairs. All steps are CPU string processing and run on Modal, never locally.

**Step A: mine French style from unlabeled test France targets**
- A1 Anchor pairs (precision-first, as in E-8b but stricter): target t and France S1 q with the same city token, the same canonical house number (after N°/#/()/leading-zero/bis-ter normalization), street-name core Jaccard ≥ 0.8, exclusive best (margin > 0), and core-name overlap ≥ 1 token.
- A2 Token alignment on anchors: count S1-token → target-token rewrites for street types, admin tokens, legal forms and accents. Keep rewrites with support ≥ N and a stable direction. Output: street-type abbreviation distribution (with period/hyphen variants, as in libpostal `add_period_hyphen_prob`), legal-form variants (E.U.R.L., lowercase, hyphen-attached), accent removal and injection rates, number-format distribution (N°x / #x / (x) / 0x / xBIS / x bis).
- A3 Département↔région **without anchors**: S1 gives (city, région). Targets give (city, admin token). For each non-région admin token, take the région with the highest city co-occurrence. That yields the département→région map and makes the admin canonicalizer in section 3 possible.
- A4 Per-source France style: S2 address all-upper 0.29, S3 0.00, name all-upper S2 0.208 / S3 0.056 (from `profile.json`). Reuse them directly.

**Step B: country-agnostic name-noise parameters from train pairs (per source)**
- Estimate per-source probabilities for: case change, double space, bracketed legal form, junk prefix, typo count and edit position histogram (compare with GeCo's middle/end prior), leet, diacritic injection, token reorder, legal suffix move/add/drop, generic token injection, truncation, alias constructs, domain/handle, acronym. Use France-measured rates where the profile shows a difference (junk, digits).
- Condition rates on field length and source. Lam et al. 2024 found that modelling error-attribute dependencies improved synthetic-data utility.

**Step C: generation (hierarchical, mirroring section 2)**
1. Sample France S1 records. Put a share aside as **distractor originals** so that the generated France targets have ~37% distractors (section 0, point 4). Their copies are generated, but their S1 rows are removed from the synthetic S1 pool.
2. Per S1 and source, draw copy counts from the fitted per-source distribution (caps 5/6).
3. Per source, build one **base variant**: address formatter/abbreviations/admin swap/number format from A2–A4, plus a source-level name variant. Then per copy, apply independent corruptions from Step B (copy propagation, section 2.1).
4. Plant **siblings**: copy an S1, apply the |Δ| distribution measured on train sibling distractors (section 2.1 test), sometimes a 1-char name edit. Label them negative.
5. Negatives for training are then the natural competitors retrieved by the normal blocking over this synthetic France universe (real France S1 neighbours + siblings + distractor copies).

**Step D: realism checks**
- Pattern-rate check: re-run the `profile.json` pattern counters on synthetic France targets. Every rate should lie within ±20% relative of real France targets.
- **C2ST** (Lopez-Paz & Oquab): a classifier (char 3-gram bag + pattern flags, LightGBM) separates synthetic from real France targets. AUC ≤ 0.6 = acceptable. The top features show which corruption is mis-specified. Iterate 2–3 times.

**Step E: gate on LOCO-sim before touching France**
- Run the same pipeline for India: mine India style from unlabeled India targets, take name rates from US pairs only, generate synthetic India pairs from India S1. Train US-real + India-sim → score on real India labels.
- Baselines: US-only 0.94475 (E-4), in-domain ≈0.984. Proceed for France only if India-sim closes a meaningful part of that gap, or if its threshold/π calibration lands close to the oracle threshold. This is the Ruiz et al. "learning to simulate" idea: judge the simulator by downstream real performance, reduced to a 2–3 knob grid.

**Step F: usage modes (in order of risk)**
1. Calibration only: choose the France threshold / π from synthetic France validation. Lowest risk, directly addresses LOCO over-confidence.
2. Training rows at low weight (tune 0.2–0.5 on LOCO-sim).
3. France-only model: not recommended (too little real signal).

Pitfalls: (i) a generator that is too clean makes the model over-trust exact matches. The C2ST check guards against this. (ii) Do not copy libpostal/GeCo dictionaries or tables into the generator: external data. (iii) Synthetic labels are 100% clean but shifted, pseudo-labels are real but noisy (E-8b: 96.5% precision on US→India). The two are complementary: prefer calibration from FR-sim and training from strict pseudo-labels.

---

## 5. Source cards

Format: **What** · **Key technique** · **Evidence** (numbers read on the page) · **Adaptation** · **Priority** · **URL** · status.

### 5.1 GeCo: Christen & Vatsalan, "Flexible and extensible generation and corruption of personal data", CIKM 2013 (pp. 1165–1168)
- What: Python generator + corruptor for personal data, Unicode-capable, 4 attribute-dependency types.
- Technique: 6 corruptors: missing value; character edit (insert/delete/substitute/transpose at a position "more likely towards the middle or end"); keyboard (same row/column neighbour, row vs column probabilities); OCR look-up (5↔S, m↔rn); phonetic look-up (ph↔f, rie↔ry); categorical swap (misspellings, nicknames). Per-attribute weighted corruptor lists, `attr_corr_prob`, `max_num_dup_per_rec`, `num_dup_dist` ∈ {poisson, uniform, zipf}, `max_num_mod_per_attr`, `num_mod_per_rec`.
- Evidence: example config `insert_prob=0.5, delete_prob=0.2, substitute_prob=0.3, transpose_prob=0.0`, `max_num_dup_per_rec=5`, `num_dup_dist='zipf'`, `max_num_mod_per_attr=1`, `num_mod_per_rec=2`. 1M originals × 8 attributes in ~6 min, 1M duplicates in 38 min (3 GHz, 4 GB). Unicode validated on Kanji/Katakana/Hiragana.
- Source code read (tarball from dmm.anu.edu.au/geco, **MPL-2.0**): `position_mod_normal` = Gaussian with mean len/2+1 and sd len/4, resampled until inside the string. Poisson mean = 1 + num_mod/num_org. Zipf θ = 0.5. `ocr-variations.csv` holds pairs such as 5,s / 0,o / l,1 / m,rn / cl,d / w,vv. `phonetic-variations.csv` (358 lines) holds rules `POSITION,pattern,replacement,conditions` (ALL,h,@ / END,e,@ / ALL,ee,i).
- Adaptation: gives the structure of section 2 and the parameterisation for FR-sim Step B. The edit-position prior motivates token-prefix features (P2). Diagnostic: histogram of edit positions in aligned train pairs. A middle/end-skewed histogram means a GeCo-like corruptor.
- Priority: P1 (as generator template). URLs: https://users.cecs.anu.edu.au/~Peter.Christen/publications/christen2013cikm.pdf · https://doi.org/10.1145/2505515.2507815 · code https://dmm.anu.edu.au/geco/geco-data-generator-corruptor.tar.gz. VERIFIED (PDF + code).

### 5.2 GeCo online: Tran, Vatsalan & Christen, CIKM 2013 demo
- What: web front-end for GeCo. Technique/evidence as 5.1 (the abstract is elided by the publisher, only metadata read).
- Adaptation: none beyond 5.1. P3. URL: https://dl.acm.org/doi/10.1145/2505515.2508207. VERIFIED (metadata only).

### 5.3 FEBRL generator: Christen, "Probabilistic data generation for deduplication and data linkage", IDEAL 2005
- What: first FEBRL generator, frequency tables + look-up tables of real misspellings and name variants.
- Technique: single char edits; insert/delete whitespace (split/merge words); set missing / insert new value; swap with look-up value; swap two attribute values.
- Evidence: evaluation on NSW Midwives Data (175,211 records, 8,442 duplicate pairs by AutoMatch); generated 3 sets with 5%, 10%, 20% duplicates.
- Adaptation: "swap two attribute values" is the analog of component reordering; "split/merge" is the analog of run-ons ("KirtanSingh", space-less domains). Both already covered by the cat view. P3. URL: https://users.cecs.anu.edu.au/~Peter.Christen/publications/ideal2005-slides.pdf. VERIFIED (slides).

### 5.4 Christen & Pudjijono, "Accurate synthetic generation of realistic personal information", PAKDD 2009
- What: second FEBRL generator. Error taxonomy by data-entry channel (typed, handwritten, OCR, dictated, speech). **Family and household generation**.
- Technique: household = copy an original record, change given name/age/gender, "with small probability also address". ~350 phonetic rules, each a position, original pattern, substitute and four conditions.
- Evidence: the ~350-rule count and rule examples (ALL 'h'→'@'; END 'le'→'ile').
- Adaptation: **the planted-sibling distractors in our data are the business version of households.** Model them explicitly (section 2.1 test; FR-sim C4). P1. URL: https://users.cecs.anu.edu.au/~Peter.Christen/publications/pakdd2009-slides.pdf · https://link.springer.com/chapter/10.1007/978-3-642-01307-2_47. VERIFIED (slides).

### 5.5 Gecko: Jugl et al., SoftwareX vol. 27 (2024) 101846
- What: modern NumPy/Pandas successor of GeCo, **MIT**.
- Technique: mutators `with_cldr_keymap_file` (keyboard typos from CLDR layouts), `with_phonetic_replacement_table` (flags ^ start, $ end, _ middle), `with_replacement_table` (`inline`, `reverse`), `with_regex_replacement_table`, `with_permute` (swap values across columns), `with_missing_value`, `with_insert/delete/substitute/transpose`, `with_categorical_values`, `with_datetime_offset`, `with_repeat` (copy-paste duplication), `with_generator` (prepend/append generated tokens), `with_group` (mutually exclusive weighted mutators), `with_lowercase/uppercase`. `mutate_data_frame` applies (probability, mutator) per column, where probability = fraction of rows.
- Evidence: README says vectorised performance gains over GeCo, no numbers.
- Adaptation: a clean, MIT-licensed engine to implement FR-sim Step C: `with_generator` = generic-token injection, `with_repeat` ≈ "Group Group", "Partners, Partners" (seen in FP samples), `with_permute` = field swaps, `with_group` = one-of-k corruption choice. Use it only as code: supply our own mined tables. P1. URLs: https://github.com/ul-mds/gecko · https://ul-mds.github.io/gecko/data-mutation/ · https://www.sciencedirect.com/science/article/pii/S2352711024002176. VERIFIED (docs + README + citation page).

### 5.6 pseudopeople: Haddock et al., Gates Open Research 2024 (IHME)
- What: census-scale simulated population + noise functions, **BSD-3**.
- Technique: column noise with `cell_probability` and a per-token `token_probability`. Typos (keyboard-adjacent, 10% insert / 90% replace), OCR and phonetic errors (GeCo tables), nicknames, fake names, wrong digits, wrong ZIP digits with position-specific probabilities, month/day swap, copy from household member. Row noise: omit, non-response, duplicate with guardian.
- Evidence (docs defaults): `cell_probability` 0.01 for most noise types, `token_probability` 0.1; ZIP `digit_probabilities` = [0.04, 0.04, 0.20, 0.36, 0.36] (errors concentrated in the last digits); duplicate-with-guardian 2% (<18 in households) / 5% (<24 in college GQ).
- Adaptation: (i) the cell × token two-level parameterisation explains partial transliteration and partial abbreviation. The generator should sample per token (FR-sim). (ii) "Copy from household member" is the analog of siblings sharing fields. (iii) The ZIP position profile suggests checking house-number digit-position error profiles in train (last digits more often wrong → weight leading digits more in number similarity). P2. URLs: https://pmc.ncbi.nlm.nih.gov/articles/PMC11518969/ · https://pseudopeople.readthedocs.io/en/latest/noise/column_noise.html · https://pseudopeople.readthedocs.io/en/latest/noise/row_noise.html · https://github.com/ihmeuw/pseudopeople. VERIFIED.

### 5.7 Splink synthetic data (UK Ministry of Justice), via a fork
- What: synthetic person records from Wikidata, with corruption configs per output column. The original repo `moj-analytical-services/splink_synthetic_data` now returns 404. The code was read in the fork.
- Technique: "error vector" per duplicate. Each column gets −1 null / 0 no-op / k-th corruption function. Weights: null 0.1, no-op 0.5, the remaining 0.4 split by function weights (e.g. full_name: alternative 0.5, per-name alternatives 0.4, typo 0.1). `max_corrupted_records = 3` with a zipf distribution. The config comment says corruption probability rises from `start_prob_corrupt` to `end_prob_corrupt` "as we generate more duplicate records". The README's "implicit pairwise comparisons" argument: between-entity pairs are only realistic if the originals are realistic, within-entity pairs only if the error process is.
- Adaptation: (i) copies of one entity have **graded** noise, so the cleanest copy is a good anchor for collective features (use max-over-copies similarity as a feature). (ii) Their README argument means FR-sim negatives must come from real France S1 neighbours. P2. URL: https://github.com/tony-stone/splink_synthetic_data (fork; license unknown). VERIFIED (code), original UNAVAILABLE.

### 5.8 Lam, Boyd, Linacre, Blackburn & Harron, "Generating synthetic identifiers to support development and evaluation of data linkage methods", IJPDS 9(1) 2024
- What: synthetic identifiers calibrated to a real cohort (ALSPAC).
- Evidence: within-person disagreement in 18% of surnames and 12% of forenames. Synthetic data reproduced linkage-quality metrics within 0.13–0.55% (missed matches) and 0.00–0.04% (false matches). "Incorporating associations between identifier errors and maternal age/ethnicity improved synthetic data utility."
- Adaptation: condition FR-sim noise rates on source, field length and script, not one global rate. Validate the simulator by comparing linkage metrics, not string statistics (= Step E). P2. URL: http://ijpds.org/index.php/ijpds/article/view/2389. VERIFIED (abstract).

### 5.9 TDGen: Bachteler & Reiher, German Record Linkage Center WP-GRLC-2012-01
- What: KNIME workflow that garbles an input file. Error types adopted from Christen's FEBRL work.
- Technique: rows → columns → cells, with a per-cell **error score limit** (each realised error adds its score, and errors stop once the limit is reached). Error-type probability = weight / sum of weights. Generic errors: Abbreviate (token → first letter + "."), Blanks↔Hyphens, Delete Blank, Delete Token, Digit To Dot, Drop Last (Two) Character(s), Drop Nondigits, Insert Blank, **Keep First Token**, **Keep Longest Token**, Replace By Empty, Resolve/Revert Umlauts, Special Characters To Blanks, Swap Fields.
- Adaptation: (i) "Keep First Token" = our truncation. "Keep Longest Token" is a non-prefix truncation that `n_trunc_prefix` misses (add subset-at-any-position, P3). (ii) The error-score cap explains why heavily corrupted copies are rare: model a per-field cap in FR-sim. (iii) "Resolve/Revert Umlauts" = accent fold/inject. P2. URL: https://www.record-linkage.de/wp-content/uploads/2014/07/publication-including-an-installation-guide-for-TDGen.pdf. VERIFIED (PDF).

### 5.10 DaPo+: Panse, Wingerath & Wollmer, "Towards Scalable Generation of Realistic Test Data for Duplicate Detection", arXiv 2312.17324
- What: 6-phase generator (profiling → … → source creation & pollution → result), Spark-scalable.
- Technique: **event-based error model** simulating "outdated values and copying processes", with copying relationships between sources and error dependencies.
- Evidence: no quantitative results in the paper.
- Adaptation: explains why within-source copies share a corrupted base. Build **consensus-of-copies** features (P1) and generate FR-sim hierarchically (base variant → copies). P1. URL: https://arxiv.org/html/2312.17324. VERIFIED.

### 5.11 SOG (Synthetic Occupancy Generator; Talburt, Zhou & Shivaiah, ICIQ 2009), as described in arXiv 2607.21614 (2026)
- What: synthetic occupancy histories. A degraded "external view" is used for ER evaluation.
- Technique: records "disrupted through duplication, heterogeneous layouts, formatting variation, abbreviation drift, attribute omission, and typographical noise". Noise increases with dataset index. Mixed layouts: one concatenated string vs structured fields.
- Adaptation: "abbreviation drift" (inconsistent abbreviation inside one dataset) matches our mixed "AVE"/"AV"/"Avenue". Treat abbreviations as a distribution, not a single map. P3. URL: https://arxiv.org/html/2607.21614. VERIFIED (quoted text). The original ICIQ 2009 paper was not opened: UNVERIFIED.

### 5.12 LANCE: Saveta et al., "LANCE: Piercing to the Heart of Instance Matching Tools", ISWC 2015
- What: linked-data benchmark generator (successor of SPIMBENCH, extends SWING).
- Technique: value transformations vt1 blank char add/delete, vt2 random char add/delete/modify, vt3 token add/delete/shuffle, vt4 date formats, vt5 country & simple abbreviations, vt6 synonym/antonym (WordNet), vt7 stemming, vt8 multilinguality (English → 64 languages). Each has a **severity** parameter.
- Adaptation: "multilinguality" is a reminder to check France targets for translated generic tokens (English "Services" vs French "Centre"). Mine translation pairs through anchors (FR-sim A2). P3. URL: http://users.ics.forth.gr/~fgeo/files/ISWC15LANCE.pdf. VERIFIED (PDF).

### 5.13 EMBench: Ioannou & Velegrakis, "EMBench: Generating Entity-Related Benchmark Data" (ISWC 2014 poster/demo)
- Technique: "Entity Modifiers" in categories: syntactic variations (misspellings, word permutations, aliases, abbreviations, homonymity), structural variations (split vs single attribute), entity evolution. Modifier order and degree are configurable.
- Adaptation: "homonymity" = deliberate same-name different-entity cases (= our shared-name FPs/FNs). The FR-sim distractor/sibling pool must include same-name, different-address France entities (France S1 names are highly repetitive). P3. URL: https://velgias.github.io/docs/IoannouV14.pdf. VERIFIED (PDF).

### 5.14 FakER and friends: Teofili, Firmani, Koudas, Merialdo & Srivastava, "Can we trust LLM Self-Explanations for Entity Resolution?", arXiv 2606.01210 (2026)
- What: new synthetic ER sets. FB and FCP are generated with Mistral Nemo. FakER is rule-generated.
- Technique: FakER has 300 records per source (150 shared + 150 source-specific entities; "source-specific entities act as natural distractors"), 105 attributes. Noise: typos, abbreviations and truncations, synonym substitutions (street→st), missing/extra characters, limited word reordering. **Asymmetric noise** (medium in source A, high in B), seeded.
- Adaptation: the same distractor design as our S2/S3. FR-sim must keep S2/S3 noise asymmetric (per-source rates). P2. URL: https://arxiv.org/html/2606.01210. VERIFIED.

### 5.15 Steiner, Peeters & Bizer, "Fine-tuning Large Language Models for Entity Matching", arXiv 2409.08185 (DAIS 2025)
- What: fine-tuning study including **LLM-generated training examples**.
- Evidence: Llama 8B non-transfer F1 69.19 → 74.04 (+4.85) with generation + relevancy filtering. GPT-4o-mini −6.42 F1 vs baseline fine-tuning. Manual inspection found many generated examples mislabeled or too easy.
- Adaptation: **do not** create French pairs with an LLM. Rule-based FR-sim gives guaranteed labels. An LLM would also bring in outside knowledge (compliance grey zone). P3 (avoid). URL: https://arxiv.org/html/2409.08185v1. VERIFIED.

### 5.16 Rotom / InvDA: Miao, Li & Wang, SIGMOD 2021 (Megagon Labs)
- What: meta-learned data augmentation. **InvDA** fine-tunes T5 to reconstruct inputs corrupted by simple DA operators, then generates augmentations.
- Technique: EM DA operators `['del', 'drop_col', 'append_col', 'swap', 'ins']`. InvDA = seq2seq inverse of the corruption. Code **BSD-3**.
- Adaptation: two variants, both heavy and P3. (a) A **denoiser**: ByT5-small (Apache-2.0) trained on (target → S1) train pairs, used to canonicalize targets before similarity. (b) A learned **corruptor** (S1 → target) conditioned on source, to generate India-/France-style copies. (b) cannot learn French-specific rewrites (unseen), so it only complements FR-sim. P3. URL: https://github.com/megagonlabs/rotom (README + invda/README). Paper: https://dl.acm.org/doi/10.1145/3448016.3457258. VERIFIED (README; paper numbers not read).

### 5.17 libpostal (Barrentine / Mapzen / OpenVenues): method only, compliance RISK as data
- What: CRF address parser + `expand_address` normaliser, **MIT**, trained on >1 billion synthetic formatted addresses.
- Technique: training data rendered from templates, with "abbreviations, drop out components at random". Code: `abbreviate(..., abbreviate_prob=0.3, separate_prob=0.2, add_period_hyphen_prob=0.3)`. **Dependency-ordered component dropout**: components dropped in an order that keeps the address valid, each with its own dropout probability. State full-name vs abbreviation probability. Per-language dictionaries (e.g. the `fr/street_types.txt` rows "boulevard|bd|…", "chemin|ch|che", "avenue|av|ave|…").
- Evidence: 99.45% full-parse accuracy on held-out addresses (README).
- Adaptation: copy the **procedure** for FR-sim address corruption: abbreviate with probability, sometimes add a period/hyphen, drop components in dependency order (number/suffix and admin first, street last). **Do not** load libpostal dictionaries or models into the pipeline (C4 in COMPLIANCE_CHECKLIST). The French dictionary may only serve as a human sanity check that our mined map is complete, never as model input. P1 (procedure). URLs: https://github.com/openvenues/libpostal (README) · scripts/geodata/address_expansions/abbreviations.py · scripts/geodata/addresses/components.py · https://www.mapzen.com/blog/libpostal-1-0/. VERIFIED.

### 5.18 Deepparse zero-shot: Yassine, Beauchemin, Laviolette & Lamontagne, "Multinational Address Parsing: A Zero-Shot Evaluation", arXiv 2112.04008
- What: seq2seq address parsing, trained on a set of countries and evaluated zero-shot on 41 other countries. Attention and domain-adversarial (ADANN) variants.
- Evidence: accuracy on **incomplete** addresses is 20–40% lower when the model was not trained on them. Training on merged complete + incomplete data recovers about 98% (fastTextADANN) and about 93% (BPEmbADANN) on incomplete holdout. France was a *training* country (holdout ≈99.7–99.8%). Library is **LGPL-3.0**.
- Adaptation: any learned address tagger or component split (FEATURE_CATALOG H "learned per-country house-number position") must be trained with component dropout, or it breaks on the 3% empty and many partial addresses. Domain-adversarial training is an option only if a neural matcher is added (P3). P3. URL: https://arxiv.org/pdf/2112.04008. VERIFIED.

### 5.19 WDC Products: Peeters, Der & Bizer, EDBT 2024 (arXiv 2301.09521)
- What: EM benchmark varied along corner-case share, unseen-entity share and dev-set size.
- Evidence (50% corner cases, medium dev set, 0% → 100% unseen): RoBERTa 78.58 → 71.14, Ditto 79.16 → 70.24, HierGAT 75.17 → 68.74, **R-SupCon 81.88 → 57.23**.
- Adaptation: entity-memorising (contrastive, entity-level) models collapse on unseen entities. Pairwise similarity models degrade less. France has entirely unseen entities *and* style, so keep the GBDT on similarity features as the France workhorse. Any contrastively fine-tuned bi-encoder should feed retrieval or features, not decide alone. P2 (design guard). URL: https://arxiv.org/html/2301.09521. VERIFIED (table read).

### 5.20 Papadakis, Kirielle, Christen & Palpanas, "A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms", ICDE 2024 (arXiv 2307.01231)
- Technique: 4 difficulty lenses: new linearity measures, existing complexity measures, the best non-linear vs linear matcher gap, and the best learned matcher vs oracle gap. 13 datasets. "Most of the popular datasets pose rather easy classification tasks". Proposes a methodology for harder benchmarks.
- Adaptation: stratify validation by corruption type × ambiguity and compute the linear-vs-GBDT gap per stratum. Strata where the gap is ~0 are "solved". Ambiguity strata (shared names, siblings) are where effort pays. This matches ERROR_ANALYSIS. P3. URL: https://arxiv.org/abs/2307.01231. VERIFIED (abstract).

### 5.21 Primpeli & Bizer, "Profiling Entity Matching Benchmark Tasks", CIKM 2020
- Technique: 5 profiling dimensions: schema complexity, textuality, sparsity, development-set size, corner cases. 21 tasks grouped, with baseline difficulty.
- Adaptation: per-country profile (sparsity = empty/missing components, corner cases = same-name different-entity share) to predict where France differs. France has high name repetition, i.e. many corner cases. P3. URL: https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Web-based_Systems/pub/CIKM2020_Primpeli_Bizer.pdf. VERIFIED.

### 5.22 Wang et al., "Bridging the Gap between Reality and Ideality of Entity Matching: A Revisiting and Benchmark Re-Construction", IJCAI 2022 (arXiv 2205.05889)
- Technique: criticises closed entity sets, balanced labels and fixed candidate sets. Re-constructs benchmarks with open entities and imbalance.
- Adaptation: supports P-dense (test-like distractor density) and P-loco. Nothing new to build. P3. URL: https://arxiv.org/abs/2205.05889. VERIFIED (abstract).

### 5.23 Classifier two-sample tests: Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests", arXiv 1610.06545
- Technique: train a classifier to separate two samples. Near-chance held-out accuracy supports the hypothesis that they come from the same distribution. Proposed for judging generative-model samples.
- Adaptation: FR-sim realism check (Step D). The C2ST's top features name the mis-specified corruption. P2. URL: https://arxiv.org/abs/1610.06545. VERIFIED.

### 5.24 Ruiz, Schulter & Chandraker, "Learning To Simulate", ICLR 2019 (arXiv 1810.02513)
- Technique: tune simulator parameters (RL) to maximise the downstream model's accuracy on *real* validation data.
- Adaptation: LOCO-sim (Step E) is a cheap grid version: pick the simulator knobs (typo rate, abbreviation rate, distractor share) that maximise real-India F0.5 when the model is trained on US-real + India-sim, then reuse them for France. P2. URL: https://arxiv.org/abs/1810.02513. VERIFIED.

### 5.25 Indic parallel-block transliteration: Indic NLP Library (`indicnlp.transliterate.unicode_transliterate`, **MIT**)
- Technique: `offset = ord(c) − SCRIPT_RANGES[lang][0]`. Map the same offset into the target script's block (the ISCII-derived parallel layout of the Brahmic Unicode blocks), with special cases (Tamil missing plosives, Sinhala via Devanagari, Malayalam chillus).
- Adaptation: normalise Bengali/Gujarati/Gurmukhi/Oriya/Tamil/Telugu/Kannada/Malayalam names to Devanagari by offset (re-implement: 10 lines, no library needed). The learned native→Latin dictionary then pools counts across scripts. Rare-script tokens gain coverage. Cross-script errors are already ≤5% of errors, so P3. URL: https://indic-nlp-library.readthedocs.io/en/latest/_modules/indicnlp/transliterate/unicode_transliterate.html · https://github.com/anoopkunchukuttan/indic_nlp_library. VERIFIED.
- Related: libindic `soundex` (Thottingal) applies one phonetic code table across Indic scripts via parallel charmaps. **LGPL-3.0**: method only. https://github.com/libindic/soundex. VERIFIED (code).

### 5.26 Brill & Moore, "An Improved Error Model for Noisy Channel Spelling Correction", ACL 2000 (pp. 286–293)
- Technique: channel model over generic string-to-string (substring) edits, learned from (misspelling, correct) pairs.
- Adaptation: learn P(target substring | S1 substring) from aligned train pairs (all corruption types at once: leet, OCR-like, abbreviation, diacritics). Features: channel log-likelihood of t given q, and a learned-cost edit distance. It also serves as a data-driven corruptor for FR-sim Step B. P2. URL: https://aclanthology.org/P00-1037/ (metadata). The "generic string to string edits" technique statement is from the MSR/ACL listing: VERIFIED (metadata); the paper body was not read.

### 5.27 Structure-Guided ER: Chourasia, Kapoor & Patil, ACL 2026 (arXiv 2605.23597), Indian KYC names
- What: Indian person-name matching with a two-phase LLM (parse to JSON, then match).
- Technique: augmentation from 20,000 curated pairs to 50,000+ via pair swapping, component permutation, and **random space removal** ("Kirtan Singh" → "KirtanSingh").
- Evidence: SFT F1 0.946 → SFT + Aug. 0.973 → SGER 0.994 (Llama 3 8B, held-out 50,000 real pairs). GPT-4o few-shot 0.911.
- Adaptation: space-removal and permutation augmentation are cheap and license-free. The model itself (Llama 3) is **not** MIT/Apache, so it is not usable. Our cat view already inverts run-ons. P3. URL: https://arxiv.org/html/2605.23597. VERIFIED.

### 5.28 nlpaug (**MIT**): KeyboardAug / OcrAug / RandomCharAug
- Technique: keyboard-distance substitution, OCR confusion map, random insert/substitute/swap/delete.
- Adaptation: ready-made char corruptors for FR-sim if Gecko is not used. Bundled keyboard/OCR maps are small generic tables. Safer to derive confusion maps from train pairs (5.26). P3. URL: https://github.com/makcedward/nlpaug. VERIFIED (license via GitHub API; augmenter descriptions from docs listing).

### 5.29 Senzing truth sets (**Apache-2.0**) and "How to create an entity resolution truth set"
- What: vendor methodology. Three data sources (customers, watchlist, reference) with an expected-cluster key.
- Technique: add "two to ten times the number of harder matches"; "records that should not match are just as important"; include foreign scripts and empty fields.
- Adaptation: FR-sim negative mix should over-represent hard negatives (siblings, same-name). P3. URLs: https://github.com/Senzing/truth-sets · https://senzing.zendesk.com/hc/en-us/articles/360051016033-How-to-create-an-entity-resolution-truth-set. VERIFIED.

### 5.30 Challenge-side public artefacts (third-party, treat as data)
- The problem statement, copied in public participant repos (e.g. https://github.com/ItsDemonic/amazon-business-entity-resolution-challenge README), lists the intended noise ("DBA/trade names", "& vs and", "municipal numbering formats", "landmark-based references"). The listed families match what we measured, with no new family.
- A participant EDA (https://github.com/ApexYash11/amazon_ml_challenge_2026/blob/main/docs/EDA_FINDINGS.md) reports test France counts S2 703,378 / S3 731,615. Our `profile.json` gives the same numbers. Used for the density calculation (section 0, point 4). VERIFIED (read). Do not use any participant code.

---

## 6. Unverified / secondary-only mentions
- Hernandez & Stolfo (SIGMOD 1995) DBGen: lists of names/cities without frequencies, edit-based corruption and categorical swaps. Bertolazzi et al. (2003): adds missing values. Arehart & Miller (LREC 2008): 70,000 culturally diverse romanized names with manual variants. All three are known only through Christen & Vatsalan 2013 related work. **UNVERIFIED** (primary not opened).
- Comber & Arribas-Bel, "Machine learning innovations in address matching" (Trans. GIS 2019): title/authors/venue verified via Semantic Scholar. The claim that they generated ~934k synthetic non-matched address pairs with the FEBRL generator came from a search snippet (publisher PDF blocked). **UNVERIFIED**.
- SOG original (Talburt, Zhou, Shivaiah, ICIQ 2009): **UNVERIFIED** (only described in 5.11).
- Splink docs claim (median 8 duplicates, range 0–35 in their Wikidata benchmark): search snippet only. **UNVERIFIED**.

---

## 7. Compliance notes
- Allowed: implementing corruption **procedures** (GeCo/Gecko/pseudopeople/libpostal ideas) in our own code, with all tables and maps **mined from the challenge files** (train pairs, unlabeled test S1/S2/S3).
- Not allowed as model input: GeCo look-up tables (OCR/phonetic/misspellings), libpostal dictionaries/models, Faker/Wikidata name lists, LLM-generated pairs (outside knowledge), CLDR keyboard files. A tiny hand-written QWERTY adjacency is generic knowledge like "St"→"Street" (C6). It still counts as a grey zone, so prefer confusion maps learned from train pairs.
- Libraries: Gecko (MIT), nlpaug (MIT), Indic NLP Library (MIT), pseudopeople (BSD-3), Rotom (BSD-3) are fine as code. GeCo is MPL-2.0, libindic and Deepparse are LGPL-3.0: method only.

## 8. Local evidence computed for this note (lightweight, sample-based)
Script: inline Python over `experiments/reports/samples_train_clusters.txt` (true clusters) and `errors_lgb_v1_samples.txt` (FP section). Compares the first number in the S1 address with the numbers in the target address.
- True-copy comparisons (1,198 with numbers on both sides): exact 1,073. |Δ| = 1: 13, 2: 9, 3: 1, 4–10: 7, 11–20: 12, 21–50: 6, >50: 77 (mostly digit deletion, leading zeros, other numbers). No number in the target: 188.
- FP pairs (106 with numbers): exact 50. |Δ| = 1: 8, 2: 4, 3: 5, 4–5: 8, 6–10: 5, 11–20: 8, 21–50: 5, >50: 13.
- Digit-edit classes: keyboard-adjacent single substitutions appear in both (true 14, FP 8). Not discriminative.
- Caveats: clusters repeat copies (not independent), the FP set is conditioned on model errors, and n is small. Re-measure on full train (distractor siblings, section 2.1) before trusting the ~12× likelihood ratio for |Δ|∈[1,5].
- Pattern-rate tables in §2.4–2.5 are read directly from `experiments/reports/profile.json` (full-data profiling, 2026-09-25).
- Target density: train 10,320,219 / 2,206,821 = 4.677. Test non-France (9,969,589 − 1,434,993) / 1,473,092 = 5.794. France 1,434,993 / 259,452 = 5.531. With 3.46 true matches per S1, the implied distractor shares are 0.26 / 0.40 / 0.37.
