# ERROR_ANALYSIS.md

Taxonomy built from the data (DATA_ANALYSIS §5). Each category lists how we detect it and what mitigates it. Measured error counts are filled from validation predictions (`stage_train` val predictions + `stage_analyze`, TODO).

| # | Failure mode | Detector (label-free) | Blocking lever | Feature lever | Model/decision lever | Measured FN / FP share |
|---|---|---|---|---|---|---|
| 1 | Name-only match (address empty / "null") | `a_t_empty`, `t_ad_scr=empty` | name, cat views | name sims, name frequency | name-frequency-aware threshold; exclusivity | TBD |
| 2 | Address-only match (pure alias "Kelojax", acronym "SC") | low name sims, high address sims | addr view, reverse view | `n_initials`, `n_alias_core_jacc`, address IDF overlap | collective support from other copies (E-6) | TBD |
| 3 | Cross-script name | `t_nm_scr ∈ {brahmic, mixed}` | learned dictionary → name view | cosines on transliterated text, phonetic Jaccard | — | TBD |
| 4 | Typos / leetspeak / injected diacritics | edit distance vs token Jaccard gap | char n-gram views | JW, Levenshtein, char cosines | — | TBD |
| 5 | Token reorder / legal suffix moved | token-set ≫ ratio | char_wb n-grams (order-free) | token_set / core-set equality | — | TBD |
| 6 | Truncation ("Deleon", "Allied") | `n_trunc_prefix`, `n_idf_cov_t` high, `n_idf_cov_q` low | name view | coverage asymmetry features | name-frequency (generic heads are ambiguous) | TBD |
| 7 | Alias constructs (dba/aka/fka) | `t_has_alias` | name view on full string | alias-part Jaccard | — | TBD |
| 8 | Handles / domains | `t_has_dom` | cat view | `n_dom_prefix_cov`, cat partial ratio | — | TBD |
| 9 | Shared / generic names (chains, "Primary Care Group", "Lille Club SARL") | `q_nm_freq_s1 > 1` | combo, reverse | address sims, competition margins | exclusivity; per-target argmax | TBD |
| 10 | Shared addresses (several businesses at one address) | many S1 with the same address | — | name sims decide | exclusivity | TBD |
| 11 | Address number noise (typo, missing, "#"/"No" prefix, leading zeros) | number-set features | addr view (other tokens) | `a_num_*`, `a_bignum_jacc` | — | TBD |
| 12 | City variants / region↔département (France) | address IDF overlap low but number + street match | combo | street-only overlap (TODO: component split by learned position) | — | TBD |
| 13 | Singleton S1 wrongly matched (similar distractor exists) | S1 with no true match | — | competition + name frequency | **match-exists model π** (E-7), expected-F empty option | TBD |
| 14 | Ambiguous candidates (two near-identical S1) | small `tmargin` | — | margins | exclusivity and a margin feature | TBD |
| 15 | Blocking miss | pair not in candidates | new views, larger K, reverse view | — | — | see BLOCKING_STRATEGIES |

## Findings log
_(dated entries; newest last)_

### 2026-09-25 — lgb_v1 (thr 0.7 + excl), val fold 0: 3,332 FP, 17,885 FN-in-candidates, 738,302 TP
| detector | FP (share) | FN (share) | TP |
|---|---|---|---|
| target address empty | 704 (0.21) | 8,737 (**0.49**) | 18,224 |
| shared S1 name (freq > 1) | 1,265 (0.38) | 10,554 (**0.59**) | 291,527 |
| not best S1 for target (trank > 1) | 443 (0.13) | 8,322 (0.47) | 4,449 |
| low address sim (< 50) | 715 (0.21) | 8,810 (0.49) | 18,407 |
| core-name equal | 1,905 (**0.57**) | 11,284 (0.63) | 508,955 |
| truncated name | 285 (0.09) | 1,161 (0.06) | 51,605 |
| cross-script | 160 (0.05) | 441 (0.03) | 54,390 |
| handle/domain | 52 (0.02) | 211 (0.01) | 39,686 |
| alias / acronym | 0 | 0–2 | 13,365 |
Singletons wrongly matched: 266. S1 with matches but empty prediction: 443.

Findings:
1. **FN = mostly undecidable ambiguity**: an empty target address plus a name shared by several S1s ("Regional Dynamic Center", "Pipefitters Local 836"). The name alone cannot pick the S1. Leave as FN; F0.5 penalises a guess more than a miss.
2. **FP = planted sibling distractors**: same or near-same name on the same street with a nearby house number ("Amber Inc 340 vs 345 Olympic Park Dr", "Quex Enhanced Group 1927 vs 1931"). The generator also offsets house numbers of TRUE copies ("Downtown Deli 104 vs 103 Main St"), so a number offset is not decisive by itself.
3. Cross-script, alias, acronym and handle cases are essentially solved (≤ 5% of errors) thanks to the learned dictionary and the alias/handle features.
4. Mitigation implemented: collective / sibling-support features (stage 2) → FP-driven precision 0.9959 → 0.9973 and +0.0021 macro F0.5. House-number geometry features coded for v2.
