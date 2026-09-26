# FEATURE_CATALOG.md

Every feature is a label-free function of the two records (plus unlabeled corpus statistics), and is computed identically for train, validation and test. Code: `src/ber/features.py`, `modal_app/stage_features.py`. Importance = LightGBM gain (filled after runs).

Notation: `q` = S1 record, `t` = target (S2/S3) record. `nm` = normalized name, `ad` = normalized address, `cat` = nm with spaces removed. "Core" tokens = name tokens whose document frequency within the country is ≤ 1% (generic tokens such as legal forms, "services" and "group" are learned from data, not listed by hand).

## A. Normalization-level (upstream; see DATA_ANALYSIS §5)
Script-aware transliteration, learned native→Latin dictionary, learned abbreviation maps, alias split (dba/aka/fka), domain/handle extraction, leetspeak repair, null-token removal, single-letter-run collapse.

## B. Retrieval features (from blocking; `stage_block.py`)
| name | definition |
|---|---|
| s_name, r_name | char_wb-3gram TF-IDF cosine / rank of t in q's name list (0 / 999 if not retrieved) |
| s_cat, r_cat | char-3gram TF-IDF on space-less name (handles, domains, run-ons) |
| s_addr, r_addr | word TF-IDF cosine on address |
| s_combo, r_combo | hybrid 0.5·name + 0.5·address cosine |
| s_rev, r_rev | hybrid cosine and rank of q in **t's** top-8 S1 list (target-side retrieval) |

## C. Exact TF-IDF cosines (every pair; `rowwise_cos`)
c_name (char_wb 3-gram), c_cat (char 3-gram, space-less), c_nword (word-level name TF-IDF, max_df 0.5), c_addr (word), c_combo = 0.5·c_name + 0.5·c_addr.

## D. RapidFuzz string similarities (`fuzz_feats`)
Name: n_ratio, n_tsort, n_tset, n_partial, n_wratio, n_jw (Jaro-Winkler), n_cat_ratio, n_cat_partial, n_cat_lev (normalized Levenshtein on the space-less name).
Address: a_ratio, a_tsort, a_tset, a_partial.

## E. Token-level features (`token_feats_batch`)
| name | definition | targets which noise |
|---|---|---|
| n_jacc | token Jaccard | reorder, generic-token injection |
| n_idf_jacc | IDF-weighted Jaccard | rare vs generic tokens |
| n_idf_cov_q / n_idf_cov_t | share of q's / t's IDF mass that is matched | truncation vs extra tokens |
| n_max_shared_idf | IDF of the rarest shared token | a distinctive surname/brand shared |
| n_core_jacc, n_core_eq | Jaccard / exact equality of core token sets | legal-form and generic-word noise |
| n_first_tok_eq | first tokens equal | truncation keeps the head |
| n_initials | t equals the initials of q's core tokens | acronym names ("SC") |
| n_phon_jacc | Jaccard of phonetic keys of core tokens | transliteration variants, typos |
| n_alias_core_jacc | best core Jaccard vs either side of "X dba/aka/fka Y" | alias records |
| n_len_q, n_len_t, n_core_missing, n_core_extra | sizes and set differences | truncation, injection |
| n_trunc_prefix | t's core tokens are a strict prefix of q's | truncated names |
| n_dom_prefix_cov | handle/domain core is a prefix of q's space-less name (coverage ratio) | "@parabolalaw", "laxmibombayservices.com" |
| a_num_jacc, a_first_num_eq, a_num_shared, a_q_num_missing, a_bignum_jacc | address number agreement (house number, 3+ digit numbers) | number typos, missing numbers, "#"/"No" prefixes |
| a_idf_jacc, a_idf_cov_t, a_max_shared_idf | IDF-weighted overlap of alphabetic address tokens | street/locality overlap under reorder |
| a_t_empty | target address empty | 3% of targets |

## F. Structural / ambiguity
t_src (2/3), t_nm_scr / t_ad_scr (latin, brahmic, mixed, other, empty), t_has_dom, t_has_alias, q_nm_freq_s1 (number of S1s with the same normalized name; ambiguity), t_nm_freq_s1, t_nm_freq_t.

## G. Competition / context (over the FULL candidate graph, label-free)
For c ∈ {c_combo, c_name, c_addr}:
- `{c}_qrank`, `{c}_qgap`: rank and gap to the best candidate **within q's list**;
- `{c}_trank`, `{c}_tgap`: rank and gap of q among all S1s competing for **the same t** (many-to-one signal);
- `{c}_tmargin`: margin over the best *other* S1 for t (> 0 only if q is the unique best);
- q_ncand, t_nq: list sizes.

## H. Planned (not yet implemented) — see EXPERIMENT_PLAN
- Collective/support features: similarity of t to q's top-1 candidate, number of near-duplicate co-candidates.
- Dense multilingual embedding cosines (name, address) from an MIT/Apache encoder (GPU).
- Cross-encoder score (fine-tuned small multilingual model) as a stacked feature.
- Learned per-country house-number position and postcode shape features.
