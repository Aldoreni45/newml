# DATA_ANALYSIS.md — measured characteristics of the supplied data

Source: full-data profiling on Modal (`modal_app/profile_data.py`, 2026-09-25). Raw numbers:
`experiments/reports/profile.json`, readable samples: `experiments/reports/samples_*.txt`.
All numbers below are measured, not estimated, unless marked *(inferred)*.

## 1. File integrity
- All 7 TSVs parse cleanly: 0 rows with a wrong tab count, 0 CRLF rows, every entity_id unique and correctly prefixed.
- GT covers **every** train S1 (2,206,821 rows = S1 file), every GT id exists in the S2/S3 files.

## 2. Sizes and country mix

| split | S1 | S2 | S3 | S2/S1 | S3/S1 |
|---|---|---|---|---|---|
| train | 2,206,821 (US 1,323,633 · India 883,188) | 5,034,616 | 5,285,603 | 2.28 | 2.40 |
| test | 1,732,544 (India 809,986 · US 663,106 · **France 259,452**) | 4,887,273 | 5,082,316 | 2.82 | 2.93 |

- **France = 15.0% of test S1** (unseen in training) → worth ~0.15 of the macro score.
- Test has **~24% more targets per S1** than train. Plausible explanation *(inferred)*: a larger share of S2/S3 records belong to entities that are absent from S1 (distractors). Consequence: train-calibrated probabilities/thresholds will be optimistic on test unless validation simulates this (see VALIDATION_STRATEGY §density).

## 3. Ground-truth structure (train)
- 7,638,365 true (S1, target) pairs: S2 3,693,619 · S3 3,944,746. Mean 3.46 matches per S1.
- Matches per S1: 0: 123,247 (**5.58% singletons**) · 1: 119,157 · 2: 375,212 · 3: 530,841 · 4: 484,115 · 5: 321,957 · 6: 164,868 · 7: 63,968 · 8: 18,680 · 9: 4,205 · 10: 534 · 11: 37.
- Per source: S2 matches per S1 range 0–5 (mode 1), S3 0–6 (mode 1). S2 and S3 each contain several noisy copies of the same entity.
- Singleton rate and cardinality are identical for US and India (5.58% / 5.59%, mean 3.46).
- **Target multiplicity = 1 for all 7.64M matched targets**: no S2/S3 record is linked to two S1 records. S1 is deduplicated, so every target belongs to **at most one** S1 → a many-to-one assignment constraint is valid.
- **Coverage**: 73.4% of S2 and 74.6% of S3 records match some S1. The other ~26% (≈2.68M records) are distractors: realistic businesses whose entity has no S1 record.
- **Country agreement = 100%** on all 7.64M true pairs (US→US, India→India). Using country-label equality as a blocking partition is lossless on train. It works for any label, so this is not hard-coding.

## 4. Field statistics (medians; rates are fractions of records)

| file/country | name len (chars/tokens) | name non-ASCII | name unique rate | addr len | addr empty | addr non-ASCII |
|---|---|---|---|---|---|---|
| train S1 US | 22 / 3 | 0 | 0.738 | 34 | 0 | 0 |
| train S1 India | 27 / 4 | 0 | 0.638 | 76 | 0 | 0.001 |
| train S2 US | 23 / 3 | 0.067 (diacritics) | 0.902 | 32 | 0.037 | 0 |
| train S2 India | 27 / 4 | **0.279** | 0.835 | 68 | 0.029 | 0.237 |
| train S3 US | 23 / 3 | 0.068 | 0.891 | 39 | 0.035 | 0 |
| train S3 India | 27 / 4 | **0.185** | 0.866 | 58 | 0.031 | 0.225 |
| test S1 France | 19 / 3 | 0.157 (accents) | 0.745 | 48 | 0 | 0.283 |
| test S2 France | 20 / 3 | 0.245 | 0.880 | 40 | 0.031 | 0.241 |

- **S1 names are highly non-unique**: only 64% (India) to 74% (US) of S1 names are distinct. Examples: "Primary Care Group", "Shree Trading Private Limited", "Bordeaux Club SARL". Many S1 entities can only be told apart by address, which makes address the primary discriminator.
- S1 is clean: no junk, no non-ASCII, Title Case (France keeps its accents).
- Indic scripts in India S2 names: Devanagari 13.4%, Telugu 1.9%, Kannada 1.8%, Tamil 1.7%, Bengali 1.5%, Gujarati 1.5%, Malayalam 0.9%, Oriya 0.4%, Gurmukhi 0.3%. Addresses: native-script state names in ~23% of India target addresses (e.g. "महाराष्ट्र", "ಕರ್ನಾಟಕ", "ગુજરાત").
- Postcodes are rare: US ZIP5 in ~11% of addresses; India PIN6 in <2%; France ~0.5% in targets. **Postcode blocking alone covers little.**
- Empty target addresses: 2.3–3.7%. In those cases only the name can decide.

## 5. Noise catalogue (from 400 sampled clusters + pattern rates)
Name-side (target vs S1):
- case change (S2 US names all-upper 21%), doubled spaces (~11%), bracketed tokens `[Inc]`, `(Ltd)` (~6–10%), junk prefixes `--`, `<<`, `***` (~2%);
- typos: insert/delete/swap/replace ("Madr", "Bdrtehc", "Tmradhg"), leetspeak (`0`↔`o`, `l`↔`I`: "Téchn0logies", "lnstitute"), injected diacritics ("Léarning", "Ínc");
- token reorder ("Center Solutions Cinderella's", "Arp Ltd Buddha"), suffix moved to the front ("LLC Moncada …", "Pvt. EFS …");
- generic token injection/deletion: legal forms (LLC/Inc/LP/Corp/Co/LLP/Ltd/Private/Limited), "Services", "Center", "Partners", "Group", titles "The", "Dr", "Mr", "Smt";
- truncation to the leading tokens ("Deleon", "Allied", "Oo, Scott &");
- **aliases**: "Kelonylalum a/k/a Allied Federation", "Onyxdeltanyla f/k/a Gouveia Ramaco LLC", "Miraquojax dba E E Johnson Offshore LLC". Pure alias records also exist ("Kelojax" matched by address only);
- handles/domains: "@parabolalaw", "laxmibombayservices.com", "www.blacktrad.com" (truncated);
- acronyms: frequent 2-letter names in S3/test S2/S3 ("SC", "SI", "SS", "CC", "PC");
- **cross-script** full or partial transliteration of Indian names ("लक्ष्मी बॉम्बे सर्विसेज प्राइवेट लिमिटेड", "Urban हेल्थकेयर प्राइवेट लिमिटेड", "బ్లాక్ ట్రేడింగ్ ప్రైవేట్ లిమిటెడ్").

Address-side:
- abbreviation (Road/Rd, Drive/DR, Avenue/AVE, Trail/TRL, West/W), state code ↔ full name (TX ↔ Texas, MH ↔ Maharashtra ↔ महाराष्ट्र);
- component reordering, missing components (house number dropped, city dropped), added PO Box / PMB / Unit / Fl;
- city-name variants ("Chesterfield County"/"Chester", "City Of Socorro", "Thurmmont CDP", "Hazel Green City", "Bombay, Mumbai", "Pune Region"), occasionally a different neighbouring city ("Cambria" vs "SNABORN");
- number noise: leading zeros ("00514"), suffixes ("5935-D", "2057-"), `#`/`##`/"No"/"Door No"/"H.NO"/"HN" prefixes, digit typos ("54" vs "64", "2771" vs "2774");
- null tokens ("null", "NULL", "<NULL>", "N/A") in ~3–4% of target addresses; landmarks "Near …" (India 9–13%).
- Within-source copies of the same entity **share the same corrupted base** (e.g. two S3 copies both have "B/80+802+803" where S1 has "B/801+802+803"). Target-to-target similarity is therefore a strong extra signal, supporting collective/graph features.

France (test only):
- region ↔ département swaps (S1 "Hauts-de-France" vs S2 "Nord"/"Pas-de-Calais"; "Pays de la Loire" vs "Loire-Atlantique"); French abbreviations "R."/"BD"/"CH"/"AV", "N°", "bis/ter/quater";
- accents injected or removed ("Àmicale", "Immôbiliere"); lower-cased names; legal forms SARL/SAS/SASU/SCI/EURL/EI/SA;
- **very repetitive names** built from a small vocabulary plus city names ("Lille Club SARL"). France will depend on address matching even more than the other countries.

## 6. Leakage probes (all negative)
- Spearman(S1 numeric id, matched target numeric id) = 0.0001; Spearman(S1 row, target row) ≈ 0.001. Matched copies are scattered through the files (median intra-entity row gap 1.09M rows). **No id or row-order leakage.** Ids and row order are never used as features.
- Train/test id overlap: 0 for every source. Exact (name, address) overlap test S1 ↔ train S1: 0. Exact name overlap: 584,933 test S1 names also occur in train S1. These are generic names, not leaked entities.

## 7. Positive-pair similarity (200k sampled true pairs, RapidFuzz on lower-cased, punctuation-stripped text)

| src\|country | name norm-equal | name token-set p10 / p50 | addr token-set p10 / p50 | target non-ASCII & S1 ASCII | shared number |
|---|---|---|---|---|---|
| S2\|US | 0.256 | 77 / 100 | 76 / 95 | 0.076 | 0.766 |
| S3\|US | 0.257 | 77 / 100 | 70 / 88 | 0.074 | 0.796 |
| S2\|India | 0.147 | **11** / 93 | 85 / 100 | 0.279 | 0.835 |
| S3\|India | 0.170 | **14** / 95 | 74 / 94 | 0.190 | 0.819 |

Random same-country pairs score about 31–33 (p50) and 52–71 (p99) on name token-set, and 34 / 51–52 on address. Without transliteration, cross-script India pairs have almost no Latin name similarity (p10 ≈ 11), while their addresses stay highly similar. Two design needs follow: (a) a cross-script bridge for names, and (b) blocking that also retrieves by address.
