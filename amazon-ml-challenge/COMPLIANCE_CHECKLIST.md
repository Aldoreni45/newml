# COMPLIANCE_CHECKLIST.md

Authoritative source: `student_resource/README.md` + problem-statement PDF (read 2026-09-25).
Every item must be re-checked on the FINAL package before upload. Status legend: ☐ open · ☑ verified · ⚠ risk.

## A. Output format (hard rejection if violated)

| # | Rule (verbatim intent) | How we guarantee it | Status |
|---|---|---|---|
| A1 | Both files tab-separated, UTF-8, `\n` line endings | `ber.io.write_id_list_file` writes `\t`, utf-8, `newline="\n"` | ☐ |
| A2 | Exact headers: `source1_entity_id<TAB>matched_entity_ids` and `source1_entity_id<TAB>candidate_entity_ids` | constants `MATCH_HEADER` / `CAND_HEADER` | ☑ validator PASS |
| A3 | Every test S1 entity has exactly one row (incl. France) | writer iterates the full test_source1 id list; raises on duplicates | ☑ sub_v1: 1,732,544 rows (validator) |
| A4 | Empty list for singletons (no placeholder text) | writer emits `S1-x\t` | ☐ |
| A5 | No duplicate IDs inside a list | `dict.fromkeys` de-dup in writer | ☑ validator PASS |
| A6 | Only S2-/S3- IDs; no S1 self-matches | writer filters prefix | ☑ validator PASS |
| A7 | IDs must exist in the test S2/S3 files | pipeline only emits ids read from test files; run validator with `--check-ids` on Modal (needs a few GB RAM) | ☑ sub_v1 (9,969,589 valid ids checked) |
| A8 | Matches ⊆ candidates (validator warns otherwise) | matching stage only selects from the final candidate list; assert in code | ☑ by construction (selection mask over the scored frame); validator raised no subset warning |
| A9 | `candidate_pairs.tsv` = the LAST candidate set the model scores (not an early blocking pass) | write it from the exact frame fed to the final scorer | ☑ stage_predict writes candidates from the scored frame F |
| A10 | Official validator passes: `python3 utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test [--check-ids]` | run on Modal before every upload | ☑ sub_v1 PASS with --check-ids (Modal) + PASS locally |

## B. Model / license rules

| # | Rule | Our position | Status |
|---|---|---|---|
| B1 | "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters" | Every pretrained weight used anywhere in the final pipeline must have an MIT or Apache-2.0 license on its model card, ≤ 8B params. Our own trained models (GBDT, small encoders) are trained only on provided data. License table → `research/MODEL_LICENSES.md` | ☑ only pretrained weight: minishlab/potion-multilingual-128M (MIT, 128M params, rev 73908c34); LightGBM models trained from scratch |
| B2 | No Gemma / Llama / CC-BY-NC / "other"-licensed weights | explicitly excluded | ☑ none used |
| B3 | Software library licenses are separate from model licenses | libraries recorded in README with their licenses | ☑ README §Licences |

## C. Fair play — NO external data lookup (disqualification risk)

| # | Prohibited | Our position | Status |
|---|---|---|---|
| C1 | Commercial ER APIs/services | none used | ☑ (design) |
| C2 | Government business-registration lookups | none used | ☑ (design) |
| C3 | Geocoding APIs for address normalization | none used; all normalization local, rule- or data-driven | ☑ (design) |
| C4 | External data augmentation from internet sources | no external datasets. ⚠ Libraries that bundle externally-derived data (e.g. libpostal's OSM-trained parser + gazetteers) are treated as RISK and not used unless explicitly judged safe | ☐ |
| C5 | Research sources (papers, blogs, repos) inform METHOD only; never data used by the model | `LITERATURE_REVIEW.md` separates "research sources" from "data used by the model" | ☐ |
| C6 | Hand-written normalization knowledge (e.g. "St"→"Street") | Allowed as generic text normalization, but preferred approach is to LEARN abbreviation/alias maps from the provided data (train pairs; unsupervised test co-occurrence) so the method generalizes to unseen countries | ☐ |

## D. Open-set country rule

| # | Rule | Our position | Status |
|---|---|---|---|
| D1 | Do not hard-code, filter, or one-hot the pipeline to {US, India} | country used only as a partition key discovered from data + for per-country unsupervised statistics (IDF, postcode patterns); never as a fixed categorical feature | ☑ feature matrices contain no country column (verified in feature lists) |
| D2 | Every test entity incl. France appears in the submission | row-per-S1 writer + count check per country | ☑ sub_v1 France 259,452 rows, 3.45 matches per S1 |
| D3 | Validate unseen-country generalization | leave-one-country-out (train US→eval India and vice-versa) in `VALIDATION_STRATEGY.md` | ☑ done (MODEL_COMPARISON LOCO table) |

## E. Final package structure

```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   (filled in)
```
| # | Rule | Status |
|---|---|---|
| E1 | README with exact end-to-end run instructions (data → blocking → matching → output) | ☑ README Option A (Modal) + Option B (run_pipeline.py) |
| E2 | requirements.txt with pinned versions | ☑ |
| E3 | Regenerates both outputs from train/test data using only what is in the folder (model weights downloaded from their public source are declared) | ☐ |
| E4 | Documentation_template.md filled (methodology, blocking, model+features, threshold method, F0.5, FP/FN analysis) | ☑ draft (team name and members to fill) |
| E5 | Seeds fixed; deterministic reruns | ☑ seeds fixed; ⚠ an end-to-end rerun has not been executed yet (compute paused) |

## F. Leakage / integrity self-audit

| # | Check | Status |
|---|---|---|
| F1 | No use of entity_id numeric values or file row order as features (probed for leakage in profiling; never exploited) | ☑ ids only used for fold assignment (id mod 10), never as features |
| F2 | Validation splits grouped by S1 entity; no target-label leakage into features | ☑ grouped by S1; normalization maps now exclude eval fold 0 (code fixed 2026-09-25; takes effect on the next run). Current reported numbers still carry the mild bias |
| F3 | Transductive use of UNLABELED test records (e.g. IDF, name frequency, reverse-rank) documented as unsupervised and label-free | ☑ FEATURE_CATALOG / VALIDATION_STRATEGY §0 |

## G. Software-licence hygiene (not a rule, but reviewed)
| # | Item | Status |
|---|---|---|
| G1 | GPL `unidecode` fallback replaced by ISC `anyascii` in code | ☑ code; ⚠ records v1 were produced with unidecode installed (only non-Latin/non-Brahmic symbols such as "°" hit the fallback). A final clean run uses anyascii |
| G2 | No AGPL/GPL libraries in the shipped pipeline (Aksharamukha, Zingg, abydos avoided) | ☑ |
