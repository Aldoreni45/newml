# License-verified pretrained models (<= 8B, MIT / Apache-2.0) for Business Entity Resolution

*Amazon ML Challenge 2026: Business Entity Resolution (S1 -> S2/S3, US + India train, France unseen at test).*
*Research date: 2026-09-25. Nothing was run on the laptop. Every license and parameter count below comes from the Hugging Face Hub API (`https://huggingface.co/api/models/<id>`, fields `cardData.license`, `license_name`, `safetensors.total`), from the model card README (`https://huggingface.co/<id>/raw/main/README.md`), or from `config.json`. The model page is `https://huggingface.co/<id>`. Repo facts come from the GitHub API (via `gh api repos/<owner>/<repo>`). Paper facts come from the arXiv API, ACL Anthology pages, and publisher PDFs that were actually fetched.*

---

## 0. TL;DR

* **Best compliant model for cross-script and cross-lingual pairing on the available evidence:** `intfloat/multilingual-e5-large-instruct` (MIT, 560M). On MMTEB its bitext-mining score is **80.13**. That is close to `Qwen3-Embedding-8B` (80.89) and above `BGE-M3` (79.11) and `Qwen3-Embedding-0.6B` (72.22), all from the table on the Qwen3-Embedding card. The model card's own Tatoeba F1 scores are hin 97.6, tam 90.6, tel 96.1, ben 89.1, fra 95.0. The MMTEB paper (ICLR 2025) also names it the best-performing publicly available model.
* **Cheap blocking and embedding tier:** `multilingual-e5-small` (MIT, 118M, only about 21M non-embedding parameters), `potion-multilingual-128M` (MIT, a static Model2Vec model distilled from bge-m3, runs on CPU), and `snowflake-arctic-embed-m-v2.0` (Apache-2.0, 113M non-embedding parameters, 256-d Matryoshka vectors).
* **Rerankers and cross-encoders to fine-tune:** `Alibaba-NLP/gte-multilingual-reranker-base` (Apache, 306M, 8192 ctx), `BAAI/bge-reranker-v2-m3` (Apache, 568M), `Qwen/Qwen3-Reranker-0.6B` (Apache), `mixedbread-ai/mxbai-rerank-base-v2` (Apache, 0.5B, Qwen2 architecture). Backbones for our own Ditto-style cross-encoder: XLM-R (MIT), mDeBERTa-v3 (MIT), MuRIL-base (Apache), IndicBERTv2 (MIT), CANINE-s (Apache).
* **Indic cross-script tooling:** `IndicXlit` (MIT, about 11M parameters, converts native script to Roman and Roman to native for 21 languages), `MuRIL-base` (Apache, pretrained on transliterated pairs), `IndicBERTv2-SS` (MIT, all scripts mapped to Devanagari), `IndicTrans2-dist-200M` (MIT; the HF repo is gated with automatic approval, so the user has to accept it).
* **Traps (a permissive license tag, but not compliant or risky):**
  * `BAAI/bge-reranker-v2-gemma` is tagged apache-2.0 but is built on gemma-2b.
  * `granite-embedding-*-multilingual-r2` has Apache weights, but the card says its tokenizer is "subject to the Gemma Terms of Use".
  * `jina-embeddings-v4` is under the Qwen Research License.
  * `Qwen2.5-3B-Instruct` is `qwen-research`, while the 0.5B, 1.5B and 7B sizes are Apache.
  * `Qwen3-8B`, `Qwen3-Reranker-8B` and `granite-3.3-8b-instruct` have more than 8.0B total parameters.
  * Jellyfish is CC-BY-NC.
  * `sarvam-1` has no license metadata and its card says "Sarvam non-commercial license".
  * `muril-large-cased` declares no license.
  * *(Added by audit)* `jhu-clsp/mmBERT-*` is tagged `mit` but ships the Gemma 2 tokenizer (card spec table). It is treated like Granite R2: **RISK**.
  * *(Added by audit)* The IndicXlit pip engine defaults to `rescore=True`, which applies a bundled word dictionary. Use `rescore=False`.
* **Compute reality check:** these are analytic estimates for about 11.7M test strings and 1.73M x 30 candidate pairs (section 6). A base encoder takes about 0.2 h on an A100 or 1.3 h on a T4. A large encoder takes about 0.8 h on an A100. Qwen3-Embedding-8B would take about 18 h on an A100, so 4B/8B models are only realistic as teachers or on a small ambiguous subset.

---

## 1. Compliance rules applied here

1. **The weights license** must be exactly `mit` or `apache-2.0` in the HF card metadata *and* must not be contradicted by the card text or the base-model lineage. Examples of contradictions: a Gemma, Llama or Qwen-research base; an explicit ToU clause on a component such as the tokenizer.
2. **The <= 8B limit counts total parameters** (`safetensors.total`, embeddings included). Anything above 8.00e9 is marked **RISK** even if the marketing name says "8B". The safe ceiling is about 7.6B.
3. **Gated repos** (`gated=auto/manual`) need the user to accept terms on HF. Claude must not click "accept". The user does this.
4. **External-data rule:** pretrained weights are allowed. Instruction datasets (e.g. Jellyfish-Instruct), gazetteers and geocoders are external data, so they are **not** allowed as training input.
5. **Library licenses are a separate question** (section 8). For example, `unidecode` is GPL-2.0, `anyascii` is ISC and `aksharamukha` is AGPL-3.0. *(Audit correction: `src/ber/normalize.py` already imports `anyascii` (ISC) as its optional fallback, aliased as `_unidecode`. It does not use the GPL `unidecode` package, and no file in the repo imports it.)* None of these is a model-license question, but GPL/AGPL code in a shipped pipeline deserves a conscious decision.

---

## 2. Master table: compliant candidates

Legend for throughput class, for short name+address strings of about 48 tokens:
* **CPU**: all ~12M strings are feasible on a 32-64 core box in hours.
* **T4**: fine on a T4.
* **L4/A10G**: comfortable on 24 GB cards.
* **A100**: use an A100 for the full corpus; otherwise use a subset.

Coverage lists the Indic languages and French found in HF `language` tags. When the tags are absent, the coverage comes from the card text.

### (a) Multilingual bi-encoders / sentence embeddings

| HF id | Params (total) | License (exact card string) | Indic + FR coverage | Max seq | Class | Usefulness for short noisy names/addresses | Verdict |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 117,654,272 | `mit` | hi ta te kn bn mr gu ml pa or fr (94 tags) | 512 | CPU/T4 | **High (blocking).** About 21M non-embedding parameters, so very fast. Card Tatoeba F1: hin 93.25, tam 81.4, tel 85.8, ben 77.8, mar 86.8, fra 89.9. Needs the `query: ` prefix. | COMPLIANT |
| intfloat/multilingual-e5-base | 278,044,162 | `mit` | same as small | 512 | T4 | High. Good middle tier for fine-tuning. | COMPLIANT |
| intfloat/multilingual-e5-large | 559,890,946 | `mit` | same | 512 | L4/A100 | High. | COMPLIANT |
| intfloat/multilingual-e5-large-instruct | 559,890,432 | `mit` | same | 512 | L4/A100 | **Highest bitext evidence among compliant <1B models.** MMTEB bitext 80.13 (Qwen3 card table). Card Tatoeba F1: hin 97.6, tam 90.6, tel 96.1, ben 89.1, mar 92.6, mal 98.9, fra 95.0; BUCC fr-en 99.1. | COMPLIANT |
| BAAI/bge-m3 | ~0.6B (API has no count; the Qwen3 card table says 0.6B; XLM-R-large backbone) | `mit` | "more than 100 working languages" (card; *re-audit: quote corrected*) | 8192 | L4/A100 | High. Dense, sparse (lexical) and ColBERT outputs from one model; the sparse head is useful for token-overlap features. MMTEB bitext 79.11. | COMPLIANT |
| Alibaba-NLP/gte-multilingual-base | 305,369,089 | `apache-2.0` | hi ta te kn bn mr gu ml pa fr (no `or`) | 8192 | T4 | High. Elastic dimension 128-768 plus sparse vectors. Needs `trust_remote_code` (code repo `Alibaba-NLP/new-impl`, `apache-2.0`). Card Tatoeba F1: hin 93.3, tam 84.3, tel 89.7, ben 73.2, fra 93.3; BUCC fr-en 97.9. | COMPLIANT |
| Snowflake/snowflake-arctic-embed-m-v2.0 | 305,368,320 | `apache-2.0` | hi ta te kn bn mr gu ml pa fr | 8192 | T4 | High for blocking. 113M non-embedding parameters (card). 256-d Matryoshka vectors, "128 bytes/vector" with int4. Built on gte-multilingual-base. | COMPLIANT |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 567,754,752 | `apache-2.0` | same | 8192 | L4/A100 | Medium-high. 303M non-embedding parameters; built on bge-m3-retromae. | COMPLIANT |
| sentence-transformers/LaBSE | 470,927,360 | `apache-2.0` | all incl. `or` (110 tags) | 256 | T4 | **High for cross-script names.** Trained specifically for translation-pair retrieval; paper reports 83.7% Tatoeba accuracy over 112 languages. Note: `google/LaBSE` returns HTTP 401 on the HF API, so use the ST port. | COMPLIANT |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 117,654,272 | `apache-2.0` | hi mr gu fr only (no ta te kn bn) | 128 | CPU | Low-medium. **No Dravidian or Bengali training languages.** | COMPLIANT (weak) |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 278,044,162 | `apache-2.0` | hi mr gu fr only | 128 | T4 | Low-medium, same language gap. | COMPLIANT (weak) |
| nomic-ai/nomic-embed-text-v2-moe | 475,292,928 (305M active) | `apache-2.0` | hi ta te kn bn mr gu ml pa fr | 512 | T4 | Medium-high. MoE, Matryoshka 768 to 256. | COMPLIANT |
| Qwen/Qwen3-Embedding-0.6B | 595,776,512 | `apache-2.0` | "100+ languages" (card) | 32K | L4/A100 | Medium. MMTEB mean 64.33 but bitext only 72.22, **below mE5-large-instruct**. Instruction-aware; MRL 32-1024. | COMPLIANT |
| Qwen/Qwen3-Embedding-4B | 4,021,774,336 | `apache-2.0` | 100+ | 32K | A100 (subset) | Teacher / distillation use. MMTEB 69.45, bitext 79.36. | COMPLIANT |
| Qwen/Qwen3-Embedding-8B | 7,567,295,488 | `apache-2.0` | 100+ | 32K | A100 (subset) | Teacher only. MMTEB 70.58, bitext 80.89. Under 8.0B total. | COMPLIANT |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | 1,776,197,120 | `apache-2.0` | (no tags) | long | A100 | Low. MMTEB bitext 62.51 (Qwen3 card table). | COMPLIANT (dominated) |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 7,612,620,288 | `apache-2.0` | (no tags) | long | A100 | Low. Bitext 73.92, below the 560M mE5-large-instruct. | COMPLIANT (dominated) |
| intfloat/e5-mistral-7b-instruct | 7,110,660,096 | `mit` | tag: en | long | A100 | Low (English-tagged). | COMPLIANT (dominated) |
| minishlab/potion-multilingual-128M | 128,090,368 | `mit` | hi ta te kn bn mr gu ml pa fr (101 langs) | unlimited (static) | **CPU** | Medium as a fast candidate generator / extra feature. Distilled from BAAI/bge-m3 (MIT, compliant lineage); card says it reaches "90.86% of the performance of LaBSE". | COMPLIANT |
| sentence-transformers/static-similarity-mrl-multilingual-v1 | (static; API has no count) | `apache-2.0` | 50 langs incl. hi mr gu fr; **no ta te kn bn** | unlimited | **CPU** | Low-medium. Card: "100x to 400x faster" on CPU than mE5-small. | COMPLIANT (weak) |
| ibm-granite/granite-embedding-278m-multilingual | 278,043,648 | `apache-2.0` | 12 langs (fr; **no Indic**) | 512 | T4 | Low for India. | COMPLIANT (weak) |
| BAAI/bge-small / base / large-en-v1.5 | 33.4M / 109.5M / 335.1M | `mit` | en | 512 | CPU/T4 | Low (English only). Possible US-only side model. | COMPLIANT (not multilingual) |
| mixedbread-ai/mxbai-embed-large-v1; nomic-embed-text-v1.5; Alibaba-NLP/gte-modernbert-base; ibm-granite/granite-embedding-english-r2 | 335M; 137M; 149M; 149M | `apache-2.0` (all) | en | 512-8192 | T4 | Low (English only). | COMPLIANT (not multilingual) |

### (b) Cross-encoders / rerankers

| HF id | Params | License | Coverage | Max seq | Class | Usefulness | Verdict |
|---|---|---|---|---|---|---|---|
| Alibaba-NLP/gte-multilingual-reranker-base | 305,959,681 | `apache-2.0` | hi ta te kn bn mr gu ml pa fr | 8192 | T4/L4 | **High.** Fastest multilingual reranker; MMTEB-R 59.44 (Qwen3 card, their run). Fine-tune on our pairs. | COMPLIANT |
| BAAI/bge-reranker-v2-m3 | 567,755,777 | `apache-2.0` | multilingual (base bge-m3) | 8194 pos (card examples use 512) | L4/A100 | High. MMTEB-R 58.36. | COMPLIANT |
| Qwen/Qwen3-Reranker-0.6B | 595,776,512 | `apache-2.0` | 100+ | 32K | L4/A100 | High. MMTEB-R 66.36, the best of the <1B rerankers in the Qwen table. A seq-cls conversion exists: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` (`apache-2.0`, 595,777,536). | COMPLIANT |
| Qwen/Qwen3-Reranker-4B | 4,021,784,576 | `apache-2.0` | 100+ | 32K | A100 (subset) | Teacher / ambiguous band. MMTEB-R 72.74. | COMPLIANT |
| mixedbread-ai/mxbai-rerank-base-v2 | 494,032,768 | `apache-2.0` | 109 tags incl. all Indic + fr | 32768 pos | L4 | Medium-high. Qwen2ForCausalLM architecture (config.json). Tech report is ProRank, arXiv 2506.03487. | COMPLIANT |
| mixedbread-ai/mxbai-rerank-large-v2 | 1,543,714,304 | `apache-2.0` | same | 32768 | A100 | Medium. | COMPLIANT |
| zeroentropy/zerank-2-reranker | 4,022,468,096 | `apache-2.0` (base Qwen/Qwen3-4B) | tag: en | long | A100 | Low-medium. | COMPLIANT |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 117,641,603 | `apache-2.0` | 15 langs incl. hi fr; no Dravidian | 512 | CPU/T4 | Low-medium. | COMPLIANT (weak) |
| mixedbread-ai/mxbai-rerank-xsmall/base/large-v1; Alibaba-NLP/gte-reranker-modernbert-base; ibm-granite/granite-embedding-reranker-english-r2 | 71M/184M/435M; 150M; 150M | `apache-2.0` | en | 512-8192 | CPU/T4 | Low (English only). | COMPLIANT (not multilingual) |

### (c) Small generative LLMs (pairwise matching, adjudication, distillation teachers)

| HF id | Params (total) | License | Languages | Class | Usefulness | Verdict |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 494,032,768 | `apache-2.0` | 29+ languages (card) | T4 | Low-medium. Cheap LoRA matcher (AnyMatch-style). | COMPLIANT |
| Qwen/Qwen2.5-1.5B-Instruct | 1,543,714,304 | `apache-2.0` | 29+ | T4/L4 | Medium. | COMPLIANT |
| **Qwen/Qwen2.5-3B-Instruct** | 3,085,938,688 | **`other` / `qwen-research`** | 29+ | - | - | **NOT COMPLIANT** |
| Qwen/Qwen2.5-7B-Instruct | 7,615,616,512 | `apache-2.0` | 29+ | A100 | Medium-high as adjudicator/teacher. Safely under 8B. | COMPLIANT |
| Qwen/Qwen3-0.6B | 751,632,384 | `apache-2.0` | 119 langs (Qwen3 tech report) | T4 | Medium. | COMPLIANT |
| Qwen/Qwen3-1.7B | 2,031,739,904 | `apache-2.0` | 119 | T4/L4 | Medium. | COMPLIANT |
| Qwen/Qwen3-4B / Qwen3-4B-Instruct-2507 | 4,022,468,096 | `apache-2.0` | 119 | L4/A100 | **High (best size/quality for LoRA pairwise matching).** Multilingual incl. Indic: the Qwen3 release blog's 119-language list names Hindi, Bengali, Marathi, Gujarati, Punjabi, Oriya, Tamil, Telugu, Kannada, Malayalam and French (*re-audit: source made specific; the tech-report abstract gives only the count*). | COMPLIANT |
| Qwen/Qwen3-8B | **8,190,735,360** (card: 8.2B, 6.95B non-embedding) | `apache-2.0` | 119 | A100 | High quality but **above 8.0B total**. | **RISK (param cap)** |
| mistralai/Mistral-7B-Instruct-v0.3 | 7,248,023,552 | `apache-2.0` | 32,768 vocab; weak Indic | A100 | Low-medium. | COMPLIANT |
| microsoft/Phi-3.5-mini-instruct | 3,821,079,552 | `mit` | multilingual (no Indic listed) | L4 | Low-medium. | COMPLIANT |
| microsoft/Phi-4-mini-instruct | 3,836,021,760 | `mit` | 23 langs incl. fr; **no Indic** in tags; 200,064 vocab | L4 | Low-medium. | COMPLIANT |
| HuggingFaceTB/SmolLM3-3B | 3,075,098,624 | `apache-2.0` | 6 languages (fr yes, Indic no) | L4 | Low. | COMPLIANT |
| ibm-granite/granite-3.3-2b-instruct | 2,533,539,840 | `apache-2.0` | - | L4 | Low-medium. | COMPLIANT |
| ibm-granite/granite-3.3-8b-instruct | **8,170,864,640** | `apache-2.0` | - | A100 | - | **RISK (param cap)** |
| allenai/OLMo-2-1124-7B-Instruct | 7,298,617,344 | `apache-2.0` | en | A100 | Low. | COMPLIANT |
| utter-project/EuroLLM-1.7B-Instruct | 1,656,850,432 | `apache-2.0` | 35 langs incl. hi fr | T4 | Low. | COMPLIANT |

### (d) Character / byte-level encoders

| HF id | Params | License | Coverage | Max input | Class | Usefulness | Verdict |
|---|---|---|---|---|---|---|---|
| google/canine-s | 132,099,328 | `apache-2.0` | 104 langs (mBERT wiki) incl. Indic, fr | 16,384 char positions (config), 4x downsampling | T4 | Medium. Tokenizer-free, so robust to typos ("Hospirlg"), injected diacritics and junk prefixes. A candidate cross-encoder backbone for name-vs-name. | COMPLIANT |
| google/canine-c | 132,099,328 | `apache-2.0` | same | same | T4 | Medium (character-loss variant). | COMPLIANT |
| google/byt5-small | ~300M (1.199 GB fp32 weights) | `apache-2.0` | mC4 (101 languages per the mT5 paper; *re-audit: was "102"*) | bytes (Devanagari is 3 bytes/char, so sequences get long) | T4/L4 | Low-medium. Encoder usable as a byte-level cross-encoder. ByT5 paper: "significantly more robust to noise". | COMPLIANT |
| google/byt5-base, google/mt5-small/base | - | `apache-2.0` | mC4 | - | L4 | Low. | COMPLIANT |

### (e) Indic-specific encoders, transliteration and translation

| HF id | Params | License | Coverage | Usefulness | Verdict |
|---|---|---|---|---|---|
| google/muril-base-cased | ~238M (953 MB fp32) | `apache-2.0` | 17 Indian languages **and their transliterated counterparts** (card) | **High for a Latin-vs-native cross-encoder.** Pretraining included translation and transliteration segment pairs. Vocab 197,285; 512 max. | COMPLIANT |
| google/muril-large-cased | ~507M | **none declared** (no `license` field in metadata or card) | 17 | - | **UNVERIFIED -> avoid** |
| ai4bharat/IndicBERTv2-MLM-only | 278M (card) | `mit` | 23 Indic + en (hi ta te kn bn mr gu ml pa or...) | Medium-high backbone. Vocab 250,000; 512. | COMPLIANT |
| ai4bharat/IndicBERTv2-SS | ~278M | `mit` | 23 Indic + en | Medium-high. "Script-standardised": Indic scripts converted to Devanagari before MLM, which matches our cross-script problem. | COMPLIANT |
| ai4bharat/indic-bert (v1, ALBERT) | small | `mit` (gated=auto) | 12 | Low. | COMPLIANT (gated) |
| ai4bharat/IndicXlit | ~11M (GitHub README) | HF `mit`; GitHub README: "code (and models) are released under the MIT License"; PyPI `ai4bharat-transliteration` 1.1.3: `MIT` | 21 Indic langs, **Roman -> native and native -> Roman** | **High.** A neural alternative/complement to our rule-based `translit_indic`, producing natural romanizations and top-k candidates. *Audit correction:* the HF repo **does** host the weights: `indicxlit-en-indic-v1.0/transformer/indicxlit.pt` and `indicxlit-indic-en-v1.0/transformer/indicxlit.pt`, plus fairseq `dict.*.txt` vocab files. The GitHub v1.0 release zips are a second source. **Audit note:** in the pip package, `XlitEngine(..., rescore=True)` is the default, and it "returns the reranked suggestions after using a dictionary" (PyPI README). That dictionary is bundled external data, and the PyPI page does not document where it comes from. Use `rescore=False` so that only the model weights are used; the PyPI native-to-Roman example already does this. | COMPLIANT (use `rescore=False`) |
| ai4bharat/indictrans2-indic-en-dist-200M | 228,316,160 | `mit` (**gated=auto**) | 22 scheduled Indian langs -> en | Medium-low. Only for truly *translated* names (not transliterated ones). The user must accept the HF gate. | COMPLIANT (gated) |
| ai4bharat/indictrans2-en-indic-dist-200M / indic-en-1B | 274,584,576 / 1,023,006,720 | `mit` (gated=auto) | same | Low. | COMPLIANT (gated) |
| ai4bharat/IndicBART | - | metadata **missing**; card text: "available under the MIT License" | 11 Indic (all in Devanagari) + en | Low. | COMPLIANT per card text (metadata gap noted) |
| facebook/mbart-large-50 | - | `mit` | 50 langs incl. hi ta te bn mr gu ml fr | Low. | COMPLIANT |

### (f) General multilingual encoder backbones (for our own fine-tuned bi- or cross-encoders)

| HF id | Params | License | Notes | Verdict |
|---|---|---|---|---|
| FacebookAI/xlm-roberta-base | 278,885,778 | `mit` | 100 langs incl. all target Indic + fr; the backbone of mE5, bge-m3 and LaBSE-era work. | COMPLIANT |
| FacebookAI/xlm-roberta-large | 561,192,082 | `mit` | | COMPLIANT |
| microsoft/mdeberta-v3-base | 86M backbone + 190M embedding (card) | `mit` | CC100 like XLM-R; strong cross-encoder backbone. | COMPLIANT |
| jhu-clsp/mmBERT-base / mmBERT-small | 307M (110M non-emb) / 140M (42M non-emb) (card) | `mit` | 1800+ langs, 8192 ctx, ModernBERT architecture (fast). **Tokenizer is Gemma 2** (card spec table: "Tokenizer: Gemma 2"). *Audit change:* this model ships the Gemma 2 tokenizer itself, while Granite R2 ships a tokenizer *derived* from Gemma 3, so the Gemma-ToU exposure here is at least as large as for Granite R2 (flagged RISK in section 3). The card's silence on the ToU does not remove that exposure. It is treated the same way as Granite R2. | **RISK (Gemma tokenizer)** |
| Alibaba-NLP/gte-multilingual-mlm-base | 306,210,496 | `apache-2.0` | MLM base of mGTE (8192 ctx). | COMPLIANT |
| EuroBERT/EuroBERT-210m / 610m | 310M / 756M (API) | `apache-2.0` | 15 langs incl. hi, fr; no Dravidian. | COMPLIANT (partial coverage) |
| answerdotai/ModernBERT-base / large | 149.7M / 395.9M | `apache-2.0` | **English only**, not useful for Indic. | COMPLIANT (not multilingual) |

---

## 3. Non-compliant or risky models (do NOT ship)

| HF id | What the card/API says | Why excluded |
|---|---|---|
| jinaai/jina-embeddings-v3 | `cc-by-nc-4.0` | Non-commercial license |
| jinaai/jina-reranker-v2-base-multilingual; jinaai/jina-reranker-v3; jinaai/jina-colbert-v2 | `cc-by-nc-4.0` | Non-commercial |
| jinaai/jina-embeddings-v4 | no license field; card: "The correct license is the Qwen Research License" | Qwen research license |
| BAAI/bge-reranker-v2-gemma | tagged `apache-2.0`, **base model gemma-2b** (card table) | Gemma ToU flows to derivatives. **Tag is misleading.** |
| BAAI/bge-multilingual-gemma2 | `gemma`; 9,241,713,152 params | Gemma license and above 8B |
| google/embeddinggemma-300m, google/gemma-2-2b-it, google/gemma-3-1b-it | `gemma` (gated=manual) | Gemma terms |
| **ibm-granite/granite-embedding-311m-multilingual-r2** (and derived 97m-r2) | tagged `apache-2.0`; 311m card: tokenizer "derived from the **Gemma 3** tokenizer ... Use of the Gemma tokenizer is subject to the Gemma Terms of Use" | **RISK.** Apache weights, but a Gemma-ToU component. *Audit precision:* the 97m card has **no** Gemma clause. It describes "layer pruning (22 → 12 layers) and vocabulary selection (262K → 180K tokens)" from the 311m, so its vocabulary is a subset of the Gemma-3-derived one and the lineage risk carries over. Avoid unless organizers confirm. |
| jhu-clsp/mmBERT-base / mmBERT-small | tagged `mit`; card spec table: "Tokenizer: Gemma 2" | **RISK (added by audit).** Same class of issue as Granite R2: it ships the Gemma 2 tokenizer. |
| meta-llama/Llama-3.2-1B-Instruct; Llama-3.1-8B-Instruct | `llama3.2`; `llama3.1` | Llama licenses |
| mistralai/Ministral-8B-Instruct-2410 | `other` / `mrl` | Mistral Research License |
| **Qwen/Qwen2.5-3B-Instruct** | `other` / `qwen-research` | Research-only (the other Qwen2.5 sizes are Apache) |
| Qwen/Qwen3-8B; Qwen/Qwen3-Reranker-8B; ibm-granite/granite-3.3-8b-instruct | `apache-2.0` but 8,190,735,360 / 8,188,548,096 / 8,170,864,640 params | **RISK: above 8.0B total parameters** |
| NECOUDBFM/Jellyfish-7B / -8B / -13B | `cc-by-nc-4.0`; 8B is fine-tuned from Meta-Llama-3-8B-Instruct, 7B from Mistral-7B-Instruct-v0.2 | Non-commercial (+ Llama). Also, the Jellyfish-Instruct data counts as external data. |
| NousResearch/Jellyfish | HF API: HTTP 401 (no such public repo) | The real Jellyfish repos are under NECOUDBFM |
| sarvamai/sarvam-1 | no license metadata; card: "Sarvam non-commercial license"; the repo's `LICENSE.md` is titled "Sarvam AI Research License" (re-audit) | Non-commercial |
| sarvamai/sarvam-2b-v0.5 (re-audit: this id now redirects to `sarvamai/sarvam-1-v0.5`) | `other` | Not MIT/Apache |
| sarvamai/sarvam-m | `apache-2.0` but 23,572,403,200 params (Mistral-Small-24B base) | Too large |
| sarvamai/sarvam-translate | `gpl-3.0`; base google/gemma-3-4b-it | GPL + Gemma |
| krutrim-ai-labs/Krutrim-2-instruct | `other` / `krutrim-community-license-agreement-version-1.0` | Not MIT/Apache |
| CohereLabs/aya-expanse-8b | `cc-by-nc-4.0` | Non-commercial |
| facebook/nllb-200-distilled-600M | `cc-by-nc-4.0` | Non-commercial |
| l3cube-pune/indic-sentence-similarity-sbert; hindi-sentence-similarity-sbert | `cc-by-4.0` | Permissive, but **not MIT/Apache** under the challenge rule |
| Linq-Embed-Mistral; Salesforce/SFR-Embedding-Mistral; nvidia/NV-Embed-v2 | `cc-by-nc-4.0` | Non-commercial |
| nvidia/Nemotron-3-Embed-1B-BF16; nvidia/llama-nemotron-rerank-1b-v2 | `other` / `openmdw-1.1` | Not MIT/Apache |
| google/muril-large-cased | no license declared | Unverifiable |
| tencent/Hunyuan-MT-7B | no license metadata; 8,030,269,440 params | Unverifiable and above 8B |
| jxm/cde-small-v2 | no license metadata | Unverifiable |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | `apache-2.0`, 30.5B total | Too large (the cap is total params, not active params) |

---

## 4. Quality evidence (only numbers actually read on the source pages)

### 4.1 MMTEB (multilingual) bitext mining and mean, from the table on the Qwen3-Embedding card

*Compared-model scores taken by Qwen from the MTEB leaderboard on 2025-05-24. Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B*

| Model | Size | Mean (Task) | **Bitext Mining** | STS | Retrieval |
|---|---|---|---|---|---|
| multilingual-e5-large-instruct | 0.6B | 63.22 | **80.13** | 76.81 | 57.12 |
| BGE-M3 | 0.6B | 59.56 | 79.11 | 74.12 | 54.60 |
| gte-Qwen2-1.5B-instruct | 1.5B | 59.45 | 62.51 | 71.61 | 60.78 |
| gte-Qwen2-7B-instruct | 7B | 62.51 | 73.92 | 73.98 | 60.08 |
| Qwen3-Embedding-0.6B | 0.6B | 64.33 | 72.22 | 76.17 | 64.64 |
| Qwen3-Embedding-4B | 4B | 69.45 | 79.36 | 80.86 | 69.60 |
| Qwen3-Embedding-8B | 8B | 70.58 | 80.89 | 81.08 | 70.88 |

**Read-across:** bitext mining (does text X in language A pair with its equivalent in language B?) is the closest public proxy to "does a Devanagari/Tamil business name pair with its Latin version?". On that axis a 560M MIT model matches a 7.6B model. The Qwen3 models win on retrieval, which is less relevant to short-record matching.

### 4.2 Tatoeba / BUCC F1, parsed from each model card's MTEB `model-index` metadata

| Model | hin-eng | tam-eng | tel-eng | ben-eng | mar-eng | mal-eng | urd-eng | fra-eng | BUCC fr-en |
|---|---|---|---|---|---|---|---|---|---|
| multilingual-e5-small | 93.25 | 81.4 | 85.8 | 77.8 | 86.8 | 94.8 | 84.7 | 89.9 | 92.7 |
| multilingual-e5-large-instruct | 97.6 | 90.6 | 96.1 | 89.1 | 92.6 | 98.9 | 93.4 | 95.0 | 99.1 |
| gte-multilingual-base | 93.3 | 84.3 | 89.7 | 73.2 | 84.4 | 94.6 | 83.9 | 93.3 | 97.9 |

(No Kannada pair is present in these cards' Tatoeba metadata. The LaBSE paper reports 83.7% average Tatoeba accuracy over 112 languages.)

**Caveat:** Tatoeba is *sentence translation*. Our cross-script pairs are mostly *transliterations of proper nouns plus loanwords* ("राम मार्केटिंग प्राइवेट लिमिटेड" = "Ram Marketing Private Limited"). Rankings must be re-measured on our own train pairs (section 9, P0-2).

### 4.3 Reranker comparison (Qwen3-Reranker card, "our runs", top-100 from Qwen3-Embedding-0.6B)

| Model | Param | MMTEB-R | MTEB-R (en) | MLDR |
|---|---|---|---|---|
| Jina-multilingual-reranker-v2-base (non-compliant) | 0.3B | 63.73 | 58.22 | 39.66 |
| gte-multilingual-reranker-base | 0.3B | 59.44 | 59.51 | 66.33 |
| BGE-reranker-v2-m3 | 0.6B | 58.36 | 57.03 | 59.51 |
| Qwen3-Reranker-0.6B | 0.6B | 66.36 | 65.80 | 67.28 |
| Qwen3-Reranker-4B | 4B | 72.74 | 69.76 | 69.97 |
| Qwen3-Reranker-8B (RISK >8B) | 8B | 72.94 | 69.02 | 70.19 |

These are passage-retrieval scores. For entity matching, **fine-tuning on our 2.2M-entity train set will dominate** zero-shot differences (Ditto, Peeters et al., Zhang et al. EDBT 2025).

### 4.4 Other card-reported facts

* Arctic-embed-m-v2.0: 113M non-embedding parameters; "MRL for this model is 256 dimensions, and high-quality 128-byte compression is achieved via 4-bit quantization". The l-v2.0 has 303M non-embedding parameters.
* nomic-embed-text-v2-moe: 475M total, 305M active, 512 max tokens, Matryoshka 768 to 256.
* potion-multilingual-128M: static, 256-d, "reaching 90.86% of the performance of LaBSE with a mean task score of 47.31".
* static-similarity-mrl-multilingual-v1: "On CPU, this model is 100x to 400x faster than ... multilingual-e5-small".
* mmBERT: 3T+ tokens, 1800+ languages, 8192 ctx, Gemma 2 tokenizer (256k vocab).
* Granite-311M-multilingual-R2 (released 2026-04-29): multilingual MTEB Retrieval 65.2; 32,768 ctx; Gemma-3-derived tokenizer (**risk**).
* ByT5 paper: byte-level models "are significantly more robust to noise". CANINE: beats mBERT by 2.8 F1 on TyDi QA with 28% fewer parameters.
* Aksharantar/IndicXlit: 26M transliteration pairs, 21 languages, 12 scripts; IndicXlit improves Dakshina accuracy by 15%.
* RomanSetu (ACL 2024): embeddings of romanized text "exhibit closer alignment with their English translations than those from the native script". This supports transliterating to Latin *before* embedding as one of the views.
* "One Script Instead of Hundreds?" (arXiv 2601.05776): negligible performance loss from romanization for segmental scripts (Indic scripts are segmental/abugida). Degradation is limited to morphosyllabic scripts such as Chinese and Japanese.

---

## 5. How each model family maps to THIS challenge

* **Scale (1.73M queries x ~10M targets).** Dense retrieval needs a small, fast encoder plus FAISS. Budget the index memory:
  * 10M x 1024-d fp32 is about 41 GB;
  * 768-d fp16 is about 15.4 GB;
  * 384-d fp16 is about 7.7 GB;
  * 256-d fp16 (Matryoshka) is about 5.1 GB.

  Partition the search by coarse keys (country/state/city/postcode, derived from the text itself, open set) so each FAISS search is local. mE5-small, arctic-m-v2 (256-d MRL), gte-multilingual-base (elastic 128-768) and potion (CPU) are the practical blocking encoders. Large models (mE5-large-instruct, bge-m3, LaBSE) are feasible once on an A100 in under 1 h (section 6).
* **Cross-script Indic names.** Use three complementary views:
  1. **Raw native text into a multilingual encoder.** LaBSE and mE5-large-instruct have the best bitext evidence.
  2. **Transliterated text** (our rule-based `translit_indic` and/or IndicXlit native to Roman, top-k) into char n-gram TF-IDF plus the Latin-heavy features.
  3. **A cross-encoder over a mixed-script pair.** Backbones: MuRIL (trained with transliterated pairs), IndicBERTv2-SS (script-standardised), XLM-R/mDeBERTa, CANINE (char-level).

  Mixed-script names ("Sun पावर Provision") are handled natively by all subword multilingual models and by CANINE/ByT5.
* **Unseen French.** Every recommended multilingual model covers French (BUCC fr-en 92.7 for mE5-small, 99.1 for mE5-large-instruct, 97.9 for gte-mb). Fine-tuning only on US+India data risks over-specialising. Mitigations:
  * keep the multilingual base;
  * use a moderate learning rate / few epochs, or LoRA;
  * create "French-style" noise ourselves (R./Rue, Av./Avenue, Bd/Boulevard, SARL/SAS/EURL moved around) from our own training records. This is self-generated augmentation, not external data, but get it confirmed against the rule text;
  * validate on that synthetic slice.
* **Per-entity macro F0.5 with singletons.** Precision counts double and one FP costs 4x one FN. What matters is a **calibrated** pair probability plus per-S1 cardinality decisions, including deciding to return the empty set. Cross-encoder logits, and Qwen3-Reranker's P("yes"), should be calibrated (isotonic/Platt on out-of-fold data) and used as LightGBM features, not thresholded raw.
* **Multi-copy S2/S3.** Once the matcher links S2/S3 copies to each other, transitive/cluster consistency among candidates is a strong precision booster (records that match each other should co-occur). Embeddings help cluster S2/S3 copies before assigning them to S1.

---

## 6. Compute estimates (analytic, not measured; +/-3x)

Assumptions:
* FLOPs = 2 x non-embedding params x tokens.
* About 48 tokens per record; 11.73M test strings (S1+S2+S3).
* Candidate pairs = 1.73M x 30 at 96 tokens.
* Effective throughput: T4 20, L4/A10G 45, A100 120 TFLOP/s.

| Job | T4 | L4/A10G | A100 |
|---|---|---|---|
| Embed all strings: mE5-small (~21M non-emb) | 0.3 h | 0.1 h | 0.1 h |
| Embed: base XLM-R class (~86M) | 1.3 h | 0.6 h | 0.2 h |
| Embed: arctic-m-v2 / gte-mb (~113M) | 1.8 h | 0.8 h | 0.3 h |
| Embed: large (mE5-large-instruct / bge-m3 / arctic-l, ~303M) | 4.7 h | 2.1 h | 0.8 h |
| Embed: Qwen3-Embedding-0.6B (~0.44B) | 6.9 h | 3.1 h | 1.1 h |
| Embed: Qwen3-Embedding-4B | 57 h | 25 h | 9.5 h |
| Embed: Qwen3-Embedding-8B | 109 h | 48 h | 18 h |
| Rerank 52M pairs: base cross-encoder (~86M) | 12 h | 5.3 h | 2.0 h |
| Rerank: gte-multilingual-reranker-base | 16 h | 7.0 h | 2.6 h |
| Rerank: bge-reranker-v2-m3 | 42 h | 19 h | 7.0 h |
| Rerank: Qwen3-Reranker-0.6B | 61 h | 27 h | 10 h |
| Rerank: Qwen3-Reranker-4B | 502 h | 223 h | 84 h |
| 7B LLM pointwise on 1M ambiguous pairs (150 tokens) | 27 h | 12 h | 4.5 h |

Conclusions:
* 4B/8B embedders and rerankers do **not** fit as full-corpus stages.
* Use them only (i) on the uncertain band (a few % of pairs), or (ii) as teachers on sampled train pairs to distill into a base-size cross-encoder.
* Cut K adaptively: K=30 is a ceiling, and most S1 have 2-6 true matches.

---

## 7. Papers (verified)

| # | Title | Authors | Venue / year | Verified via | Relevance / adaptation |
|---|---|---|---|---|---|
| 1 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Reimers, Gurevych | EMNLP 2019 | https://aclanthology.org/D19-1410/ | The bi-encoder recipe (MNRL fine-tuning) for our blocking encoder |
| 2 | Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation | Reimers, Gurevych | EMNLP 2020 | https://aclanthology.org/2020.emnlp-main.365/ | Teacher-student alignment; basis of the paraphrase-multilingual-* models. Could align transliterated and native views |
| 3 | Unsupervised Cross-lingual Representation Learning at Scale (XLM-R) | Conneau et al. | ACL 2020 | https://aclanthology.org/2020.acl-main.747/ | MIT backbone behind mE5, bge-m3 |
| 4 | Language-agnostic BERT Sentence Embedding (LaBSE) | Feng, Yang, Cer, Arivazhagan, Wang | ACL 2022 | https://aclanthology.org/2022.acl-long.62/ | 83.7% Tatoeba over 112 langs; translation-pair objective suits cross-script names |
| 5 | Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5) | Wang et al. | arXiv 2212.03533 (2022) | arXiv API | Recipe behind mE5 |
| 6 | Multilingual E5 Text Embeddings: A Technical Report | Wang et al. | arXiv 2402.05672 (2024) | arXiv API | mE5 small/base/large/instruct; 1B multilingual pairs pretraining |
| 7 | M3-Embedding (BGE-M3) | Chen, Xiao, Zhang, Luo, Lian, Liu | Findings of ACL 2024 | https://aclanthology.org/2024.findings-acl.137/ | Dense + sparse + multi-vector from one model; the sparse head gives lexical weights |
| 8 | C-Pack: Packed Resources For General Chinese Embeddings (BGE) | Xiao et al. | SIGIR 2024 (arXiv comment) | arXiv API 2309.07597 | BGE training recipe/hard negatives |
| 9 | mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval | Zhang et al. | EMNLP 2024 Industry | https://aclanthology.org/2024.emnlp-industry.103/ | gte-multilingual-base + reranker; "match ... BGE-M3" at base size |
| 10 | Arctic-Embed 2.0: Multilingual Retrieval Without Compromise | Yu, Merrick, Nuti, Campos | arXiv 2412.04506 (2024) | arXiv API | Matryoshka + quantization-aware for a compact 10M-vector index |
| 11 | Training Sparse Mixture Of Experts Text Embedding Models (Nomic Embed v2) | Nussbaum, Duderstadt | arXiv 2502.07972 (2025) | arXiv API | MoE embedder, 305M active |
| 12 | Granite Embedding Models | Awasthy et al. | arXiv 2502.20204 (2025) | arXiv API | R1 multilingual (12 langs, no Indic). The R2 models carry the Gemma-tokenizer risk |
| 13 | Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | Zhang et al. | arXiv 2506.05176 (2025) | arXiv API | 0.6/4/8B embed + rerank; teachers |
| 14 | Qwen3 Technical Report | Yang et al. | arXiv 2505.09388 (2025) | arXiv API | "expands multilingual support from 29 to 119 languages" |
| 15 | ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking (mxbai-rerank-v2 report) | Li, Shakir, Huang, Lipp, Clavié, Li (Anthology list) | Findings of ACL 2026 | https://aclanthology.org/2026.findings-acl.51/ (audit-fetched) + arXiv API 2506.03487 | 0.5B SLM reranker recipe |
| 16 | MTEB: Massive Text Embedding Benchmark | Muennighoff, Tazi, Magne, Reimers | EACL 2023 | https://aclanthology.org/2023.eacl-main.148/ | Benchmark definitions (BitextMining, STS) |
| 17 | MMTEB: Massive Multilingual Text Embedding Benchmark | Enevoldsen et al. | ICLR 2025 (poster) | arXiv API 2502.13595 (comment: "Accepted for ICLR" + OpenReview `zl3pfz4VCV`) + https://iclr.cc/virtual/2025/poster/27651 (page title "ICLR Poster MMTEB ...") + https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html (re-audit fetched; OpenReview itself now returns a browser check) | "the best-performing publicly available model is multilingual-e5-large-instruct with only 560 million parameters" |
| 18 | Matryoshka Representation Learning | Kusupati et al. | NeurIPS 2022 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html | Truncatable dims mean a cheaper FAISS index |
| 19 | Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT) | Warner, Chaffin, Clavié, Weller et al. | ACL 2025 (Long) (*audit fix: title and venue corrected*) | https://aclanthology.org/2025.acl-long.127/ (audit-fetched) + arXiv 2412.13663 | Architecture of mmBERT/Granite-R2 (fast unpadded encoders) |
| 20 | mmBERT: A Modern Multilingual Encoder with Annealed Language Learning | Marone, Weller et al. | arXiv 2509.06888 (2025) | arXiv API | 1800+ langs MIT encoder backbone. The Gemma 2 tokenizer puts it in the RISK class (audit). |
| 21 | DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing | He, Gao, Chen | ICLR 2023 (poster) | arXiv API 2111.09543 + ICLR 2023 virtual site (paper listed at https://iclr.cc/virtual/2023/poster/11295) + https://mlanthology.org/iclr/2023/he2023iclr-debertav3/ (re-audit fetched; OpenReview forum `sE7-XhLxHA` now returns a browser check) | mDeBERTa-v3 cross-encoder backbone; abstract: mDeBERTa Base gets 79.8% zero-shot XNLI, +3.6% over XLM-R Base |
| 22 | ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models | Xue et al. | TACL 2022 | https://aclanthology.org/2022.tacl-1.17/ | Noise/typo robustness |
| 23 | CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation (*re-audit: full title restored*) | Clark, Garrette, Turc, Wieting | TACL 2022 | https://aclanthology.org/2022.tacl-1.5/ | Char-level encoder for noisy names |
| 24 | MuRIL: Multilingual Representations for Indian Languages | Khanuja et al. | arXiv 2103.10730 (2021) | arXiv API | Trained with transliterated pairs, so it fits Latin-vs-native |
| 25 | Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages (IndicBERTv2 / IndicXTREME; *re-audit: full title restored*) | Doddapaneni et al. | ACL 2023 | https://aclanthology.org/2023.acl-long.693/ | IndicBERTv2 incl. the Script-Standardised variant |
| 26 | Aksharantar: Open Indic-language Transliteration datasets and models for the Next Billion Users (IndicXlit; *re-audit: full title restored*) | Madhani et al. | Findings of EMNLP 2023 | https://aclanthology.org/2023.findings-emnlp.4/ | IndicXlit native<->Roman transliteration |
| 27 | IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages | Gala et al. | TMLR (12/2023) | arXiv API 2305.16307 (comment "Accepted at TMLR") + https://mlanthology.org/tmlr/2023/gala2023tmlr-indictrans2/ (re-audit fetched: "Transactions on Machine Learning Research"; OpenReview `vfT4YuzAYA` now returns a browser check) | Translation of truly translated names (rare) |
| 28 | RomanSetu: Efficiently unlocking multilingual capabilities of Large Language Models via Romanization | Husain (listed as "Jaavid J" on the Anthology), Dabre et al. | ACL 2024 (Long) | https://aclanthology.org/2024.acl-long.833/ (audit-fetched) + arXiv API 2401.14280 | Romanized embeddings align better with English, supporting a transliterate-then-embed view |
| 29 | One Script Instead of Hundreds? On Pretraining Romanized Encoder Language Models | Ebing, Keller, Glavas | arXiv 2601.05776 (2026) | arXiv API | Romanization is nearly lossless for segmental scripts |
| 30 | Deep Entity Matching with Pre-Trained Language Models (Ditto) | Li, Li, Suhara, Doan, Tan | PVLDB 14(1), 2021 | https://www.vldb.org/pvldb/vol14/p50-li.pdf (fetched) | Cross-encoder EM; "up to 29%" F1 gain; domain-knowledge injection and summarization |
| 31 | Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation | Wang, Li, Wang | ICDE 2023 | arXiv API 2207.04122 + Crossref DOI 10.1109/ICDE55515.2023.00391 ("2023 IEEE 39th International Conference on Data Engineering (ICDE)"; audit-confirmed; the earlier IEEE Xplore doc number 10184556 was not confirmed and has been replaced) | Contrastive pretraining on unlabeled records (S2/S3 self-pairs) |
| 32 | Unicorn: A Unified Multi-tasking Model for Supporting Matching Tasks in Data Integration | Tu, Fan, Tang et al. | SIGMOD 2023 (PACMMOD vol. 1, no. 1) | Crossref record for DOI 10.1145/3588938 (audit-confirmed: title; authors Tu, Fan, Tang, Wang, Li, Du, Jia, Gao; *Proceedings of the ACM on Management of Data* 1(1), 2023) + GitHub repo | Unified encoder+matcher; repo has **no license** |
| 33 | Entity Matching using Large Language Models | Peeters, Steiner, Bizer | EDBT 2025 | https://openproceedings.org/2025/conf/edbt/paper-81.pdf (fetched) | Best LLMs need few examples; "higher robustness to unseen entities" (French!) |
| 34 | A Deep Dive Into Cross-Dataset Entity Matching with Large and Small Language Models | Zhang, Groth, Calixto, Schelter | EDBT 2025 | https://openproceedings.org/2025/conf/edbt/paper-224.pdf (fetched) | "fine-tuned small models can perform on par with prompted large models" |
| 35 | AnyMatch - Efficient Zero-Shot Entity Matching with a Small Language Model | Zhang, Groth, Calixto, Schelter | arXiv 2409.04073 (2024) | arXiv abs page | Within 4.4% of MatchGPT at 3,899x lower inference cost |
| 36 | Fine-tuning Large Language Models for Entity Matching | Steiner, Peeters, Bizer | arXiv 2409.08185 (2024) | arXiv abs page | Fine-tuning helps small models, but hurts cross-domain transfer (a caution for the unseen-French domain shift) |
| 37 | Jellyfish: Instruction-Tuning Local Large Language Models for Data Preprocessing (*re-audit: exact Anthology title; the arXiv title is "Jellyfish: A Large Language Model for Data Preprocessing"*) | Zhang, Dong, Xiao, Oyamada | EMNLP 2024 | https://aclanthology.org/2024.emnlp-main.497/ | Method is instructive; **weights CC-BY-NC, data is external**, so do not use |

Model tech reports used for model facts: Qwen2.5 (arXiv 2412.15115), Phi-4-Mini (arXiv 2503.01743), EuroBERT (arXiv 2503.05500). All were confirmed via the arXiv API.

---

## 8. Repos (GitHub API, 2026-09-25)

| Repo | License | Stars | Last push | Use |
|---|---|---|---|---|
| huggingface/sentence-transformers | Apache-2.0 | 19,118 | 2026-09-24 | Bi-/cross-encoder training (MNRL, CachedMNRL, Matryoshka loss) |
| FlagOpen/FlagEmbedding | MIT | 12,191 | 2026-08-24 | bge-m3 / bge-reranker fine-tuning scripts |
| facebookresearch/faiss | MIT | 40,973 | 2026-09-24 | GPU kNN over 10M vectors (IVF-PQ, pq256x4fs) |
| vllm-project/vllm | Apache-2.0 | 92,638 | 2026-09-24 | Fast LLM/reranker scoring on A100 |
| huggingface/text-embeddings-inference | Apache-2.0 | 5,060 | 2026-09-23 | High-throughput embed/rerank server on Modal |
| michaelfeil/infinity | MIT | 2,945 | 2026-03-24 | Alternative embedding server |
| MinishLab/model2vec | MIT | 2,211 | 2026-09-21 | Distil our fine-tuned encoder into a static CPU model |
| embeddings-benchmark/mteb | Apache-2.0 | 3,434 | 2026-09-24 | Harness for our own bitext-style eval |
| AI4Bharat/IndicXlit | MIT | 142 | 2023-10-13 | Transliteration model + `ai4bharat-transliteration` |
| AI4Bharat/IndicTrans2 | MIT | 475 | 2025-10-03 | Translation (gated HF weights) |
| AI4Bharat/IndicBERT | MIT | 124 | 2025-04-06 | IndicBERTv2 code |
| QwenLM/Qwen3-Embedding | **none detected** | 2,037 | 2025-09-30 | Reference code only; weights are Apache on HF |
| Snowflake-Labs/arctic-embed | Apache-2.0 | 91 | 2025-11-03 | MRL + int4 compression examples |
| nomic-ai/contrastors | Apache-2.0 | 804 | 2025-03-26 | Contrastive training code |
| AnswerDotAI/ModernBERT | Apache-2.0 | 1,744 | 2026-03-01 | Encoder training code |
| JHU-CLSP/mmBERT | **none detected** | 161 | 2026-01-20 | Reference; weights MIT on HF |
| google-research/byt5 | Apache-2.0 | 548 | archived | Reference |
| megagonlabs/ditto | Apache-2.0 | 318 | 2024-04-17 | EM cross-encoder reference |
| ruc-datalab/Unicorn | **none detected** | 32 | 2023-04-15 | Do not copy code (no license) |
| wbsg-uni-mannheim/MatchGPT | **none detected** | 68 | 2024-10-18 | Prompts reference only |
| indic-transliteration/indic_transliteration_py | MIT | 212 | 2026-09-08 | Rule-based scheme conversion (ISO/ITRANS) |
| anyascii/anyascii | ISC | 424 | 2026-06-06 | Permissive replacement for unidecode |
| avian2/unidecode | **GPL-2.0** | 609 | 2026-01-05 | *Audit correction:* **not used**. `src/ber/normalize.py` already imports `anyascii` (ISC) under the alias `_unidecode`. Keep it out of the pipeline. |
| virtualvinodh/aksharamukha-python | **AGPL-3.0** | 62 | 2026-08-30 | Avoid in shipped code |

---

## 9. What we should do

### P0 (do first; decides the architecture)
1. **Freeze a model whitelist and a license ledger.** Record the HF id, **commit hash (revision)**, exact license string, total params and the date checked in `research/MODEL_LICENSES.md`, and pin `revision=` in code. Whitelist: mE5 family, LaBSE (ST port), bge-m3, bge-reranker-v2-m3, gte-multilingual-base/reranker, arctic-embed-m/l-v2.0, nomic-embed-v2-moe, Qwen3-Embedding 0.6/4/8B, Qwen3-Reranker 0.6/4B, mxbai-rerank-v2, XLM-R, mDeBERTa-v3, MuRIL-base, IndicBERTv2(-SS), IndicXlit, CANINE, ByT5, potion-multilingual-128M, Qwen2.5-0.5/1.5/7B-Instruct, Qwen3-0.6/1.7/4B(-Instruct-2507), Phi-3.5/4-mini, Mistral-7B-v0.3.
2. **Zero-shot retrieval bake-off on Modal (L4/A10G, about 1-2 GPU-h).**
   * Hold out 20-50k S1 entities from train, stratified into (i) Indic cross-script, (ii) Latin India, (iii) US, and (iv) a synthetic French-style slice made from our own records.
   * Measure **recall@10/50/200** for S1 to S2/S3 on raw and transliterated text with mE5-small, mE5-large-instruct, LaBSE, bge-m3 (dense and dense+sparse), gte-mb, arctic-m-v2 (256-d), Qwen3-Emb-0.6B and potion-128M.
   * Keep the top 1-2, unioned with char n-gram TF-IDF blocking.
3. **Fine-tune the chosen bi-encoder** (expected: mE5-base or mE5-large-instruct, or LaBSE) with MNRL plus in-batch and mined hard negatives (same city/postcode, similar name). Use GroupKFold by S1 entity. Positives are every train S1 to S2/S3 link, plus S2 to S3 copy pairs. Add a Matryoshka loss so 256-d vectors work for the 10M index.

### P1
4. **Pairwise cross-encoder**, Ditto-style, serialising `name | address | city | state | postcode | country`. Compare fine-tuned gte-multilingual-reranker-base (fast), bge-reranker-v2-m3, and Qwen3-Reranker-0.6B-seq-cls on top-K candidates. Feed calibrated out-of-fold probabilities into the LightGBM matcher with char/rapidfuzz features, then choose per-S1 thresholds and cardinality to maximise macro F0.5, including the empty-set decision.
5. **IndicXlit native-to-Roman as a second transliteration view** (CPU, native-script records only). Use its top-k romanizations as extra TF-IDF documents and features next to `translit_indic`. Call `XlitEngine(src_script_type="indic", rescore=False)`, because the default `rescore=True` applies a bundled external word dictionary. Load the weights from the HF repo `ai4bharat/IndicXlit` (MIT) at a pinned revision.
6. ~~Swap the `unidecode` fallback to `anyascii`~~ *Audit: already done.* `src/ber/normalize.py` uses `anyascii` (ISC). Just make sure nobody adds `unidecode` (GPL-2.0) or `aksharamukha` (AGPL-3.0) later. When IndicXlit is added (item 5), call it with `rescore=False`.

### P2
7. **LLM adjudicator for the ambiguous band only** (for example 0.3 < p < 0.7, about 1-3% of pairs). LoRA-tune Qwen3-4B-Instruct-2507 or Qwen2.5-7B-Instruct on train pairs (A100, a few hours) and serve with vLLM. Alternatively, use Qwen3-Reranker-4B / Qwen3-Embedding-8B as **teachers** on sampled train pairs to distill into the base cross-encoder.
8. **Indic-specialised cross-encoder ensemble member**: MuRIL-base or IndicBERTv2-SS for the cross-script slice, and CANINE-s for typo-heavy Latin names. Blend via LightGBM.

### P3
9. Distil the fine-tuned bi-encoder into a Model2Vec static model for CPU-only re-blocking passes and fast ablations.
10. Use IndicTrans2-dist-200M only if error analysis shows *translated* (not transliterated) names. The user must accept the HF gate personally.
11. Avoid unless the organizers explicitly confirm: anything above 8.0B total (Qwen3-8B, Qwen3-Reranker-8B, granite-3.3-8b), and models that ship a Gemma or Gemma-derived tokenizer (Granite-Embedding R2 multilingual 311m/97m; mmBERT-base/small, upgraded to RISK by the audit).

---

## 10. Open items / not verified

* ~~The venue for DeBERTaV3 (ICLR 2023) was not re-fetched~~ **Resolved by audit:** ICLR 2023 poster. *Re-audit:* the OpenReview API/forum now returns a browser-verification page, so this was re-confirmed via the ICLR 2023 virtual site (`/virtual/2023/poster/11295`) and ML Anthology instead.
* ~~Unicorn: the ACM DL page was blocked~~ **Resolved by audit:** the Crossref record for DOI 10.1145/3588938 confirms title, authors and PACMMOD 1(1), 2023.
* Exact bge-m3 total parameter count: the HF API exposes none. "0.6B" comes from the Qwen3-Embedding comparison table. *Audit cross-check:* `pytorch_model.bin` is 2,271,145,830 bytes in fp32, which is about 568M parameters. That matches `bge-reranker-v2-m3` (567,755,777), which is built on bge-m3.
* The CANINE usable input length was taken from `max_position_embeddings=16384` in config.json. The paper's recommended length was not re-checked.
* All throughput figures are analytic estimates. They must be measured on Modal before planning final runs.

---

## Audit log (independent citation/license audit, 2026-09-25)

Nothing was run on the laptop apart from HTTP metadata queries. Every item was assumed hallucinated until it was confirmed on a primary source.

### What was checked and how

* **Models: 124 HF ids** (every id in sections 2 and 3). Each was queried on `https://huggingface.co/api/models/<id>`, reading `cardData.license`, `license_name`, `license:*` tags, `gated`, `safetensors.total` and `base_model:*` tags.
  * **Every license string and every total-parameter count in the tables matches the API exactly.** That includes `other`/`qwen-research` for Qwen2.5-3B, `other`/`mrl` for Ministral-8B, `other`/`openmdw-1.1` for the Nemotron models, `gpl-3.0` for sarvam-translate, and missing metadata for sarvam-1, muril-large-cased, IndicBART, jina-embeddings-v4, Hunyuan-MT-7B and cde-small-v2.
  * `google/LaBSE` and `NousResearch/Jellyfish` return HTTP 401, as stated.
* **Card text and config claims.** Raw `README.md` and `config.json` were fetched for 47 models, and every quoted string was confirmed. The confirmed items were:
  * **Gemma lineage:** bge-reranker-v2-gemma is based on gemma-2b; the Granite-311m-R2 card carries the Gemma ToU clause, is dated April 29, 2026, reports 65.2 MTEB-ML retrieval and has 32,768 ctx.
  * **Other license text:** the jina-v4 Qwen Research License sentence; the sarvam-1 "Sarvam non-commercial license"; the IndicBART MIT sentence.
  * **Qwen3 tables:** the Qwen3-Embedding MMTEB table (all 7 rows, including the "May 24th, 2025" note); the Qwen3-Reranker table (all 6 rows, with the top-100 from Qwen3-Embedding-0.6B note); Qwen3-8B "8.2B / 6.95B non-embedding".
  * **Tatoeba/BUCC F1:** all 27 cells were parsed from the `model-index` YAML of mE5-small, mE5-large-instruct and gte-multilingual-base, and all match after rounding. *(Re-audit: mE5-small hin-eng is exactly 93.25, a rounding boundary, so it is now written as 93.25.)*
  * **Embedder card claims:**
    * arctic-m-v2: "113m non-embedding", "128 bytes/vector", "MRL ... 256 dimensions", built on GTE-multilingual-base.
    * arctic-l-v2: "303m", built on bge-m3-retromae.
    * potion: "90.86% ... LaBSE", 47.31, distilled from bge-m3.
    * static-mrl: "100x to 400x faster".
    * nomic-v2-moe: 475M/305M, 512 tokens, 768 to 256.
    * gte-mb: elastic dims [128, 768]; remote code `Alibaba-NLP/new-impl` is `apache-2.0`.
    * Qwen3-Emb: output dims 32 to 1024.
  * **Encoder specs:** mmBERT 307M/110M and 140M/42M with the Gemma 2 tokenizer; MuRIL "transliterated counterparts" with vocab 197,285; IndicBERTv2 vocab 250,000 and 278M; IndicBERT-SS converts Indic scripts to Devanagari; mDeBERTa 86M + 190M; CANINE `max_position_embeddings=16384`.
  * **Other architecture facts:** Mistral vocab 32,768; Phi-4-mini vocab 200064; mxbai-rerank-base-v2 `Qwen2ForCausalLM`; bge-reranker-v2-m3 8194 positions.
  * **Jellyfish base models:** 8B from Meta-Llama-3-8B-Instruct, 7B from Mistral-7B-Instruct-v0.2.
  * **Language-tag counts:** mE5 94, LaBSE 110, mxbai 109; paraphrase-multilingual has no ta/te/kn/bn; gte-mb has no `or`; granite-278m has 12; SmolLM3 has 6 main languages; Phi-4-mini has 23 plus `multilingual`.
  * **Size estimates from weight files:** MuRIL-base 953 MB (about 238M); MuRIL-large 2.03 GB (about 507M); ByT5-small 1.199 GB; IndicBERTv2-SS 1.11 GB.
* **Papers: 40** (the 37 table rows plus the Qwen2.5, Phi-4-Mini and EuroBERT tech reports).
  * All arXiv ids were resolved through the arXiv API; titles and authors match.
  * Title, authors and venue were checked on 15 ACL Anthology pages: D19-1410, 2020.emnlp-main.365, 2020.acl-main.747, 2022.acl-long.62, 2024.findings-acl.137, 2024.emnlp-industry.103, 2023.eacl-main.148, 2022.tacl-1.17, 2022.tacl-1.5, 2023.acl-long.693, 2023.findings-emnlp.4, 2024.emnlp-main.497, 2024.acl-long.833, 2025.acl-long.127 and 2026.findings-acl.51.
  * Other primary sources:
    * The NeurIPS 2022 MRL page returned HTTP 200 with the matching title.
    * The PVLDB PDF p50-li gives "PVLDB, 14(1): 50 - 60, 2021", DOI 10.14778/3421424.3421431 and "up to 29% of F1".
    * EDBT 2025 paper-81 contains "higher robustness to unseen entities".
    * EDBT 2025 paper-224 lists authors Zhang, Groth, Calixto, Schelter and says "eight matchers on 11 benchmark datasets" and "fine-tuned small models can perform on par with prompted large models".
    * Crossref confirms Unicorn (PACMMOD 1(1) 2023), Sudowoodo (ICDE 2023) and C-Pack (SIGIR 2024).
    * The OpenReview API confirms MMTEB (ICLR 2025 Poster), DeBERTaV3 (ICLR 2023 poster) and IndicTrans2 (TMLR). *(Re-audit: not reproducible on 2026-09-25, because OpenReview returns HTTP 403 / a browser check. All three venues were re-confirmed from other primary pages; see the re-audit section below.)*
  * **Every quoted number or phrase was found in the source abstract:** 83.7% Tatoeba (LaBSE); within 4.4% of MatchGPT and 3,899x (AnyMatch); "hurting cross-domain transfer" (Steiner); "560 million" (MMTEB); "closer alignment" (RomanSetu); "negligible ... segmental ... morphosyllabic" (One Script); 26M pairs, 21 languages, 12 scripts and +15% Dakshina (Aksharantar); "significantly more robust to noise" (ByT5); 2.8 F1 and 28% fewer parameters (CANINE); 29 to 119 languages (Qwen3); "match ... BGE-M3" (mGTE); "1 billion multilingual text pairs" and "on par" (mE5); "no particular text embedding method dominates" (MTEB).
* **Repos: 24.** Each was checked with `gh api repos/<owner>/<repo>` for SPDX license, stars, `pushed_at` and `archived`, and each repo with no license was checked for a LICENSE file in its root.
  * All 24 licenses match: Qwen3-Embedding, mmBERT, Unicorn and MatchGPT really have no LICENSE file; unidecode is GPL-2.0; aksharamukha is AGPL-3.0; anyascii is ISC.
  * Stars had drifted by at most 13 on the audit date (e.g. vllm 92,651, faiss 40,976, sentence-transformers 19,120, Qwen3-Embedding 2,038). The listed values are approximate and were left as they were.
  * google-research/byt5 is `archived=true`, with its last push on 2024-02-13.
* **Local code claim.** `src/ber/normalize.py` was grepped, along with every `.py/.txt/.toml/.cfg` file in the repo, for `unidecode`.
* **Compute table.** The arithmetic was re-derived (for example 2 x 303M x 563M tokens / 120 TFLOP/s = 0.79 h, and Qwen3-Emb-8B about 18 h). It is internally consistent, but it is still an estimate, not a measurement.

### What was changed and why

1. **Wrong local-code claim (high impact on the P1 list).** The notes said the GPL `unidecode` was a fallback in `src/ber/normalize.py`. In fact the file imports `anyascii` (ISC) under the alias `_unidecode`, and nothing in the repo imports `unidecode`. Section 1 rule 5, the section 8 unidecode row and P1-6 were corrected. The "swap to anyascii" task is already done.
2. **ModernBERT citation (paper #19).** The title was wrong ("ModernBERT: Smarter, Better, Faster, Longer") and the venue was listed as arXiv only. It was corrected to the real title and ACL 2025 Long (`2025.acl-long.127`).
3. **IndicXlit model row.**
   * The claim "weights are GitHub release zips, not HF files" was false: the HF repo hosts `indicxlit-en-indic-v1.0/transformer/indicxlit.pt` and `indicxlit-indic-en-v1.0/transformer/indicxlit.pt`.
   * A missing compliance flag was added. The pip `XlitEngine` defaults to `rescore=True`, which reranks with a bundled word dictionary, and that dictionary is external data. The row, TL;DR and P1-5 now say to use `rescore=False`.
4. **mmBERT verdict: COMPLIANT -> RISK.** The card's spec table lists the tokenizer as "Gemma 2". This is the same issue for which Granite R2 was marked RISK, and arguably a stronger one, because mmBERT ships the Gemma tokenizer itself rather than a derived one. Keeping one COMPLIANT and the other RISK was inconsistent. mmBERT was not on the P0 whitelist, so the architecture plan is unaffected. It was added to section 3, the TL;DR traps and P3-11.
5. **Granite-97m-R2 precision.** The 97m card has no Gemma ToU clause, which the notes implied it had. Its risk comes from "vocabulary selection (262K → 180K tokens)" from the 311m's Gemma-3-derived vocabulary. The RISK verdict stays, with the correct reason.
6. **Sudowoodo verified_via.** The IEEE Xplore document number 10184556 came from a search result and could not be confirmed. It was replaced with the Crossref-confirmed DOI 10.1109/ICDE55515.2023.00391.
7. **Upgraded from "arXiv comment only" to primary-confirmed:**
   * MMTEB: ICLR 2025, via OpenReview.
   * DeBERTaV3: ICLR 2023, via OpenReview. This resolves section 10.
   * Unicorn: PACMMOD 1(1) 2023, via Crossref. This resolves section 10.
   * IndicTrans2: TMLR, via OpenReview.
   * RomanSetu: ACL 2024, `2024.acl-long.833`. The Anthology renders the first author as "Jaavid J".
   * ProRank: Findings of ACL 2026, `2026.findings-acl.51`; the full title was restored.
8. **bge-m3 parameter count.** It was cross-checked at about 568M from the size of the fp32 weight file (section 10).

### Items not changed but worth knowing

* No paper, model, repo, license string or quoted number was found to be fabricated. All 188 audited items exist.
* A policy-level caveat remains outside the model-license question. Several compliant models were trained on data with its own terms, for example MS MARCO-derived retrieval data. The challenge rule concerns the *model license*, so these stay COMPLIANT, but that assumption should be confirmed against the organizers' rule text if it is ever questioned.

### Re-audit (second independent pass, 2026-09-25)

Everything above was re-checked from scratch. Every item was treated as possibly hallucinated until a primary source confirmed it. Nothing was run on the laptop except HTTP metadata queries and text extraction from fetched PDFs. No model was loaded.

**How it was checked**

* **Models (124 HF ids).** A script queried `https://huggingface.co/api/models/<id>` for every id in sections 2 and 3, reading `cardData.license`, `license_name`, `gated`, `safetensors.total` and the `base_model:*` tags.
  * All 124 license strings match.
  * Every exact parameter count matches, and so does every rounded one (bge-en, mxbai-v1, ModernBERT, EuroBERT and others).
  * `google/LaBSE` and `NousResearch/Jellyfish` still return HTTP 401.
  * `sarvamai/sarvam-2b-v0.5` now redirects to `sarvamai/sarvam-1-v0.5`. Its license is still `other`.
* **Card text and configs.**
  * The raw `README.md` or `config.json` was fetched again for about 45 models, and every quoted string was found.
  * Both Qwen3 tables were checked row by row: all 7 rows × 4 columns of the embedding table, and all 6 rows × 3 columns of the reranker table.
  * The 27 Tatoeba/BUCC cells were re-parsed. They match, apart from the hin-eng boundary case listed below.
  * The Gemma/ToU lines were re-checked: the bge-reranker-v2-gemma base is gemma-2b; the Granite-311m card says "Gemma 3 ... Gemma Terms of Use"; the Granite-97m card has no Gemma sentence but says "262K → 180K"; the mmBERT card lists "Tokenizer | Gemma 2".
  * Also re-confirmed: the jina-v4 Qwen Research License sentence; the IndicBART MIT sentence; the sarvam-1 non-commercial line.
  * Configs: CANINE 16384; Mistral 32768; Phi-4-mini 200064; mxbai-rerank-v2 `Qwen2ForCausalLM` / 32768; bge-reranker-v2-m3 8194; MuRIL 197,285 / 512; IndicBERTv2 250,000.
  * ST configs: LaBSE max_seq 256; paraphrase-multilingual 128; mE5 512.
  * Remote code: `Alibaba-NLP/new-impl` is `apache-2.0`.
  * Weight-file sizes: MuRIL-base 953,477,430 B; MuRIL-large 2,028,800,502 B; ByT5-small 1,198,627,927 B; IndicBERTv2-SS 1,113,240,299 B; bge-m3 2,271,145,830 B.
  * Language tags were re-counted: mE5 94, LaBSE 110, mxbai-v2 109, gte-mb 75 without `or`, paraphrase-multilingual 50 without ta/te/kn/bn, granite-278m 12, Llama-3.1 8.
* **IndicXlit.**
  * The HF repo lists `indicxlit-{en-indic,indic-en}-v1.0/transformer/indicxlit.pt`.
  * The GitHub README says "(~11M)", "21 Indic languages" and "The IndicXlit code (and models) are released under the MIT License".
  * PyPI `ai4bharat-transliteration` 1.1.3 is MIT, and its README says `rescore` uses "a dictionary (Default: True)".
* **Repos (24).** `gh api repos/...` was run for each.
  * All SPDX licenses match.
  * Qwen3-Embedding, mmBERT, Unicorn and MatchGPT have no LICENSE file in their root listings.
  * Stars drifted by at most 20 (vllm 92,658), and some push dates moved by one day. The values stay as approximate.
  * byt5 is still archived.
* **Papers (37 in the table + 3 tech reports).**
  * All 38 arXiv ids resolve through the arXiv API with matching titles and authors.
  * All 15 ACL Anthology pages were re-fetched, and their `citation_title`, `citation_author` and venue meta tags match. The ProRank Anthology author list is Li, Shakir, Huang, Lipp, Clavié, Li.
  * Crossref confirms Unicorn (PACMMOD 1(1), 2023), Sudowoodo (ICDE 2023), Ditto (PVLDB 14(1)) and C-Pack (SIGIR 2024).
  * The NeurIPS 2022 MRL page was fetched.
  * PDF text was extracted and checked:
    * PVLDB p50-li: "PVLDB, 14(1): 50 - 60, 2021", the DOI, and "up to 29% of F1".
    * EDBT paper-81: "higher robustness to unseen entities", EDBT 2025.
    * EDBT paper-224: "eight matchers on 11 benchmark datasets" and "fine-tuned small models can perform on par with prompted large models".
  * Every quoted abstract phrase or number in sections 4 and 7 was found in the arXiv abstract (21 checks).
* **Compute table.** Every cell of section 6 and all four FAISS memory figures were recomputed from the stated assumptions, and all are consistent.

**What was changed in this pass, and why**

1. **Paper venue evidence (#17 MMTEB, #21 DeBERTaV3, #27 IndicTrans2).** The earlier `verified_via` entries relied on the OpenReview API. It now returns HTTP 403 or a browser check, so that evidence could not be reproduced. Each venue was re-confirmed on another primary page and the `verified_via` was replaced:
   * MMTEB: iclr.cc virtual poster 27651 and the ICLR 2025 proceedings page.
   * DeBERTaV3: iclr.cc virtual poster 11295 and ML Anthology.
   * IndicTrans2: ML Anthology TMLR 2023, plus the arXiv comment "Accepted at TMLR".
2. **Truncated or abbreviated paper titles restored:**
   * #23 CANINE: "... for Language Representation".
   * #25 IndicBERTv2: the full ACL 2023 title.
   * #26 Aksharantar: "... for the Next Billion Users".
   * #37 Jellyfish: the exact title is "Instruction-Tuning Local Large Language Models ...", not "Local LLMs". The arXiv title differs and is noted.
3. **bge-m3 quote.** The card says "more than 100 working languages", not the quoted "100+ working languages".
4. **ByT5 coverage.** "mC4 102 langs" was corrected to 101. The mT5 abstract says "covering 101 languages", and the ByT5 card gives no count.
5. **mE5-small Tatoeba hin-eng.** The metadata value is exactly 93.25, so both tables now show 93.25 instead of the ambiguous rounding 93.2.
6. **Qwen3 Indic coverage.** The source is now the specific one, the Qwen3 release blog's language list, which names the ten Indic languages plus French. The tech-report abstract only gives the count of 119.
7. **sarvam rows.** The `sarvam-2b-v0.5` → `sarvam-1-v0.5` redirect is noted. It was also recorded that the sarvam-1 `LICENSE.md` is titled "Sarvam AI Research License". Both rows stay non-compliant.

**Unchanged conclusions**

* No fabricated paper, model, repo, license string or number was found.
* Every compliance verdict stands. That includes all the RISK flags: Granite-R2 and mmBERT for Gemma tokenizer lineage, and Qwen3-8B, Qwen3-Reranker-8B and granite-3.3-8b for exceeding 8.0B total parameters.
* The P0-P3 plan does not change.
