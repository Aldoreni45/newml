# Company-name and address matching, transliteration, and cross-script matching

Research notes for the Amazon ML Challenge 2026 (Business Entity Resolution).
Compiled 2026-09-25. Every paper, repo, model and license below was checked against a primary page
(arXiv abs, ACL Anthology, VLDB PDF, OpenProceedings PDF, Crossref DOI metadata, publisher page, GitHub API,
Hugging Face API). Numbers are quoted only where they were read on that source. Anything not confirmed is marked **UNVERIFIED**.
**Independently audited 2026-09-25** (see "Audit log" at the end): all 48 papers, 28 repos/libraries and 16 HF models
were re-checked on primary sources; wrong venues, counts and license strings have been corrected in place. A third
independent audit the same day (last subsection of the Audit log) re-verified everything from scratch. It fixed
three small factual errors (a libpostal line count, the USPS suffix count, the ISO anusvara in a worked example)
and refreshed three star counts. A **fourth independent audit** (final subsection) re-checked all items again and
fixed: the ICDE author order of Deng et al., an over-general Aksharantar En→Indic claim, the USPS count (again: 206
names / 202 abbreviations; the third audit had counted an empty table row), and the base model of Chourasia et al.
(**Llama 3 8B, license `llama3`, not compliant**). It also added verified details to three rows.

> **Compute note:** nothing in this document needs to run on the laptop. Rule-based normalization and
> transliteration run on cloud CPU boxes (Modal/Kaggle/Lightning, 64+ cores). Neural transliteration
> (IndicXlit) and any encoder run on cloud GPUs (T4/L4/A10G/A100).

---

## 0. TL;DR

1. **Normalization should be learned from the data, not hand-coded.** Arasu et al. (PVLDB 2009) learn rules such as
   `corp→corporation` and `ltd→limited` from matching pairs, and Deng et al. (ICDE 2019) do it without labels. Our
   training positives (2.2M S1 entities with 2-6 matches each) are a large source of such rules. For **unseen
   France**, mine the rules on the test set itself from high-precision pseudo-matches (self-training). No external
   data is involved.
2. **Cross-script Indic is mainly a transliteration and phonetic-key problem.** Before comparing, romanize every
   non-Latin token with a rule-based tool (ICU `Any-Latin; Latin-ASCII` or `indic_transliteration`). Then build two
   phonetic keys that collapse schwa, aspiration, vowel length, retroflex/dental, sh/s, v/w and gemination. The
   neural IndicXlit model is MIT-licensed, but its **top-1 Indic→En accuracy on named entities is only 35-61%**
   for the major languages (README tables: hin 59.22, tam 35.46, tel 57.57, kan 60.77; lower still for urd 23.14 and
   kas 12.87). Use its top-k beams through fuzzy/phonetic comparison, never exact match.
3. **Script-aware Unicode handling is a correctness issue, not polish.** Stripping all combining marks to remove
   injected diacritics ("Léarning") also deletes Devanagari/Tamil vowel signs, virama and nukta (all combining
   marks). Strip marks **only when the base character is Latin**, and resolve the Script property per UAX #24.
4. **Keep legal form as a separate attribute; do not just delete it.** Under F0.5 (1 FP costs 4× one FN), treat
   "Ram Marketing Pvt Ltd" vs "Ram Marketing LLP" as a *conflict* feature and a missing legal form as *neutral*.
   cleanco (MIT) is a seed list only. Its *India* country list is just `ltd.`, `pse`, `psu`, `pvt. ltd.`, but
   `basename()` matches the union of all type and country terms, which does include `private`, `limited`, `llp`,
   `llc`, `sa`, `co`, `company` (so it strips far more than India-specific forms, including risky ones like `co`).
   It has no `opc`, and no single-token `pvt`. `basename()` strips only suffixes by default (`prefix=False,
   middle=False`), so it misses "LLC Moncada Learning Center" unless `prefix=True`.
5. **IDF per country replaces stop-word lists and generalizes to the open set.** Compute document frequency per
   country over S1∪S2∪S3 (transductive, no labels). Legal suffixes, street types and "null" tokens then get low
   weight automatically, including in French. Cohen et al. (2003) found that TF-IDF combined with Jaro-Winkler was
   the best name metric.
6. **Postcodes carry the most signal per byte, and their structure is country-specific but learnable.** US ZIP5
   (first 3 digits = sectional center), India PIN6 (first 3 = sorting district), France 5-digit (first 2 =
   département; Corsica "20"; overseas 97x). Learn each country's postcode digit length from the data, then block on
   and compare prefixes.
7. **Compliance flags.** The **libpostal** parser model is MIT code, but it is trained on OpenStreetMap (ODbL) and
   OpenAddresses, so it embeds external geo data: **RISK, keep it out of the final pipeline**. **deepparse** is
   LGPL-3.0 and its weights are trained on libpostal-derived OSM data: **not an MIT/Apache model and a RISK**.
   **pgeocode** downloads GeoNames at runtime: **violates the no-external-lookup rule**. **Aksharamukha** is
   AGPL-3.0 and **Unidecode**/**abydos** are GPL (software licenses, not model licenses), so prefer the
   MIT/ISC alternatives.

---

## 1. Problem-specific framing

| Noise type (measured in data) | Mechanism to handle it | Learned or rule | Where it enters |
|---|---|---|---|
| Casing, doubled spaces, junk prefixes (`--`, `<<`, `##`) | Unicode NFKC, lowercase, collapse whitespace, strip leading/trailing non-alphanumeric runs | rule (universal) | preprocessing |
| Injected diacritics ("Léarning") | NFKD, then drop combining marks **only on Latin base chars** | rule, script-aware | preprocessing |
| Abbreviations (Corp/Corporation, Pvt/Private, St/Street, R./Rue) | transformation rules mined from positive pairs (Arasu 2009); self-mined on test for FR (Deng 2019) | learned | canonicalization |
| Legal suffix moved ("LLC Moncada Learning Center") | legal-form token set learned by DF plus boundary position; extract from anywhere | learned | separate attribute |
| DBA names ("X dba Y") | split on learned DBA markers (`dba`, `d/b/a`, `doing business as`, `t/a`) into 2 name variants | rule seed + learned | variant generation |
| Website as name ("heassociates.com") | detect `stem.tld`; compare stem against the space-removed core name ("compact key"); optional word segmentation with an in-corpus unigram LM | rule + learned LM | variant generation |
| Typos ("Hospirlg") | char n-gram TF-IDF, Jaro-Winkler, Monge-Elkan | similarity | features |
| Cross-script names (Devanagari/Tamil/Kannada ↔ Latin), mixed-script ("Sun पावर Provision") | per-token script detection, then romanization and phonetic keys; optional IndicXlit top-k | rule + pretrained (MIT) | parallel fields |
| Native-script state names ("महाराष्ट्र") | romanize, key, then map to canonical state learned from pairs | learned | address canonicalization |
| US state full name vs code | map learned from aligned positive pairs (no hard-coded list needed) | learned | address canonicalization |
| "null" tokens, missing components | learned null-token set (whole components with very high DF); missing = neutral, not mismatch | learned | features |
| Component reordering ("OH, Columbus, 5559 Orville Avenue") | order-insensitive token-set similarity; component tagging by type (digits, postcode, state) | rule | features |
| Landmarks ("Near Fortis Hospital") | learned trigger words (near/opp/behind/beside…), extracted as a soft field | learned | features |
| Unseen country (France) | IDF per country; postcode length learned from data; self-training rule mining on test | transductive | whole pipeline |

**Metric implications (macro F0.5 per S1 entity).**
- Over-normalization creates false positives. If you strip the legal form, generic tokens and the city, then
  "Sharma Traders Pvt Ltd, Pune" and "Sharma Traders LLP, Pune" collapse together. Every canonicalization should
  feed both a *merged* key (for blocking/recall) and a *conflict* feature (for precision).
- Singleton S1 entities score 1 only when predicted empty, so the matcher needs calibrated "no match" behaviour.
  Name keys that are too loose (for example the Tamil voicing-collapsed key) should be features with learned
  weights, never hard match rules.

---

## 2. Verified papers

| # | Paper | Venue / year | What it gives us | Verified via |
|---|---|---|---|---|
| 1 | Arasu, Chaudhuri, Kaushik: *Learning String Transformations From Examples* | PVLDB 2(1), pp. 514-525 / VLDB 2009 (DOI 10.14778/1687627.1687686) | Learn a concise set of token transformation rules (MDL-style, greedy approximation of an NP-hard problem) from matching pairs. Fig. 2 lists rules learned for **organization names**: co→company, corp→corporation, inc→incorporated, ltd→limited, assoc→associates, intl→international, mfg→manufacturing, bros→brothers, univ→university | http://www.vldb.org/pvldb/vol2/vldb09-226.pdf (PDF p.1-2 read) |
| 2 | Deng, Tao, Abedjan, Elmagarmid, Ilyas, Li, Madden, Ouzzani, Stonebraker, Tang (ICDE author order per Crossref; the arXiv version lists Li before Ilyas): *Unsupervised String Transformation Learning for Entity Consolidation* | ICDE 2019, pp. 196-207 (DOI 10.1109/ICDE.2019.00026) | Groups value pairs that share a transformation pattern with no labels (a human then approves groups); 75% recall at 99.5% precision on 17,497 records with 100 yes/no questions. **Template for self-mining FR rules on test.** | https://arxiv.org/abs/1709.10436 ; Crossref 10.1109/ICDE.2019.00026 |
| 3 | Lu, Lin, Wang, Li, Wang: *String similarity measures and joins with synonyms* | SIGMOD 2013, pp. 373-384 | Similarity joins where synonym/abbreviation rules expand strings (selective expansion, SI-tree). How to apply learned rules at join time without blowing up candidates. | Crossref 10.1145/2463676.2465313 |
| 4 | Li, Cheng, Chu, He, Chaudhuri: *Auto-FuzzyJoin: Auto-Program Fuzzy Similarity Joins Without Labeled Examples* | SIGMOD 2021 | Unsupervised search over preprocessing, tokenization, distance function and threshold to **meet a precision target τ while maximizing recall**, using the fact that one table is a reference table (our S1 is one). Directly fits F0.5 and unseen France. | https://arxiv.org/abs/2103.04489 |
| 5 | Cohen, Ravikumar, Fienberg: *A Comparison of String Distance Metrics for Name-Matching Tasks* | IIWeb 2003 (IJCAI workshop), pp. 73-78 | Best performer: a hybrid of TF-IDF weighting and Jaro-Winkler (SoftTFIDF, JW secondary distance, θ = 0.9). Baseline name metric for all token-level comparisons. | https://www.cs.cmu.edu/~wcohen/postscript/ijcai-ws-2003.pdf (p.1 read); ACM DL 10.5555/3104278.3104293 |
| 6 | Christen: *A Comparison of Personal Name Matching: Techniques and Practical Issues* | ICDM Workshops 2006, pp. 290-294 | Taxonomy of name variation sources and a comparison of phonetic and edit-based matchers; on four large name data sets there is "no clear best matching technique". Reference for phonetic-key design. | Crossref 10.1109/ICDMW.2006.2 |
| 7 | Churches, Christen, Lim, Zhu: *Preparation of name and address data for record linkage using hidden Markov models* | BMC Med. Inform. Decis. Mak. 2, 2002 | Lexicon-based tokenization plus HMM standardization of names and addresses. HMMs were equal or better than rule-based on addresses but worse on simple name data. Foundational for a learned address segmenter. | Europe PMC (PMID 12482326 / PMC140019) |
| 8 | Loster, Zuo, Naumann, Maspfuhl, Thomas: *Improving Company Recognition from Unstructured Text by using Dictionaries* | EDBT 2017, pp. 610-619 | Company **alias generation** in 5 steps: (1) remove legal form (regexes from Wikipedia legal-entity lists), (2) remove special chars, (3) normalize, (4) remove country name, (5) stem. Up to 9 aliases per name. German company NER reached P = 91.11%, R = 78.82%. Blueprint for our name-variant generation. | https://openproceedings.org/2017/conf/edbt/paper-257.pdf (pp.1-5 read) |
| 9 | Kruse, Awick, Marx Gómez, Loos: *Developing a Legal Form Classification and Extraction Approach for Company Entity Matching: Benchmark of Rule-Based and Machine Learning Approaches* | BIS 2021 (Vol. 1), pp. 13-26 | Legal forms are represented inconsistently across sources. The best of four approaches was a **hybrid of rules and supervised ML** for extracting legal form as an attribute. | https://www.tib-op.org/ojs/index.php/bis/article/view/44 (DOI 10.52825/bis.v1i.44) |
| 10 | Arimond, Molteni, Jany, Manolova, Borth, Hoepner: *Transformer-based Entity Legal Form Classification* | arXiv 2023 | BERT variants classify the legal form from raw names, trained on 1.1M+ GLEIF LEI entities across 30 jurisdictions, and beat traditional baselines in F1. (Their training data is external; we would only reuse the idea and train on our own data.) | https://arxiv.org/abs/2310.12766 |
| 11 | Basile, Crupi, Grasso, Mercanti, Regoli, Scarsi, Yang, Cosentini: *Disambiguation of Company names via Deep Recurrent Networks* | Expert Syst. Appl. 238 (Part C), 122035, 2024 (DOI 10.1016/j.eswa.2023.122035) | Siamese LSTM embeddings of company names beat string-matching baselines when enough labels exist. Active learning cuts labels needed. | https://arxiv.org/abs/2303.05391 |
| 12 | Gschwind, Miksovic, Minder, Mirylenka, Scotton: *Fast Record Linkage for Company Entities* | arXiv 2019 | Company name normalization, MinHash blocking, then scoring with name, location and industry. Recall 91% vs 73% for baseline approaches, with linear scaling. | https://arxiv.org/abs/1907.08667 |
| 13 | Ziv, Gronau, Fire: *CompanyName2Vec: Company Entity Matching Based on Job Ads* | arXiv 2022 | Learns company-name semantics and synonyms from a corpus. 89.3% average success rate. Shows corpus co-occurrence can reveal name aliases. | https://arxiv.org/abs/2201.04687 |
| 14 | Li, Li, Suhara, Doan, Tan: *Deep Entity Matching with Pre-Trained Language Models (Ditto)* | PVLDB 14(1), pp. 50-60 (issue dated 2020; presented at VLDB 2021) | Domain-knowledge injection (typing and normalizing spans) plus summarization and augmentation. **96.5% F1 matching two company datasets (789K and 412K records).** Normalized fields plug into a cross-encoder. | https://arxiv.org/abs/2004.00584 |
| 15 | Neculoiu, Versteegh, Rotaru: *Learning Text Similarity with Siamese Recurrent Networks* | RepL4NLP @ ACL 2016, pp. 148-157 | Char-level BiLSTM Siamese network that maps variable-length strings into a fixed embedding using only pair similarity labels. Cheap cross-script name encoder design. | https://aclanthology.org/W16-1617/ |
| 16 | Schwartz, Hearst: *A Simple Algorithm for Identifying Abbreviation Definitions in Biomedical Text* | PSB 2003, pp. 451-462 | Training-free short-form/long-form alignment. 96% P / 82% R, and 95% P / 82% R on a larger set. Use as the **candidate generator** for abbreviation pairs (St↔Street, Pvt↔Private, acronyms) mined from positive pairs. | http://psb.stanford.edu/psb-online/proceedings/psb03/abstracts/p451.html |
| 17 | Gorman, Kirov, Roark, Sproat: *Structured abbreviation expansion in context* | Findings EMNLP 2021, pp. 995-1005 | Ad hoc abbreviations are intentional and cannot be resolved by dictionary lookup alone. Context is needed, which argues for mining from pairs. | https://aclanthology.org/2021.findings-emnlp.85/ |
| 18 | Madhani, Parthan, Bedekar, Nc, Khapra, Kunchukuttan, Kumar, Khapra: *Aksharantar: Open Indic-language Transliteration datasets and models* | Findings EMNLP 2023, pp. 40-57 | 26M transliteration pairs, 21 Indic languages, 12 scripts. The IndicXlit model improves accuracy by 15% on the Dakshina test set. Named-entity top-1 is low in both directions (see §4). Note: for **Indic→En** the README shows named-entity accuracy is *not* uniformly lower than native-word accuracy (NE is higher for ben 52.54 vs 11.76, guj, hin 59.22 vs 52.30, mar, pan and tel 57.57 vs 54.55). For En→Indic, NE is lower in 18 of 19 languages; Hindi is the exception (58.87 NE vs 55.59 native). | https://aclanthology.org/2023.findings-emnlp.4/ |
| 19 | Gala et al. (14 authors): *IndicTrans2* | TMLR 2023 (per arXiv comment) | MT for all 22 scheduled languages. It is **translation, not transliteration**, so it may translate the meaning of name tokens. Low priority for names. | https://arxiv.org/abs/2305.16307 |
| 20 | Roark, Wolf-Sonkin, Kirov, Mielke, Johny, Demirsahin, Hall: *Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset* | LREC 2020, pp. 2413-2423 | Native and Latin script text plus romanization lexicons for 12 South Asian languages. Documents how varied attested romanizations are (why exact transliteration fails). | https://aclanthology.org/2020.lrec-1.294/ |
| 21 | Hermjakob, May, Knight: *Out-of-the-box Universal Romanization Tool uroman* | ACL 2018 Demos, pp. 13-18 | Any script to Latin, which makes string similarity across scripts possible without phonetic resources. **Open-set fallback for unseen scripts.** | https://aclanthology.org/P18-4003/ |
| 22 | J (Jaavid Aktar Husain on arXiv), Dabre, M (Aswanth Kumar on arXiv), Gala, Jayakumar, Puduppully, Kunchukuttan: *RomanSetu* | ACL 2024, pp. 15593-15615 | Romanized Indic text cuts token fertility 2-4× and **its embeddings align more closely with English** than native script does. Supports "romanize first, then embed" for cross-script names. | https://aclanthology.org/2024.acl-long.833/ |
| 23 | Ebing, Keller, Glavaš: *One Script Instead of Hundreds? On Pretraining Romanized Encoder Language Models* | arXiv 2026 | Encoders pretrained from scratch on six high-resource languages, **including Hindi (Devanagari)** alongside Arabic, Russian, Vietnamese, Chinese and Japanese, using two romanizers (uroman and uconv). Romanizing segmental scripts (Arabic, Hindi, Russian, Vietnamese) costs negligible performance. Morphosyllabic scripts (Chinese/Japanese) degrade. Direct evidence that romanizing Devanagari is safe. | https://arxiv.org/abs/2601.05776 |
| 24 | Khanuja et al. (14 authors): *MuRIL* | arXiv 2021 | Indic BERT pre-trained with translated **and transliterated** segment pairs, with results on native→Latin transliterated test sets. Candidate cross-script encoder (Apache-2.0). | https://arxiv.org/abs/2103.10730 |
| 25 | Xue et al.: *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models* | TACL 10, 2022, pp. 291-306 | Byte models handle any script, are more robust to noise, and do better on spelling/pronunciation-sensitive tasks. Candidate for a learned transliteration or cross-script matcher. | https://aclanthology.org/2022.tacl-1.17/ |
| 26 | Clark, Garrette, Turc, Wieting: *CANINE* | TACL 10, 2022, pp. 73-91 | Character-level encoder with no tokenizer, +5.7 F1 over mBERT on TyDi QA with fewer parameters. | https://aclanthology.org/2022.tacl-1.5/ |
| 27 | Sajjad, Fraser, Schmid: *A Statistical Model for Unsupervised and Semi-supervised Transliteration Mining* | ACL 2012, pp. 469-477 | EM model separating transliteration pairs from noise, language-pair independent. **Mine char alignments from our own cross-script training positives** with no external data. | https://aclanthology.org/P12-1049/ |
| 28 | Benites, Duivesteijn, von Däniken, Cieliebak: *TRANSLIT: A Large-scale Name Transliteration Resource* | LREC 2020, pp. 3265-3271 | 1.6M entries, 180+ languages, about 3M name variants, 92% accuracy identifying transliterated pairs. **External data: context only, do not train on it.** | https://aclanthology.org/2020.lrec-1.399/ |
| 29 | Sälevä, Lignos: *ParaNames 1.0* | LREC-COLING 2024, pp. 12599-12610 | 140M names, 400+ languages, 16.8M entities (PER/LOC/ORG) from Wikidata. **External data: context only.** | https://aclanthology.org/2024.lrec-main.1103/ |
| 30 | Gadd: *Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching* | arXiv 2026 | Built for **toponyms** (place names) from 20 writing systems, with reported transfer to personal names in archival sources. Teacher-student distillation from IPA articulatory features into a char-level student, giving a 128-d shared space. Trained on 32.7M triplets from GeoNames, Wikidata and Getty TGN. MEHDIE benchmark (medieval Hebrew/Arabic toponyms): Recall@1 85.2%, MRR 90.8%. Design idea for a phonetic cross-script embedding only (their training data is external). | https://arxiv.org/abs/2601.06932 |
| 31 | Chourasia, Kapoor, Patil: *Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts* | ACL 2026 **Industry Track** (Vol. 6), pp. 1461-1468 (DOI 10.18653/v1/2026.acl-industry.101) | **Person-name** matching on Indian KYC identity records (Dream11), not business names. Two-phase curriculum (parse name structure, then match). 99.02% accuracy, F1 0.994 on 50K held-out real-world pairs. Beats GPT-4o few-shot. The paper body (§3.1, §4.3) states the base is **Meta Llama 3 8B Instruct**, fine-tuned with LoRA (rank 4, alpha 8). Its HF card license is `llama3`, so **their model is not compliant**. Only the two-phase curriculum idea can be reused, and only on an MIT/Apache base of 8B or fewer parameters whose card license has been checked. | https://arxiv.org/abs/2605.23597 ; https://aclanthology.org/2026.acl-industry.101/ |
| 32 | Singh, Choudhary, Shrivastava: *Automatic Normalization of Word Variations in Code-Mixed Social Media Text* | CICLing 2018 (proceedings in LNCS, published 2023, pp. 371-381, DOI 10.1007/978-3-031-23793-5_30) | Spelling variants of romanized Hindi share context, so cluster them with unsupervised embeddings. Corpus-internal variant discovery. | https://arxiv.org/abs/1804.00804 |
| 33 | Yadav, Akhtar, Chakraborty: *Normalization of Spelling Variations in Code-Mixed Data* | ICON 2022, pp. 269-279 | Cheap phonetic normalization of Hinglish spelling variants. The paper notes there is no universal protocol or alphabet mapping for code-mixing. | https://aclanthology.org/2022.icon-main.33/ |
| 34 | Yassine, Beauchemin, Laviolette, Lamontagne: *Leveraging Subword Embeddings for Multinational Address Parsing* | IEEE CiSt 2020, pp. 353-360 (DOI 10.1109/CiSt49399.2021.9357170) | Single RNN parser over many countries, about 99% on training countries, zero-shot good on 33/41. (Basis of deepparse; see compliance.) | https://arxiv.org/abs/2006.16152 |
| 35 | Yassine, Beauchemin, Laviolette, Lamontagne: *Multinational Address Parsing: A Zero-Shot Evaluation* | iJIST (accepted, per arXiv) | Attention and domain-adversarial training for **zero-shot countries**, plus incomplete-address robustness. Relevant to unseen France. | https://arxiv.org/abs/2112.04008 |
| 36 | Beauchemin, Yassine: *Deepparse: An Extendable, and Fine-Tunable State-Of-The-Art Library for Parsing Multinational Street Addresses* | **NLP-OSS 2023** (3rd Workshop for NLP Open Source Software, Singapore, Dec 2023), pp. 19-24. (The arXiv comment says "EMNLP 2024 NLP-OSS", but the ACL Anthology places it in NLP-OSS 2023.) | Library paper. LGPL-3.0. Evaluated on 60+ countries. | https://arxiv.org/abs/2311.11846 ; https://aclanthology.org/2023.nlposs-1.3/ |
| 37 | Mangalgi, Kumar, Tallamraju: *Deep Contextual Embeddings for Address Classification in E-commerce* | KDD 2020 workshop | Indian e-commerce addresses have no fixed format. A RoBERTa pre-trained on an address corpus reaches about 90% sub-region classification, and preprocessing uses edit distance and **phonetic algorithms**. | https://arxiv.org/abs/2007.03020 |
| 38 | Rustogi, Bhattacharya, Church, Raskar: *What is the right addressing scheme for India?* | arXiv 2018 (MIT Emerging Worlds) | **80% of Indian addresses are written relative to a landmark, which typically lies 50-1500 m from the actual address.** Justifies a separate landmark field. | https://arxiv.org/abs/1801.06540 |
| 39 | Lin, Kang, Wu, Du, Liu: *A deep learning architecture for semantic address matching* | IJGIS 34(3), 2020, pp. 559-576 | word2vec address vectors plus an ESIM text-matching model for address pairs (code MIT on GitHub). On the Shenzhen Address Database (Chinese addresses) precision, recall and F1 reach 0.97 on the test set. | Crossref 10.1080/13658816.2019.1681431 ; abstract via Semantic Scholar API (DOI lookup) |
| 40 | Comber, Arribas-Bel: *Machine learning innovations in address matching: A practical comparison of word2vec and CRFs* | Trans. in GIS 23(2), 2019, pp. 334-348 | CRF segmentation vs word2vec for address matching. | https://livrepository.liverpool.ac.uk/3038194/ |
| 41 | Duarte, Oliveira: *Improving Address Matching using Siamese Transformer Networks* | EPIA 2023, LNCS pp. 413-425 (DOI 10.1007/978-3-031-49011-8_33) | Bi-encoder retrieves the top-10 from a normalized DB, then a cross-encoder reranks. >95% door-level accuracy on Portuguese addresses, about 4.5× faster than BM25 on GPU. Retrieval then rerank template. | https://arxiv.org/abs/2307.02300 |
| 42 | Yang, Hoang, Mikolov, Han: *Place Deduplication with Embeddings* | WWW 2019, pp. 3420-3426 (DOI 10.1145/3308558.3313456) | Facebook place graph (name + address + location, multi-source duplicates, the same shape as S2/S3). Nearest-neighbour blocking, then a matcher. | https://arxiv.org/abs/1910.04861 |
| 43 | Ganesan, Gupta, Mathew: *Mining Points of Interest via Address Embeddings* | arXiv 2021 | Indian delivery addresses: locality-name preprocessing plus a RoBERTa trained on an internal address corpus. It finds 74.8% more PoIs than the Mummidi-Krumm baseline. The median polygon F-score is 0.15 before and 0.69 after post-processing with OSM building footprints, which we cannot use. | https://arxiv.org/abs/2109.04467 |
| 44 | Linacre, Lindsay, Manassis, Slade, Hepworth: *Splink: Free software for probabilistic record linkage at scale* | IJPDS 7(3), 2022 (conference abstract) | Fellegi-Sunter with EM, Spark for scale; the abstract reports ~2M package downloads. Its **term-frequency adjustments** are exactly the "IDF per value" idea (TF adjustments and the DuckDB backend are documented in the Splink docs, not in the IJPDS abstract). | https://ijpds.org/index.php/ijpds/article/view/1794 ; https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html |
| 45 | Peeters, Bizer: *Cross-Language Learning for Product Matching* (arXiv title: "...for Entity Matching") | WWW '22 Companion (poster), pp. 236-238 (DOI 10.1145/3487553.3524234) | Adding English pairs to a small German training set improved transformer matchers in all cases, most in low-resource settings. Supports **training on US+IN and transferring to FR**. | https://arxiv.org/abs/2110.03338 |
| 46 | Tarannum, Mohammed, Cakmak, Al Mandalawi, Talburt: *A System for Name and Address Parsing with Large Language Models* | arXiv 2026 | Prompted LLM, constrained decoding and a rule validator to a 17-field schema. **Too slow for 10M records**; useful only for labelling a small sample offline (cloud). | https://arxiv.org/abs/2601.18014 |
| 47 | Gupta: *Corporate-Family Resolution Is Not a String-Matching Problem* (CorpFam) | arXiv 2026 | Stratifies pairs by "name visibility" (identical after normalization / shared distinctive token / none). 93.2% of invisible links never entered the candidate set. **Use the stratification for our error analysis.** | https://arxiv.org/abs/2609.04269 |
| 48 | Gaere, von Wangenheim: *MESSY STREETS: A Benchmark for Geocoding Real-World Addresses* | arXiv 2026 | Non-canonical surface form alone accounts for up to 25 pp of recall loss, and one unrecognized token can zero a query in Nominatim's conjunctive matching. Argues for **disjunctive/soft token matching and normalization**. | https://arxiv.org/abs/2609.01612 |

Standards and reference facts (not papers):
- **Unicode UAX #24 Script property** (Unicode 18.0.0, rev. 41, 2026-08-28). Each code point has a Script value.
  `Common` and `Inherited` must be resolved from context, and combining marks inherit the script of their base.
  https://www.unicode.org/reports/tr24/
- **ICU transforms**: `Any-Latin; Latin-ASCII` exists. Indic scripts go through an internal "Inter-Indic" form and
  romanize per ISO 15919. The page warns that "script transliteration is not translation". https://unicode-org.github.io/icu/userguide/transforms/general/
- **USPS Publication 28, Appendix C1**: 206 primary street-suffix names mapping to 202 distinct standard
  abbreviations (counted from the C1 table's 3-cell rows after excluding the header and one empty trailing row),
  each with common variants and a standard abbreviation (for example AVENUE:
  AV/AVEN/AVENU/AVN/AVNUE → AVE). This is external reference data, so see the compliance note.
  https://pe.usps.com/text/pub28/28apc_002.htm
- **AFNOR NF Z10-011** (Jan 2013), the French postal address standard. https://www.boutique.afnor.org/en-gb/standard/nf-z10011/mailing-address-how-to-write-a-mailing-address-rules-for-the-presentation-o/fa178533/40541
  From secondary sources only (**UNVERIFIED on primary**): max 6 lines × 38 characters; the *indice de répétition*
  is bis/ter/quater/quinquies (B/T/Q/C) or a letter A-D after the house number.
- **Postal code structure** (Wikipedia, secondary source):
  - US ZIP: 1st digit = national area (0 = New England/NJ/PR … 9 = West Coast/Pacific), first 3 = sectional
    center facility, ZIP+4 optional, leading zeros exist (00501).
  - India PIN: 6 digits. 1st = zone (1 Delhi/HR/HP/PB/J&K, 2 UP/UK, 3 GJ/RJ, 4 MH/MP/CG/Goa, 5 AP/KA/TG,
    6 KL/TN, 7 East/NE, 8 BR/JH, 9 Army Postal). First 3 = sorting district, last 2 = delivery office.
  - France: 5 digits, first 2 = département. Corsica uses "20" (2A/2B), overseas uses 971-976 and later, CEDEX
    codes exist, Paris/Lyon/Marseille last 2 digits = arrondissement, and adjacent villages often share a postcode.
  - Sources: https://en.wikipedia.org/wiki/ZIP_Code, https://en.wikipedia.org/wiki/Postal_Index_Number,
    https://en.wikipedia.org/wiki/Postal_codes_in_France

---

## 3. Repositories and libraries (license, stars, last push as of 2026-09-25)

Library licenses are **software** licenses and are separate from the MIT/Apache **model** rule. They still matter
for redistribution. "Data risk" means the package ships or downloads externally derived data.

| Library | URL | License (software) | Stars | Last push | Data risk | Use for us |
|---|---|---|---|---|---|---|
| cleanco | https://github.com/psolin/cleanco | MIT | 360 | 2026-06-23 | Low: hand-built legal-term list (from Wikipedia "types of business entity") | Seed legal-form list. `basename(suffix=True, prefix=False, middle=False)` defaults, so set prefix/middle=True for moved suffixes. The *India* country list is only `ltd.`, `pse`, `psu`, `pvt. ltd.`; the *France* list is `ei, eurl, fcp, gie, sarl, sas, sasu, sca, sci, scop, scs, sem, sep, sicav, snc, societe civile immobiliere, sogepa`. But `basename()` uses the **union of all type + country terms**, which includes `private`, `limited`, `ltd`, `llp`, `llc`, `sa`, `co`, `company` (no `opc`, no bare `pvt`). Stripping `co`/`company` is aggressive, so prefer our own learned list |
| libpostal | https://github.com/openvenues/libpostal | MIT (code) | 4,894 | 2026-05-13 | **HIGH**: CRF parser and language classifier trained on OSM (ODbL), OpenAddresses (mostly CC-BY) and Yahoo GeoPlanet (CC-BY); about 1.8-2.2 GB model download. 99.45% full-parse accuracy | **Do not use `parse_address` in the final pipeline.** Text dictionaries under `resources/dictionaries/<lang>/` (fr: 28 files incl. `street_types.txt` 165 lines, `company_types.txt` 40 lines, `near.txt`; en `company_types.txt` 61 lines incl. `doing business as|d / b / a|dba|d b a`; hi `street_types.txt` has only 4 lines; **no ta/kn/te/mr/bn/gu/ml**) are hand-curated, but they are external knowledge. Use them offline only to *validate* self-mined rules |
| pypostal | https://github.com/openvenues/pypostal | MIT | 880 | 2025-11-01 | same as libpostal | same as above |
| deepparse | https://github.com/GRAAL-Research/deepparse | **LGPL-3.0** | 354 | 2026-09-19 | **HIGH**: training data generated from libpostal (i.e. OSM/OpenAddresses) per deepparse-address-data (MIT); 20 training countries incl. France and US, India only zero-shot | **Pretrained weights are not MIT/Apache and are OSM-derived: not compliant as a model.** The architecture could be retrained on our own labels, but we lack component labels |
| deepparse-address-data | https://github.com/GRAAL-Research/deepparse-address-data | MIT | 32 | 2022-10-27 | derived from libpostal (OSM) | external data, do not use |
| usaddress | https://github.com/datamade/usaddress | MIT | 1,637 | 2025-08-07 | **Med**: CRF (python-crfsuite) trained on bundled labelled US data, and the `training/` folder includes OSM/OpenAddresses-derived files (`synthetic_osm_data_xml.xml`, `synthetic_clean_osm_data.xml`, `openaddress_us_ia_linn.xml`), so the shipped model is partly trained on external geo data | US-only component tagger (house number, street, suffix, state, ZIP). Optional feature extractor for US; a small pretrained model, not a gazetteer |
| indic_transliteration (sanscript) | https://github.com/indic-transliteration/indic_transliteration_py | MIT | 212 | 2026-09-08 | None (rule tables) | **Primary rule-based Indic↔Latin** (ITRANS, IAST, HK, Velthuis, ISO; Devanagari, Tamil, Kannada, Telugu, Bengali, Gujarati, Gurmukhi, Malayalam, Oriya). Has `detect.py`, `tamil_tools.py`, `deduplication.get_approx_deduplicating_key` (Sanskrit-oriented orthographic key). Scheme tables come from the `common_maps` submodule (MIT). Install plain `indic-transliteration`, **not** `[extras]`, because the `extras` extra pulls in AGPL Aksharamukha |
| indic_nlp_library | https://github.com/anoopkunchukuttan/indic_nlp_library | MIT | 648 | 2024-06-07 | None | Script conversion between Indic scripts (Unicode offset), normalizers (nukta/anusvara), ITRANS romanization |
| Aksharamukha | https://github.com/virtualvinodh/aksharamukha | **AGPL-3.0** per README and PyPI classifier; the repo's only license file is `gpl-3.0.txt` and the GitHub API detects no license. Copyleft either way | 220 | 2025-03-25 | None | 120 scripts, 21 romanizations. AGPL is a software-license concern; prefer the MIT tools |
| uroman | https://github.com/isi-nlp/uroman | **Custom MIT-like** (adds a publication acknowledgement clause; PyPI classifier says Apache) | 251 | 2024-07-26 | None | Universal any-script→Latin fallback for unseen scripts (Python v1.3.1.1) |
| PyICU (ICU) | https://pyicu.org (PyPI `PyICU`) | MIT (PyPI); ICU is under the Unicode license | n/a (moved off GitHub) | PyPI 2.16.2 | Low: CLDR transliteration rule data | `Transliterator.createInstance("Any-Latin; Latin-ASCII")`: fast C++ open-set romanization (ISO 15919 for Indic) |
| anyascii | https://github.com/anyascii/anyascii | ISC | 424 | 2026-06-06 | None | Permissive alternative to Unidecode for ASCII folding |
| Unidecode | https://github.com/avian2/unidecode | **GPL-2.0** | 609 | 2026-01-05 | None | Avoid (GPL); use anyascii/ICU |
| RapidFuzz | https://github.com/rapidfuzz/RapidFuzz | MIT | 4,139 | 2026-09-12 | None | Vectorized JW/Levenshtein/token-set ratios (`process.cdist`) for candidate scoring |
| jellyfish | https://github.com/jamesturk/jellyfish | MIT | 2,233 | 2026-07-24 | None | Soundex/Metaphone/NYSIIS/Match Rating (English-centric, a weak feature only) |
| abydos | https://github.com/chrislit/abydos | **GPL-3.0** | 194 | 2022-11-10 | None | Many phonetic algorithms. GPL, so avoid in the shipped code |
| sparse_dot_topn | https://github.com/ing-bank/sparse_dot_topn | Apache-2.0 | 424 | 2026-09-14 | None | Top-n sparse cosine for char n-gram TF-IDF blocking at 10M scale |
| name_matching (DNB) | https://github.com/DeNederlandscheBank/name_matching | MIT | 169 | 2026-07-02 | None | Reference implementation of company-name matching (TF-IDF preselection, multiple distance metrics) |
| Splink | https://github.com/moj-analytical-services/splink | MIT | 2,429 | 2026-09-22 | None | Fellegi-Sunter plus term-frequency adjustment, DuckDB/Spark backends; a baseline and a feature-engineering reference |
| wordninja | https://github.com/keredson/wordninja | MIT | 875 | 2023-02-19 | **Med**: default English word list (provenance not stated in README) | Use **`LanguageModel(custom.txt.gz)` built from our own name vocabulary** to split "heassociates" |
| SymSpell / symspellpy | https://github.com/wolfgarbe/SymSpell / https://github.com/mammothb/symspellpy | MIT / MIT | 3,467 / 876 | 2026-07-04 / 2026-08-21 | **Med**: bundled frequency dictionaries are external | Use only with a dictionary built from training data |
| pycountry | https://github.com/pycountry/pycountry | LGPL-2.1 | 975 | 2026-09-21 | Med: ships ISO 3166/subdivision data | Avoid; learn state maps from data instead |
| pgeocode | https://github.com/symerio/pgeocode | BSD-3 | 270 | 2026-07-06 | **VIOLATES**: downloads GeoNames postal data at runtime | Do not use |
| AutomaticFuzzyJoin | https://github.com/chu-data-lab/AutomaticFuzzyJoin | **MIT** text in a misspelled `LISENCE` file (boilerplate "Python Packaging Authority" copyright) plus an MIT classifier in `setup.py`; the GitHub API reports none because of the misspelling | 26 | 2021-09-09 | None | Reference code for Auto-FuzzyJoin. Effectively MIT, but the copyright line is boilerplate, so keep attribution if code is reused |
| semantic_address_matching | https://github.com/linyue-gis/semantic_address_matching (old URL linyuehzzz/... redirects) | MIT | 37 | 2020-02-09 | None | ESIM address-matching reference (Lin et al. 2020) |
| IndicXlit | https://github.com/AI4Bharat/IndicXlit | MIT (code and models) | 142 | 2023-10-13 | Model trained on Aksharantar (pretrained weights are allowed) | Neural Indic↔Latin, see §4 |
| IndicTrans2 | https://github.com/AI4Bharat/IndicTrans2 | MIT | 475 | 2025-10-03 | pretrained weights | Translation, which risks translating names. Low priority |

External reference **lists** (flag): the GLEIF ISO 20275 ELF code list (3,600+ legal forms in 200+
jurisdictions, v1.6 Feb 2026; the GLEIF page states no license), USPS Pub 28, and the libpostal dictionaries. These
are *external knowledge*. Under the "no external data" rule the safe position is to **derive our lists from training
and test data** and use these lists only to sanity-check offline. Ask the organizers before shipping them.

---

## 4. Models (Hugging Face card license checked via huggingface.co/api/models/<id>)

| HF id | License (card) | Compliant? | Params | Kind | Use |
|---|---|---|---|---|---|
| ai4bharat/IndicXlit | mit | Yes | ~11M (GitHub README: 6 enc/dec layers, 256-d, 4 heads) | fairseq transformer, en↔21 Indic, both directions (`indicxlit-indic-en-v1.0/transformer/indicxlit.pt`) | Top-k romanization of non-Latin name tokens, **on cloud GPU**. Indic→En top-1 on **named entities**: hin 59.22, ben 52.54, guj 50.56, kan 60.77, mal 45.23, mar 56.76, tam 35.46, tel 57.57, pan 48.00, urd 23.14 (README). **Never exact-match its output** |
| ai4bharat/indictrans2-indic-en-dist-200M | mit | Yes (HF repo is **gated**, auto-approve: accept access conditions first) | 228,316,160 (safetensors) | NMT Indic→En | Low priority (translates, does not transliterate) |
| ai4bharat/indictrans2-indic-en-1B | mit | Yes (HF repo is **gated**, auto-approve) | 1,023,006,720 (safetensors) | NMT | Low priority |
| google/muril-base-cased | apache-2.0 | Yes | not stated on card | BERT, 17 Indic languages + transliterated pairs | Cross-script name encoder, fine-tuned Siamese on train positives |
| google/muril-large-cased | **no license in card metadata and none in README text** | **No / UNVERIFIED** | BERT-large (24L) per README | — | Do not use unless a license is confirmed |
| ai4bharat/IndicBERTv2-MLM-Sam-TLM | mit | Yes | not stated | Indic BERT with TLM | Alternative Indic encoder |
| google/byt5-small, google/byt5-base | apache-2.0 | Yes | not stated in API | byte-level T5 | Train our own translit/normalizer on train cross-script pairs |
| google/canine-s | apache-2.0 | Yes | 132,099,328 | char encoder | Char-level Siamese name matcher |
| sentence-transformers/LaBSE | apache-2.0 | Yes | 470,927,360 | multilingual sentence encoder | Cross-script embedding feature |
| intfloat/multilingual-e5-small / -base / -large | mit | Yes | small 117,654,272; base 278,044,162; large 559,890,946 | multilingual embeddings | Dense blocking on romanized strings |
| BAAI/bge-m3 | mit | Yes | not stated on card | dense+sparse+multi-vector | Hybrid retrieval |
| Alibaba-NLP/gte-multilingual-base | apache-2.0 | Yes | 305,369,089 | multilingual embeddings | Dense blocking alternative. Needs `trust_remote_code`: its `config.json` loads code from `Alibaba-NLP/new-impl` (also apache-2.0 on HF), so pin that revision |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0 | Yes | 117,654,272 | small multilingual encoder | Cheap dense feature |
| libpostal CRF parser (not on HF) | MIT code, **ODbL/CC-BY training data** | **RISK** | ~1.8-2.2 GB | CRF | Do not ship |
| deepparse pretrained weights (not on HF) | **LGPL-3.0** | **No** | — | RNN seq2seq | Do not ship |

---

## 5. Concrete designs for this challenge

### 5.1 Script-aware Unicode pipeline (P0, CPU cloud)
1. `unicodedata.normalize("NFKC", s)`, then casefold and collapse whitespace.
2. Tokenize and give each token a **script** using the `regex` module `\p{Script=...}` (UAX #24).
   `Common`/`Inherited` characters (digits, punctuation, ZWJ/ZWNJ, combining marks) take the script of the
   neighbouring letters. Tag each record `scripts = {Latin, Devanagari, Tamil, ...}` and `is_mixed`.
3. Remove diacritics **only for Latin tokens**: NFKD, then drop `Mn` marks whose base is Latin, then NFC.
   *Never* apply this to Indic tokens, because matras, virama (्), nukta (़) and anusvara (ं) are combining marks.
4. Strip junk: leading/trailing runs of `[^\p{L}\p{N}]+`, repeated punctuation, and learned null tokens (see 5.4).

### 5.2 Romanization and Indic phonetic keys (P0)
- Romanize each non-Latin token with one engine, applied consistently to S1, S2 and S3:
  - Indic: `indic_transliteration.sanscript.transliterate(tok, <script>, ISO/ITRANS)`, then ASCII fold. Or ICU
    `Any-Latin; Latin-ASCII`, which covers any script at C++ speed.
  - Other or unseen scripts: uroman or ICU `Any-Latin`.
- Build **three name fields** in parallel: `name_latin` (romanized, cleaned), `key_strict`, `key_loose`.
- `key_strict` (per token): aspirate collapse `kh gh ch jh th dh ph bh → k g c j t d p b`; `sh, ṣ, ś, s → s`;
  `z→j`, `f→p`, `q→k`, `w→v`; long vowels `aa/ee/ii/oo/uu → a/i/i/u/u`; `ai/ay→e`, `au/ow→o`;
  retroflex→dental (`ṭ ḍ ṇ ḷ → t d n l`); nasal before a consonant (`ṅ ñ ṇ ṃ ṁ m`) → `n` (ISO 15919 writes
  anusvara as `ṁ`, IAST as `ṃ`); collapse doubled letters
  (gemination such as Tamil `kk`, `ṭṭ`); drop the word-final inherent `a` (schwa) when the token came from Indic
  script.
- `key_loose` = consonant skeleton (keep the first letter, drop vowels) **plus a voicing collapse**
  (`g→k, j→c, d→t, b→p`). Tamil script does not mark voicing or aspiration, so rule-based Tamil romanization gives
  "kanesan" for Ganesan. Because this key is loose, use it only as a weighted feature and a secondary blocking key.
- Worked examples:
  - "मार्केटिंग" → ISO "mārkeṭiṁga" (anusvara ं → `ṁ` in indic_transliteration's `iso.toml`) → strict "marketing"
    and skeleton "mrktng", identical to "Marketing".
  - "प्राइवेट लिमिटेड" → "prāiveṭa limiṭeḍa" → "prvt lmtd", which is the Latin "Private Limited".
  - "महाराष्ट्र" → "mahārāṣṭra" → "mhrstr", identical to the key of "Maharashtra".
  - The skeletons above are shown *before* the voicing collapse. Applying it changes both sides the same way
    ("mrktng" → "mrktnk"; "lmtd" → "lmtt" → "lmt" after the gemination collapse), so the equalities still hold.
- **Validate on train.** On training positives where the S1 and S2 scripts differ, measure the key-equality rate
  and the JW distribution per key. Keep a key only if it lifts cross-script recall at a fixed FP rate.
- **Optional neural step (P1).** Run IndicXlit indic→en with beam 4-5 on unique non-Latin tokens only; unique tokens
  number far fewer than records. Take the max similarity over the beams. Run it on Modal/Kaggle GPU and cache it as
  a token→romanizations table.
- **Mine alignments from our own data (P2).** Sajjad-style EM (2012) over cross-script train positives gives
  char-level Indic→Latin substitutions that match *this* synthetic generator's romanization. This is internal data
  and probably beats generic rules if the generator uses a specific scheme.

### 5.3 Learned abbreviation, synonym and state maps (P0)
Adapted from Arasu 2009 (supervised) and Deng 2019 (unsupervised), per field (name, street, city, state):
1. For each train positive pair, align tokens. Remove exact matches, then pair the residual tokens by position and
   char similarity.
2. Candidate rule `a→b` if one of these holds: b is a prefix of a (St/Street), b is a subsequence of a with the
   same first letter (Pvt/Private, Blvd/Boulevard, Schwartz-Hearst style), the initials of an a-phrase equal b
   (acronyms), or the pair co-occurs frequently (state code↔full name, native-script↔Latin state).
3. Score each rule by support, confidence and **lift vs. random non-matching pairs from the same country**. Then
   select a concise rule set greedily (Arasu's MDL objective).
4. Canonicalize by union-find over accepted rules, mapping each cluster to its most frequent surface form.
5. **France (unseen):** run the same miner on **test pseudo-pairs**. These are high-precision matches (same
   postcode, name JW ≥ 0.95, and the other fields agreeing). Iterate 2-3 rounds, as in Deng 2019 and the
   Auto-FuzzyJoin precision target. Expected discoveries: `r.`/`rue`, `av`/`avenue`, `bd`/`boulevard`,
   `pl`/`place`, `imp`/`impasse`, `ch`/`chemin`, `sarl`/`s.a.r.l.`, `st`/`saint`, `ste`/`sainte`.
   **This needs no external data.**
6. Apply rules at query time as *expansions* (Lu et al. SIGMOD 2013), not destructive rewrites. Keep the original
   string as well.

### 5.4 Stop tokens and weights learned per country (P0)
- IDF per (country, field) over S1∪S2∪S3, fitted transductively on test with no labels. Fall back to global IDF
  when a country has fewer than N records. Legal suffixes, street types, "near", "null", "india" and "usa" drop
  automatically.
- Null-like tokens: tokens that often form an *entire* address component and have near-uniform co-occurrence
  (`null`, `none`, `na`, `n/a`, `-`). Treat as missing.
- Landmark triggers: tokens that start phrases whose tokens are *usually absent* from the matched counterpart
  (learned on train IN pairs: `near`, `opp`, `opposite`, `behind`, `beside`, `next to` and their Hindi
  equivalents). Extract into `landmark` and compare softly. Missing is neutral.

### 5.5 Legal form, DBA, domains and acronyms: name variants (P0/P1)
Following Loster et al. EDBT 2017 and Kruse et al. BIS 2021:
- `legal_form`: find a learned legal-term token or n-gram **anywhere** in the name (boundary position preferred).
  Map it to a canonical class via the learned rules (pvt ltd = private limited = pvt. ltd.; llc = l.l.c.;
  sarl = s.a.r.l.). Remove it from the core name. Features: `lf_equal`, `lf_conflict`, `lf_missing_one_side`.
- DBA: split on learned markers (`dba`, `d/b/a`, `doing business as`, `t/a`, `trading as`, `aka`). Each side
  becomes a variant, and the score is the max over variant pairs.
- Domain names: a regex for `^[a-z0-9-]+(\.[a-z]{2,}){1,2}$`, with the TLD set learned from data. Build
  `compact = core_name.replace(" ", "")` and `domain_stem`. Features: equality, containment, JW(compact, stem).
  Optionally segment the stem with a wordninja **custom LM built from our S1 name unigram counts**.
- Acronym key: the first letters of core tokens (after dropping learned low-IDF tokens), to catch "ABC Pvt Ltd" vs
  "Anand Brothers Corporation".
- Mixed-script names: romanize per token (5.2), so "Sun पावर Provision" → "sun pavar provision". The phonetic key
  then matches "Sun Power Provision" through `w→v` and the vowel rules.

### 5.6 Address features (P0/P1)
- Postcode detection: for each country, learn the mode digit-length of the numeric token in the postcode position
  (US 5, IN 6, FR 5), so it works for any country. Features: `pc_exact`, `pc_prefix3` (IN sorting district, US
  SCF), `pc_prefix2` (FR département), `pc_missing`. Use postcode or its prefix as a **blocking key** together with
  name keys.
- Number agreement: Jaccard of the digit tokens (house numbers, unit numbers) excluding the postcode. Treat
  bis/ter/A-D suffixes as part of the house number.
- Order-insensitive similarity: token-set ratio, Monge-Elkan(JW) and SoftTFIDF on the canonicalized address
  (reordering like "OH, Columbus, 5559 Orville Avenue" is common).
- State/region: canonicalized via the learned map, then equality or conflict. City: JW after rules.
- Skip heavy address parsers. libpostal and deepparse are compliance risks, and our records already have separate
  address and country fields. A light, learned component tagger (digits, postcode, state and city lexicons *learned
  from S1*) is enough.

### 5.7 Scale
- About 16M normalized strings in total. All steps are vectorizable: polars/pyarrow string ops, PyICU in
  `multiprocessing` over 64 cores, and romanization cached **per unique token**. Unique tokens are probably
  10^5-10^6, which is an assumption to measure.
- Char 3-gram TF-IDF + `sparse_dot_topn` (Apache-2.0) gives top-k blocking per country block. Combine it with
  postcode-prefix and key-equality blocks.

---

## 6. What we should do (prioritized)

**P0 (do first, all CPU cloud):**
1. Build the script-aware normalizer (5.1). Unit-test it on the noise catalogue, especially that Devanagari/Tamil
   survive diacritic stripping.
2. Romanize non-Latin tokens (ICU or indic_transliteration) and add `name_latin`, `key_strict` and `key_loose`
   (5.2). Measure cross-script positive recall on train by key.
3. Mine abbreviation, synonym and state rules from train positives (5.3). Apply them as expansions.
4. IDF per country and field. Learn null tokens (5.4).
5. Legal form as a separate attribute with conflict/missing features (5.5).
6. Postcode detection with learned length, prefix blocking keys and features (5.6).

**P1:**
7. Name-variant generation: DBA split, domain stem vs compact key, acronym key. Score = max over variants.
8. **Self-training rule mining on test for France** (and any unseen country), using a precision-targeted
   pseudo-label threshold (Deng 2019, Auto-FuzzyJoin).
9. IndicXlit top-k romanization for unique non-Latin tokens on a cloud GPU. Keep it only if it adds cross-script
   recall over the rule-based keys on train.
10. Landmark extraction as a soft field.

**P2:**
11. A Siamese char-level encoder (CANINE-s, ByT5-small or MuRIL-base, all Apache-2.0) trained on train positive
    pairs, especially cross-script ones. Use it as a feature or a dense blocker. Alternatively, embed
    **romanized** strings with multilingual-e5 or LaBSE (RomanSetu: romanized embeddings align better).
12. Char-alignment mining (Sajjad EM) from cross-script train positives to learn the generator's romanization
    scheme.
13. A wordninja custom LM for concatenated names.

**P3:**
14. libpostal dictionaries and USPS Pub 28 **offline only**, to sanity-check the mined rules. Do not ship without
    organizer clearance.
15. LLM parsing or labelling of hard residual cases, on a small offline sample in the cloud only.
16. Error analysis stratified by name visibility (CorpFam): identical after normalization, shared distinctive
    token, or no shared token.

**Do not use:** libpostal `parse_address` or the language classifier (OSM/ODbL-trained), deepparse weights (LGPL
and OSM-derived), pgeocode (GeoNames download), pycountry data, GLEIF ELF list or any gazetteer in the final
pipeline. Avoid GPL/AGPL code (Unidecode, abydos, Aksharamukha) in shipped code when MIT/ISC alternatives exist.
Do not use muril-large-cased (no license metadata), or any Llama-based checkpoint such as the SGER model of
Chourasia et al. (Llama 3 8B, license `llama3`). Treat the pretrained **usaddress** model as a data-provenance
risk too (its training folder includes OSM/OpenAddresses-derived files); ask the organizers before shipping it.

---

## 7. Open questions to measure on cloud (cheap checks)
- What share of train positives are cross-script, split by script? What share are mixed-script?
- Does the synthetic generator use one fixed romanization scheme? Compare Latin S1 names with ISO/ITRANS output of
  the native S2 name. If one scheme dominates, mine it exactly (P2 #12) and skip IndicXlit.
- How often does the legal form differ between true matches? If the generator swaps legal forms freely, downgrade
  `lf_conflict`.
- What is the postcode presence rate per source and country? How often is the postcode missing or wrong in S2/S3?
- What is the distribution of landmark phrases in IN addresses, and how often do they differ between true matches?

---

## Audit log

Independent audit on 2026-09-25. Every item was treated as possibly hallucinated until confirmed on a primary source.
Only HTTP metadata lookups and PDF text extraction ran locally. No model was run.

### How items were checked
- **arXiv (29 IDs):** a batch query to the arXiv API (`export.arxiv.org/api/query?id_list=...`). Title, authors,
  comments/journal-ref/DOI and abstract were compared with the notes. All 29 exist and their titles and authors match.
- **ACL Anthology (14 papers):** `.bib` and abstract for W16-1617, 2021.findings-emnlp.85, 2023.findings-emnlp.4,
  2020.lrec-1.294, P18-4003, 2024.acl-long.833, 2022.tacl-1.17, 2022.tacl-1.5, P12-1049 (PDF abstract),
  2020.lrec-1.399, 2024.lrec-main.1103, 2022.icon-main.33, 2026.acl-industry.101, 2023.nlposs-1.3. Page ranges match.
- **Crossref DOI metadata:** Arasu (PVLDB 2(1) 514-525), Lu (SIGMOD 2013 373-384), Christen (ICDMW 2006 290-294),
  Churches (BMC MIDM 2), Kruse (BIS 2021 13-26), Lin (IJGIS 34(3) 559-576), Comber (TGIS 23(2) 334-348), Splink (IJPDS
  7(3)), Auto-FuzzyJoin (SIGMOD 2021 1064-1076), Peeters (WWW'22 Companion 236-238), Ditto (PVLDB 14(1) 50-60), Basile
  (ESWA 238, 122035), Deng (ICDE 2019 196-207), Place Dedup (WWW 2019 3420-3426), Duarte (EPIA 2023, LNCS 413-425).
  The Loster EDBT DOI resolves (doi.org 302) to the OpenProceedings PDF.
- **PDF text read:** Arasu PVLDB 2009 (Fig. 2 rule list, NP-hard/greedy/MDL confirmed), Loster EDBT 2017 (P 91.11%,
  R 78.82%, five alias steps, max nine aliases, Wikipedia-derived legal-form regexes), Cohen 2003 (TF-IDF + Jaro-Winkler
  best; IIWeb 2003 pp. 73-78 per ACM DL and DBLP search listing).
- **Other primary pages:** PSB 2003 abstract (96%P/82%R and 95%P/82%R), TIB-OP BIS page (hybrid best), IJPDS page,
  Europe PMC (Churches result sentences), UAX #24 (Unicode 18.0.0, Rev. 41, 2026-08-28), ICU transforms guide, USPS
  Pub 28 C1 (AVENUE: AV/AVEN/AVENU/AVN/AVNUE to AVE), AFNOR NF Z10-011 page (January 2013), GLEIF ELF page (v1.6, Feb
  2026, 3,600+ forms, 200+ jurisdictions, no license on the page), Wikipedia PIN page (secondary, zones as listed).
  For Lu et al. (SIGMOD 2013), "selective-expansion" and "SI-tree" were confirmed from the ACM DL abstract via a
  search listing (the ACM page returned 403 to direct fetch).
- **Hugging Face (16 models):** `huggingface.co/api/models/<id>` `license:` tag, `cardData.license`, `safetensors.total`
  and `gated`. All license strings in §4 match the card metadata exactly.
- **GitHub (28 repos):** `gh api repos/<owner>/<repo>` for `license.spdx_id`, `stargazers_count`, `pushed_at`, plus
  reading the LICENSE/README/source files where the claim depended on them (cleanco `termdata.py`/`clean.py`,
  libpostal README and `resources/dictionaries`, deepparse and deepparse-address-data READMEs, pgeocode README, usaddress
  `training/`, indic_transliteration package files, uroman `LICENSE.txt`, Aksharamukha README and root files,
  AutomaticFuzzyJoin `LISENCE`, IndicXlit README tables). PyPI JSON was checked for Aksharamukha, uroman and PyICU.

### What was changed and why
| Item | Problem found | Fix |
|---|---|---|
| Paper 36 Deepparse | Venue given as "NLP-OSS @ EMNLP 2024" (copied from the arXiv comment). The ACL Anthology lists it in **NLP-OSS 2023** (Singapore, Dec 2023), pp. 19-24, 2023.nlposs-1.3 | Venue corrected, Anthology URL added |
| Paper 31 Chourasia et al. | "ACL 2026 (accepted)" was imprecise. It is the **ACL 2026 Industry Track** (Vol. 6), pp. 1461-1468. The paper is about **person names** on KYC data, not business names | Venue and scope corrected |
| Paper 45 Peeters & Bizer | The published WWW'22 Companion title is "Cross-Language Learning for **Product** Matching" (pp. 236-238); "Entity Matching" is only the arXiv title | Title corrected, arXiv title noted |
| Paper 19 IndicTrans2 | "13 authors" is wrong. arXiv lists **14** | Corrected |
| Paper 18 Aksharantar | The claim "named-entity accuracy is much lower than on native words" is false for Indic→En. The README shows NE > native for ben (52.54 vs 11.76), guj, hin, mar, pan | Statement corrected |
| Paper 48 MESSY STREETS | "about 25 pp" should be "up to 25 pp" per the abstract | Corrected |
| Paper 44 Splink | The IJPDS abstract mentions EM, Fellegi-Sunter and Spark, but not term-frequency adjustments or DuckDB | Attributed TF/DuckDB to the Splink docs (verified) |
| Papers 1, 2, 5, 9, 11, 14, 34, 41, 42 | Missing page ranges, DOIs or full titles (Kruse subtitle) | Added from Crossref, ACM and arXiv metadata |
| Paper 6 Christen | Key finding omitted | Added "no clear best matching technique" (abstract) |
| Paper 22 RomanSetu | Author names were garbled | Now gives the Anthology form (Jaavid J, Aswanth M) and the arXiv form |
| TL;DR 4 and cleanco row | The claim that cleanco lacks `private limited` and `llp` was misleading. The *India country list* is only `ltd., pse, psu, pvt. ltd.`, but `basename()` uses the union of all terms, which includes `private`, `limited`, `llp`, `llc`, `sa`, `co`, `company`. The France list also has more entries than stated | Rewritten with the verified term lists |
| libpostal row | Wrong line counts: fr `street_types.txt` has **165** lines (not 226) and en `company_types.txt` has **61** (not 72) on master. hi `street_types.txt` = 4 is correct. No ta/kn/te/mr/bn/gu/ml directories is correct | Counts corrected; fr `company_types.txt` added (first given as 39; the third audit below found **40** and corrected it) |
| usaddress row | Rated "Low-Med", but `training/` contains OSM/OpenAddresses-derived files | Raised to **Med** data risk; added to the "Do not use / ask organizers" note |
| AutomaticFuzzyJoin row | Listed as "no license", but the repo has MIT text in a misspelled `LISENCE` file and an MIT classifier in `setup.py` | Corrected to MIT (with the boilerplate-copyright caveat) |
| semantic_address_matching row | Stars and activity were "n/a", and the repo has moved | Now linyue-gis/semantic_address_matching, MIT, 37 stars, last push 2020-02-09 |
| Aksharamukha row | The license nuance was missing | README/PyPI say AGPL-3.0, the repo file is `gpl-3.0.txt`, and the GitHub API detects none |
| IndicTrans2 models | Params for 1B were "~1.1B (card)", and gating was not mentioned | 1,023,006,720 params (safetensors). Both HF repos are **gated (auto)**, so access conditions must be accepted on HF (user action) |
| multilingual-e5-small | Params "not read" | 117,654,272 (safetensors) |
| muril-large-cased | — | Confirmed: no license tag and no license text in the README. It stays non-compliant |
| ICU quote | Paraphrased as a direct quote | Now the exact phrase "script transliteration is not translation" |

### Confirmed without change (selection)
- All 16 HF license strings (mit / apache-2.0; muril-large none). Param counts: indictrans2-dist-200M 228,316,160;
  canine-s 132,099,328; LaBSE 470,927,360; e5-base 278,044,162; e5-large 559,890,946; gte-multilingual-base 305,369,089;
  paraphrase-multilingual-MiniLM-L12-v2 117,654,272. IndicXlit is ~11M params (6 layers, 256-d, 4 heads, MIT) per the
  GitHub README. The HF checkpoint `indicxlit-indic-en-v1.0/transformer/indicxlit.pt` exists.
- IndicXlit Indic→En named-entity top-1 values match the README exactly: hin 59.22, ben 52.54, guj 50.56, kan 60.77,
  mal 45.23, mar 56.76, tam 35.46, tel 57.57, pan 48.00, urd 23.14.
- The license, star count and last push of all other GitHub repos match the notes exactly: cleanco, libpostal, pypostal,
  deepparse (LGPL-3.0), deepparse-address-data, usaddress, indic_transliteration, indic_nlp_library, uroman
  (NOASSERTION: MIT text plus an acknowledgement clause; PyPI classifier Apache; v1.3.1.1), anyascii (ISC), Unidecode
  (GPL-2.0), RapidFuzz, jellyfish, abydos (GPL-3.0), sparse_dot_topn (Apache-2.0), name_matching, Splink, wordninja,
  SymSpell, symspellpy, pycountry (LGPL-2.1), pgeocode (BSD-3-Clause; README confirms a GeoNames download), IndicXlit,
  IndicTrans2. PyICU: PyPI license "MIT", v2.16.2, GitHub mirror ovalhub/pyicu archived, homepage gitlab.pyicu.org.
- libpostal: the README confirms training on OSM (ODbL), OpenAddresses (various, mostly CC-BY) and GeoPlanet (CC-BY),
  99.45% held-out accuracy, and a 1.8 GB default / 2.2 GB Senzing model. deepparse-address-data is derived from libpostal
  (20 training countries incl. France and the US; India only in the zero-shot set).
- Numbers confirmed on the source: Deng 75%R/99.5%P/17,497/100 questions; Ditto 96.5% F1 on 789K/412K; Gschwind 91% vs
  73%; CompanyName2Vec 89.3%; Arimond 1.1M entities/30 jurisdictions; Rustogi 80% landmark-relative, 50-1500 m;
  Duarte >95% door-level, ~4.5x faster than BM25; Ganesan 74.8% more PoIs, median F 0.69; Yassine 2020 ~99%, 33/41;
  Mangalgi ~90%; RomanSetu 2x-4x fertility; CANINE +5.7 F1 on TyDi QA; TRANSLIT ~1.6M entries/180+ languages/~3M
  variants/92%; ParaNames 140M/400+/16.8M; Symphonym R@1 85.2%, MRR 90.8%, 128-d, 20 writing systems; CorpFam 93.2%;
  Chourasia 99.02%/F1 0.994/50K; Schwartz-Hearst 96/82 and 95/82; Loster 91.11/78.82.

### Re-verification pass (2026-09-25, 06:4x IST, after a session resume)
A second, independent metadata pass re-ran the checks above from scratch:
- arXiv API batch (28 IDs): all exist, and titles, author lists and comments match (IndicTrans2 has 14 authors,
  Chourasia's journal-ref is ACL 2026 Vol. 6 Industry Track pp. 1461-1468, and Deepparse's arXiv comment says
  "EMNLP 2024 NLP-OSS" while the Anthology bib says NLP-OSS 2023, Singapore, pp. 19-24).
- ACL Anthology abstract for 2026.acl-industry.101: it covers person names, KYC, Dream11, and 99.02% / F1 0.994 on
  50,000 pairs. Confirmed.
- Crossref: Deng ICDE 2019 196-207; Arasu PVLDB 2 514-525; Peeters "Cross-Language Learning for Product Matching",
  236-238; Basile ESWA 238, 122035; Duarte LNCS 413-425; Yang WWW 2019 3420-3426; Ditto PVLDB 14(1) 50-60 (issued
  2020-09). All match.
- HF API (16 models): all license tags, parameter counts and gating flags match §4 exactly.
- GitHub API (28 repos): all license SPDX ids and last-push dates match. Stars differ by at most 1 (RapidFuzz 4,139,
  Splink 2,429, wordninja 875). AutomaticFuzzyJoin still has the misspelled `LISENCE` file. The ovalhub/pyicu mirror is
  archived.
No further corrections were needed.

### Nothing deleted
Every listed paper, repo and model was confirmed to exist. No item had to be removed or marked UNVERIFIED beyond what
the notes already flagged (the AFNOR line-format details and Wikipedia postcode facts remain labelled as secondary
sources, and the muril-large-cased license stays UNVERIFIED).

### Third independent audit (2026-09-25, from scratch)
This audit treated every item as possibly hallucinated, including the claims made by the two passes above. Only HTTP
metadata lookups and PDF text extraction ran locally. No model was run and nothing was installed.

**How items were checked**
- **arXiv API, one batch of 29 IDs:** all 29 exist. Titles, full author lists, comments, journal-refs and DOIs match
  the table. Every quoted number was compared against the abstract text and all match: Deng 75%/99.5%/17,497/100;
  Ditto 96.5% F1 on 789K/412K; Gschwind 91% vs 73%; CompanyName2Vec 89.3%; Arimond 1.1M/30; Rustogi 80%/50-1500 m;
  Duarte >95%/~4.5x; Ganesan 74.8%/median F 0.69 after OSM post-processing; Yassine 2020 ~99%/33 of 41; Mangalgi ~90%;
  RomanSetu 2x-4x; Symphonym 85.2/90.8/128-d/20 writing systems; CorpFam 93.2%; Chourasia 99.02%/0.994/50,000;
  MESSY STREETS "up to 25 percentage points"; Deepparse LGPL-3.0/60+ countries/99%; ParaNames 140M/400+/16.8M;
  IndicTrans2 14 authors, "Accepted at TMLR"; MuRIL 14 authors.
- **ACL Anthology `.bib` files and abstracts (14 papers):** titles, authors, venues, pages and DOIs all match.
  Chourasia is in Vol. 6 (Industry Track) pp. 1461-1468. Deepparse is NLP-OSS 2023 pp. 19-24. CANINE +5.7 F1,
  TRANSLIT ~1.6M/180+/~3M/92%, Dakshina 12 languages, ByT5 noise robustness and Yadav's "no universal protocol" were
  all read in the abstracts. Sajjad (EM training) and Neculoiu (char BiLSTM Siamese, job titles) were read in the PDF
  first pages.
- **Crossref (18 DOIs):** Arasu, Deng, Lu, Christen, Churches, Kruse, Lin, Comber, Splink, Auto-FuzzyJoin (pp.
  1064-1076), Peeters ("...for Product Matching", pp. 236-238), Ditto (PVLDB 14(1) 50-60), Basile (ESWA 238, 122035),
  Place Dedup (pp. 3420-3426), Duarte (LNCS pp. 413-425), Yassine CiSt (pp. 353-360) and Singh (LNCS pp. 371-381).
  All exist and match. The Loster EDBT DOI is not in Crossref, but doi.org redirects it (302) to the OpenProceedings
  PDF, whose first page carries DOI 10.5441/002/edbt.2017.82 and page number 610. The PDF is 10 pages long, so
  pp. 610-619 is confirmed.
- **PDF text read:** Arasu (Fig. 2 org-name rules, NP-hard, greedy, MDL), Loster (91.11/78.82, five steps, at most
  nine aliases, legal-form regexes derived from Wikipedia), Cohen (TF-IDF + Jaro-Winkler best, JW as the secondary
  distance with θ = 0.9) and Place Dedup (names and addresses as attributes, kNN candidate fetch).
- **Other primary pages:** TIB-OP (Kruse's full subtitle, pp. 13-26, hybrid approach best), PSB abstract (96/82 and
  95/82, no training data), Europe PMC (Churches' HMM result sentences), IJPDS (Splink "downloaded 2 million times",
  EM, Fellegi-Sunter, Spark), the Splink docs (the TF adjustments page and DuckDB), Semantic Scholar API abstracts
  (Christen's "no clear best matching technique" and Lin et al.'s 0.97), and a web-search listing of the ACM DL for
  Lu et al.'s selective expansion and SI-tree. Also checked: UAX #24 (Unicode 18.0.0, Rev. 41, 2026-08-28), the ICU
  transforms guide (the exact quote, Inter-Indic, ISO 15919, "Any-Latin; Latin-ASCII"), USPS Pub 28 C1, the AFNOR
  product page (January 2013, 47 pp.), the GLEIF ELF page (v1.6, February 2026, 3,600+ forms, 200+ jurisdictions,
  no license text) and the Wikipedia PIN, ZIP and France postcode pages, which are secondary and match the notes.
- **Cohen 2003 pages:** DBLP served a bot-check page, so "pp. 73-78" is confirmed only through the search-engine
  listing of the DBLP record conf/ijcai/CohenRF03 and ACM DL 10.5555/3104278.3104293. It is kept, with that caveat.
- **Hugging Face API (16 models):** the `license:` tag and `cardData.license` match §4 exactly for all 16:
  mit ×8 (IndicXlit, both IndicTrans2 models, IndicBERTv2, e5 small/base/large, bge-m3) and apache-2.0 ×7 (muril-base,
  byt5-small/base, canine-s, LaBSE, gte-multilingual-base, paraphrase-multilingual-MiniLM). That is 15; the 16th is
  muril-large, which has no license tag, no cardData license and no license text in its README. The safetensors
  parameter counts and `gated: auto` for both IndicTrans2 repos match. The IndicXlit HF repo holds both
  `indicxlit-indic-en-v1.0/transformer/indicxlit.pt` and `indicxlit-en-indic-v1.0/transformer/indicxlit.pt`.
- **GitHub API (28 repos) and file reads:** every license SPDX id and last-push date matches. Aksharamukha and
  AutomaticFuzzyJoin show NONE and uroman shows NOASSERTION, each explained in the table. The following were read
  directly:
  - cleanco `termdata.py` and `clean.py`: the India and France lists are verbatim; the union of terms contains
    private/limited/ltd/llp/llc/sa/co/company but not opc, pvt or "private limited"; `custom_basename` defaults to
    `suffix=True, prefix=False, middle=False`; the README credits Wikipedia's "Types of business entity" article.
  - libpostal: the README (OSM ODbL, OpenAddresses mostly CC-BY, GeoPlanet CC-BY, 99.45%, 1.8 GB vs 2.2 GB Senzing)
    and exact dictionary line counts.
  - deepparse-address-data README: derived from libpostal; 20 training countries including France and the US; India
    in the zero-shot set.
  - usaddress: `training/` holds the OSM and OpenAddresses XML files; `usaddr.crfsuite` ships as package data.
  - uroman: `LICENSE.txt` is MIT text plus an acknowledgement clause; the PyPI classifier says Apache; v1.3.1.1.
  - Aksharamukha: README and PyPI say AGPL 3.0; the root file is `gpl-3.0.txt`; the README claims 120 scripts and
    21 romanizations.
  - AutomaticFuzzyJoin: MIT text in `LISENCE`, and an MIT classifier in `setup.py`.
  - IndicXlit README: ~11M parameters, 6 layers, 256-d, 4 heads, MIT; all ten Indic→En named-entity values match.
  - Also read: pgeocode's GeoNames `DOWNLOAD_URL`, wordninja's custom `LanguageModel`, the bundled symspellpy
    frequency dictionaries, pycountry's `databases/`, name_matching (char n-gram TF-IDF cosine top-n, then fuzzy
    metrics), and PyPI for PyICU (MIT, 2.16.2, gitlab.pyicu.org).

**What this audit changed and why**
| Item | Problem found | Fix |
|---|---|---|
| libpostal row (§3) | fr `company_types.txt` has **40** lines on master (counted with `splitlines()`; there is no trailing newline, so `wc -l` gives 39) | 39 → 40, and the earlier audit-log row annotated |
| USPS Pub 28 (§2 standards) | "about 230 street suffixes" is not supported. The C1 table has 207 primary suffix names mapping to 203 distinct standard abbreviations | Corrected, with the full AVENUE variant list. *(The fourth audit found 206 / 202: the count of 207 / 203 included an empty trailing table row.)* |
| Worked example (§5.2) | indic_transliteration's ISO scheme (`common_maps/roman/iso.toml`) maps anusvara ं to **ṁ**, not ṅ, so "मार्केटिंग" → "mārkeṭiṁga". The strict-key nasal set also lacked ṁ | Example corrected; `ṁ` added to the nasal-collapse set |
| Star counts (§3) | RapidFuzz 4,139, Splink 2,429, wordninja 875 today (the table was off by 1) | Updated |
| Paper 39 Lin et al. | "Not read (abstract unavailable)" in the summary. The abstract is available and gives concrete facts | Added word2vec + ESIM, Shenzhen Address Database (Chinese), P/R/F1 0.97 |
| Papers 32 and 34 | Missing bibliographic detail | Singh: LNCS 2023, pp. 371-381, DOI added. Yassine CiSt: pp. 353-360 |
| gte-multilingual-base (§4) | Not mentioned: loading needs `trust_remote_code` from `Alibaba-NLP/new-impl` | Noted; that repo is also apache-2.0 on HF |
| indic_transliteration (§3) | Not mentioned: the optional `extras` extra depends on AGPL `aksharamukha` | Advised installing without `[extras]` |

**Discrepancies between the researcher's JSON summary and this (already corrected) notes file.** Anyone who reuses
the JSON should use the notes values instead:
- RomanSetu authors: the Anthology gives "Jaavid J", the arXiv "Jaavid Aktar Husain".
- IndicTrans2 has 14 authors (the JSON says 13).
- Deepparse venue: NLP-OSS 2023 (the JSON says NLP-OSS @ EMNLP 2024).
- Peeters & Bizer's published title is "...for Product Matching".
- Chourasia et al. is the ACL 2026 Industry Track, and the paper is about person names (KYC), not business names.
- AutomaticFuzzyJoin is effectively MIT (misspelled `LISENCE`), not "none declared".
- libpostal fr `street_types` has 165 lines, not 226.
- multilingual-e5-small has 117,654,272 parameters, not "not read". IndicTrans2-1B has 1,023,006,720, not "~1.1B".
- semantic_address_matching moved to linyue-gis; MIT, 37 stars, last push 2020-02-09.

**Nothing deleted.** All 48 papers, 28 repos and 16 models exist and their metadata is now correct. Items still
labelled UNVERIFIED or secondary: the AFNOR line-format details, the Wikipedia postcode facts, the muril-large-cased
license (none declared, so non-compliant), and Cohen 2003's page range (confirmed only through a DBLP/ACM search
listing).

### Fourth independent audit (2026-09-25, from scratch, after "Try again")
This audit treated every item, and every claim made by the three passes above, as possibly hallucinated. Only HTTP
metadata lookups and text extraction from PDFs and READMEs ran locally. No model was run and nothing was installed.
DBLP served a bot-check page and was not bypassed. The ACM DL returned 403. IEEE Xplore is rendered by JavaScript,
so its publisher metadata was taken from Crossref instead.

**How items were checked**
- **arXiv API, one batch of 29 IDs.** All 29 exist, and the titles, full author lists, comments, journal-refs and
  DOIs match. Every quoted number was re-read in the abstracts and all match. The Deepparse arXiv comment reads
  "Accepted in EMNLP 2024 NLP-OSS workshop", but the Anthology bib says NLP-OSS 2023 (Singapore, Dec 2023,
  pp. 19-24), which is why the table cites the Anthology.
- **ACL Anthology `.bib` files (14 papers).** Titles, *full* author lists, venues, pages and DOIs all match. The
  abstracts were re-read for Dakshina (12 languages), uroman, ByT5 (noise, spelling and pronunciation), CANINE
  (+5.7 F1 TyDi QA, fewer parameters), TRANSLIT (~1.6M/180+/~3M/92%), Yadav ("no universal protocol") and Gorman
  (no dictionary-only resolution). The PDF first pages were read for Neculoiu (char BiLSTM Siamese, job titles) and
  Sajjad (language-pair independent, EM, mixture of transliteration and non-transliteration models). The full
  Chourasia PDF was read (8 pages).
- **Crossref (18 DOIs).** All exist and match the table: titles, venues, volumes, issues and pages. Crossref gives
  **Ilyas before Li** for Deng et al. (see the fixes below). The Loster DOI is not in Crossref, but doi.org returns a
  302 to the OpenProceedings PDF, whose first page shows DOI 10.5441/002/edbt.2017.82 and page 610 (10 pages).
- **PDF text re-read.** Arasu Fig. 2 verbatim: co→company, corp→corporation, assoc→associates *and*
  assoc→association, univ→university, inc→incorporated, ltd→limited, intl→international, mfg→manufacturing and
  bros→brothers, together with the NP-hard, greedy and MDL wording. Loster: 91.11/78.82, German texts, five steps,
  at most nine aliases, legal-form regexes derived from Wikipedia. Cohen: TF-IDF plus Jaro-Winkler best, SoftTFIDF
  with JW secondary and θ = 0.9. Place Dedup: names and addresses as attributes, multi-source imports, k-NN
  candidate fetch.
- **Other primary pages.** Checked: the PSB abstract (96/82, 95/82, no training data); TIB-OP (Kruse subtitle,
  hybrid best); Europe PMC (Churches HMM result sentences); IJPDS ("downloaded 2 million times", EM, Fellegi-Sunter,
  Spark); the Splink TF-adjustment docs and README (DuckDB/Spark); Semantic Scholar (Christen's "no clear best
  matching technique" and Lin et al.'s word2vec + ESIM, Shenzhen, 0.97); the Liverpool repository (Comber, TGIS
  23(2) 334-348); UAX #24 (Unicode 18.0.0, Rev. 41, 2026-08-28); the ICU transforms guide (exact quote, Inter-Indic,
  ISO 15919, "Any-Latin; Latin-ASCII"); the AFNOR page (January 2013, 47 pp.); GLEIF ELF (v1.6, February 2026,
  3,600+ forms, 200+ jurisdictions, no license text); and Wikipedia ZIP, PIN and France (secondary; zone list, SCF,
  Corsica "20", arrondissement and CEDEX all match). Lu et al.'s selective expansion and SI-tree are still confirmed
  only via a web-search listing of the ACM DL abstract. The Semantic Scholar record shows the abstract elided by the
  publisher.
- **Hugging Face API (16 models + `Alibaba-NLP/new-impl` + `meta-llama/Meta-Llama-3-8B-Instruct`).** The `license:`
  tags and `cardData.license` match §4 exactly: mit ×8, apache-2.0 ×7, and muril-large-cased with none (its README
  also has no license text). The safetensors counts match (dist-200M 228,316,160; 1B 1,023,006,720; canine-s
  132,099,328; LaBSE 470,927,360; e5 small/base/large 117,654,272 / 278,044,162 / 559,890,946; gte 305,369,089;
  MiniLM-L12 117,654,272). Both IndicTrans2 repos are `gated: auto`. The IndicXlit repo holds both `.pt` checkpoints.
  gte's `config.json` `auto_map` points to `Alibaba-NLP/new-impl` (apache-2.0). Llama 3 8B Instruct is `llama3`
  (gated: manual).
- **GitHub API (28 repos).** Every license SPDX id, star count and last-push date matches the §3 table *exactly*
  today (for example RapidFuzz 4,139, Splink 2,429, wordninja 875). The following files were re-read:
  - cleanco `termdata.py`/`clean.py`: India and France lists verbatim; the union holds private, limited, ltd, llp,
    llc, sa, co and company, but not opc, pvt or "private limited"; `custom_basename(suffix=True, prefix=False,
    middle=False)`; the README credits Wikipedia.
  - libpostal: fr has 28 files; `street_types` 165, fr `company_types` 40, en `company_types` 61 (line 13 = "doing
    business as|d / b / a|dba|d b a"), hi `street_types` 4; there are no ta/kn/te/mr/bn/gu/ml directories. The
    README confirms ODbL OSM, mostly CC-BY OpenAddresses and CC-BY GeoPlanet, 99.45%, and 1.8 GB vs 2.2 GB (Senzing).
  - deepparse-address-data: generated from libpostal; France and the US are in the 20 training countries; India is
    in the zero-shot set.
  - usaddress: `training/` holds `openaddress_us_ia_linn.xml`, `synthetic_osm_data_xml.xml` and
    `synthetic_clean_osm_data.xml`. `usaddr.crfsuite` is gitignored but declared as package data in `pyproject.toml`
    and loaded at import.
  - indic_transliteration: `detect.py`, `tamil_tools.py`, `deduplication.get_approx_deduplicating_key`; the
    `common_maps` submodule (MIT); `extras.txt` includes `aksharamukha`; `roman/iso.toml` maps anusvara to "ṁ".
  - uroman `LICENSE.txt`: MIT text plus an acknowledgement clause; PyPI 1.3.1.1 has an Apache classifier.
  - Aksharamukha: the README says "GNU AGPL 3.0", 120 scripts and 21 romanizations; the root file is `gpl-3.0.txt`;
    PyPI 2.3 says AGPL.
  - AutomaticFuzzyJoin: MIT text in `LISENCE` and an MIT classifier in `setup.py`.
  - pgeocode: `DOWNLOAD_URL` = download.geonames.org.
  - wordninja: `LanguageModel(word_file)`.
  - symspellpy: bundled `frequency_dictionary_en_82_765.txt`.
  - pycountry: `databases/`.
  - name_matching: TF-IDF char n-gram cosine top-n, then fuzzy metrics.
  - IndicXlit README: ~11M parameters, 6 layers, 256-d, 4 heads, MIT, and every Indic→En and En→Indic table value.
- **USPS Pub 28 C1.** The HTML table was parsed: 505 rows, 209 of them with 3 cells. After dropping the header and
  one empty trailing row, **206** primary names map to **202** distinct standard abbreviations. AVENUE → AVE with
  variants AV, AVEN, AVENU, AVN and AVNUE.

**What this audit changed and why**
| Item | Problem found | Fix |
|---|---|---|
| Paper 2 Deng et al. | The author order followed arXiv (…Elmagarmid, **Li, Ilyas**…). The published ICDE 2019 version (IEEE metadata via Crossref) lists **Ilyas, Li** | ICDE order used; the arXiv order is noted |
| Paper 18 Aksharantar | "for En→Indic NE is lower" is not true for every language: Hindi En→Indic NE is 58.87 vs 55.59 native. The Indic→En examples of NE > native also missed tel (57.57 vs 54.55) | Stated as 18 of 19 with the Hindi exception, and tel added |
| Paper 31 Chourasia et al. | "Base LLM … not given in the abstract" left the compliance question open. The paper body says the base is **Llama 3 8B Instruct** with LoRA (rank 4, alpha 8), and the HF card license is `llama3` | Marked non-compliant, only the curriculum idea kept, and added to "Do not use" |
| USPS Pub 28 (§2 standards) | "207 names / 203 abbreviations" counted an empty trailing `<tr>` | Corrected to **206 / 202**; the third-audit row is annotated |
| TL;DR 2 | "35-61%" read as a global range, but urd is 23.14 and kas 12.87 | Scoped to the major languages, with urd/kas added |
| Paper 23 Ebing et al. | Missing the key fact that makes it relevant | Added: Hindi (Devanagari) is one of the six languages, and uroman and uconv are the romanizers (arXiv HTML v1) |
| Paper 30 Symphonym | It was not said that the system is built for toponyms | Added: toponyms, 32.7M triplets from GeoNames/Wikidata/Getty, transfer to personal names |
| Paper 43 Ganesan et al. | The key numbers were missing, and the 0.69 depends on OSM | Added: 74.8% more PoIs, median F 0.15 before and 0.69 after OSM post-processing |
| §5.2 worked examples | The skeletons were shown without the voicing collapse that `key_loose` defines | Clarified; the equalities still hold after the collapse |

**Discrepancies between the researcher's JSON summary and the notes (in addition to the list above).** Anyone
reusing the JSON should use the notes values:
- Deng et al. author order: in the ICDE version Ilyas comes before Li.
- Chourasia et al.: the JSON's "possible ≤8B cross-encoder reranker (license check required)" is settled. The
  published model is Llama 3 8B (`llama3`), so it is **not compliant**.
- RapidFuzz has 4,139 stars (the JSON says 4,138).
- The JSON's Ebing claim ("Romanization is safe for Indic abugidas") is now directly supported, because Hindi
  (Devanagari) was one of the six test languages.

**Nothing deleted.** All 48 papers, 28 repos and 16 HF models (plus the 2 non-HF model entries) exist, and their
metadata is correct after the fixes above. Items still labelled UNVERIFIED or secondary: the AFNOR line-format
details, the Wikipedia postcode facts (secondary, consistent), the muril-large-cased license (none declared, so
non-compliant), Cohen 2003's page range and Lu et al.'s abstract terms (both confirmed only via search-engine
listings, because DBLP/ACM block direct fetches).
