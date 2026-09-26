# Neural and transformer entity matching; LLM-based matching

Research notes for the Amazon ML Challenge 2026 (business entity resolution: S1 to S2/S3, per-S1 macro F0.5).
Compiled 2026-09-25. Every paper below was checked against a primary page (arXiv abs/API record, ACL Anthology,
IJCAI proceedings, VLDB PDF, OpenProceedings PDF, SIGMOD Record PDF, publisher or author-hosted PDF). Model licenses
come from the Hugging Face model-card metadata (`huggingface.co/api/models/<id>`, the same `license:` field
the card shows). Repo licenses, stars and last-push dates come from the GitHub API, read on 2026-09-25.

**Labels used below**
- **[paper]**: the number was read in the paper itself.
- **[estimate]**: my own back-of-envelope arithmetic, which has to be benchmarked on cloud hardware before we rely on it.
- **UNVERIFIED**: I could not confirm this on a primary page.

Nothing here was run on the laptop. All compute suggestions target Modal, Kaggle, Colab or Lightning.

---

## 0. TL;DR for this challenge

1. **On structured, short records, neural matchers barely beat a strong feature-plus-tree model.** The main
   evidence is DeepMatcher (SIGMOD'18) on structured EM: DL averaged 87.9 F1 against Magellan's 88.8 [paper],
   and training took 5.4 h against 1.5 min. In their training-size sweep on a structured dataset, Magellan also
   beat the Hybrid DL model until there were about 50K labeled pairs. Neural wins are large on textual and dirty data: +3.0 to 22.0 % F1 on textual and +6.2 to
   32.6 % relative F1 on dirty [paper]. Papadakis et al. (ICDE'24) showed that most popular EM benchmarks are
   close to linearly separable or already solved, so published neural gains overstate what we will see in
   practice. **Our data is synthetic-noised "dirty" data plus cross-script names.** Rules and GBDT features can
   normalize most of the noise. Neural earns its cost on:
   - cross-script Indic names, which share no characters with the Latin form;
   - DBA and website-as-name records;
   - unseen abbreviations;
   - the unseen French data.

   **Plan:** GBDT stays as the backbone. Neural scores go in as features and as a stage-2 re-scorer.
2. **Scale decides the architecture.** 1.73M S1 × 10 to 50 candidates is 17 to 85M pairs.
   - **Measured anchors [paper]:**
     - Ditto (BERT-class cross-encoder, V100, fp16, max length 256) scored all 10.65M employer candidate pairs in
       22,823 s, about 467 pairs/s.
     - A Qwen3-0.6B cross-encoder on an H100 needs about 3,131 s per 1M pairs on benchmark-length inputs (Beyond
       Scale and Generation, 2026). This is the paper's own Table 7 *estimate*, derived from its measured
       H100 throughput, not a wall-clock run over 1M pairs.
     - A Llama-3-8B name matcher serves about 10K requests/min on 3× L4 with vLLM (SGER, ACL'26 Industry).
   - **What fits one GPU in a few hours:**
     - A small multilingual cross-encoder (MiniLM-L6/L12 class) can score all pairs.
     - A base-size cross-encoder (XLM-R-base / MuRIL / mDeBERTa-base) can score about 17M pairs on an A100, or
       the uncertain band.
     - 0.6B to 1.7B decoders can score about 1 to 5M uncertain pairs.
     - 7 to 8B LLMs can score at most about 0.1 to 0.5M of the hardest pairs.
   - **Bi-encoders are nearly free** once each record is encoded once. There are about 12M records, and
     representations are cached (Beyond Scale Table 7).
3. **Cross-encoder beats bi-encoder at every size, but the bi-encoder is what makes it affordable.**
   - At 0.6B/4B/8B Qwen3, bi-encoders average 72.5/79.0/81.5 F1 against 83.0/84.1/84.1 for cross-encoders [paper].
   - In Ditto's own case study, an SBERT bi-encoder reached only 92 % F1 but cut the pipeline from 6.49 h to
     1.69 h. It did this as a top-10 pre-filter in front of the cross-encoder [paper].

   **Plan:** use the bi-encoder for retrieval and as a feature, and the cross-encoder only as a re-scorer.
4. **Don't use entity-ID or contrastive "memorize the entity" objectives.** Test S1 entities are new.
   - JointBERT gains 1 to 5 % F1 on seen products and underperforms on unseen ones [paper].
   - R-SupCon is best on seen products but drops about 25 % F1 on unseen ones in WDC Products [paper].
   - PLMs fine-tuned on *other* benchmark datasets and then applied to the WDC Products test set (cross-dataset
     transfer to unseen products) lose 36 to 56 % F1 (Ditto) and 22 to 61 % (RoBERTa); every LLM stays at least
     8 % F1 above the best transferred PLM [paper]. That shift (new domain and schema) is harsher than ours (same
     schema, new entities, new country), so read it as an upper bound on the risk.

   **Plan:** use pairwise matchers trained with hard negatives, and validate leave-one-country-out to mimic France.
5. **LLMs are a precision tool for a tiny, hard slice, not a full-pipeline matcher.**
   - The "select among candidates" prompt beats pairwise prompting by +16.02 % F1 on average (ComEM, COLING'25) [paper].
   - In-context clustering (LLM-CER, SIGMOD'26) fits our "several S2/S3 copies per S1" structure.
   - Distilling LLM labels into small models works: SFT on teacher labels was the best knowledge-distillation
     strategy in DistillER (2026) [paper].
6. **License traps.** None of these is compliant:
   - **Jellyfish-7B/8B/13B:** CC-BY-NC-4.0.
   - **Llama-3.x:** llama3.1 license.
   - **Gemma:** gemma license.
   - **Qwen2.5-3B-Instruct (and the Qwen2.5-3B base):** "qwen-research", even though the 0.5B, 1.5B, 7B, 14B
     and 32B Qwen2.5 models are Apache-2.0. (Qwen2.5-72B uses the separate "qwen" license, and is over the cap anyway.)
   - **jina-embeddings-v3 and jina-reranker-v2:** CC-BY-NC-4.0.
   - **l3cube indic-sentence-similarity-sbert:** CC-BY-4.0, which is not MIT/Apache.

   **Qwen3-8B is Apache but has 8.19B parameters by its safetensors count, which is over the 8B cap.** Use
   Qwen2.5-7B-Instruct (7.62B), Mistral-7B-Instruct-v0.3 (7.25B) or Qwen3-4B instead.

---

## 1. The metric, and where neural scores plug in

F_β = (1+β²)·TP / (β²·M + k), where M is the number of true matches of this S1 and k is the number predicted.
With β = 0.5 this becomes **F0.5 = 1.25·TP / (0.25·M + k)**. It is identical to the challenge's
1.25TP/(1.25TP+0.25FN+FP), because 1.25TP + 0.25(M−TP) + (k−TP) = 0.25M + k. (This is my own algebra, and
it is exact.)

What follows from it:
- **A wrong prediction costs a full unit in the denominator; a missed match costs 0.25.** Given calibrated
  match probabilities p_i for one S1's candidates, sorted descending, the approximately expected-F-optimal set is
  a prefix. Adding candidate i helps only if **1.25·p_i > F_current**. At a running F of about 0.8, the next
  candidate needs p above about 0.64. This gives a per-S1 adaptive cut-off instead of a global threshold. [estimate / derivation]
- **Predict empty when P(no match) = Π(1−p_i) × P(the blocker missed nothing) beats the best non-empty
  expected F.** Singletons score 1 only if the prediction is empty.
- **Neural models are therefore needed for well-calibrated probabilities, not just a ranking.** Calibrate
  (temperature or isotonic) on held-out train S1s, stratified by country and by script (Latin vs Indic).
- **Multi-copy structure.** Each S1 typically has 2 to 6 true matches. Once a high-confidence match `a` is
  found, other candidates that are near-duplicates of `a` (by S2↔S2 or S2↔S3 similarity) should be boosted, and
  candidates that contradict `a` should be demoted. See TransClean, MultiEM, LLM-CER and ComEM below.

---

## 2. Verified papers

"Verified via" gives the primary page I actually opened. Numbers are the ones I read on that page.

### 2.1 Classic neural EM and the "does DL beat features?" evidence

| # | Paper | Venue | Verified via | Key facts read |
|---|---|---|---|---|
| 1 | Mudgal, Li, Rekatsinas, Doan, Park, Krishnan, Deep, Arcaute, Raghavendra. *Deep Learning for Entity Matching: A Design Space Exploration* (DeepMatcher) | SIGMOD 2018, pp. 19–34, doi:10.1145/3183713.3196926 | author PDF https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf | Four DL designs (SIF, RNN, Attention, Hybrid) against Magellan (RF and other learners over similarity features). **Structured: 87.9 vs 88.8 avg F1 (DL vs Magellan); training 5.4 h vs 1.5 min.** Best DL beats Magellan on 8/11 structured sets, but the gain is usually <0.7 % except DBLP-Scholar (2.4 %) and Amazon-Google (20.2 %, synonym-heavy). **Textual: +3.0 to 22.0 % F1 (abstract; the textual-results section gives 5.8 to 22.0 % relative F1); dirty: +6.2 to 32.6 % relative F1.** Company (textual): 92.7 vs 79.8. Magellan beats Hybrid on structured data below about 50K labeled pairs; Hybrid is slightly better from about 200K. |
| 2 | Ebraheem, Thirumuruganathan, Joty, Ouzzani, Tang. *Distributed Representations of Tuples for Entity Resolution* (DeepER) | PVLDB 11(11), 2018, pp. 1454–1467, doi:10.14778/3236187.3236198 (arXiv title: *DeepER – Deep Entity Resolution*) | https://arxiv.org/abs/1710.00597 (arXiv API record) and the Crossref record for the DOI | Tuple embeddings via GloVe plus RNN/LSTM, with LSH-based blocking. Historical baseline. |
| 3 | Konda et al. *Magellan: Toward Building Entity Matching Management Systems* | PVLDB 9(12), 2016, pp. 1197–1208, doi:10.14778/2994509.2994535 | https://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf (PDF header: Vol. 9, No. 12; 12 pages) and the Crossref record for the DOI (audit 3) | The "features + classic ML" reference system (py_entitymatching). Conceptually, our GBDT pipeline is this. |
| 4 | Brunner, Stockinger. *Entity Matching with Transformer Architectures – A Step Forward in Data Integration* (EMTransformer) | EDBT 2020 (Industry and Applications), pp. 463–473, doi:10.5441/002/edbt.2020.58 | https://openproceedings.org/2020/conf/edbt/paper_205.pdf (DOI resolved via DataCite) | Transformers beat the "classical" DL EM methods by an average margin of 27.5 % on inherently difficult (dirty/textual) sets. The abstract cites refs [7, 20] (DeepER and DeepMatcher) for this; the contributions list states the same 27.5 % as "the best transformer outperforms DeepMatcher[20]", so both phrasings appear in the paper. The authors explicitly skip structured sets because Magellan already scores about 86 % avg F1 on the 11 structured sets (about 90 % without Amazon-Google). |
| 5 | Fu, Han, He, Sun. *Hierarchical Matching Network for Heterogeneous Entity Resolution* (HierMatcher) | IJCAI 2020, pp. 3665–3671, doi:10.24963/ijcai.2020/507 | https://www.ijcai.org/proceedings/2020/507 and the Crossref record for the DOI | Token-, attribute- and entity-level matching with cross-attribute token alignment. Relevant idea: **cross-attribute alignment**, because our address components are reordered and fields can bleed into each other ("LLC" moved to the front, landmark text in addresses). |
| 6 | Papadakis, Kirielle, Christen, Palpanas. *A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms* (arXiv title; the ICDE version is titled *A Critical Re-evaluation of Record Linkage Benchmarks for Learning-Based Matching Algorithms*) | ICDE 2024, pp. 3435–3448, doi:10.1109/ICDE60146.2024.00265 | https://arxiv.org/abs/2307.01231 (PDF read), Zenodo record https://zenodo.org/records/10350420 (lists ICDE 2024, Utrecht), Crossref record for the DOI | Across 13 standard datasets, "most of the popular datasets pose rather easy classification tasks". They define two measures: non-linear boost (best non-linear minus best linear F1) and learning-based margin (distance to a perfect oracle). A challenging dataset should have both ≥5 %, ideally 10 %. Only four datasets qualify on these practical measures: Ds4 and Ds6 (structured), Dd4 (dirty) and Dt1 (textual). **Lesson:** measure our own "non-linear boost". Train logistic regression and GBDT on the same similarity features, then add a cross-encoder, and look at the per-slice gaps. |
| 7 | Wang, Lin, Fu, Han, Sun, Xiong, Chen, Lu, Zhu. *Bridging the Gap between Reality and Ideality of Entity Matching: A Revisiting and Benchmark Re-Construction* | IJCAI 2022, pp. 3978–3984, doi:10.24963/ijcai.2022/552 | https://www.ijcai.org/proceedings/2022/552 | Benchmarks with restricted entities and balanced labels significantly overestimate progress. Open-entity, imbalanced settings are much harder. That matches us: 1.7M S1 against about 10M targets, extremely imbalanced. |
| 8 | Barlaug, Gulla. *Neural Networks for Entity Matching: A Survey* | ACM TKDD 15(3), 2021, doi:10.1145/3442200 | https://arxiv.org/abs/2010.11075 (journal ref in the arXiv record) | Taxonomy survey; background only. |

### 2.2 Pre-trained LM (cross-encoder) matchers and contrastive / bi-encoder matchers

| # | Paper | Venue | Verified via | Key facts read |
|---|---|---|---|---|
| 9 | Li, Li, Suhara, Doan, Tan. *Deep Entity Matching with Pre-Trained Language Models* (Ditto) | PVLDB 14(1), 2021, pp. 50–60, doi:10.14778/3421424.3421431 | https://arxiv.org/abs/2004.00584 (PDF read) | Serializes records as `[COL] name [VAL] ...` pairs for a BERT/RoBERTa/DistilBERT sequence-pair classifier. Up to +29 % F1 over the previous state of the art; the three optimizations add up to +9.8 %. **Employer case study (closest to our task):** 789K × 412K employer records with name/addr/city/state/zip/phone. Blocking by zipcode plus TF-IDF top-20 on name+addr gave **10.65M candidates**. 20K labels (39 % positive) gave **96.53 F1**. Domain knowledge tagged the street number and last 4 phone digits; augmentation used `attr_del` for robustness to missing values. **Matching all 10.65M pairs took 22,823 s** (V100, fp16, max length 256), about 467 pairs/s. Table 9 breakdown: basic blocking 537 s; SBERT "advanced blocking" = encoding 2,229 s (GPU) + vector search 1,982 s (CPU); top-10 matching 1,339 s. End to end that is 1.69 h with advanced blocking vs 6.49 h without (3.8×). SBERT alone reached only about 92 % F1. |
| 10 | Peeters, Bizer. *Dual-Objective Fine-Tuning of BERT for Entity Matching* (JointBERT) | PVLDB 14(10), 2021, pp. 1913–1921, doi:10.14778/3467861.3467878 | preprint PDF https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Research/Web-based_Systems/pub/Peeters-Bizer-Dual-Objective-Fine-Tuning-VLDB2021-preprint.pdf and GitHub | Binary match plus entity-ID multi-class heads. +1 to 5 % F1 on **seen** products, **underperforms on unseen**. **Do not use:** test S1 entities are unseen. |
| 11 | Peeters, Bizer. *Supervised Contrastive Learning for Product Matching* (R-SupCon) | WWW'22 Companion (poster), pp. 248–251, doi:10.1145/3487553.3524254 | https://arxiv.org/abs/2202.02098 | Two stages: supervised contrastive pre-training of the Transformer encoder (with source-aware sampling), then a binary cross-entropy head over the pooled pair representation (u, v, abs(u−v), u∗v); the encoder is either frozen or further fine-tuned, and both variants are reported. Strong when entities are seen and data is scarce. |
| 12 | Peeters, Der, Bizer. *WDC Products: A Multi-Dimensional Entity Matching Benchmark* | EDBT 2024, pp. 22–33, doi:10.48786/edbt.2024.03 | https://arxiv.org/abs/2301.09521 (PDF read) and https://openproceedings.org/html/pages/2024_edbt.html | With 80 % corner cases, R-SupCon, Ditto and RoBERTa score 72.18 to 79.99 F1. **R-SupCon drops about 25 % on unseen products**; all DL methods still beat the symbolic baselines. With a small dev set, R-SupCon still exceeds 78 % F1 (contrastive learning is more data-efficient). Corner cases mostly hurt **precision**, i.e. very similar negatives get called matches. That is exactly our F0.5 risk: chains and franchises with the same brand at different addresses. |
| 13 | Wang, Li, Wang. *Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation* | ICDE 2023, pp. 1502–1515, doi:10.1109/ICDE55515.2023.00391 | https://arxiv.org/abs/2207.04122 (PDF read); venue per the authors' repo https://github.com/megagonlabs/sudowoodo and the Crossref record | SimCLR-style self-supervised record embeddings with EM augmentations (token deletion, span cutoff, ...), used for blocking and then fine-tuned with few labels (500) plus pseudo-labels. **Adapt:** our S2/S3 noise generator is a known set of operators, so we can mirror it as augmentations when pre-training a bi-encoder on all ~24M records (train 2.2+5.0+5.3 = 12.5M plus test 1.73+4.9+5.1 ≈ 11.7M), test ones included. This is self-supervised, uses no labels and no external data. |
| 14 | Tu, Fan, Tang, Wang, Li, Du, Jia, Gao. *Unicorn: A Unified Multi-tasking Model for Supporting Matching Tasks in Data Integration* (SIGMOD 2023). The SIGMOD Record highlight has a different title (*Unicorn: A Unified Multi-Tasking Matching Model*) and author order (Fan, Tu, Li, Wang, Du, Jia, Gao, Tang) | SIGMOD 2023 = PACMMOD 1(1), doi:10.1145/3588938; highlight in SIGMOD Record 53(1):44–53, 2024 | SIGMOD Record PDF https://sigmodrecord.org/publications/sigmodRecord/2403/pdfs/12_unicorn-fan.pdf and https://github.com/ruc-datalab/Unicorn | A single DeBERTa encoder over (a, b) pairs, plus a mixture-of-experts layer and a binary matcher. Trained on 20 datasets across 7 matching tasks; supports zero-shot. **Adapt:** one shared cross-encoder for name-pair, address-pair and full-record-pair tasks, with MoE or task tokens. |
| 15 | Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* | EMNLP-IJCNLP 2019, pp. 3980–3990, doi:10.18653/v1/D19-1410 | https://arxiv.org/abs/1908.10084 (PDF read) and the Crossref record for the DOI | Siamese bi-encoder. Canonical cost: finding the most similar pair among 10K sentences with a BERT cross-encoder takes about 50M inferences and about 65 h, against about 5 s with SBERT. |
| 16 | Thakur, Reimers, Daxenberger, Gurevych. *Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks* | NAACL 2021, pp. 296–310, doi:10.18653/v1/2021.naacl-main.28 | https://arxiv.org/abs/2010.08240 (PDF read) and the Crossref record for the DOI | Label many extra pairs with the cross-encoder, then train the bi-encoder on them. Up to +6 points in-domain and +37 under domain adaptation. **Directly usable:** distill our cross-encoder into the retrieval bi-encoder so that blocking recall and the cosine feature both improve. |
| 17 | Humeau, Shuster, Lachaux, Weston. *Poly-encoders* | ICLR 2020 | https://arxiv.org/abs/1905.01969 | A middle ground between bi- and cross-encoders: candidate embeddings are cached (as in a bi-encoder), the context is summarized into m global features via m learned "codes", and a light attention layer scores the candidate against them. An option if a cross-encoder over all pairs is too slow. |
| 18 | Khattab, Zaharia. *ColBERT* | SIGIR 2020 | https://arxiv.org/abs/2004.12832 | Late interaction with cached per-token vectors. bge-m3 ships a ColBERT-style multi-vector head, which gives token-level alignment without a full cross-encoder. |
| 19 | Zeakis, Papadakis, Skoutas, Koubarakis. *Pre-trained Embeddings for Entity Resolution: An Experimental Analysis* | PVLDB 16(9), 2023, pp. 2225–2238, doi:10.14778/3598581.3598594 | https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf (read) | 12 LMs on 17 datasets. Sentence-BERT-family models dominate for blocking and unsupervised matching; S-GTR-T5 is singled out as the best, and the paper says differences among the SBERT models are minor on average. **For supervised matching this does not hold:** there "all BERT-based models excel", RoBERTa ranks first on average, and static models should be avoided. S-GTR-T5 recall is 15 % higher than DeepBlocker on 8 datasets. An S-GTR-T5 kNN (k=10) pipeline takes under 1 min, while ZeroER did not finish in 6 h on half the datasets. **Caveat:** those are English models, so for us use multilingual equivalents. |
| 20 | Thirumuruganathan et al. *Deep Learning for Blocking in Entity Matching: A Design Space Exploration* (DeepBlocker) | PVLDB 14(11), 2021, pp. 2459–2472, doi:10.14778/3476249.3476294 | https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf (read) | Autoencoder self-supervised blocking is best on structured and dirty data, Hybrid on textual. |
| 21 | Paulsen, Govind, Doan. *Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching* | PVLDB 16(6), 2023, pp. 1507–1519, doi:10.14778/3583140.3583163 | https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf (read) | Top-k TF-IDF (Lucene, 3-grams) beats 8 state-of-the-art blockers, including the DL Autoencoder and Hybrid. Blocks 10M-tuple tables in under 100 minutes on 10 AWS nodes for $12.5. **Lesson:** sparse char-gram top-k plus a dense multilingual bi-encoder is the right union for recall. Neural alone is not better than a good TF-IDF. |

### 2.3 Foundational text-matching architectures (older, but asked for)

| # | Paper | Venue | Verified via | Note |
|---|---|---|---|---|
| 22 | Huang, He, Gao, Deng, Acero, Heck. *Learning Deep Structured Semantic Models for Web Search using Clickthrough Data* (DSSM) | CIKM 2013, pp. 2333–2338, doi:10.1145/2505515.2505665 | https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/ and the Crossref record | The original two-tower model, built on letter-trigram word hashing. Its char-trigram hashing is still a cheap, script-agnostic record encoder, and it is trainable on our pairs on CPU or GPU in minutes. |
| 23 | Chen, Zhu, Ling, Wei, Jiang, Inkpen. *Enhanced LSTM for Natural Language Inference* (ESIM) | ACL 2017, pp. 1657–1668, doi:10.18653/v1/P17-1152 | https://arxiv.org/abs/1609.06038 | Soft-alignment (cross-attention) matcher. The ancestor of DeepMatcher's Attention and Hybrid models. |
| 24 | Pang, Lan, Guo, Xu, Wan, Cheng. *Text Matching as Image Recognition* (MatchPyramid) | AAAI 2016 | https://arxiv.org/abs/1602.06359 | CNN over a token-similarity matrix. **Adapt:** a tiny CNN over a token × token (or char-gram) similarity grid of name and address is cheap and order-invariant, which suits component reordering. |
| 25 | Feng, Yang, Cer, Arivazhagan, Wang. *Language-agnostic BERT Sentence Embedding* (LaBSE) | ACL 2022, pp. 878–891, doi:10.18653/v1/2022.acl-long.62 | https://arxiv.org/abs/2007.01852 and the Crossref record | Translation-pair-trained multilingual bi-encoder (Apache-2.0 weights). Candidate for cross-script retrieval, but check it on transliterated names first, because translation training ≠ transliteration. |
| 26 | Kasai, Qian, Gurajada, Li, Popa. *Low-resource Deep Entity Resolution with Transfer and Active Learning* | ACL 2019, pp. 5851–5861, doi:10.18653/v1/P19-1586 | https://arxiv.org/abs/1906.08042 and the Crossref record | Transfer plus active learning. Less relevant, since we have millions of labels. |

### 2.4 LLM-based matching

| # | Paper | Venue | Verified via | Key facts read |
|---|---|---|---|---|
| 27 | Peeters, Steiner, Bizer. *Entity Matching using Large Language Models* | EDBT 2025 (Experiments & Analyses), pp. 529–541, doi:10.48786/edbt.2025.42 | https://arxiv.org/abs/2310.11244 (PDF read; the comment gives EDBT 2025) and the DataCite record for the DOI (OpenProceedings paper-81) | **GPT-4 zero-shot vs fine-tuned PLMs:** higher on 3 of 6 datasets (+2.65 to 4.71 % F1), lower on the others (−3.69, −4.49, −0.73 per the text; Table 4 prints −0.33 for the last one, which is an inconsistency in the paper itself). **PLM transfer to unseen WDC Products:** RoBERTa and Ditto models fine-tuned on the *other* benchmark datasets (not on WDC Products) were applied to the WDC Products test set. Against Ditto fine-tuned directly on WDC Products (84.90 F1), the transferred models drop 36 to 56 % F1 (Ditto) and 22 to 61 % (RoBERTa). This is cross-dataset transfer, a harsher shift than our new-country setting. Every LLM beats the best transferred PLM by at least 8 % F1. There is no universal best prompt. Open LLMs match hosted ones given a few examples or rules. Fine-tuning helps, but it reduced generalization for the Llama models. |
| 28 | Peeters, Bizer. *Using ChatGPT for Entity Matching* (MatchGPT repo) | ADBIS 2023 short papers (CCIS), pp. 221–230, doi:10.1007/978-3-031-42941-5_20 | https://arxiv.org/abs/2305.03423 (comment) | The prompt library behind MatchGPT. |
| 29 | Steiner, Peeters, Bizer. *Fine-tuning Large Language Models for Entity Matching* | ICDE 2025 Workshops (ICDEW), pp. 9–17, doi:10.1109/ICDEW67478.2025.00006 (arXiv 2409.08185) | https://arxiv.org/abs/2409.08185 and the Crossref record for the DOI | Fine-tuning significantly helps smaller LLMs, with mixed results for larger ones. It helps in-domain transfer and hurts cross-domain transfer. Structured explanations in the training data helped 3 of 4 LLMs. |
| 30 | Zhang, Groth, Calixto, Schelter. *AnyMatch – Efficient Zero-Shot Entity Matching with a Small Language Model* | arXiv 2409.04073 (2024). The authors' lab page lists it under the AAAI 2025 workshop "Preparing Good Data for Generative AI" (GoodData), which is **non-archival**. The OpenReview entry (id Nees1OD5td) could not be opened from here, so treat the venue as *workshop, per the author lab page*. | https://arxiv.org/abs/2409.04073 (PDF read) and https://deem.berlin/publication/2024-09-09-anymatch-efficient-zero-shot-entity-matching-with-a-small-language-model/ | Fine-tunes **GPT-2 (124M, MIT)** in a transfer setup. Training data is chosen by an **AutoML filter: an AutoML classifier is trained on the transfer data, and the *matching* (positive) pairs it misclassifies, i.e. its false negatives, are kept**, plus attribute-level examples and label-imbalance control (downsampling). Average quality is within 4.4 % of MatchGPT/GPT-4 at 3,899× lower inference cost. **Adapt:** train the neural stage mostly on pairs our GBDT gets wrong or is unsure about. |
| 31 | Zhang, Li, Calixto, Groth, Schelter. *Beyond Scale and Generation: Understanding Language Model-based Entity Matching* | arXiv 2607.24688 (Jul 2026, under review) | https://arxiv.org/abs/2607.24688 (PDF read) | 1,215 fine-tuning runs of Qwen3 0.6B/4B/8B across bi-encoder, cross-encoder and generative matchers. **Bi-encoder 72.5/79.0/81.5 vs cross-encoder 83.0/84.1/84.1 avg F1.** Embedding-variant checkpoints are the best bi-encoder initializations. Generative is about equal to cross-encoder in-distribution and better under distribution shift. Larger models lean more on shortcuts. **Cost on H100, seconds per 1M pairs (Table 7, which the paper labels "estimated inference time"):** cross-encoder 3,131 (0.6B), 11,114 (4B), 16,921 (8B); bi-encoder with k=10 cached, 313 / 1,112 / 1,693. On the 7 datasets excluding Semi-Heter and Semi-Rel, the 4B cross-encoder reaches 86.4 F1 at about 165 pairs/s, and 8B adds little. The training runs used more than 850 H100 GPU-hours. **Measured vs estimated:** the paper's measured throughput (Figure 5 text: 0.6B cross-encoder about 473 pairs/s, 4B about 165 pairs/s) is faster than Table 7 implies (3,130.7 s/1M is about 319 pairs/s; 11,113.6 s/1M is about 90 pairs/s), so treat Table 7 as a conservative estimate. |
| 32 | Fan, Han, Fan, Chai, Tang, Li, Du. *Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration* (BatchER) | ICDE 2024, pp. 3696–3709, doi:10.1109/ICDE60146.2024.00284 | https://arxiv.org/abs/2312.03987 and https://pure.bit.edu.cn/en/publications/cost-effective-in-context-learning-for-entity-resolution-a-design/ | Batch prompting: many questions per prompt, with covering-based demonstration selection. More cost-effective than PLM fine-tuning and manual prompting (hosted-LLM $ setting). **Adapt:** batch 10 to 20 pair questions per prompt for the LLM stage to amortize the prompt. |
| 33 | Wang, Chen, Lin, Chen, Han, Sun, Wang, Zeng (COLING author order; arXiv lists Sun last). *Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching* (ComEM) | COLING 2025, pp. 96–109 | https://arxiv.org/abs/2405.16884 (PDF read) and https://aclanthology.org/2025.coling-main.8/ | **Selecting** (anchor plus candidate list) beats pairwise **matching** by +16.02 % avg F1. ComEM (a medium LLM ranks and filters, then a stronger LLM selects from the top-k) adds up to +4 % while lowering cost. **Position bias** means selection accuracy varies with the candidate's position in the list. **Adapt:** our S1 has 2 to 6 matches, so use "select ALL that match, or none" over the top 5 to 8 candidates, and shuffle or permute to de-bias. |
| 34 | Zhang, Dong, Xiao, Oyamada. *Jellyfish: Instruction-Tuning Local Large Language Models for Data Preprocessing* | EMNLP 2024, pp. 8754–8782, doi:10.18653/v1/2024.emnlp-main.497 | https://aclanthology.org/2024.emnlp-main.497/ | Instruction-tuned Mistral-7B, Llama-3-8B and OpenOrca-Platypus2-13B for data preprocessing, including EM. **All Jellyfish weights are CC-BY-NC-4.0: NOT compliant.** Only the recipe is reusable. |
| 35 | Fu, Tang, Khan, Mehrotra, Ke, Gao. *In-context Clustering-based Entity Resolution with Large Language Models: A Design Space Exploration* (LLM-CER) | SIGMOD 2026 = PACMMOD 3(4), 2025, doi:10.1145/3749170 | https://arxiv.org/abs/2506.02509 (arXiv comment "Accept by SIGMOD26") and the Crossref record for the DOI | The LLM clusters a set of records directly. Set size, diversity and ordering all matter. Across nine real-world datasets it reports up to 150 % higher accuracy, a 10 % increase in *FP-measure* (the paper's clustering metric, not plain pairwise F1) and up to 5× fewer API calls. **Adapt:** give the LLM {S1 record + top-k S2/S3 candidates} and ask for the cluster that contains S1. This is the natural format for multi-copy matches. |
| 36 | Zeakis, Papadakis, Skoutas, Koubarakis. *DistillER: Knowledge Distillation in Entity Resolution with Large Language Models* | arXiv 2602.05452 (Feb 2026) | https://arxiv.org/abs/2602.05452 (PDF read) | **Teachers:** Llama-3.1 8B/70B and Qwen-2.5 14B/32B. **Students:** LLMs and SLMs (all-MiniLM-L6-v2, RoBERTa-base). For data selection, **ranking query tuples by their blocking similarity (either the Max or the Top-2 score; the paper does not single out Max)** was the most reliable unsupervised strategy, matching the labeled "Sampled" baseline. SFT on the teacher's noisy labels beats GRPO/DPO. SLM *training* times are up to an order of magnitude faster than LLMs. **Adapt:** label hard unlabeled test-candidate pairs with a compliant LLM teacher (Qwen2.5-7B), then SFT a small cross-encoder on them. This is transductive and uses no external data. |
| 37 | Chourasia, Kapoor, Patil. *Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts* (SGER) | ACL 2026 Industry Track, pp. 1461–1468, doi:10.18653/v1/2026.acl-industry.101 | https://arxiv.org/abs/2605.23597 (PDF read) and https://aclanthology.org/2026.acl-industry.101/ (BibTeX) | Indian person-name matching for KYC at Dream11. **Scope caveat (audit 3):** every example and all data in the paper are *Latin-script* spellings of Indian names (e.g. "Subham" vs "Shubham", "Kirtan Singh" vs "SinghKirtan"); "transliteration across scripts" refers to how names were romanized, and the paper contains no native-script (Devanagari/Tamil) vs Latin pairs. So it is evidence for romanization-variant matching, **not** for our Devanagari/Tamil/Kannada ↔ Latin slice. **Two-phase curriculum:** (1) parse a name into first/middle/last JSON, trained on about 10K annotated names; (2) binary match, trained on 20K labeled pairs expanded to 50K+ with three augmentations. Evaluated on a name-disjoint held-out set of 50K real pairs. F1 0.994 against Levenshtein 0.726, fine-tuned BERT 0.806, GPT-4o few-shot 0.911, Llama-3-8B single-stage SFT 0.946, and single-stage SFT + augmentation 0.973. So augmentation alone accounts for 0.946 → 0.973, and the curriculum for the remaining 0.973 → 0.994. Served on 3× L4 with vLLM at up to 10K requests/min, P99 120 ms. Base model **Llama-3-8B (license NOT compliant)**; replicate with Qwen2.5-7B / Qwen3-4B. **Adapt:** phase 1 = parse a business name into {core name, legal form, DBA alias, domain}, and an address into {number, street, city, state, postcode}, using our own train records; phase 2 = match. |
| 38 | Peeters, Bizer. *Cross-Language Learning for Product Matching* (the arXiv listing title is "...for Entity Matching"; the PDF and the published version say "Product Matching") | WWW'22 Companion (poster), pp. 236–238, doi:10.1145/3487553.3524234 | https://arxiv.org/abs/2110.03338 (PDF read) and the Crossref record for the DOI | Adding English training pairs to a small German set improves multilingual transformer matchers, especially in low-resource settings. Supports **zero-shot transfer to French** from US+India training with a multilingual backbone. |

### 2.5 Multi-source and cluster-level matching (our S2/S3 multi-copy structure)

| # | Paper | Venue | Verified via | Key facts read |
|---|---|---|---|---|
| 39 | Zeng, Wang, Mao, Chen, Liu, Gao. *MultiEM: Efficient and Effective Unsupervised Multi-Table Entity Matching* | ICDE 2024, pp. 3421–3434, doi:10.1109/ICDE60146.2024.00264 | https://arxiv.org/abs/2308.01927 and the Crossref record for the DOI (the GitHub README does **not** name a venue; the earlier "README gives ICDE 2024" claim was wrong) | Enhanced representation, then table-wise hierarchical merging (which resolves transitive conflicts), then density-based pruning of outliers. **Adapt:** merge S2 and S3 copies first (dedupe within and across S2/S3), then match the merged clusters to S1. |
| 40 | de Meer Pardo, Hadji Misheva, Braschler, Stockinger. *TransClean: Finding False Positives in Multi-Source Entity Matching under Real-World Conditions via Transitive Consistency* | IEEE Access 13 (2025), pp. 195856–195870, doi:10.1109/ACCESS.2025.3632400 (arXiv 2506.04006) | https://arxiv.org/abs/2506.04006 and the Crossref record for the DOI | Uses the pairwise model's predictions on implied pairs to prune false positives iteratively. Reports +24.42 avg F1 over plain pairwise matching in the multi-source setting. **Caveat:** the results section says this average is "between Pre and Post TransClean scores **when run with manual labeling**" (the method is designed to need only limited manual labeling). It was measured with DistilBERT and CLER base matchers. **Adapt:** a precision filter for F0.5. Drop predicted S2/S3 candidates that the model does not also match to the other accepted copies. |
| 41 | Tang, Su, Zhang, Guo, Liu. *Unlocking the Power of Large Language Models for Multi-table Entity Matching* (LLM4MEM) | NLPCC 2025 (LNCS), pp. 208–220, doi:10.1007/978-981-95-3352-7_17 | https://arxiv.org/abs/2604.21238 | LLM attribute coordination, transitive-consensus embedding matching and density-aware pruning. +5.1 % avg F1 on 6 multi-table EM datasets. |
| 42 | Wang, Yang, Zheng, Ke. *Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution* (Alper) | KDD 2026, pp. 4906–4915, doi:10.1145/3770855.3817856 (arXiv 2605.25814) | https://arxiv.org/abs/2605.25814 and the Crossref record for the DOI | Iterative label propagation over an evolving graph. It spends a budget of LLM pair queries where they gain most (greedy, with guarantees). **Adapt:** a budgeted choice of which pairs go to the LLM. |
| 43 | Gupta. *Corporate-Family Resolution Is Not a String-Matching Problem: A Public Benchmark Stratified by Name Visibility* (CorpFam) | arXiv 2609.04269 (Sep 2026) | https://arxiv.org/abs/2609.04269 | This is a different task (parent/subsidiary). Its lesson carries over: 93.2 % of "name-invisible" links never enter the candidate set, so the intervention point is candidate generation. For our DBA and website-name records, add a blocking channel that does not rely on name overlap. |

Also checked, lower priority:
- Wang et al., *PromptEM: Prompt-tuning for Low-resource Generalized Entity Matching* (PVLDB 16(2):369–378, 2022, doi:10.14778/3565816.3565836; arXiv 2207.04802).
- Wang, Li, Hirota, *Machamp: A Generalized Entity Matching Benchmark* (CIKM 2021, pp. 4633–4642, doi:10.1145/3459637.3482008; arXiv 2106.08455).
- Moslemi, Mousavi, Behkamal, Milani, *Heterogeneity in Entity Matching: A Survey and Experimental Analysis* (Data & Knowledge Engineering 164, 2026, article 102575; arXiv 2508.08076).
- Bopardikar, Wang, Zou, *Structured Multi-Step Reasoning for Entity Matching Using Large Language Model* (arXiv 2511.22832; no venue found).
- Wang et al., *Neural Locality Sensitive Hashing for Entity Blocking* (NLSHBlock; SDM 2024, pp. 887–895, doi:10.1137/1.9781611978032.101; arXiv 2401.18064).
- Arora, Dell, *LinkTransformer* (ACL 2024 System Demonstrations, pp. 221–231, doi:10.18653/v1/2024.acl-demos.21; arXiv 2309.00789; its repo is **GPL-3.0**).

---

## 3. How much does neural add over strong GBDT + similarity features? (the evidence, then my read)

**Evidence [paper]:**
- **Structured EM:** DeepMatcher DL averages 87.9 against Magellan's 88.8. The best DL gain is usually under 0.7 %.
  The big exception is synonym-heavy Amazon-Google (+20.2 %).
- **Dirty EM:** DL is +6.2 to 32.6 % relative F1. Hybrid and Attention lose at most 1.1 F1 going from clean to
  dirty on 4 of 6 sets, so DL is robust to misplaced values.
- **Textual EM:** DL is +3.0 to 22.0 % F1. On the purely textual Company dataset: 92.7 vs 79.8.
- **Data regime:** with fewer than about 50K labeled pairs, Magellan beats DL on structured data. With far more
  labels (we have millions), DL catches up and slightly overtakes.
- **Transformer generation (EMTransformer, Ditto):** +27.5 % average over the classical DL matchers (DeepER and
  DeepMatcher) on dirty/textual data, and up to +29 % over the previous state of the art. Neither paper claims large gains on clean structured data.
- **Papadakis ICDE'24:** in most of the 13 popular datasets, the best linear matcher is within a few points of the
  best non-linear one, or everything is near-perfect. Reported neural gains on the classic benchmarks are
  therefore mostly not "neural beating features".

**My read for this dataset [estimate, to be measured]:**
- **Latin-script US/India records:** after deterministic normalization, most variation is recoverable by
  similarity features. Those normalizations are casefold, NFKD with diacritics stripped, junk-prefix strip,
  legal-form canonicalization (Corp/Corporation, Pvt/Private, LLC moved), US state full name ↔ code, token-sort,
  "null" removal, and street-type abbreviations. The features are char-3gram TF-IDF cosine, Jaro-Winkler,
  token-set ratio, number/postcode agreement, and so on. Expected neural lift on that slice: **about +0 to 1
  macro-F0.5**.
- **Cross-script Indic slice:** Devanagari, Tamil, Kannada and other scripts against Latin. Without
  transliteration, char features are near zero, and neural or multilingual encoders are the main signal.
  **Lift inside that slice could be large (10 to 30+ points).** Rule-based transliteration features would shrink
  that gap.
- **DBA and website-as-name slice:** a cross-encoder learns "heassociates.com" ≈ "H E Associates" and "X dba Y"
  equivalences better than fixed features. **Moderate lift.**
- **France (unseen):** multilingual encoders transfer. Hand-built features can only transfer if they are written
  country-agnostically. Neural helps robustness; generative/LLM is most robust under shift (Beyond Scale).
- **Overall:** likely **+1 to 3 macro-F0.5** from adding neural scores as GBDT features plus a stage-2
  re-scorer. Most of it will come from calibration on hard cases and from the cross-script subset. **Verify with a
  per-slice ablation** (country × script × noise type) before investing in bigger models.

---

## 4. What is realistic for 17 to 85M pair inferences on one GPU in a few hours?

**Measured anchors [paper]:**
- Ditto (2020 code, BERT-class, V100, fp16, max length 256): about 467 pairs/s (10.65M pairs in 22,823 s).
- Beyond Scale (2026, H100, Qwen3 cross-encoders, benchmark-length inputs; Table 7 is the paper's estimate
  derived from its measured throughput): 0.6B takes 3,131 s per 1M pairs; 4B takes 11,114 s; 8B takes 16,921 s. A bi-encoder with cached records at k=10: 313 / 1,112 / 1,693 s per 1M
  decisions. The same paper's *measured* throughput is higher: about 473 pairs/s for the 0.6B cross-encoder
  (≈ 2,100 s per 1M) and about 165 pairs/s for 4B on seven datasets (≈ 6,100 s per 1M). Table 7 is the
  conservative figure.
- SGER (Llama-3-8B, vLLM, 3× L4, short name pairs): up to 10K requests/min, i.e. about 55 requests/s per L4.
- SBERT paper: about 50M BERT cross-encoder inferences took about 65 h (2019 hardware).

**Back-of-envelope [estimate].** Our pair is name+address × 2, roughly 64 to 96 tokens. I take 80 tokens,
dynamic padding with length bucketing, bf16, and forward FLOPs ≈ 2 × non-embedding params × tokens.

I assume an *effective* throughput of about 30 TFLOP/s on an L4/A10G and about 120 TFLOP/s on an A100 with large
batches. Real code is often 2 to 5× slower (see the anchors above). **Benchmark with a quick Modal job before
committing.**

| Model (non-embedding params) | GFLOP/pair | L4/A10G pairs/s | A100 pairs/s | 17M pairs (A100) | 85M pairs (A100) | Verdict |
|---|---|---|---|---|---|---|
| MiniLM-L6-H384 cross-encoder (about 11M) | about 1.7 | about 15K | about 50K+ (launch/memory bound) | minutes | about 0.5 h | **Score every pair.** |
| multilingual MiniLM-L12-H384 (about 21M), e.g. mmarco-mMiniLMv2-L12 cross-encoder, multilingual-e5-small backbone | about 3.4 | about 8K | about 30K | about 10 min | about 45 min | **Score every pair (multilingual). Best default.** |
| XLM-R-base / MuRIL / mDeBERTa-v3-base (about 86M; DeBERTa about 1.3 to 1.5× slower) | about 14 | about 2K | about 8K | about 35 to 50 min | about 3 to 4 h | Feasible on A100 for ≤ about 30M pairs; otherwise uncertain band only. |
| XLM-R-large / bge-reranker-v2-m3 (about 300M) | about 48 | about 600 | about 2.5K | about 2 h | about 9 h | Uncertain band only (≤ about 10M). |
| Qwen3-0.6B / Qwen3-Reranker-0.6B (about 0.44B) | about 70 | about 400 | about 1.5K | about 3 h | 15 h+ | ≤ about 5M pairs. |
| Qwen2.5-7B / Mistral-7B (about 7B) | about 1,100 | about 25 to 55 | about 100 to 150 | infeasible | infeasible | ≤ about 0.1 to 0.5M hardest pairs (listwise prompts are better). |
| **Bi-encoder encoding** of about 12M records at about 40 tokens (base size) | about 7 per record | | | about 12M records in about 10 to 15 min on A100, about 45 min on L4 | then cosine is about free | **Always do this.** |

**Conclusion.** Architecture by stage:
- **Stage 0:** multilingual bi-encoder for retrieval and features, using cached embeddings.
- **Stage 1:** GBDT on all candidates, on a CPU box.
- **Stage 2:** small multilingual cross-encoder on every candidate or on the GBDT-uncertain band (p in about
  0.03 to 0.97, likely 5 to 20 % of pairs).
- **Stage 3 (optional):** base or large cross-encoder or a 0.6B/1.7B decoder on the residual.
- **Stage 4 (optional):** 7B LLM listwise "select all matches or none" on under about 1 % of S1s, the ones with
  the most uncertainty. These are mostly cross-script, DBA and website-name cases.
- **Stage 5:** per-S1 expected-F0.5 set selection plus transitive-consistency pruning.

**Training cost [estimate].** Train S1 = 2.2M, with typically 2 to 6 positives each.
- **Data:** sample about 3 to 5M pairs, with all positives plus hard negatives from our own blocker. Hard
  negatives are same brand at a different address, chain stores, and same address with a different business.
- **Epochs:** 1 to 2 epochs of a MiniLM-L12 cross-encoder on an A100 take about 30 to 90 min. A base-size model
  takes about 2 to 4 h.
- **LoRA SFT of a 1.5B to 7B decoder:** a few hundred thousand hard pairs, a few hours on an A100.

---

## 5. Models (license checked on Hugging Face; the 8B cap is checked with the safetensors parameter count)

**Compliant (MIT / Apache-2.0, ≤ 8B params):**

| HF id | License | Params (HF safetensors) | Kind | Languages | Use here |
|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | mit | 118M | bi-encoder | about 100 langs, incl. hi/ta/kn/fr | Default retrieval and feature encoder, fast. |
| intfloat/multilingual-e5-base / -large / -large-instruct | mit | 278M / 560M / 560M | bi-encoder | multilingual | Stronger retrieval; fine-tune with MNRL plus hard negatives. |
| BAAI/bge-m3 | mit | about 568M (XLM-R-large based; the card gives no count) | dense + sparse + multi-vector | 100+ langs, 8192 context | Dense, sparse and ColBERT scores as features, with token alignment. |
| BAAI/bge-reranker-v2-m3 | apache-2.0 | 568M | cross-encoder | multilingual | Stage-3 re-scorer; fine-tune on our pairs. |
| Alibaba-NLP/gte-multilingual-base / gte-multilingual-reranker-base | apache-2.0 | 305M / 306M | bi-encoder / cross-encoder | multilingual | Mid-size alternative. |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | apache-2.0 | 118M | multilingual cross-encoder | trained on machine-translated MS MARCO (mMARCO, 14 languages); card metadata lists hi and fr but **no ta/kn/bn/te/ml** | **Best stage-2 all-pairs candidate** (fine-tune). **License-chain caveat (audit 3):** its card declares `base_model: nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`, which has **no license on its card** (listed as non-compliant below). The upstream MiniLM code repo (microsoft/unilm) is MIT, and this card itself says apache-2.0, so it formally passes the rule. The cleaner alternative is to fine-tune multilingual-e5-small (MIT, same 118M size) as the cross-encoder backbone. |
| cross-encoder/ms-marco-MiniLM-L6-v2 | apache-2.0 | 22.7M | cross-encoder | English | Fastest; only for the Latin slice. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0 | 118M | bi-encoder | about 50 langs by the card's metadata; includes hi/mr/gu/ur/fr but **not ta/kn/bn/te/ml** | Cheap multilingual baseline. Weak candidate for the Tamil/Kannada cross-script slice. |
| sentence-transformers/LaBSE | apache-2.0 | 471M | bi-encoder | 109 langs | Test cross-script recall; it is translation-trained, not transliteration-trained. |
| sentence-transformers/gtr-t5-base | apache-2.0 | 110M | bi-encoder | English | S-GTR-T5 was best in Zeakis et al.; English-only. |
| google/muril-base-cased | apache-2.0 | about 237M by my arithmetic from config.json (BERT-base encoder about 86M plus a 197,285 × 768 embedding table about 152M); the card gives no count and there is no safetensors file | encoder | 17 languages per the card's footnote: 16 Indian languages plus English. Pretrained with **transliterated segment pairs** (Wikipedia only) | Cross-script Indic cross-encoder backbone. No French. |
| ai4bharat/IndicBERTv2-MLM-only | mit | 278M (stated in the card text; no safetensors count) | encoder | 23 Indic languages + English (card) | Alternative Indic backbone. |
| FacebookAI/xlm-roberta-base / -large | mit | 279M / 561M | encoder | 100 langs | Generic multilingual cross-encoder backbone (Ditto-style). |
| microsoft/mdeberta-v3-base | mit | 86M backbone + 190M embeddings | encoder | CC100 multilingual | Strong multilingual cross-encoder backbone. |
| microsoft/deberta-v3-xsmall / -small / -base | mit | n/a (small) | encoder | English | Latin-only slice. |
| google/canine-s / canine-c | apache-2.0 | 132M | char-level encoder | tokenizer-free | Typo and diacritic robust; worth a probe on cross-script. |
| google/byt5-small | apache-2.0 | about 300M, inferred from the 1,198,627,927-byte fp32 pytorch_model.bin (the card gives no count) | byte-level seq2seq | tokenizer-free | Byte-level matcher or transliteration-style normalizer. |
| Qwen/Qwen3-Embedding-0.6B / -4B / -8B | apache-2.0 | 596M / 4.02B / 7.57B | LLM bi-encoder | multilingual | Embedding-variant checkpoints are the best bi-encoder initializations (Beyond Scale). |
| Qwen/Qwen3-Reranker-0.6B / -4B | apache-2.0 | 596M / 4.02B | LLM reranker (cross) | multilingual | Stage-3 re-scorer. |
| Qwen/Qwen3-0.6B / 1.7B / 4B | apache-2.0 | 752M / 2.03B / 4.02B | decoder LLM | multilingual | SFT generative matcher; 4B is the sweet spot per Beyond Scale. |
| Qwen/Qwen2.5-7B-Instruct | apache-2.0 | 7.62B | decoder LLM | card: "over 29 languages"; the named examples include French but no Indic language, so test it on the cross-script slice before using it as a teacher | **Largest safe LLM**: stage-4 listwise judge or distillation teacher. |
| Qwen/Qwen2.5-1.5B-Instruct / 0.5B-Instruct | apache-2.0 | 1.54B / 494M | decoder LLM | multilingual | Small SFT matchers. |
| mistralai/Mistral-7B-Instruct-v0.3 | apache-2.0 | 7.25B | decoder LLM | the card states no language coverage ("mostly European / good at French" is UNVERIFIED) | Alternative teacher; test its French quality before relying on it. |
| microsoft/Phi-4-mini-instruct | mit | 3.84B | decoder LLM | 23 languages in the card metadata, incl. fr, but **no Indic language** | Alternative small LLM; poor fit for the cross-script slice. |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | apache-2.0 | 1.71B | decoder LLM | English | Latin-only. |
| openai-community/gpt2 | mit | 137M (HF count) | decoder | English | AnyMatch base; weak for multilingual. |
| ibm-granite/granite-embedding-278m-multilingual / 107m-multilingual | apache-2.0 | 278M / 107M | bi-encoder | card metadata lists 12 languages (en, ar, cs, de, es, fr, it, ja, ko, nl, pt, zh); **no Indic language** | Extra candidates for the Latin/French slice only. |
| Snowflake/snowflake-arctic-embed-m-v2.0 | apache-2.0 | 305M | bi-encoder | multilingual | Extra candidate. |
| ai4bharat/indictrans2-indic-en-1B | mit (gated=auto) | 1.02B | translation | Indic→En | Transliteration/translation probe for the cross-script slice (inference on cloud). |
| ai4bharat/IndicXlit | mit | n/a (the card has no text beyond the license) | transliteration | Indic↔Latin | Transliteration candidate. The repo ships fairseq checkpoints for both directions (`indicxlit-en-indic-v1.0/…/indicxlit.pt`, `indicxlit-indic-en-v1.0/…/indicxlit.pt`). |

**NOT compliant (do not ship):**

| HF id | License / issue |
|---|---|
| NECOUDBFM/Jellyfish-7B / -8B / -13B | cc-by-nc-4.0 (13B is also over 8B) |
| meta-llama/Llama-3.1-8B-Instruct | llama3.1 (and 8.03B > 8B) |
| google/gemma-3-4b-it | gemma |
| **Qwen/Qwen2.5-3B-Instruct** (and Qwen/Qwen2.5-3B) | **other: qwen-research** (a trap: the 0.5B/1.5B/7B/14B/32B Qwen2.5 models are Apache-2.0, while 72B is "other: qwen") |
| **Qwen/Qwen3-8B** | apache-2.0, **but 8,190,735,360 params > 8B cap**: treat as non-compliant, or get an explicit ruling |
| jinaai/jina-embeddings-v3, jinaai/jina-reranker-v2-base-multilingual | cc-by-nc-4.0 |
| l3cube-pune/indic-sentence-similarity-sbert | cc-by-4.0 (not MIT/Apache) |
| nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large | no license on card: treat as non-compliant |

Note: pretrained weights trained on web corpora are allowed. That is not an "external data lookup". **Do not**
ship anything that embeds an external gazetteer or registry, e.g. an address parser trained on OpenStreetMap, as a
runtime resource. Flag it as a compliance risk.

---

## 6. Repos (GitHub API, read 2026-09-25)

Library and code licenses are separate from model licenses. Code with **no license** is "all rights reserved":
read it for ideas, don't vendor it.

| Repo | License | Stars | Last push | Stage | Usefulness |
|---|---|---|---|---|---|
| huggingface/sentence-transformers | Apache-2.0 | 19,121 | 2026-09-24 | production lib | **Core:** bi- and cross-encoder training (MNRL, CachedMNRL, CrossEncoder trainer, hard-negative mining). |
| megagonlabs/ditto | Apache-2.0 | 318 | 2024-04-17 | research code | Serialization, span/attr augmentation, domain-knowledge tagging. |
| anhaidgroup/deepmatcher | BSD-3-Clause | 624 | 2024-06-18 | research lib (old torch) | Reference only. |
| anhaidgroup/py_entitymatching (Magellan) | BSD-3-Clause | 195 | 2024-05-29 | lib | Feature-generation reference. |
| wbsg-uni-mannheim/jointbert | BSD-3-Clause | 16 | 2021-06-07 | research | Skip (seen-entity bias). |
| wbsg-uni-mannheim/contrastive-product-matching (R-SupCon) | BSD-3-Clause | 38 | 2022-02-11 | research | SupCon loss code, if we pre-train a bi-encoder. |
| megagonlabs/sudowoodo | BSD-3-Clause | 19 | 2023-05-24 | research | Contrastive-augmentation ideas. |
| megagonlabs/rotom | BSD-3-Clause | 23 | 2022-05-31 | research | Meta-learned augmentation (low priority). |
| qcri/DeepBlocker | BSD-3-Clause | 30 | 2023-04-05 | research | Blocking reference. |
| ruc-datalab/Unicorn | none | 32 | 2023-04-15 | research | Ideas only. |
| wbsg-uni-mannheim/MatchGPT | none | 68 | 2024-10-18 | research | Prompt library (ideas only). |
| wbsg-uni-mannheim/TailorMatch | none | 10 | 2025-06-17 | research | LLM fine-tuning for EM (ideas only). |
| Jantory/anymatch | none | 10 | 2025-01-17 | research | AutoML-filter data selection (ideas). |
| Jantory/llm-trained-matcher | none | 2 | 2026-07-28 | research | Beyond Scale code (ideas). |
| tshu-w/ComEM | none | 20 | 2026-05-27 | research | Selecting-strategy prompts (ideas). |
| fmh1art/BatchER | none | 12 | 2023-12-17 | research | Batch-prompt design (ideas). |
| ZJU-DAILY/MultiEM | none | 16 | 2023-11-05 | research | Hierarchical merging and density pruning (ideas). |
| gpapadis/DLMatchers | none | 3 | 2024-04-18 | research | The re-evaluation's Docker images (ideas). |
| moj-analytical-services/splink | MIT | 2,429 | 2026-09-22 | production | Fellegi-Sunter plus clustering at scale on DuckDB/Spark. Useful for m/u-probability features. |
| dedupeio/dedupe | MIT | 4,514 | 2025-07-29 | production | Active-learning dedupe (less relevant with millions of labels). |
| dell-research-harvard/linktransformer | **GPL-3.0** | 145 | 2026-09-06 | lib | Transformer record linkage; copyleft, so check before use. Its hub models have their own licenses. |
| zinggAI/zingg | **AGPL-3.0** | 1,251 | 2026-09-13 | production | Copyleft; avoid. |

---

## 7. Adaptation details specific to this challenge

- **Serialization.** Use a Ditto-style `[NAME] ... [ADDR] ... [CTRY] ...`. Put the normalized name and the raw
  name both in the input. The model sees the raw form (for cross-script and domain cues) and the canonical form
  (for cheap alignment). Randomly drop or permute address components during training, as Ditto's
  `attr_del`/`span_shuffle` does. This mirrors the S2/S3 noise generator, including component reordering,
  "null" tokens and missing parts.
- **Hard negatives decide F0.5.** WDC Products shows that corner cases mainly destroy **precision**. Mine them
  from our own blocker's top-k:
  - same brand or core name at a different street number;
  - the same address shared by different businesses (malls, office towers);
  - legal-form-only differences that are actually different entities;
  - near-identical names across cities.
- **Cross-script.** Three options, measured against each other:
  - (a) A multilingual cross-encoder fine-tuned on the train cross-script pairs, which already exist in the data.
  - (b) Features on a rule-based transliteration (Indic→Latin, then phonetic key). The library license must be
    checked; MIT/Apache Python libs exist, but verify them.
  - (c) MuRIL as an Indic-specialist backbone. It was pretrained on transliterated pairs. Route only
    Indic-script records to it; since MuRIL lacks French, route France to XLM-R or mDeBERTa.

  **Probe first on cloud:** recall@k of cross-script positives for e5-small/base, bge-m3, LaBSE, MuRIL-mean-pool
  and CANINE.
- **Unseen France.**
  - **Validation:** leave-one-country-out. Train on US and test on India, and the reverse. Train on everything
    except Indic-script records and test on them.
  - **Model choice:** pick the model family whose drop is smallest. Beyond Scale says generative/LLM matchers
    degrade least under shift, and Peeters et al. (EDBT'25) show PLMs fine-tuned on other benchmark datasets
    collapse when applied to unseen WDC products (cross-dataset transfer). Our shift is milder: same schema,
    new country.
  - **Normalization rules:** keep them country-agnostic, e.g. learned legal-form lists from train token
    statistics rather than hand-coded US/India lists.
- **Don't memorize entities.** Test S1 entities are new, so avoid multi-class entity heads (JointBERT) and
  contrastive-only classifiers (R-SupCon). Pairwise cross-encoders plus bi-encoders trained with in-batch and
  hard negatives are fine.
- **Transductive self-supervision is allowed and cheap.** Sudowoodo/SimCLR-style contrastive pre-training on all
  records (train + test S1/S2/S3, no labels) uses only challenge data. **Confirm the rules allow unlabeled
  test-record usage** (usually yes; it is not "external data").
- **Distillation.**
  - (i) AugSBERT: label about 20 to 50M blocker pairs with the stage-2 cross-encoder, then retrain the bi-encoder.
    This improves recall and the cosine feature.
  - (ii) DistillER/AnyMatch: use the 7B LLM teacher only on GBDT-uncertain pairs (a few hundred thousand), then
    SFT the small cross-encoder on them.
  - (iii) "Distill into GBDT": feed bi-encoder cosine, bge-m3 sparse/ColBERT scores and the cross-encoder
    probability (where computed) as features to one final GBDT or calibrator, with a missing-indicator where the
    cross-encoder was not run.
- **Set-level post-processing (precision for F0.5).**
  - For each S1, compute pairwise similarity among its accepted S2/S3 candidates (cheap: cached bi-encoder
    vectors).
  - Remove a candidate that is not consistent with the other accepted copies (TransClean).
  - Add a borderline candidate that is a near-duplicate of an accepted one (MultiEM merge logic).
  - Then apply the expected-F0.5 prefix rule from §1.
- **LLM stage prompt.** Use "Here is a reference business record R and N candidate records. Return the IDs of ALL
  candidates that are the same business as R, or NONE." (ComEM select plus LLM-CER clustering, adapted to
  multi-match.) Handle position bias with 2 permutations and a majority vote. Batch several S1 anchors per
  prompt, BatchER-style, only if quality holds. **Only compliant ≤ 8B models:** Qwen2.5-7B-Instruct,
  Mistral-7B-v0.3, Qwen3-4B. **Serve with vLLM on cloud.**

---

## 8. What we should do (prioritized)

### P0 (do first; cheap, highest expected value)
1. **Baseline ablation to measure the neural headroom.** Build GBDT on normalized similarity features. Report
   macro-F0.5 per slice (US / India-Latin / India-cross-script / DBA / website-name / country-holdout) and the
   "non-linear boost" (logistic regression vs GBDT) per slice, following Papadakis ICDE'24. Only spend GPU where
   slices are weak.
2. **Multilingual bi-encoder as retrieval and feature (cloud GPU).**
   - Encode all about 12M records once with intfloat/multilingual-e5-small, then e5-base or bge-m3 (MIT). Use the
     top-k union with char-3gram TF-IDF (Sparkly lesson).
   - Add cosine, rank and sparse scores as GBDT features.
   - Measure recall@k on train, per slice (especially cross-script).
3. **Expected-F0.5 decision layer.** Calibrate probabilities (isotonic, per country/script), apply the per-S1
   prefix rule (add a candidate if 1.25·p > F_running), and predict empty when P(no match) is high. This is pure
   CPU and model-agnostic, and likely worth more than any single model upgrade.
4. **License gate in code.** Whitelist the HF ids from §5. Block Qwen2.5-3B (base and Instruct), Qwen2.5-72B, Qwen3-8B (8.19B), Llama, Gemma,
   Jellyfish and Jina.

### P1 (the main neural win)
5. **Fine-tune a small multilingual cross-encoder** (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 or
   multilingual-e5-small as backbone, Apache/MIT). Prefer the e5-small backbone if we want a clean license chain:
   the mmarco model's declared base checkpoint has no license on its card (see §5).
   - Data: about 3 to 5M train pairs with blocker-mined hard negatives and Ditto-style augmentation.
   - Score all candidates, or the GBDT-uncertain band, on an A100/L4.
   - Feed its probability into the final GBDT or calibrator. Expected budget: about 1 to 2 GPU-hours to train and
     under 1 h to infer [estimate].
6. **Fine-tune the bi-encoder** (e5-base or bge-m3) with MNRL plus hard negatives, then run the AugSBERT loop
   (cross-encoder labels more pairs). This improves blocking recall and the cosine feature at the same time.
7. **Leave-one-country-out validation** as the model-selection criterion, to protect against the France shift.
8. **Set-level consistency pruning** (TransClean/MultiEM style) using cached embeddings among each S1's accepted
   candidates.

### P2 (targeted, if P0/P1 slices still show gaps)
9. **Cross-script specialist.** Route Indic-script records (Unicode-block detection) to a MuRIL- or
   XLM-R-base-based cross-encoder, and/or add rule-based transliteration features. Compare with the generic model
   on the cross-script slice.
10. **Base-size cross-encoder re-scorer** (bge-reranker-v2-m3 or mDeBERTa-v3-base, fine-tuned) on the residual
    uncertain band (≤ about 10M pairs, A100).
11. **SGER-style two-phase SFT of Qwen3-1.7B/4B** (Apache).
    - Phase 1: parse business names and addresses into structured JSON. Labels come from our own train records
      and the normalization rules.
    - Phase 2: match. Include SGER's three augmentations (pair swapping, component permutation, random space
      removal), which map directly onto our reordering and doubled/missing-space noise. In SGER, augmentation
      alone gave 0.946 → 0.973 F1.
    - Note: SGER only tested Latin-script romanization variants. Its result does not show the recipe works for
      native-script ↔ Latin pairs, so validate that on our cross-script slice separately.
    - Use it as a generative matcher on about 1 to 5M hard pairs (vLLM on A100). Beyond Scale says generative
      matchers are the most robust under shift, which helps with France.

### P3 (nice to have / only if time remains)
12. **7B LLM listwise judge** (Qwen2.5-7B-Instruct or Mistral-7B-v0.3), vLLM on cloud. Use it on under about 1 %
    of S1s with the highest decision entropy, with the ComEM/LLM-CER "select all or none" prompt and 2
    permutations. Or use it only as a **teacher** to label those pairs, then SFT the small cross-encoder
    (DistillER).
13. **Sudowoodo-style self-supervised contrastive pre-training** on all train+test records with
    noise-mirroring augmentations, before supervised fine-tuning. Confirm the rules allow unlabeled test usage.
14. **Poly-encoder / ColBERT (bge-m3 multi-vector)** as a cheaper stand-in for a cross-encoder, if cross-encoder
    inference is the bottleneck.
15. **Skip:** JointBERT, R-SupCon multi-class, DeepMatcher RNNs, Jellyfish weights, hosted APIs. None of these is
    allowed or useful here.

---

## Audit log

Independent audit, 2026-09-25. Every item was treated as possibly hallucinated until confirmed on a primary
source. Nothing was run on the laptop except HTTP metadata fetches and PDF text extraction (PyMuPDF); no model was
loaded or run.

### How each item was checked
- **Papers (43 in §2 plus the 6 in "Also checked"):**
  - All 37 arXiv IDs were resolved through the arXiv API (`export.arxiv.org/api/query?id_list=...`). Title,
    authors, comment, journal-ref and DOI all matched the notes except where listed under "Changes".
  - Venues, pages and DOIs were checked against Crossref (`api.crossref.org/works`), ACL Anthology, IJCAI
    proceedings pages, OpenProceedings (EDBT 2020 and 2024 index pages) and Zenodo. DBLP refused connections from
    this network, so Crossref stood in for it.
  - Every quoted number was checked by downloading the PDF and searching its text. That covers DeepMatcher (author
    PDF), EMTransformer (OpenProceedings PDF), Papadakis, Ditto, the JointBERT preprint, WDC Products, Sudowoodo,
    the Unicorn SIGMOD Record PDF, Zeakis/DeepBlocker/Sparkly/Magellan (VLDB PDFs), Peeters EDBT'25, AnyMatch,
    Beyond Scale, ComEM, DistillER, SGER, LLM-CER, TransClean, R-SupCon and Cross-Language. For LLM4MEM, CorpFam,
    Alper, Augmented SBERT, SBERT and Steiner et al., the arXiv abstract was the source.
- **Models (55 HF ids):** the `huggingface.co/api/models/<id>` JSON was read for each one: `cardData.license`,
  `license:` tag, `safetensors.total` and `gated`. README cards were read where the notes make language or
  parameter claims (MuRIL, mDeBERTa, Mistral-7B-v0.3, e5-small, bge-m3, ByT5, IndicXlit, IndicBERTv2, LaBSE,
  SmolLM2, Qwen2.5-7B, Qwen3-4B, all three Jellyfish models).
- **Repos (22):** checked through the authenticated GitHub API (`gh api repos/<r>`) for `license.spdx_id`, stars,
  `pushed_at` and archived status. For every repo reported as "none", the root listing was checked for a
  LICENSE/COPYING file. None had one, so "no license" is correct. The README venue statements were also read.

### Confirmed with no change needed
- **All 22 repos:** the license, star count and last push date match the GitHub API exactly as of 2026-09-25.
  - sentence-transformers is Apache-2.0 with 19,118 stars.
  - ditto is Apache-2.0 with 318.
  - deepmatcher and py_entitymatching are BSD-3-Clause, and so are jointbert, contrastive-product-matching,
    sudowoodo, rotom and DeepBlocker.
  - splink and dedupe are MIT.
  - linktransformer is GPL-3.0 and zingg is AGPL-3.0.
  - Unicorn, MatchGPT, TailorMatch, anymatch, llm-trained-matcher, ComEM, BatchER, MultiEM and DLMatchers have no
    license.
- **All 55 HF license strings are exact.** In particular:
  - Qwen2.5-3B-Instruct = `other` / `qwen-research`.
  - Qwen3-8B = apache-2.0 with 8,190,735,360 params, which exceeds the 8B cap.
  - Jellyfish-7B/8B/13B = cc-by-nc-4.0. Their bases are Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct and
    OpenOrca-Platypus2-13B.
  - Llama-3.1-8B = llama3.1 (8.03B).
  - gemma-3-4b-it = gemma.
  - jina v3 and jina-reranker-v2 = cc-by-nc-4.0.
  - l3cube = cc-by-4.0.
  - nreimers mMiniLMv2 has no license field.
  - indictrans2-indic-en-1B is mit with gated=auto.
  - Every safetensors parameter count quoted in §5 matches.
- **Key numbers confirmed in the source PDFs:**
  - **DeepMatcher:** 87.9 vs 88.8; 5.4 h vs 1.5 min; 8/11; <0.7 %; 2.4 % and 20.2 %; +3.0 to 22.0 %; +6.2 to
    32.6 %; Company 92.7 vs 79.8; 50K/200K; ≤1.1 %.
  - **EMTransformer:** 27.5 %; Magellan 86 % / 90 %.
  - **Ditto:** 10,652,249 candidates; 20,000 labels at 39 % positive; 96.53; 22,823.43 s; 1.69 h vs 6.49 h;
    SBERT 92 %; V100, fp16, max length 256; tagging of the street number and last 4 phone digits; attr_del; +29 %
    and +9.8 %.
  - **JointBERT:** +1 to 5 % on seen products, underperforms on unseen.
  - **WDC Products:** 72.18 to 79.99; ~25 % R-SupCon drop; >78 % with a small dev set; corner cases hurt precision.
  - **Sudowoodo:** 500 labels.
  - **Unicorn:** DeBERTa, 20 datasets, 7 tasks.
  - **Zeakis:** 12 LMs × 17 datasets; +15 % recall on 8 datasets; <1 min vs ZeroER >6 h.
  - **DeepBlocker:** Autoencoder best on structured/dirty data, Hybrid on textual.
  - **Sparkly:** 8 blockers; 10M tuples in <100 min on 10 nodes for $12.5; 3-grams.
  - **Peeters EDBT'25:** +2.65 to 4.71; Ditto 84.90 with 36 to 56 % drops; RoBERTa 22 to 61 %; ≥8 %.
  - **AnyMatch:** GPT-2 124M; 4.4 %; 3,899×.
  - **Beyond Scale:** 72.5/79.0/81.5 vs 83.0/84.1/84.1; 3,130.7 / 11,113.6 / 16,920.5 s; 312.9 / 1,111.9 /
    1,693.2 s; 86.4 F1 at ~165 pairs/s; >850 H100 GPU-hours.
  - **ComEM:** +16.02 %, up to +4 %, position bias.
  - **DistillER:** teachers, students, max-blocking-similarity selection, SFT beats GRPO/DPO, up to an order of
    magnitude faster.
  - **SGER:** 0.994 / 0.726 / 0.806 / 0.911 / 0.946; ~10K annotated names; 3× L4; 10,000 RPM; P99 120 ms;
    Llama-3-8B.
  - **LLM-CER:** 150 % / 10 % / 5×.
  - **TransClean:** +24.42.
  - **LLM4MEM:** +5.1 %.
  - **CorpFam:** 93.2 %.
  - **SBERT:** 65 h vs 5 s.
  - **AugSBERT:** +6 / +37.

### Changes made (what and why)
1. **Ditto (row 9):** the timing breakdown was mis-assigned. Table 9 lists basic blocking at 537 s, SBERT
   encoding at 2,229 s (GPU), search at 1,982 s (CPU), top-10 matching at 1,339 s and matching all pairs at
   22,823 s. The old text called 537 s the "encode" time and 2,229 s the "search" time. Fixed.
2. **Papadakis (row 6):**
   - The list of challenging datasets left out **Dd4**. The paper names Ds4, Ds6, Dd4 and Dt1. Fixed.
   - Added the published ICDE title ("...Record Linkage Benchmarks..."), pages 3435–3448 and
     doi:10.1109/ICDE60146.2024.00265.
3. **MultiEM (row 39):** "README gives ICDE 2024" was **false**, because the README names no venue. The venue was
   re-verified through Crossref (ICDE 2024, pp. 3421–3434, doi:10.1109/ICDE60146.2024.00264) and the
   verified-via entry corrected.
4. **Unicorn (row 14):**
   - The notes gave the SIGMOD Record author order next to the SIGMOD'23 title.
   - The PACMMOD 1(1) order is Tu, Fan, Tang, Wang, Li, Du, Jia, Gao (doi:10.1145/3588938).
   - The SIGMOD Record 53(1):44–53 highlight has a different title.
   - Both are now stated.
5. **ComEM (row 33):**
   - The author order now follows the COLING 2025 proceedings: Sun is 6th there and last on arXiv.
   - The title is written out in full, and pp. 96–109 plus the ACL Anthology URL were added.
6. **Cross-Language Learning (row 38):** the paper PDF and the published WWW'22 Companion version are titled
   "*Cross-Language Learning for **Product** Matching*". Only the arXiv listing says "Entity Matching". Fixed, and
   pp. 236–238 added.
7. **Steiner et al. (row 29):** the venue was UNVERIFIED. It is now confirmed as ICDE 2025 Workshops, pp. 9–17,
   doi:10.1109/ICDEW67478.2025.00006.
8. **TransClean (row 40):** it was listed as an arXiv preprint but was published in IEEE Access 13 (2025),
   pp. 195856–195870, doi:10.1109/ACCESS.2025.3632400. Updated.
9. **Alper (row 42):** it was listed as an arXiv preprint but was published at KDD 2026, pp. 4906–4915,
   doi:10.1145/3770855.3817856. Updated.
10. **"Also checked" list:** venues were added.
    - PromptEM: PVLDB 16(2), 2022. It was previously marked "venue UNVERIFIED".
    - Machamp: CIKM 2021.
    - NLSHBlock: SDM 2024.
    - LinkTransformer: ACL 2024 Demos.
    - Moslemi et al.: DKE 164 (2026).
    - Bopardikar et al. remain arXiv-only.
11. **Zeakis (row 19):** "S-GTR-T5 best, then S-MiniLM" was not supported. The paper singles out S-GTR-T5 and
    says the differences among SBERT models are minor on average. Reworded.
12. **Beyond Scale:** the throughput figures are the paper's Table 7 *"estimated inference time"*, derived from
    measured throughput. They are not wall-clock runs over 1M pairs. Labeled as such in §0, row 31 and §4.
13. **Peeters EDBT'25 (row 27):** noted an inconsistency inside the paper: the text says −0.73 and Table 4 prints
    −0.33.
14. **DeepMatcher, TL;DR:** the "Magellan wins below ~50K labels" result comes from a training-size sweep of the
    Hybrid model on a structured dataset. It is not about "the best DL model" in general. Reworded.
15. **DistillER (row 36):** the title is now written in full ("...with Large Language Models").
16. **Venue metadata added where the notes lacked it:**
    - EMTransformer: pp. 463–473.
    - R-SupCon: pp. 248–251.
    - WDC Products: EDBT 2024, pp. 22–33, doi:10.48786/edbt.2024.03.
    - Sudowoodo: pp. 1502–1515 and DOI.
    - LLM-CER: PACMMOD 3(4), doi:10.1145/3749170.
    - LLM4MEM: pp. 208–220.
    - Using ChatGPT: CCIS pp. 221–230.
17. **Model table (§5):**
    - **IndicBERTv2-MLM-only:** the card text gives 278M params and 23 Indic languages + English. The old entry
      said "n/a on card".
    - **MuRIL:** the card's 17 languages *include* English (16 Indian + English).
    - **paraphrase-multilingual-MiniLM-L12-v2:** its metadata lists ~50 languages and **excludes Tamil, Kannada,
      Bengali, Telugu and Malayalam**. Flagged as weak for the cross-script slice.
    - **Phi-4-mini-instruct:** its 23 card languages include **no Indic language**. Flagged.
    - **Mistral-7B-Instruct-v0.3:** the card states no language coverage, so "mostly European / good at French" is
      now marked UNVERIFIED.
    - **byt5-small:** "about 300M" is marked UNVERIFIED, because the card gives no count.
    - **IndicXlit:** confirmed that both en→indic and indic→en fairseq checkpoints are in the repo.

### Still UNVERIFIED (kept, clearly marked)
- The venue of AnyMatch (arXiv 2409.04073). Neither the PDF nor Crossref names one.
- Mistral-7B-v0.3's language strengths, and byt5-small's parameter count (see above).
- Everything tagged **[estimate]**: the §3 lift estimates, the §4 FLOP/throughput table and the training costs.
  These are the original author's arithmetic, not sourced numbers, and must be benchmarked on cloud hardware.

### Nothing was deleted
Every paper, model and repo in the notes exists and is correctly attributed. No fabricated item was found. The
corrections above are metadata and number-level fixes.

---

## Audit log (second independent re-audit, 2026-09-25)

A second auditor checked this file from scratch. The first audit log above was treated as unverified claims,
not as evidence. Only HTTP metadata fetches and PDF text extraction (PyMuPDF) ran locally; no model was loaded
or run.

### Scope and method
- **Papers (49 = 43 in §2 + 6 in "Also checked"):**
  - All 37 arXiv IDs were resolved in one arXiv API query. Every ID exists, and the titles and author lists match
    the notes.
  - 34 DOIs were resolved on Crossref (30 by DOI and 4 by title search: NLSHBlock, DSSM, LaBSE and Kasai). The
    three EDBT DOIs (OpenProceedings, 2020/2024/2025) were resolved on DataCite.
  - ComEM and LinkTransformer were checked against the ACL Anthology BibTeX. The Papadakis ICDE listing was
    checked on the Zenodo API, and DSSM on its Microsoft Research page.
  - Every quoted number was re-checked in the extracted PDF text of 30 papers: DeepMatcher, EMTransformer,
    Papadakis, Ditto, JointBERT, WDC Products (arXiv and EDBT versions), Sudowoodo, Unicorn (SIGMOD Record),
    Zeakis, DeepBlocker, Sparkly, Magellan, Peeters EDBT'25, AnyMatch, Beyond Scale, ComEM, DistillER, SGER,
    LLM-CER, TransClean, R-SupCon, Cross-Language, LLM4MEM, CorpFam, Alper, SBERT, AugSBERT, Steiner et al.,
    BatchER and MultiEM.
- **Models (55 HF ids):** `huggingface.co/api/models/<id>` was read for each one: `cardData.license`,
  `license_name`, `safetensors.total`, `gated` and `cardData.language`. Raw README cards were read for MuRIL,
  mDeBERTa, IndicBERTv2, Mistral-7B-v0.3, ByT5, bge-m3, e5-small, Qwen2.5-7B, Qwen3-4B and the three Jellyfish
  models. The file trees were read for IndicXlit and ByT5, and MuRIL's config.json was read. As extra context, I
  also checked the licenses of Qwen2.5-3B (base), 14B, 32B and 72B, of mistralai/Mistral-7B-v0.3 (base), and of
  microsoft/Multilingual-MiniLM-L12-H384 (the e5-small backbone, mit).
- **Repos (22):** `gh api repos/<r>` returned license SPDX, stars, `pushed_at` and archived status. For the 9
  repos reported as "none", the root listing was searched for LICENSE/COPYING files, and none has one. README
  venue claims were read for sudowoodo (says ICDE 2023), MultiEM (names no venue) and jointbert (says PVLDB 2021).

### Confirmed (no substantive change)
- **All 49 papers exist, and their titles, authors and years are correct.** Every venue, page range and DOI in
  the notes matched Crossref, DataCite or the ACL Anthology. That includes the new-in-2026 items:
  - SGER: ACL 2026 Industry, pp. 1461–1468.
  - Alper: KDD 2026, pp. 4906–4915.
  - LLM-CER: PACMMOD 3(4).
  - TransClean: IEEE Access 13.
  - LLM4MEM: LNCS pp. 208–220.
  - Moslemi et al.: DKE 164, article 102575.
  - Beyond Scale (arXiv 2607.24688) and CorpFam (arXiv 2609.04269) both exist as arXiv preprints.
- **Numbers re-found verbatim in the PDFs.** Every number the first audit listed as confirmed was found again.
  Examples: Ditto Table 9 (537.26 / 2,229.26 / 1,981.97 / 1,339.36 / 22,823.43 s); Beyond Scale Table 7
  (3,130.7 / 11,113.6 / 16,920.5 and 312.9 / 1,111.9 / 1,693.2 s); SGER (0.994 / 0.726 / 0.806 / 0.911 / 0.946,
  3× L4, 10,000 RPM, P99 120 ms); TransClean +24.42; ComEM +16.02 %; Peeters −0.73 (text) vs −0.33 (Table 4).
- **All 55 HF license strings are exact**, and every safetensors parameter count quoted in §5 matches. That
  includes the 8B-cap blocker for Qwen3-8B (8,190,735,360 parameters).
- **All 22 repo licenses and last-push dates are exact.** Star counts were exact apart from two drifts of +1 or
  +2 (fixed below).
- **Prior-audit fixes that I independently confirmed:**
  - The Ditto Table 9 re-assignment.
  - Dd4 added to the Papadakis list of challenging datasets.
  - The false claim that the MultiEM README names a venue.
  - The Unicorn author orders (PACMMOD vs SIGMOD Record).
  - The COLING author order for ComEM (Sun is 6th).
  - The Cross-Language title ("Product Matching").
  - The Steiner, TransClean and Alper venues.

### Changes made in this pass (what and why)
1. **EMTransformer (row 4, §3):** the paper says transformers beat "classical deep learning methods in EM
   [7, 20]" by 27.5 %. Its refs [7] and [20] are DeepER and DeepMatcher, so the gain is not over DeepMatcher alone.
   Reworded, and noted that the DOI resolves through DataCite.
2. **Peeters EDBT'25 (row 27, TL;DR item 4, §7):** the "36 to 56 %" drop comes from RoBERTa and Ditto models
   fine-tuned on the *other* benchmark datasets (the paper excludes the WDC-trained models) and then applied to
   the WDC Products test set. That is cross-dataset transfer, not an unseen-entity split of one dataset. The
   earlier wording ("moved to unseen WDC products") overstated how directly it applies to our same-schema,
   new-country shift. Clarified it and added the RoBERTa 22 to 61 % and "LLMs ≥ 8 % above the best transferred
   PLM" figures. Also added doi:10.48786/edbt.2025.42 (DataCite).
3. **R-SupCon (row 11):** "then a frozen encoder with a CE head" was inaccurate. The paper states that the encoder
   "can either be frozen or further tuned", and it reports both variants (the F and UF columns). Fixed.
4. **LLM-CER (row 35):** the abstract says a "10% increase in the **FP-measure**", which is the paper's clustering
   metric, not a plain "+10 % F-measure". Fixed, and added "nine real-world datasets".
5. **AnyMatch (row 30):** the venue was "UNVERIFIED". The authors' DEEM lab page lists it under the AAAI 2025
   workshop "Preparing Good Data for Generative AI". The workshop CFP describes that workshop as non-archival.
   OpenReview (id Nees1OD5td) was blocked by a browser check, so this rests on the author page and is marked
   that way.
6. **Qwen2.5 license claim (TL;DR item 6, §5 non-compliant table):** "its Qwen2.5 siblings are Apache" is
   **wrong** as a general statement. On HF, Qwen2.5-72B-Instruct is `other` / `qwen`, and the Qwen2.5-3B base
   model is also `other` / `qwen-research`. Only 0.5B/1.5B/7B/14B/32B are apache-2.0. Corrected. This doesn't
   change the whitelist, because 72B is over the cap anyway, but the gate should list both 3B ids.
7. **MuRIL params (§5):** "about 110M+" understated the size. config.json has vocab_size 197,285 × hidden 768,
   which is about 152M in embeddings on top of a BERT-base encoder of about 86M. That gives about 237M; this is
   my arithmetic, and the card gives no count. Updated.
8. **byt5-small params (§5):** it was "about 300M UNVERIFIED". The fp32 `pytorch_model.bin` is 1,198,627,927
   bytes, i.e. about 300M parameters. The UNVERIFIED tag was removed, and the entry now says how the figure was
   derived.
9. **Qwen2.5-7B-Instruct languages (§5):** the card says "over 29 languages" and names no Indic language. Since
   the notes propose it as the teacher and judge, which includes cross-script cases, I added a caveat: test it on
   the cross-script slice first. (Qwen3 cards claim 100+ languages.)
10. **DeepMatcher (row 1):** the textual gain is "3.0-22.0% F1" in the abstract but "5.8 to 22.0% relative F1" in
    the textual-results section. Both are now shown. Added pp. 19–34.
11. **Metadata added (Crossref-verified, except Magellan):**
    - DeepER: PVLDB 11(11):1454–1467, plus its arXiv title.
    - Magellan: pp. 1197–1208, taken from the 12-page VLDB PDF starting at p. 1197. Note that
      doi:10.14778/3007263.3007314 is the separate 4-page *demo* paper in PVLDB 9(13), pp. 1581–1584, so don't
      cite that DOI for this paper.
    - Ditto: pp. 50–60.
    - JointBERT: pp. 1913–1921.
    - HierMatcher: doi:10.24963/ijcai.2020/507.
    - Zeakis: doi:10.14778/3598581.3598594.
    - DeepBlocker: doi:10.14778/3476249.3476294.
    - Sparkly: doi:10.14778/3583140.3583163.
    - DSSM: pp. 2333–2338, doi:10.1145/2505515.2505665.
    - ESIM: pp. 1657–1668.
    - LaBSE: pp. 878–891, doi:10.18653/v1/2022.acl-long.62.
    - Kasai et al.: pp. 5851–5861, doi:10.18653/v1/P19-1586.
    - CorpFam: full title.
12. **Repo stars (§6):** sentence-transformers is now 19,118 → 19,120, and splink 2,428 → 2,429 (live drift
    on 2026-09-25).

### Supersedes in the first audit log
- "Still UNVERIFIED: AnyMatch venue" now has a workshop listing on the author page (item 5 above).
- "Still UNVERIFIED: byt5-small parameter count" is resolved (item 8).
- "All 55 HF license strings are exact" still holds. What was wrong was the prose claim about the Qwen2.5 siblings
  (item 6), not any license field.

### Still UNVERIFIED
- Mistral-7B-Instruct-v0.3's language strengths: the card makes no language claim.
- AnyMatch's venue on OpenReview itself, which could not be opened.
- Everything tagged **[estimate]** (§3 lift, the §4 FLOP/throughput table, training costs). This is the original
  author's arithmetic, and it must be benchmarked on cloud hardware.

### Deleted
Nothing. No fabricated paper, model or repo was found; all 126 listed items exist on their primary sources.

---

## Audit log (third independent re-audit, 2026-09-25)

A third auditor re-checked the file from scratch and treated both earlier audit logs as unverified claims.
Locally, only HTTP metadata fetches and PDF text extraction (PyMuPDF) ran. No model was loaded or run.

### Scope and method
- **Papers (49 = 43 in §2 + 6 in "Also checked"):**
  - All 37 arXiv IDs were resolved in one `export.arxiv.org/api/query` call. All exist, and the titles, author
    lists, comments and journal-refs match the notes.
  - 31 DOIs were resolved on `api.crossref.org/works/<doi>`, and the 3 OpenProceedings DOIs (EDBT 2020/2024/2025)
    on `api.datacite.org`. Title, authors, venue and pages match in every case.
  - Title searches on Crossref found the missing DOIs for Magellan, SBERT, AugSBERT, Machamp, NLSHBlock,
    LinkTransformer and Moslemi et al.
  - The ACL Anthology BibTeX was read for ComEM (COLING order: Sun is 6th, pp. 96–109), Jellyfish and SGER. The
    IJCAI pages were read for HierMatcher and Wang et al. 2022.
  - 38 PDFs were downloaded and their text searched for every quoted number: DeepMatcher, EMTransformer,
    Papadakis, Ditto, JointBERT, WDC Products (EDBT), Sudowoodo, Unicorn (SIGMOD Record), Zeakis, DeepBlocker,
    Sparkly, Magellan, HierMatcher, Wang'22, Peeters EDBT'25, AnyMatch, Beyond Scale, BatchER, ComEM, DistillER,
    LLM-CER, SGER, Cross-Language, MultiEM, TransClean, LLM4MEM, Alper, CorpFam, Steiner, R-SupCon, LaBSE,
    Poly-encoders, DeepER, SBERT, AugSBERT and Machamp.
  - The AnyMatch venue was checked on the DEEM lab page ("Workshop on Preparing Good Data for Generative AI at
    AAAI"). The GoodData CFP confirms the workshop is non-archival. OpenReview returned a bot challenge, which was
    not bypassed.
- **Models (56 HF ids named in §5):** `huggingface.co/api/models/<id>` was read for each one: `cardData.license`,
  `license_name`, the license tag, `safetensors.total`, `gated` and `cardData.language`.
  - Raw README cards were read for MuRIL, mDeBERTa, IndicBERTv2, Mistral-7B-v0.3, Qwen2.5-7B, Qwen3-4B, bge-m3,
    the mmarco cross-encoder, IndicXlit and all three Jellyfish models.
  - File trees were read for ByT5 and IndicXlit, and MuRIL's config.json was read.
  - The license chain of the mmarco cross-encoder was followed: its base checkpoint's HF card, then the
    microsoft/unilm GitHub repo.
  - Qwen2.5 0.5B/1.5B/7B base, 14B, 32B and 72B were also checked, to verify the "Qwen2.5 siblings" claim.
- **Repos (22):** `gh api repos/<r>` gave SPDX license, stars, `pushed_at` and archived status. For the 9 repos
  reported as "none", the root listing and README were checked, and none has a LICENSE/COPYING file or a license
  statement. README venue claims were read for sudowoodo (ICDE 2023), jointbert (PVLDB 2021) and MultiEM (no
  venue).

### Confirmed (no change needed)
- **Every paper, model and repo exists** and is attributed correctly. No fabricated item was found.
- **Paper numbers found verbatim in the PDFs:**
  - DeepMatcher: 87.9/88.8, 5.4h/1.5m, 8 of 11, <0.7 %, 2.4 %/20.2 %, 3.0–22.0 %, 5.8–22.0 %, 6.2–32.6 %,
    92.7/79.8, 50K/200K, 1.1 %.
  - EMTransformer: 27.5 %, 86 %/90 %.
  - Papadakis: "rather easy"; NLB/LBM ≥ 5 %, ideally 10 %; practical-measure challenging sets Ds4, Ds6, Dd4, Dt1.
  - Ditto: 789,409 × 412,418; 10,652,249 candidates; 20,000 labels; 39 %; 96.53; Table 9 537.26 / 2,229.26 /
    1,981.97 / 1,339.36 / 22,823.43 s; 1.69 h vs 6.49 h; 92 %; V100; fp16; max length 256; 29 %; 9.8 %.
  - JointBERT: 1 % to 5 % on seen products, underperforms on unseen.
  - WDC Products: 72.18–79.99; R-SupCon −25 %; >78 %; corner cases hurt precision.
  - Sudowoodo: 500 labels. Unicorn: DeBERTa, MoE, 20 datasets, 7 tasks.
  - SBERT: 50M inferences, 65 h vs 5 s. AugSBERT: +6 / +37.
  - Zeakis: 12 LMs, 17 datasets, +15 % recall on 8, <1 min vs ZeroER >6 h, "differences are minor".
  - DeepBlocker: Autoencoder/Hybrid. Sparkly: 8 blockers, 10M tuples <100 min, 10 nodes, $12.5, 3-grams.
  - Peeters EDBT'25: 2.65–4.71; 3.69/4.49/0.73 in the text vs −0.33 in Table 4; 84.90; 36–56 %; 22–61 %; ≥8 %.
  - AnyMatch: 124M, 4.4 %, 3,899×.
  - Beyond Scale: 1,215 runs; 72.5/79.0/81.5 vs 83.0/84.1/84.1; Table 7 values; 86.4 F1 at ~165 pairs/s;
    >850 GPU-hours.
  - ComEM: 16.02 %, up to 4 %, position bias. BatchER: covering-based selection.
  - LLM-CER: 150 %, 10 % FP-measure, 5×, nine datasets.
  - SGER: 0.994 / 0.726 / 0.806 / 0.911 / 0.946; 10K names; 3× L4; 10,000 RPM; P99 120 ms; Llama 3 8B.
  - TransClean +24.42; LLM4MEM +5.1 % on 6 datasets; CorpFam 93.2 %.
- **Venue metadata:** every venue, page range and DOI already in the notes matches Crossref, DataCite or the ACL
  Anthology. That includes SGER (ACL 2026 Industry, 1461–1468), Alper (KDD 2026, 4906–4915), LLM-CER (PACMMOD
  3(4)), TransClean (IEEE Access 13), LLM4MEM (LNCS 208–220), Steiner (ICDEW 2025, 9–17), Papadakis (ICDE 2024,
  3435–3448), MultiEM (ICDE 2024, 3421–3434) and BatchER (ICDE 2024, 3696–3709).
- **All 56 HF license strings are exact**, and every safetensors parameter count in §5 matches the API. Key
  values:
  - Qwen3-8B: 8,190,735,360.
  - Qwen2.5-3B and 3B-Instruct: other/qwen-research.
  - Qwen2.5-72B: other/qwen.
  - Qwen2.5 0.5B/1.5B/7B/14B/32B: apache-2.0.
  - Jellyfish 7B/8B/13B: cc-by-nc-4.0, with bases Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct and
    OpenOrca-Platypus2-13B.
  - Llama-3.1-8B: llama3.1. gemma-3-4b-it: gemma. jina v3 and reranker-v2: cc-by-nc-4.0. l3cube: cc-by-4.0.
  - nreimers mMiniLMv2: no license.
  - The card-text claims (MuRIL 17 langs incl. English and IndicTrans-transliterated Wikipedia; mDeBERTa 86M +
    190M; IndicBERTv2 278M and 23 Indic + en; Qwen2.5 "over 29 languages"; Qwen3 "100+ languages"; bge-m3 100+
    and 8192 tokens; ByT5 bin 1,198,627,927 bytes; both IndicXlit `indicxlit.pt` checkpoints) are all confirmed.
- **All 22 repo licenses, stars and last-push dates are exact**, apart from a +1 star drift (below).

### Changes made in this pass (what and why)
1. **SGER (row 37, §8 item 11): scope overstated.** The paper's data and every example are *Latin-script*
   romanizations of Indian names ("Subham"/"Shubham"). The extracted text contains zero Indic-script characters,
   and "inconsistent transliteration across scripts" refers to romanization. So it is **not** evidence for our
   native-script ↔ Latin slice; the structured summary's "Indian cross-script name matching" wording was wrong.
   Two further fixes:
   - The 0.946 → 0.994 gain was attributed to the curriculum alone. The paper says curriculum *plus*
     augmentation, and SFT + augmentation alone reaches 0.973.
   - Added: 20K → 50K+ training pairs, a 50K name-disjoint test set, the Dream11 KYC setting, and the three named
     augmentations.
2. **AnyMatch (row 30):** the AutoML filter keeps only the *positive (matching)* pairs that the AutoML model
   misclassifies, i.e. its false negatives, not "pairs it misclassifies" in general. Fixed.
3. **DistillER (row 36):** "maximum blocking similarity gives the best trade-off" was wrong. The paper finds that
   *Ranking* by blocking similarity is the most reliable unsupervised selection method under **both** the Max and
   Top-2 scores. It is the SLM *training* time that is "up to a whole order of magnitude" faster. The RL baselines
   are named (GRPO/DPO). Fixed.
4. **TransClean (row 40):** the +24.42 average is reported "when run with manual labeling". Caveat added.
5. **EMTransformer (row 4):** the second audit's claim that the 27.5 % is "not over DeepMatcher alone" is
   contradicted by the paper's own contributions list ("the best transformer outperforms DeepMatcher[20] by an
   average margin of 27.5%"). Both phrasings are now shown.
6. **Zeakis (row 19):** added that in *supervised* matching all BERT-based models excel and RoBERTa ranks first on
   average. The summary's global "SBERT-family best" only holds for blocking and unsupervised matching.
7. **Poly-encoders (row 17):** "m cached context codes" misdescribed the model. The *candidate* embeddings are
   cached; the context is summarized by m learned codes. Fixed.
8. **Beyond Scale (row 31, §4):** added the paper's *measured* throughput (0.6B cross-encoder ~473 pairs/s, 4B
   ~165 pairs/s), which is faster than Table 7 implies (~319 and ~90 pairs/s). Table 7 is now labeled as the
   conservative estimate.
9. **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (§5, §8 item 5): license-chain caveat.**
   - Its card is apache-2.0, but it declares `base_model: nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`,
     whose HF card has **no license**. The notes themselves list that model family as non-compliant.
   - Upstream microsoft/unilm is MIT, so the risk is low but not zero.
   - It was trained on machine-translated mMARCO (14 languages), and its metadata lists no Tamil/Kannada/Bengali/
     Telugu/Malayalam.
   - Recommended the multilingual-e5-small (MIT) backbone as the clean-chain alternative.
10. **granite-embedding multilingual (§5):** the card metadata lists 12 languages with **no Indic language**.
    Flagged as Latin/French-slice only.
11. **Sudowoodo adaptation (row 13):** "all 22M records" was wrong arithmetic. Train is 12.5M and test is
    ~11.7M, about 24M in total. Fixed.
12. **Metadata added (Crossref-verified):**
    - Magellan: doi:10.14778/2994509.2994535.
    - SBERT: EMNLP-IJCNLP 2019, pp. 3980–3990, doi:10.18653/v1/D19-1410.
    - AugSBERT: NAACL 2021, pp. 296–310, doi:10.18653/v1/2021.naacl-main.28.
    - Peeters EDBT'25: pp. 529–541 (Experiments & Analyses).
    - Machamp: doi:10.1145/3459637.3482008.
    - NLSHBlock: doi:10.1137/1.9781611978032.101.
    - LinkTransformer: doi:10.18653/v1/2024.acl-demos.21.
13. **Repo stars:** sentence-transformers 19,120 → 19,121 (live drift).

### Still UNVERIFIED
- Mistral-7B-Instruct-v0.3's language strengths. The card makes no language claim.
- AnyMatch on OpenReview itself (bot challenge). The venue rests on the authors' lab page and the GoodData CFP
  (a non-archival workshop).
- Everything tagged **[estimate]**: the §3 lift, the §4 FLOP/throughput table and the training costs. This is the
  original author's arithmetic and must be benchmarked on cloud hardware.

### Deleted
Nothing. All 127 items (49 papers, 56 models, 22 repos) exist on their primary sources. The corrections are
about scope, wording, license-chain risk and metadata, not fabrication.
