# Blocking and candidate generation at scale: research notes

Topic owner: research workflow, topic "blocking". Written 2026-09-25. All sources were opened on 2026-09-25. The table in section 9 gives the URL used to confirm each one.
Scope: blocking, filtering and candidate generation for the Amazon ML Challenge 2026 Business Entity Resolution task. There are 1.73M test S1 queries against about 10M S2+S3 targets, and the corpus is multi-script (Latin, Devanagari, Tamil, Kannada and others). France is unseen in train. The score is per-S1 F0.5, macro-averaged.

Compute rule (from the user, binding): **nothing in this document is to be run on the laptop.** All indexing, embedding and ANN work goes to Modal, Kaggle, Colab or Lightning AI. Any throughput or memory figure marked *(estimate)* is back-of-envelope arithmetic, not a measurement.

---

## 0. TL;DR

1. **Use top-k (cardinality) blocking, not similarity thresholds.** Three studies reach this conclusion: Sparkly (PVLDB 2023), DeepBlocker (PVLDB 2021) and Papadakis et al. (ICDE 2023). *(Audit: Sparkly and DeepBlocker share authors (Govind, Paulsen, Doan), so only two of the three are fully independent.)* Real-world match scores spread across the whole [0,1] range, so a single threshold either loses recall or blows up the output size.
2. **Sparse TF-IDF/BM25 over character 3-grams is a very strong baseline.** Sparkly beat 8 SOTA blockers, including DeepBlocker's best DL models, on 15 datasets. Its recall was 92.5-100% at k=10, 96.4-100% at k=20 and 98.7-100% at k=50, and its automatic variant (Sparkly Auto) blocked 10M-tuple tables in under 100 minutes on 10 AWS m5.4xlarge nodes (16 cores each). **IDF is essential and TF barely matters** on short fields. 3-grams beat 2-grams and 4-grams.
3. **Dense and sparse blockers complement each other, so union them.** In DeepBlocker's experiments, unioning DL with an industrial rule-based blocker raised recall on textual data from 68.7 to 83.0 (Walmart-Amazon2) and from 70.5 to 85.0 (Amazon-Google2). *(Audit correction: the paper says size grew "minimally to moderately" (up to 57.6% on textual data), but that is measured against the larger member. Against DL alone, the Walmart-Amazon2 union went from 12.8k to 7.9M candidates (Table 6) because RBB alone produced 7.9M. A union costs about the sum of its members, so budget each member's k separately.)* For this challenge, dense multilingual encoders are the natural bridge for **cross-script** names where n-grams have nothing in common.
4. **Our data has a structural gift.** DATA_ANALYSIS.md measured that every matched S2/S3 record belongs to **exactly one** S1. The mapping is many-to-one, so we should **index the small side (S1, 1.73M) and probe from the large side (each of the ~10M targets retrieves its top-k S1)**. This is also Sparkly's recommendation: index the smaller table and probe from the larger one. It handles a variable number of copies per S1 without any per-S1 k, it keeps the GPU index small (S1 fits a T4), and the matcher can later enforce "at most one S1 per target".
5. **Country-label partitioning is lossless on train** (100% agreement) and works for any label, so it is not hard-coding. Apply it first.
6. **S1 names are highly non-unique** (only 64-74% distinct), so name-only blocking produces massive ties. The workhorse must be a **name+address multi-field BM25** (Sparkly-Auto style: the sum of per-attribute BM25 scores), plus an **address-only** blocker for alias, domain and cross-script names.
7. **Allocate the candidate budget with a learned pair-probability, not a fixed k.** Train a small classifier on blocker ranks, scores, score gaps, reciprocity and multi-blocker agreement (this is supervised meta-blocking). Keep pairs above a global probability cut chosen to hit the budget. This is the greedy-optimal allocation, and it spends more candidates on hard queries where the score gaps are small.
8. **Optimize the right blocking objective.** That is the *oracle-matcher macro-F0.5 ceiling* computed from per-S1 found/missed counts, not pair recall alone. Missing 1 of 4 copies caps that entity at 0.9375; missing 1 of 2 caps it at 0.833.

---

## 1. What the challenge implies for blocking (from measured data, DATA_ANALYSIS.md)

| Fact (measured) | Blocking consequence |
|---|---|
| Test: 1.73M S1, 4.89M S2, 5.08M S3; France 15% of test S1 | Brute force is 1.7e13 pairs, so blocking is mandatory. France is worth about 0.15 of the score and blocking must be language-agnostic. |
| Target multiplicity = 1 (no target linked to 2 S1) | Probe targets into an S1 index (target to S1, top-k) and enforce assignment at match time. |
| Country agreement = 100% on 7.64M true pairs | Partition by country label first. This alone removes about 60% or more of the cross product *(estimate from the country mix)*. |
| 26% of S2/S3 are distractors (no S1); test has about 24% more targets per S1 than train | Candidate volume on test will be higher than on train. Budget rules must be relative per query, not global constants tuned on train. |
| S1 name uniqueness is 0.64 (India) to 0.74 (US) | Name-only top-k is tie-dominated, so address must be part of the retrieval score. |
| India target names: 18-28% non-ASCII; India positive-pair name token-set p10 = 11 | Without transliteration, about 20% or more of India pairs have near-zero Latin name overlap. Need a cross-script bridge (transliteration, a multilingual dense encoder, or both). |
| India positive-pair address token-set p50 = 94-100 | Address-only retrieval rescues cross-script and alias names. |
| Native-script state names in about 23% of India target addresses | Transliterate addresses too, not just names. |
| Postcodes rare (US ZIP5 about 11%, India PIN <2%, France about 0.5%) | Postcode hash-blocking is only a minor union member. |
| About 80% of positive pairs share a number token | A house-number + street-token hash key is a good cheap union member (purge oversized blocks). |
| Frequent 2-letter acronym names ("SC", "SS", "PC") | Char 3-grams are nearly empty for these. Pad boundaries (`char_wb`) or add a whole-token / 2-gram key, and lean on address. |
| Within-source copies share the same corrupted base | Target-to-target similarity is strong, so **graph expansion** (add near-duplicate neighbours of retrieved targets) raises recall cheaply. |

---

## 2. Literature by theme (verified; details and URLs in section 9)

### 2.1 Surveys and taxonomy
- **Papadakis, Skoutas, Thanos, Palpanas. "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys 53(2), 2020.** *(Audit pass 2: Crossref gives this title and subtitle; the arXiv 1905.06167 version is titled "A Survey of Blocking and Filtering Techniques for Entity Resolution".)* This is the canonical taxonomy. It covers blocking (schema-based and schema-agnostic), filtering (threshold-based similarity joins) and hybrids, plus block processing (block purging and filtering, meta-blocking). Use it as the map of the design space.
- **Christophides, Efthymiou, Palpanas, Papadakis, Stefanidis. "An Overview of End-to-End Entity Resolution for Big Data." ACM CSUR 53(6), 2020/21.** Covers end-to-end workflows: indexing, matching, and execution at scale. *(Audit: the arXiv 1905.06397 title is "End-to-End Entity Resolution for Big Data: A Survey"; the published CSUR title is confirmed on Crossref.)*
- **Christen. "A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication." IEEE TKDE 24(9), 2012.** *(Audit pass 2: metadata confirmed on Crossref, and the content confirmed in the author's PDF at users.cecs.anu.edu.au/~christen/publications/christen2011indexing.pdf. The abstract says it surveys "twelve variations of six indexing techniques".)* This is the classic reference for the traditional family: traditional blocking, sorted neighbourhood, q-gram, suffix-array, canopy clustering and string-map indexing, with Soundex-style phonetic encodings as blocking keys. It uses the measures "pairs completeness", "pairs quality" and "reduction ratio".
- **Papadakis, Svirsky, Gal, Palpanas. "Comparative Analysis of Approximate Blocking Techniques for Entity Resolution." PVLDB 9(9), 2016.** Evaluates 17 methods on 13 datasets up to 2M entities.
  - Taxonomy: Block Building, then Block Cleaning, then Comparison Cleaning.
  - Measures: PC = |D(B)|/|D(E)|, PQ = |D(B)|/||B||, RR = 1 - ||B||/||E||.
  - Findings:
    - The lazy (schema-agnostic token) methods keep PC above 0.80 on almost all datasets.
    - The default configuration of Block Filtering keeps each entity in the smallest r ≈ 0.45-0.55 fraction of its blocks.
    - Standard Blocking plus cleaning resolves D2M in about 1.5 h, against an estimated year for brute force.
    - All methods depend heavily on fine-tuning.
    - The meta-blocking pruning family it uses (CEP, CNP, ReCNP, WEP, WNP, ReWNP) is exactly the vocabulary for "per-query adaptive budget" (section 5). *(Audit correction, pass 2: the paper describes all six schemes in a short paragraph ("Six algorithms have been proposed [27, 28]"). It attributes them to [27], the TKDE 2014 meta-blocking paper, and [28], "Scaling Entity Resolution to Large, Heterogeneous Data with Enhanced Meta-blocking" (Papadakis, Papastefanatos, Palpanas, Koubarakis, EDBT 2016, pp. 221-232, DOI 10.5441/002/edbt.2016.22). The EDBT paper introduces "Reciprocal Pruning", which is where ReCNP and ReWNP come from.)*

### 2.2 Token blocking and meta-blocking (schema-agnostic)
- **Meta-blocking (Papadakis, Koutrika, Palpanas, Nejdl, TKDE 26(8), 2014).** Build a blocking graph with entities as nodes and co-occurrence as edges. Weight the edges with ARCS, CBS, ECBS, JS or EJS, then prune:
  - globally with WEP/CEP;
  - per node with WNP/CNP (WNP prunes the edges below a local, automatically set per-node threshold, which makes it an adaptive per-query threshold);
  - reciprocally with ReWNP/ReCNP. *(Audit: the reciprocal variants were added later, in EDBT 2016 "enhanced meta-blocking", not in the TKDE 2014 paper.)*
- **BLAST (Simonini, Bergamaschi, Jagadish, PVLDB 9(12), 2016).** "Loosely schema-aware" meta-blocking. It clusters attributes by value similarity (using an LSH step) and uses attribute-aware entropy to weight blocks. This is relevant because our fields (name vs address) should be weighted differently.
- **Generalized Supervised Meta-blocking (Gagliardelli, Papadakis, Simonini, Bergamaschi, Palpanas, PVLDB 15(9), 2022).** Combines multiple weighting schemes per candidate pair into a feature vector. A binary classifier gives a match probability, which then drives any pruning algorithm. The paper studies minimal training-set sizes. *(Audit pass 2, read in the vldb.org PDF: 9 real-world datasets; "50 labelled instances (25 per class) suffice for high performance"; scalability tested on synthetic datasets of up to 300,000 entities.)* **This is the template for our budget allocator (section 5).**
- **JedAI (Information Systems 93, 2020) / pyJedAI (Apache-2.0).** Reference implementations of all of the above. For us, pyJedAI is useful on **samples** (100k-1M) to compare token blocking + meta-blocking against Sparkly-style top-k. It is not designed around 12M-record multi-script data on one box, so use it for prototyping, not production.
- **Papadakis et al., ICDE 2023, "Benchmarking Filtering Techniques for ER"** (pp. 653-666, DOI 10.1109/ICDE55515.2023.00389; the current arXiv listing title is "How to reduce the search space of ER: with Blocking or Nearest Neighbor search?"). Compares 14 filters and 4 baselines on 10 datasets. *(Audit: the current arXiv abstract (v5) is broader. It also covers string similarity joins and concludes that "blocking workflows and string similarity joins" are superior. The conclusions below were re-read in the ICDE-titled PDF (v4) and match it.)* Its conclusions:
  1. Fine-tuning parameters matters a lot.
  2. **Schema-agnostic beats schema-based** in robustness.
  3. **Cardinality thresholds (top-k) beat similarity thresholds.** With top-k the number of candidates is linear in input size, while with a similarity threshold it is quadratic, and in almost all cases k ≪ 100 sufficed.
  4. **Syntactic representations beat semantic (pre-trained) ones** on their data.
  5. **kNN-Join** (character q-gram, cardinality-based sparse NN) is named the most effective filter. *(Audit nuance: the paper says the Standard Blocking Workflow (SBW) actually "performs better in the schema-agnostic settings". kNN-Join is preferred for two qualitative reasons: its candidate count is linear in input size, and it is easy to configure.)*
  6. **FAISS is by far the fastest and most scalable.** It handles D2M within 1.3 h, and SCANN comes second. *(Audit pass 2: in that benchmark FAISS used an approximate IVF index and SCANN used AH, so the 1.3 h is ANN speed. Section 6.2's exact flat search is slower per query but loses no recall to the ANN step.)*
  7. The LSH variants reach high recall only by producing excessive candidates.

### 2.3 Classical families (foundational)
- **Sorted neighbourhood / merge-purge (Hernández and Stolfo, SIGMOD 1995).** Sort by a key and compare within a sliding window. Multi-pass with different keys, then take the transitive closure. Cheap, but key-order sensitive. Our noise (reordered components, moved legal suffix, junk prefixes) breaks single-key sorting unless the key is built after normalization (for example, sorted rare tokens).
- **Canopy clustering (McCallum, Nigam, Ungar, KDD 2000).** Uses a cheap metric (TF-IDF on an inverted index) with loose and tight thresholds to create overlapping canopies. The comparative study found canopy-style methods effective but not scalable to the largest data.
- **q-gram / suffix-array / extended variants.** In the 2016 comparison, Q-grams Blocking (QGBl) uses q=6 by default. *(Audit correction: the earlier text said Q-grams Blocking scales linearly. The paper says the opposite: the lazy methods, QGBl among them, scale "slightly super-linearly". Only Suffix Arrays, Extended Suffix Arrays and Sorted Neighborhood scale linearly, and they are about 3× faster than Standard Blocking. Their PC falls from above 0.97 at D10K to below 0.80 at D2M.)*
- **Phonetic keys** (Soundex/Metaphone families, covered in Christen 2012). They are English-centric and weak for Indic transliterations and French. A **consonant-skeleton key after transliteration** is the language-agnostic analogue we recommend.
- **MinHash/LSH (Broder, SEQUENCES 1997) and all-pairs similarity search with prefix filtering (Bayardo, Ma, Srikant, WWW 2007).** These are the exact/approximate Jaccard machinery. Evidence against using them as the main blocker:
  - ICDE 2023 found that LSH needs far more candidates than top-k NN for equal recall.
  - DeepBlocker found that top-K cosine beats LSH.
  - Keep MinHash (datasketch, MIT) only for **near-duplicate grouping within S2/S3**, where Jaccard thresholds are meaningful.
- **Adaptive blocking (Bilenko, Kamath, Mooney, ICDM 2006).** Learns a DNF of blocking predicates from labeled pairs, maximizing recall under a cost constraint. This is the ancestor of learned blocking and of the dedupe library (MIT). We have millions of labeled train pairs, so learned key selection is feasible, but top-k retrieval makes it largely unnecessary.
- **Steorts, Ventura, Sadinle, Fienberg, "A Comparison of Blocking Methods for Record Linkage" (arXiv 2014).** Traditional blocking vs LSH on recall, reduction ratio and complexity. *(Audit: the venue is now confirmed. Privacy in Statistical Databases (PSD) 2014, LNCS, pp. 253-268, DOI 10.1007/978-3-319-11257-2_20, per Crossref.)*

### 2.4 TF-IDF / BM25 top-k blocking: Sparkly (PVLDB 16(6), 2023)
Paulsen, Govind, Doan. Key facts read from the paper:
- **Design.**
  - Build a Lucene inverted index on the **smaller** table.
  - Ship the index to all workers and probe with each tuple of the **larger** table, doing top-k per probe.
  - Doing top-k from both sides improved recall only minimally but cost much more runtime.
  - Share-nothing on Spark, using Lucene block-max WAND for fast top-k.
- **Sparkly Manual (SM).** Concatenate the blocking attributes, lowercase, tokenize into a bag of 3-grams, remove all non-alphanumeric tokens, and score with BM25 (k1 = 1.2, b = 0.75). *(Audit pass 2: the paper drops non-alphanumeric 3-gram tokens after tokenizing; it does not strip characters before tokenizing.)*
- **Sparkly Auto (SA).** Greedy search over (attribute, tokenizer) configs of up to 3 attributes. The score is the sum of the per-attribute BM25 scores. Configs are ranked by a label-free "discriminativeness" measure (normalized AUC of the top-k score curve on a 10k sample, k = 250), with early pruning by Wilcoxon tests.
- **Results.**
  - SM recall is 92.5-100% at k=10, 96.4-100% at k=20 and 98.7-100% at k=50, with output capped at k·|B|. The paper's point is **predictability** against PBW, DBW, JD and Union(DL, RBB) (Table 2). *(Audit: this is not strict dominance. PBW can reach 100% recall but with output up to 4.2B pairs; JD recall is 35.4-96.4%; Union(DL, RBB) recall is 83-99.9%.)*
  - SA beats SM on 10 of 15 datasets, and is at most 0.7% larger in output at 98% recall where it loses.
- **Scale.**
  - SA blocks 10M-tuple tables in under 100 min on 10 × m5.4xlarge (16 cores each), costing about $12.5.
  - It blocks WDC 26M in 130 min on 30 nodes (under $67.5).
  - Indexing 10M tuples produces a 1.3-2 GB index.
  - Autoencoder (DL) managed recall@50 of only 40 on MB 10M and 85 on BC 2.5M, against Sparkly's 94-100 on the same datasets (table 3). It took 691 min on MB 10M and 925 min on WDC 10M.
- **Ablations.**
  - **idf is crucial.** Removing idf hurts a lot.
  - **tf has minimal effect on short attributes.**
  - The 3-gram tokenizer is best; 2-grams and 4-grams are worst.
  - b = 0.75 is a good default, and k1 in [1, 2] is insensitive.
  - TFIDF-cosine beats BM25-SM on many datasets because it treats query and document symmetrically. **SM+**, which adds query-side tf/idf to BM25, is best or near-best on all datasets.
- **Top-k vs thresholding.** Gold-match score histograms (tf/idf, Jaccard, cosine) are spread across [0,1] on 2 of the 3 datasets plotted, so no single threshold gives high recall without blowing up the output. The paper calls top-k search "critical".

**Adaptation.** Implement Sparkly-style **target to S1** probing per country. Use name 3-grams (normalized and transliterated) as one field, address word tokens and numbers as another, and sum the per-field BM25 scores. Also try the SM+/TFIDF-cosine symmetric variant, because our "documents" (S1) and "queries" (targets) are the same kind of short record.

### 2.5 Deep-learning blockers
- **DeepBlocker (Thirumuruganathan et al., PVLDB 14(11), 2021).**
  - Design space: word embedding (fastText char-level), then tuple embedding (SIF aggregation or self-supervised: autoencoder, seq2seq, CTT, hybrid), then vector pairing (top-K cosine via FAISS-GPU).
  - The Autoencoder is best on structured and dirty data, and Hybrid is best on textual data.
  - **Top-K cosine outperforms threshold cosine and is much better than LSH.**
  - FAISS-GPU pairing takes under 1 min for K=100 on all datasets except the 1M × 1M Song-Song (under 35 min).
  - Union(DL, RBB) raises recall substantially (table 6): +0.3-6.7 points on structured, +8.5-14.5 on textual and +0.2-9.9 on dirty data. The candidate set grows by up to 49.9%, 57.6% and 12.3% relative to the larger member; against DL alone the growth can be far larger (Walmart-Amazon2: 12.8k to 7.9M). *(Audit pass 2: the paper does not name the base of these percentages. Table 6 shows it is the larger member: on Abt-Buy, DL gives 21.6k, RBB 28.3k and the union 44.6k, which is +57.6% over RBB.)*
  - Sparkly later showed that these DL blockers do not scale (the Autoencoder crashed on large tables and needed 691-925 min on 10M).
- **AutoBlock (Zhang, Wei, Sisman, Dong, Faloutsos, Page, WSDM 2020, Amazon).** Supervised similarity-preserving representation learning plus NN search, with sub-quadratic total time and scaling to millions of records. It needs labels, and we have them.
- **SC-Block (Brinkmann, Shraga, Bizer, ESWC 2024).**
  - Supervised contrastive learning places records of the same entity close together, then NN search builds the candidate set.
  - It keeps 99.5% pair completeness with smaller candidate sets, and pipelines run 1.5-2× faster with the same F1.
  - On a large benchmark the pipeline was 8× faster (2.5 h down to 18 min), with about 5 min of training.
  - **Most directly applicable DL idea**: we have 7.6M labeled train pairs grouped into entity clusters, which is ideal supervised-contrastive data (a cluster = a class).
- **UniBlocker (Wang et al., arXiv 2404.14831, 2024).** A dense blocker pre-trained on domain-independent tabular data with self-supervised contrastive learning. It is "comparable and complementary" to SOTA sparse blocking without domain training. The repo has **no license**, so treat it as reference only.
- **NLSHBlock (Wang et al., arXiv 2401.18064, 2024).** A PLM fine-tuned with an LSH-based loss, so that hashing approximates a task-specific similarity. Reference only. *(Audit: the venue is now confirmed as SIAM SDM 2024, pp. 887-895, DOI 10.1137/1.9781611978032.101, per Crossref.)*
- **DIAL (Jain, Sarawagi, Sen, PVLDB 15(1):31-45, 2021; presented at VLDB 2022).** Index-By-Committee active learning that jointly trains a transformer blocker and matcher. It shows that the **blocker and the matcher should be trained differently** (different training data construction and objectives), and it includes a multilingual record-matching dataset.
- **Sudowoodo (Wang, Li, Wang, arXiv 2207.04122).** Contrastive self-supervised representations that serve both blocking and matching and beat specialized blockers. *(Audit: the venue is now confirmed as ICDE 2023, pp. 1502-1515, DOI 10.1109/ICDE55515.2023.00391, per Crossref.)*

### 2.6 Pre-trained embeddings as blockers
- **Zeakis, Papadakis, Skoutas, Koubarakis, PVLDB 16(9), 2023.** Covers 12 language models on 17 datasets. **SentenceBERT models are best for blocking**, with S-GTR-T5 ranked first. Non-fine-tuned BERT/XLNet/ALBERT are poor. On synthetic Dirty ER scaling, S-GTR-T5 recall stays around 0.8 at 2M entities, only 17% below its 10k value, while FastText drops 54% (0.901 to 0.415). S-GTR-T5 recall was 15% higher than DeepBlocker on 8 datasets. The blocking used FAISS HNSW for approximate kNN on the large sets.
- **Implication.** Use *sentence-embedding* checkpoints trained with contrastive objectives (e5, bge-m3, LaBSE, gte, arctic-embed), never raw MLM checkpoints (MuRIL or BERT without fine-tuning) as blockers. For multilingual and cross-script data, the English-only GTR is not suitable. Use multilingual e5, bge-m3 or LaBSE (section 11).

### 2.7 ANN libraries and GPU search
- **FAISS.**
  - "Billion-scale similarity search with GPUs" (Johnson, Douze, Jégou, arXiv 1702.08734; IEEE Trans. Big Data 7(3):535-547, 2021, DOI 10.1109/TBDATA.2019.2921572, confirmed on Crossref): 8.5× faster than the prior GPU SOTA, and a kNN graph on 95M images in 35 min.
  - "The Faiss library" (Douze et al., arXiv 2401.08281).
  - MIT license.
  - For our size, **exact** `IndexFlatIP` per country on GPU is feasible (section 6), which avoids ANN recall loss entirely.
- **HNSW (Malkov and Yashunin, arXiv 1603.09320; IEEE TPAMI 42(4):824-836, 2020, DOI 10.1109/TPAMI.2018.2889473, confirmed on Crossref)**, hnswlib (Apache-2.0). This is the CPU graph ANN; build time and RAM grow with the number of points.
- **ScaNN (Guo et al., ICML 2020, PMLR 119:3887-3896).** Anisotropic quantization for MIPS. It ranked second in scalability in ICDE 2023.
- **CAGRA (Ootomo et al., ICDE 2024, pp. 4236-4247)**, available in cuVS (Apache-2.0; the repo is now `NVIDIA/cuvs`, and `rapidsai/cuvs` redirects to it). This is GPU graph ANN. Graph build is 2.2-27× faster than HNSW, and large-batch queries at 90-95% recall are 33-77× faster than HNSW. The large-batch regime is exactly ours (millions of queries).

### 2.8 Fusion of multiple blockers and per-query truncation
- **Reciprocal Rank Fusion (Cormack, Clarke, Buettcher, SIGIR 2009).** score(d) = Σ_b 1/(60 + rank_b(d)). It is parameter-light, robust, and beat Condorcet and learned rank fusion in their study. Use it as a strong **unsupervised** fusion of the sparse, dense and address lists before truncation.
- **Arampatzis, Kamps, Robertson, "Where to stop reading a ranked list?" SIGIR 2009.** Chooses a per-query rank cut-off that optimizes a given effectiveness measure (including F-measures) by modelling score distributions. This is conceptually what our per-S1 F0.5 needs at the *matching* stage. At the blocking stage the same idea sets per-query k.
- **Choppy (Bahri, Tay, Zheng, Metzler, Tomkins, SIGIR 2020).** A transformer over the score list that directly optimizes a user-defined metric for ranked-list truncation, using only the scores.

### 2.9 Adaptive, progressive and auto-configured blocking
- **pBlocking (Galhotra, Firmani, Saha, Srivastava, VLDB Journal 30(4):537-557, 2021).** Feeds partial ER output back to refine blocking. Reported 5× efficiency and 60% effectiveness gains for blocking, with O(n log² n) convergence. This is relevant if we iterate: match on a first candidate set, then use confident matches (cluster structure) to re-block the hard remainder.
- **Progressive Entity Matching: A Design Space Exploration (Maciejewski, Nikoletos, Papadakis, Velegrakis, PACMMOD 3(1), 2025).** *(Audit: the arXiv 2503.08298 title is "Progressive Entity **Resolution**: A Design Space Exploration". The published title "Progressive Entity Matching…" is confirmed on Crossref, DOI 10.1145/3709715.)* The pipeline is filtering, then weighting, then scheduling, then matching, under a budget. It is a framework for "spend comparisons where matches are likely".
- **Auto-Configuring ER Pipelines (Nikoletos, Efthymiou, Papadakis, Stefanidis, arXiv 2503.13226, 2025).** Hyperparameter-optimization samplers for tuning ER pipelines when ground truth exists, and regression-based transfer when it does not. It is relevant for tuning the k values of each blocker on train.
- **BlockingPy (Strojny and Beręsewicz, arXiv 2504.04266; SoftwareX 34:102583, 2026, DOI 10.1016/j.softx.2026.102583, confirmed on Crossref; MIT).** A Python package for ANN-based blocking plus graph components. Backends are FAISS (flat, HNSW, LSH), Annoy, Voyager, HNSW, mlpack and NND, with a GPU variant (faiss-gpu flat, IVF, IVFPQ, CAGRA). It is a convenient reference, but for 10M records we will write thin FAISS or cuVS code directly.

### 2.10 Multi-source and multi-table
- **MultiEM (Zeng et al., ICDE 2024, pp. 3421-3434, confirmed on Crossref).** *(Audit, corrected in pass 2: the GitHub repo ZJU-DAILY/MultiEM has **no license**, so it is for reading only. The README body does not mention ICDE, but the repo's About description does: "Code for the paper 'MultiEM: …'. ICDE 2024." The venue and pages come from Crossref.)* Unsupervised multi-table EM that uses table-wise hierarchical merging (merge tables pairwise, representing merged entities) plus density-based pruning. It parallelizes. Idea for us: **first collapse S2 and S3 near-duplicate copies into target clusters, then block S1 against clusters** (cluster representative plus member union). This works because copies share the same corrupted base.

### 2.11 Cross-script and multilingual candidate generation
- **Zhou, Rijhwani, Wieting, Carbonell, Neubig. "Improving Candidate Generation for Low-resource Cross-lingual Entity Linking." TACL 8:109-124, 2020.** Candidate generation across languages and scripts for KB linking gave +16.9% average top-30 gold-candidate recall over SOTA baselines on 7 datasets. It shows that cross-lingual candidate generation is a recall bottleneck worth dedicated modelling.
- **Aksharantar / IndicXlit (Madhani et al., Findings of EMNLP 2023, pp. 40-57).** 26M transliteration pairs, 21 Indic languages, 12 scripts. The IndicXlit model is MIT (both the HF card and the repo). It is about 11M parameters (per the repo README), and both the HF repo and the GitHub releases ship an `indic-en` (native-to-Roman) checkpoint as well as `en-indic`. It gives **learned** native-to-Roman transliteration, which is better than rule-based transliteration at schwa deletion and English loanwords ("मार्केटिंग" to "marketing").
- **LaBSE (Feng et al., ACL 2022).** Language-agnostic sentence embedding over 109+ languages, with 83.7% bitext retrieval accuracy on Tatoeba (112 languages). Apache-2.0 on HF.
- **M3-Embedding / BGE-M3 (Chen et al., Findings of ACL 2024).** One model gives dense, sparse-lexical and multi-vector outputs, covers more than 100 languages, and is MIT. The built-in **sparse + dense hybrid** is attractive for blocking.
- **Multilingual E5 (Wang et al., arXiv 2402.05672).** Small, base and large sizes, contrastively pre-trained on about 1B multilingual pairs. MIT. The small model has 12 layers and 384 dimensions, and about 118M parameters, most of them in the 250k-vocabulary embedding table *(computed from config.json)*. Its transformer compute is small, so it is fast.

---

## 3. Metrics to track for every blocker (implemented in `src/ber/metric.py` diagnostics)

Let T(e) be the true targets of S1 entity e and C(e) the candidates.
- **Pair completeness (PC / recall)** = Σ|T∩C| / Σ|T|.
- **Entity full-recall rate** = the fraction of non-singleton S1 with T(e) ⊆ C(e).
- **Oracle-matcher F0.5 ceiling (the primary blocking objective).** With f = |T∩C| and m = |T \ C|:
  - singleton: 1;
  - f = 0 and |T| > 0: 0;
  - otherwise: 1.25f / (1.25f + 0.25m).

  Macro-average this over all S1. Examples: missing 1 of 2 gives 0.833, 1 of 4 gives 0.9375, and 1 of 6 gives 0.962 (6.25/6.5). *(Audit pass 2: the earlier value of 0.968 was an arithmetic error.)* Blocking losses therefore hurt most on **low-multiplicity** entities (1-2 matches, about 22% of train S1).
- **Candidates per S1 (CPQ)**: mean, p50 and p99. Also the total number of pairs, and **|C| / (|S1|·|T|)** (the CSSR used by Sparkly and DeepBlocker).
- **Per-slice versions**: country, source (S2/S3), script of the target name (Latin vs each Indic script), and name frequency bucket.
- **Marginal recall of each blocker in the union.** For each blocker, report the recall it alone contributes that no other blocker finds. This is the evidence for keeping or dropping it.

---

## 4. Recommended blocking architecture for this challenge

### 4.1 Normalization (shared by all blockers; learned or generic, no external data)
Run each step in this order:
1. **Unicode NFC, then casefold.** Strip combining marks **only for Latin-script characters** (é to e, Í to I). **Gotcha:** running NFKD and dropping every combining mark would delete Indic vowel signs (matras) and viramas and destroy Devanagari, Tamil and other names. Transliterate Indic scripts first, then fold.
2. **Script-aware transliteration** of names and addresses to Latin:
   - **P0:** rule-based `indic_transliteration` (MIT) to ISO-15919/IAST, then ASCII fold, then drop a word-final inherent "a" after a consonant (schwa-deletion heuristic).
   - **P2:** IndicXlit (MIT) native-to-Roman for better loanword handling.
   - Mixed-script strings ("Sun पावर Provision") are transliterated per token.
   - Other scripts go through `anyascii` (ISC). **Avoid Unidecode (GPL-2.0) and Aksharamukha (AGPL-3.0 on PyPI)** for licensing hygiene. These are library licenses, not model licenses, but copyleft is best avoided in the shipped code.
3. **Junk removal**: leading and trailing non-alphanumerics (`--`, `<<`, `##`, `***`), brackets around legal forms, doubled spaces, and null tokens (`null`, `N/A`, `<NULL>`).
4. **Aliases**: split on ` dba `, ` a/k/a `, ` f/k/a `, `|` and produce **multiple name variants** per record. Index or query each variant, then take the max score per pair.
5. **Domains and handles**: strip the scheme, `www.`, the TLD and `@`, and keep the stem ("heassociates"). Also compute n-grams on the **space-stripped name**, so "he associates" and "heassociates" share all interior 3-grams. Index both the spaced and the de-spaced strings.
6. **Leetspeak folding** for n-gram keys only (0 to o, 1/l to i in a canonical shadow string).
7. **Legal forms and generic tokens**: do **not** hand-delete them. BM25/TF-IDF IDF down-weights them automatically, and that generalizes to unseen French forms (SARL, SAS, SCI) with no hard-coding. Optionally, a shadow "core name" string can drop the top-N highest-df tokens per country, with N learned from the data.
8. **Address**:
   - Learn the abbreviation and alias maps from train positive pairs (aligned token substitutions such as Road/Rd, Texas/TX, Maharashtra/MH/महाराष्ट्र). This follows COMPLIANCE C6, which prefers learned over hand-written maps.
   - Strip the "Near …" landmark clause into a separate low-weight field.
   - Canonicalize numbers: remove leading zeros, split "5935-D" into "5935" and "d", and drop the `#`/`No`/`H.NO` prefixes.

### 4.2 Blockers (union; each has its own top-k; all run within a country partition)

| id | Blocker | Direction | Index | Why it exists here |
|---|---|---|---|---|
| **B1** | BM25 (or TFIDF-cosine / SM+) over **char 3-grams of the normalized, transliterated name** (plus the de-spaced variant) **plus address word tokens and numbers**, as the sum of per-field scores (Sparkly-Auto style) | target → S1 (k_t ≈ 10-20) | S1 per country | Main workhorse. Names are non-unique, so address breaks ties. Robust to typos, reordering and moved suffixes. |
| **B2** | Same as B1 | S1 → targets (k_s ≈ 30-50) | targets per country | Symmetric pass. It catches S1s whose copies all rank low from the target side (for example, a very common name where many S1 compete). Measure its marginal recall, because Sparkly found two-sided search gave minimal gains. |
| **B3** | **Address-only BM25** (word tokens and numbers, transliterated) | target → S1 (k ≈ 10) | S1 | Pure-alias records ("Kelojax" matched by address only), domain names, cross-script names, and France (repetitive names). |
| **B4** | **Dense multilingual encoder** on raw `name \| city` or `name \| address` (no transliteration, so it is complementary): multilingual-e5-small/base or bge-m3 dense; later fine-tuned with supervised contrastive loss on train clusters (SC-Block) | target → S1 (k ≈ 10) | FAISS-GPU exact per country | Cross-script bridge. Also captures abbreviation and semantic variants that 3-grams miss. |
| **B5** | Hash keys (block purging above a size cap): (house number, first street token); (consonant skeleton of the rarest name token, city token); (two rarest name tokens, sorted) | both | hash map | Cheap, exact, complementary recall. Cap each block at a size like 200 (Block Purging) so that no key explodes. |
| **B6** | **Target-graph expansion**: build target↔target kNN within S2∪S3 per country (the same sparse representation, k ≈ 5, high similarity only). For each S1, add the 1-hop neighbours of its accepted top candidates. | derived | target kNN | Copies share a corrupted base, so a copy that is far from S1 but near its sibling copy is recovered. |

Fusion: take the union of all pairs, then attach a feature vector (section 5) and prune to the budget. For an unsupervised first cut, rank by **RRF** across B1-B4 and keep the top-N per S1 plus all pairs found by at least 2 blockers.

### 4.3 Why target → S1 is the primary direction here
- Many-to-one is measured (target multiplicity = 1). Each target needs only its **one** correct S1 in its short list, so k_t can be small (about 10) for all 10M targets. That gives about 100M raw pairs before dedup and pruning *(estimate)*.
- S1 entities with 7-11 copies are served automatically. There is no per-S1 k to size.
- The index is the small side (1.73M, or ≤0.81M per country). The sparse matrix or the FAISS index fits comfortably in one GPU or one process, and the 10M queries are embarrassingly parallel. This is the Sparkly architecture.
- Distractor targets (26%) will still propose k S1s. Their top-1 score profile (low absolute score, flat gap) is a strong feature for pruning them later.
- The **reverse-rank** feature (the rank of the S1 in the target's list, and vice versa) is a reciprocity signal (ReCNP analogue) that is highly predictive of true matches.

---

## 5. Candidate budget allocation (dynamic top-k)

Evidence base: top-k beats thresholds (Sparkly, ICDE 2023, DeepBlocker). Supervised meta-blocking uses a pair classifier over blocking features to drive pruning (Gagliardelli 2022). WNP/CNP are per-node adaptive pruning (PVLDB 2016). Per-query rank truncation is covered by Arampatzis 2009 and Choppy 2020.

**Algorithm (to run on Modal, CPU box):**
```
1. For each blocker b and query q, retrieve top-K_max(b) (generous: 50 for B1/B2, 20 for B3/B4).
2. Union all pairs. Per pair (s1, t) compute features:
   - per blocker: rank r_b, score s_b, s_b / s_b@1 (relative), s_b@1 - s_b (gap), s_b@r - s_b@(r+1)
   - reciprocal: rank of s1 in t's list and rank of t in s1's list
   - n_blockers_hit, RRF score
   - query hardness: top-1 score, gap(1,2), sum-IDF of name tokens, S1 name frequency in S1
   - cheap string sims (RapidFuzz token_set on transliterated name, address number agreement)
   - B6 support: #neighbours of t already in s1's candidate list
3. Train a GBDT (LightGBM/XGBoost on CPU) on train pairs, grouped by S1, with y = true match.
4. Choose a global cut τ on validation to meet a budget (e.g. mean ≤ 20 candidates/S1, or the capacity
   of the next-stage matcher) while maximizing the oracle-F0.5 ceiling.
   Always keep the top-m (m = 1-2) of B1 per S1 as a safety floor.
5. Output: final candidate set = pairs with p ≥ τ (∪ floor). This is the set whose ceiling we report.
```
- Why this is the right allocation: keeping every pair with p ≥ τ under a total budget is the greedy-optimal fractional-knapsack solution when each pair costs the same. Hard queries (flat score profiles, common names, cross-script) automatically get more candidates, and easy queries (a dominant top-1) get one.
- A no-label fallback (for a quick first submission) keeps rank r while s_r / s_1 ≥ ρ (for example 0.7-0.85), bounded by k_min = 2 and k_max = 30 (WNP-style), and tunes ρ on train.
- **Calibration shift warning:** test has about 24% more targets per S1 and a new country, so thresholds on raw scores will drift. Use **relative** features (ratios, ranks, gaps) and tune τ as a **budget** (quantile) rather than a fixed probability. Simulate the density shift by injecting extra distractor targets into validation.
- Compliance (A9): `candidate_pairs.tsv` must be the **last** candidate set that the final model scores. If a cascade is used (blocking, then GBDT prune, then heavy matcher), write the post-prune set that the heavy matcher actually scores.

---

## 6. Scale engineering for 1.7M × 10M (cloud only)

### 6.1 Sparse path (single big CPU box: 64+ cores, 256 GB RAM)
- **Vectorize.** Use scikit-learn `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), sublinear_tf=True, min_df=2, max_df=<cap>)`. Fit IDF **per country on S1∪S2∪S3 of that split**; this is transductive and label-free, which COMPLIANCE F3 allows.
  - `max_df` or explicit **stop-gram purging** of the most frequent 3-grams (" pv", "ltd", "ing") is the main runtime lever. The cost of a sparse product scales with the square of posting-list lengths. This is the same idea as Block Purging.
- **Top-k sparse product.** Use `sparse_dot_topn` (Apache-2.0, multi-threaded, blockwise A·Bᵀ with top-n per row) or `string_grouper` (MIT, which wraps it). Chunk the 10M query rows (for example into 200k-row chunks) and parallelize across processes. The S1 matrix is shared read-only.
- **BM25 alternative.** Use Pyserini/Lucene (Apache-2.0; JVM; the Sparkly-proven path with block-max WAND) or `bm25s` (MIT; numpy and numba). Lucene is the safest route to BM25 top-k at 10M queries on one box. Sparkly: 10 × 16 cores handled 10M in under 100 min. One 64-core box should therefore be within a few hours *(rough estimate; measure on a 1M sample first)*.
- **Memory.** A 12M-row sparse char-3-gram matrix at about 25 non-zeros per row is about 3e8 non-zeros. That is roughly 2.4 GB in CSR with float32 values and int32 indices, or about 3.6 GB if scipy promotes the indices to int64 *(estimate)*. Either way it is fine on 256 GB. *(Audit pass 2: the earlier figure paired 3.6 GB with int32 indices, which was an arithmetic slip.)*

### 6.2 Dense path (one GPU)
- **Encoding.** About 12.4M short strings (names and short addresses, 16-48 tokens). Throughput with multilingual-e5-small is dominated by a 12-layer 384-dimensional transformer. Measure it on a 100k sample on T4/L4/A10G before committing. Use fp16, dynamic padding and length-sorted batches.
- **Index S1 per country** (≤0.81M × 384): 1.2 GB in fp32, 0.6 GB in fp16. Exact `faiss.IndexFlatIP` on the GPU, with the 10M targets streamed as queries in batches of about 64k, and top-k = 10-20.
  - Arithmetic *(estimate)*: with country partitioning, the total work is Σ_c |S1_c|·|T_c|·d·2 ≈ 5e15 FLOP. That is minutes on an A100 in fp16 and tens of minutes on a T4. Exact search means **zero ANN recall loss**.
- **S1 → targets direction (B2 dense variant).** A per-country target index of ≤5M × 384 fp16 is about 3.8 GB, so exact flat search is still feasible. Otherwise use IVF-Flat, or CAGRA (cuVS) for large-batch graph ANN.
- **bge-m3** (1024-dim, about 568M params *(computed from config.json)*) is about 3× the storage and encode cost of e5-small. Use it only if the e5 variants underperform on cross-script recall.

### 6.3 Pipeline shape on Modal
```
[parquet on volume] → normalize+transliterate (CPU, multiprocess) → per-country shards
   ├─ B1/B2/B3 sparse top-k (CPU 64c)      ─┐
   ├─ B4 dense encode + FAISS-GPU (1 GPU)    ├→ union + features → GBDT prune (CPU) → candidates.parquet
   ├─ B5 hash keys (CPU)                     │
   └─ B6 target↔target kNN (CPU sparse)     ─┘
```
Everything is keyed by entity_id only. Row order and ids are never used as features (no leakage, per DATA_ANALYSIS section 6).

---

## 7. Validation protocol for blocking

1. **Split train by S1 entity** (grouped). Tune the k values, ρ and τ on a 10-20% validation split. Report the section 3 metrics per slice.
2. **Leave-one-country-out** (train on US and evaluate India, and the reverse) for any learned component: the learned abbreviation map, the fine-tuned dense encoder, and the GBDT pruner. This is the proxy for unseen France. Prefer configurations whose ceiling drops least.
3. **Density shift.** On validation, add unmatched targets so that targets per S1 match test (about 2.8-2.9 per source rather than 2.3-2.4). Budget rules must hold the ceiling under this shift.
4. **Script slice.** Report recall separately for targets whose name is Devanagari, Tamil, Kannada, Telugu, Bengali or another script. This is the acceptance test for transliteration and dense cross-script retrieval.
5. **Budget curve.** Plot the ceiling against mean candidates per S1 for each blocker, and for the union, at k ∈ {1, 2, 5, 10, 20, 50}. Pick the knee.

---

## 8. Risks and gotchas
- **Blanket NFKD mark stripping destroys Indic text.** Transliterate first (section 4.1).
- **Hand-coded gazetteers.** Using a US state name↔code table is generic knowledge, but it is safer to **learn** it from train pairs, and learning also generalizes to India and France (region↔département). **libpostal** (MIT code, but "powered by … open geo data", meaning OSM-derived models and gazetteers) is a **compliance RISK**. Do not use it.
- **Country label.** Partitioning by the label is lossless on train. If a test record had a missing or odd label, route it to all partitions (open set).
- **Frequent n-grams** make the sparse product explode. Purge them or cap df.
- **Short and acronym names.** 3-grams break down; rely on `char_wb` padding, the address blocker and whole-token keys.
- **Two-sided top-k** costs runtime. Measure its marginal recall before keeping it (Sparkly saw minimal gains).
- **Semantic embeddings introduce false positives** on out-of-vocabulary, domain-specific tokens (ICDE 2023: syntactic beat semantic). Keep dense as a union member with its own small k. Do not make it the only blocker.
- **Train/test calibration drift.** More distractors on test, and France. Use rank and ratio features and budget-based cuts.
- **Library vs model licenses.** Libraries (FAISS MIT, sparse_dot_topn Apache-2.0, Pyserini Apache-2.0, and so on) are separate from model weights. Only model weights must be MIT/Apache-2.0 and ≤8B parameters.

---

## 9. Papers (all confirmed on a primary page on 2026-09-25)

| # | Title | Authors | Venue / year | Primary URL used | What we take |
|---|---|---|---|---|---|
| 1 | Blocking and Filtering Techniques for Entity Resolution: A Survey (arXiv title: "A Survey of Blocking and Filtering Techniques for Entity Resolution") | Papadakis, Skoutas, Thanos, Palpanas | ACM CSUR 53(2):1-42, 2020 | https://arxiv.org/abs/1905.06167 ; Crossref 10.1145/3377455 | Taxonomy: blocking, filtering, hybrid, block processing |
| 2 | An Overview of End-to-End Entity Resolution for Big Data | Christophides, Efthymiou, Palpanas, Papadakis, Stefanidis | ACM CSUR 53(6), 2020/21 | https://arxiv.org/abs/1905.06397 ; Crossref 10.1145/3418896 | End-to-end ER at scale |
| 3 | A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication | Christen | IEEE TKDE 24(9):1537-1555, 2012 | Crossref 10.1109/TKDE.2011.127 ; author PDF https://users.cecs.anu.edu.au/~christen/publications/christen2011indexing.pdf | Classic blocking families; PC, PQ and RR |
| 4 | Comparative Analysis of Approximate Blocking Techniques for ER | Papadakis, Svirsky, Gal, Palpanas | PVLDB 9(9):684-695, 2016 | https://www.vldb.org/pvldb/vol9/p684-papadakis.pdf ; Crossref 10.14778/2947618.2947624 | 17 methods; PC above 0.80 for lazy methods; Block Filtering r ≈ 0.5; CNP/WNP |
| 5 | Meta-Blocking: Taking Entity Resolution to the Next Level | Papadakis, Koutrika, Palpanas, Nejdl | IEEE TKDE 26(8):1946-1960, 2014 | Crossref 10.1109/TKDE.2013.54 | Blocking-graph weighting and pruning |
| 6 | BLAST: a Loosely Schema-aware Meta-blocking Approach for ER | Simonini, Bergamaschi, Jagadish | PVLDB 9(12):1173-1184, 2016 | https://www.vldb.org/pvldb/vol9/p1173-simonini.pdf ; Crossref 10.14778/2994509.2994533 | Attribute-aware weighting, LSH attribute clustering |
| 7 | Generalized Supervised Meta-blocking | Gagliardelli, Papadakis, Simonini, Bergamaschi, Palpanas | PVLDB 15(9):1902-1910, 2022 | https://arxiv.org/abs/2204.08801 ; https://www.vldb.org/pvldb/vol15/p1902-gagliardelli.pdf | Pair classifier over blocking features, then pruning (our budget allocator) |
| 8 | Benchmarking Filtering Techniques for ER (arXiv: How to reduce the search space of ER…) | Papadakis, Fisichella, Schoger, Mandilaras, Augsten, Nejdl | ICDE 2023, pp. 653-666 | https://arxiv.org/abs/2202.12521 (PDF v4 read) ; Crossref 10.1109/ICDE55515.2023.00389 | Top-k beats thresholds; syntactic beats semantic; kNN-Join best; FAISS fastest (D2M in 1.3 h) |
| 9 | Deep Learning for Blocking in Entity Matching: A Design Space Exploration (DeepBlocker) | Thirumuruganathan, Li, Tang, Ouzzani, Govind, Paulsen, Fung, Doan | PVLDB 14(11):2459-2472, 2021 | https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf | Top-K cosine beats threshold and LSH; union DL+RBB raises recall |
| 10 | Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for EM | Paulsen, Govind, Doan | PVLDB 16(6):1507-1519, 2023 | https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf | BM25 3-gram top-k; recall 92.5-100% at k=10; 10M in under 100 min on 10 nodes (Sparkly Auto) |
| 11 | Pre-trained Embeddings for ER: An Experimental Analysis | Zeakis, Papadakis, Skoutas, Koubarakis | PVLDB 16(9):2225-2238, 2023 | https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf ; https://arxiv.org/abs/2304.12329 | SBERT-type models best for blocking; S-GTR-T5 first; raw BERT poor |
| 12 | SC-Block: Supervised Contrastive Blocking within ER Pipelines | Brinkmann, Shraga, Bizer | ESWC 2024 (LNCS 14664), pp. 121-142 | https://arxiv.org/abs/2303.03132 ; https://2024.eswc-conferences.org/accepted-papers/ ; Crossref 10.1007/978-3-031-60626-7_7 | 99.5% PC with smaller sets; 8× faster pipeline |
| 13 | AutoBlock: A Hands-off Blocking Framework for EM | Zhang, Wei, Sisman, Dong, Faloutsos, Page | WSDM 2020, pp. 744-752 | https://arxiv.org/abs/1912.03417 ; Crossref 10.1145/3336191.3371813 | Supervised representation learning + NN; millions of records |
| 14 | Deep Indexed Active Learning for Matching Heterogeneous Entity Representations (DIAL) | Jain, Sarawagi, Sen | PVLDB 15(1):31-45, 2021 (VLDB 2022) | https://arxiv.org/abs/2104.03986 ; Crossref 10.14778/3485450.3485455 | Joint blocker and matcher; train them differently |
| 15 | Towards Universal Dense Blocking for ER (UniBlocker) | Wang, Lin, Han, Chen, Cao, Sun | arXiv 2404.14831, 2024 (venue unverified) | https://arxiv.org/abs/2404.14831 | Domain-independent dense blocker, complementary to sparse |
| 16 | Neural Locality Sensitive Hashing for Entity Blocking | Wang, Kong, Tao, Borthwick, Golac, Johnson, Hijazi, Deng, Zhang | SIAM SDM 2024, pp. 887-895 (arXiv 2401.18064) | https://arxiv.org/abs/2401.18064 ; Crossref 10.1137/1.9781611978032.101 | Learned LSH for task-specific similarity |
| 17 | MultiEM: Efficient and Effective Unsupervised Multi-Table EM | Zeng, Wang, Mao, Chen, Liu, Gao | ICDE 2024, pp. 3421-3434 | https://arxiv.org/abs/2308.01927 ; Crossref 10.1109/ICDE60146.2024.00264 (the README body does not mention ICDE, but the GitHub About description says "ICDE 2024"; the repo has no license) | Hierarchical table merging; density pruning |
| 18 | Efficient and Effective ER with Progressive Blocking | Galhotra, Firmani, Saha, Srivastava | VLDB Journal 30(4):537-557, 2021 | https://arxiv.org/abs/2005.14326 ; Crossref 10.1007/s00778-021-00656-7 | Feedback-driven re-blocking; 5× / 60% gains |
| 19 | Progressive Entity Matching: A Design Space Exploration | Maciejewski, Nikoletos, Papadakis, Velegrakis | PACMMOD 3(1), 2025 (SIGMOD) (arXiv title says "Progressive Entity Resolution…") | https://arxiv.org/abs/2503.08298 ; Crossref 10.1145/3709715 | Filtering, weighting, scheduling under a budget |
| 20 | Auto-Configuring Entity Resolution Pipelines | Nikoletos, Efthymiou, Papadakis, Stefanidis | arXiv 2503.13226, 2025 | https://arxiv.org/abs/2503.13226 | HPO samplers for ER configs |
| 21 | Three-dimensional Entity Resolution with JedAI | Papadakis, Mandilaras, Gagliardelli, Simonini, Thanos, Giannakopoulos, Bergamaschi, Palpanas, Koubarakis | Information Systems 93:101565, 2020 | Crossref 10.1016/j.is.2020.101565 | JedAI toolkit (the pyJedAI ancestor) |
| 22 | BlockingPy: approximate nearest neighbours for blocking of records for ER | Strojny, Beręsewicz | SoftwareX 34:102583, 2026 (arXiv 2504.04266) | https://arxiv.org/abs/2504.04266 ; Crossref 10.1016/j.softx.2026.102583 | ANN blocking package, CPU and GPU |
| 23 | The merge/purge problem for large databases | Hernández, Stolfo | SIGMOD 1995, pp. 127-138 | Crossref 10.1145/223784.223807 | Sorted neighbourhood, multi-pass |
| 24 | Efficient clustering of high-dimensional data sets with application to reference matching | McCallum, Nigam, Ungar | KDD 2000, pp. 169-178 | Crossref 10.1145/347090.347123 | Canopy clustering |
| 25 | Adaptive Blocking: Learning to Scale Up Record Linkage | Bilenko, Kamath, Mooney | ICDM 2006, pp. 87-96 | Crossref 10.1109/ICDM.2006.13 | Learned DNF blocking predicates |
| 26 | On the resemblance and containment of documents | Broder | SEQUENCES 1997, pp. 21-29 | Crossref 10.1109/SEQUEN.1997.666900 | MinHash |
| 27 | Scaling up all pairs similarity search | Bayardo, Ma, Srikant | WWW 2007, pp. 131-140 | Crossref 10.1145/1242572.1242591 | Prefix-filtering similarity joins |
| 28 | A Comparison of Blocking Methods for Record Linkage | Steorts, Ventura, Sadinle, Fienberg | PSD 2014, LNCS, pp. 253-268 (arXiv 1407.3191) | https://arxiv.org/abs/1407.3191 ; Crossref 10.1007/978-3-319-11257-2_20 | Traditional vs LSH blocking |
| 29 | Billion-scale similarity search with GPUs | Johnson, Douze, Jégou | IEEE Trans. Big Data 7(3):535-547, 2021 (arXiv 1702.08734) | https://arxiv.org/abs/1702.08734 ; Crossref 10.1109/TBDATA.2019.2921572 | FAISS-GPU; 8.5× over prior GPU SOTA |
| 30 | The Faiss library | Douze et al. | arXiv 2401.08281, 2024 | https://arxiv.org/abs/2401.08281 | Index design trade-offs |
| 31 | Efficient and robust ANN search using HNSW graphs | Malkov, Yashunin | IEEE TPAMI 42(4):824-836, 2020 (arXiv 1603.09320) | https://arxiv.org/abs/1603.09320 ; Crossref 10.1109/TPAMI.2018.2889473 | CPU graph ANN |
| 32 | Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN) | Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar | ICML 2020, PMLR 119:3887-3896 | https://proceedings.mlr.press/v119/guo20h.html | Quantized MIPS |
| 33 | CAGRA: Highly Parallel Graph Construction and ANN Search for GPUs | Ootomo, Naruse, Nolet, Wang, Feher, Wang | ICDE 2024, pp. 4236-4247 | https://arxiv.org/abs/2308.15136 ; Crossref 10.1109/ICDE60146.2024.00323 | 33-77× over HNSW, large-batch |
| 34 | Reciprocal rank fusion outperforms Condorcet and individual rank learning methods | Cormack, Clarke, Buettcher | SIGIR 2009, pp. 758-759 | Crossref 10.1145/1571941.1572114 | Fusion of blocker lists |
| 35 | Where to stop reading a ranked list? Threshold optimization using truncated score distributions | Arampatzis, Kamps, Robertson | SIGIR 2009, pp. 524-531 | Crossref 10.1145/1571941.1572031 | Per-query cut-off optimizing F-measures |
| 36 | Choppy: Cut Transformer for Ranked List Truncation | Bahri, Tay, Zheng, Metzler, Tomkins | SIGIR 2020, pp. 1513-1516 | https://arxiv.org/abs/2004.13012 ; Crossref 10.1145/3397271.3401188 | Learned per-query truncation |
| 37 | Improving Candidate Generation for Low-resource Cross-lingual Entity Linking | Zhou, Rijhwani, Wieting, Carbonell, Neubig | TACL 8:109-124, 2020 | https://aclanthology.org/2020.tacl-1.8/ | +16.9% top-30 candidate recall across languages |
| 38 | Aksharantar: Open Indic-language Transliteration datasets and models | Madhani, Parthan, Bedekar, Nc, Khapra, Kunchukuttan, Kumar, Khapra | Findings of EMNLP 2023, pp. 40-57 | https://aclanthology.org/2023.findings-emnlp.4/ | IndicXlit (MIT) transliteration |
| 39 | Language-agnostic BERT Sentence Embedding (LaBSE) | Feng, Yang, Cer, Arivazhagan, Wang | ACL 2022, pp. 878-891 | https://aclanthology.org/2022.acl-long.62/ | Cross-lingual dense retrieval; 83.7% Tatoeba |
| 40 | M3-Embedding (BGE-M3) | Chen, Xiao, Zhang, Luo, Lian, Liu | Findings of ACL 2024, pp. 2318-2335 | https://aclanthology.org/2024.findings-acl.137/ | Dense + sparse + multi-vector in one model |
| 41 | Multilingual E5 Text Embeddings: A Technical Report | Wang, Yang, Huang, Yang, Majumder, Wei | arXiv 2402.05672, 2024 | https://arxiv.org/abs/2402.05672 | mE5 small/base/large |
| 42 | Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation | Wang, Li, Wang | ICDE 2023, pp. 1502-1515 (arXiv 2207.04122) | https://arxiv.org/abs/2207.04122 ; Crossref 10.1109/ICDE55515.2023.00391 | Contrastive representations for blocking |
| 43 | Scaling Entity Resolution to Large, Heterogeneous Data with Enhanced Meta-blocking | Papadakis, Papastefanatos, Palpanas, Koubarakis | EDBT 2016, pp. 221-232, DOI 10.5441/002/edbt.2016.22 | https://openproceedings.org/html/pages/2016_edbt.html ; PDF https://openproceedings.org/2016/conf/edbt/paper-193.pdf | Reciprocal Pruning (ReCNP/ReWNP); added in audit pass 2 because sections 2.1 and 2.2 cite it |

---

## 10. Repositories and libraries (GitHub API or repo page, read 2026-09-25)

| Repo | License | Stars (approx.) | Last push seen | Use for us | Compliance |
|---|---|---|---|---|---|
| facebookresearch/faiss | MIT | ~41.0k | 2026-09-24 | Exact and IVF GPU kNN for dense blockers | OK (library) |
| NVIDIA/cuvs (formerly rapidsai/cuvs; the old URL redirects) | Apache-2.0 | ~0.86k | 2026-09-24 | CAGRA GPU graph ANN | OK |
| nmslib/hnswlib | Apache-2.0 | ~5.3k | 2026-09-15 | CPU HNSW | OK |
| unum-cloud/USearch (the API now returns this capitalization) | Apache-2.0 | ~4.3k | 2026-08-31 | Fast CPU/GPU-agnostic HNSW | OK |
| ing-bank/sparse_dot_topn | Apache-2.0 | ~0.42k | 2026-09-14 | Multithreaded sparse A·Bᵀ top-n (char-gram TF-IDF) | OK |
| Bergvca/string_grouper | MIT | ~0.37k | 2026-07-26 | TF-IDF char-gram matching wrapper. *Audit: it now defaults to the Rust backend `Bergvca/sp_matmul_rs` (Apache-2.0); `sparse_dot_topn` is the optional original backend.* | OK |
| castorini/pyserini | Apache-2.0 | ~2.2k | 2026-09-24 | Lucene BM25 top-k at scale (Sparkly path) | OK |
| xhluca/bm25s | MIT | ~1.8k | 2026-09-18 | Fast in-Python BM25 | OK |
| ekzhu/datasketch | MIT | ~3.0k | 2026-08-09 | MinHash/LSH for within-source near-dup grouping | OK |
| rapidfuzz/RapidFuzz | MIT | ~4.1k | 2026-09-12 | Cheap string-sim features for pruning | OK |
| AI-team-UoA/pyJedAI | Apache-2.0 | ~0.10k | 2026-03-22 | Prototype token and meta-blocking on samples | OK |
| anhaidgroup/sparkly | BSD-3-Clause | ~19 | 2026-08-28 | Reference implementation of Sparkly (PyLucene + Spark) | OK (BSD) |
| qcri/DeepBlocker | BSD-3-Clause | ~30 | 2023-04-05 | Reference DL blockers | OK, stale |
| wbsg-uni-mannheim/SC-Block | BSD-3-Clause | ~10 | 2024-06-10 | Supervised contrastive blocker reference | OK |
| tshu-w/Uniblocker | **none** | ~9 | 2025-10-08 | Reading only | **No license: do not copy code** |
| Gaglia88/sparker | **GPL-3.0** | ~67 | 2024-03-29 | Meta-blocking on Spark (reference) | Copyleft: avoid shipping |
| ncn-foreigners/BlockingPy | MIT | ~21 | 2026-03-09 (250 commits) | ANN blocking package (CPU and GPU) | OK |
| moj-analytical-services/splink | MIT | ~2.4k | 2026-09-22 | Blocking-rule design patterns (Fellegi-Sunter) | OK |
| dedupeio/dedupe | MIT | ~4.5k | 2025-07-29 | Learned blocking predicates (Bilenko-style) | OK |
| indic-transliteration/indic_transliteration_py | MIT | ~0.21k | 2026-09-08 | Rule-based Indic to Latin transliteration (P0) | OK |
| AI4Bharat/IndicXlit | MIT | ~0.14k | 2023-10-13 | Learned transliteration model (P2) | OK (the model card is also MIT) |
| anyascii/anyascii | ISC | ~0.42k | 2026-06-06 | Generic Unicode to ASCII | OK |
| avian2/unidecode | **GPL-2.0** | ~0.61k | 2026-01-05 | Unicode to ASCII | Copyleft: prefer anyascii |
| virtualvinodh/aksharamukha | GitHub API detects no license, but the repo root has a `gpl-3.0.txt` (GPL-3.0 text); **PyPI: "GNU AGPL 3.0"** (classifier AGPLv3) | ~0.22k | 2025-03-25 | Script conversion | Copyleft (GPL/AGPL): avoid |
| ZJU-DAILY/MultiEM | **none** | ~16 | 2023-11-05 | MultiEM reference code | **No license: do not copy code** |
| openvenues/libpostal | MIT (code) | ~4.9k | 2026-05-13 | Address parsing | **RISK: models and gazetteers built from open geo data. The repo description says "Powered by statistical NLP and open geo data", and the README says the parser is trained on OpenStreetMap (ODbL) and OpenAddresses addresses. Do not use.** |

---

## 11. Models usable in blocking (license read from the HF model card or HF API, 2026-09-25)

| HF id | License (exact) | Params | Compliant? | Role |
|---|---|---|---|---|
| intfloat/multilingual-e5-small | mit | 117.7M (HF safetensors; matches the config-based estimate) | Yes | **Default dense blocker (B4)**; tags include hi, ta, kn, mr, bn, fr |
| intfloat/multilingual-e5-base | mit | 278.0M (HF safetensors) | Yes | Upgrade if small underperforms |
| intfloat/multilingual-e5-large | mit | 559.9M (HF safetensors) | Yes | Upper-bound check only (cost) |
| BAAI/bge-m3 | mit | ~568M *(computed from config: 24 layers, 1024-dim, 250k vocab)* | Yes | Dense + sparse hybrid in one model; 8192 max length |
| sentence-transformers/LaBSE | apache-2.0 | 470.9M (HF safetensors; the card rounds to 0.5B) | Yes | Cross-lingual alternative, 109 languages |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0 | 117.7M (HF safetensors) | Yes | Cheap multilingual baseline, 384-dim |
| Alibaba-NLP/gte-multilingual-base | apache-2.0 | 305.4M (HF safetensors) | Yes (needs `trust_remote_code`) | Alternative multilingual dense model |
| Snowflake/snowflake-arctic-embed-m-v2.0 | apache-2.0 | 305.4M (HF safetensors) | Yes | Alternative multilingual dense model |
| Qwen/Qwen3-Embedding-0.6B | apache-2.0 | 595.8M (safetensors) | Yes | Stronger but heavier; only if cost allows |
| sentence-transformers/gtr-t5-base | apache-2.0 | 109.6M (safetensors) | Yes, but **English-centric** | The Zeakis best-blocker family; not suitable for cross-script |
| google/canine-s | apache-2.0 | 132.1M | Yes | Character-level encoder (needs fine-tuning; not an out-of-box retriever) |
| google/muril-base-cased | apache-2.0 | about 237M *(computed from config.json: 197,285 vocab × 768, 12 layers; not stated on the card)* | Yes, but it is an **MLM** checkpoint | Do not use raw as a blocker (Zeakis: raw BERT is poor); fine-tune only |
| ai4bharat/IndicXlit | mit | ~11M (GitHub README) | Yes | Transliteration model (P2). The HF repo ships both `indicxlit-indic-en-v1.0` (native to Roman) and `indicxlit-en-indic-v1.0` fairseq checkpoints |
| jinaai/jina-embeddings-v3 | **cc-by-nc-4.0** | 572.3M | **NO** | Excluded (non-commercial) |
| google/embeddinggemma-300m | **gemma** (the HF repo is also gated, with manual approval) | 302.9M | **NO** | Excluded (Gemma license) |

---

## 12. What we should do (priorities)

### P0: baseline blocking that already reaches a high ceiling (first 1-2 days, CPU-only on Modal)
1. **Normalization and transliteration module** (section 4.1): rule-based Indic transliteration (indic_transliteration, MIT), then Latin diacritic fold, junk and null removal, alias splitting (dba, aka, fka), domain stem plus de-spaced variant, and number canonicalization. Unit-test it on the DATA_ANALYSIS examples ("राम मार्केटिंग प्राइवेट लिमिटेड", "Sun पावर Provision", "heassociates.com", "LLC Moncada Léarning Center", "महाराष्ट्र").
2. **Partition by country label** (lossless on train, open set).
3. **B1: target → S1 top-k** with TF-IDF char-3-gram (`char_wb`) on the name plus word TF-IDF on the address, as a summed score. Use sparse_dot_topn on a 64-core box, IDF per country fitted on all records, and df-capped stop-grams. Start with k_t = 10.
4. **B3: address-only top-k** (k = 10), unioned.
5. **Measure** PC, entity full-recall, **oracle-F0.5 ceiling**, CPQ, and per-script slices on a grouped validation split. Also measure leave-one-country-out. Target *(proposal)*: ceiling ≥ 0.98 at mean CPQ ≤ 30.
6. Write `candidate_pairs.tsv` from exactly the set the matcher scores (A9).

### P1: close the cross-script and hard-query gaps
7. **B4 dense blocker**: multilingual-e5-small (MIT) on raw `name | city`. Use exact FAISS-GPU per country with the S1 index, and targets as queries (T4 or L4 is enough). Union it and measure marginal recall on the Indic-script slice.
8. **Fine-tune the dense blocker with supervised contrastive loss** on train clusters (SC-Block recipe; S1 plus its copies form a class; BM25 hard negatives). Train on A100 or L4 on Modal. Re-measure leave-one-country-out.
9. **B2: S1 → targets pass** (k_s = 30-50). Keep it only if its marginal ceiling gain beats its cost.
10. **Budget allocator**: GBDT over blocking features (ranks, ratios, gaps, reciprocity, blocker agreement, query hardness, cheap RapidFuzz sims). Pick a global cut chosen as a budget quantile, simulate the test density shift, and keep a top-1 floor.

### P2: exploit multi-copy structure and harden
11. **B6 target-graph expansion**: within-source near-duplicate grouping (char-gram top-k among targets plus a high-similarity cut, or MinHash). Add 1-hop siblings of the accepted candidates, then use the sibling-support feature in the allocator.
12. **B5 hash keys** with block purging (number + street token; consonant skeleton + city; rarest-token pairs).
13. **IndicXlit** (MIT) learned transliteration if rule-based transliteration leaves a measurable residual on the Indic slice.
14. **BM25 via Pyserini/Lucene** and an SM+ / TFIDF-cosine comparison against the sklearn TF-IDF baseline (Sparkly ablation: symmetric scoring is often better on short records).
15. **Learned address alias map** (train-pair token alignment) to replace any hand-written abbreviation lists (COMPLIANCE C6). Apply it transductively to France by self-training on confident matches.

### P3: optional or research
16. bge-m3 sparse+dense hybrid as a single-model alternative to B1+B4. Compare it on the ceiling-vs-CPQ curve.
17. Progressive re-blocking (pBlocking-style): after a first matching pass, re-block unresolved S1 with relaxed keys and cluster-aware expansion.
18. Sparkly-Auto style label-free config search (discriminativeness AUC) to pick fields and tokenizers for France without labels.
19. pyJedAI token blocking + meta-blocking on a 200k sample, as a sanity comparison against top-k retrieval.

---

## 13. Open items and unverified claims
- **UniBlocker's venue is still UNVERIFIED.** Only the arXiv page (2404.14831 v2) is confirmed. Its arXiv comment gives only the code link. Crossref and web searches in audit pass 2 again found no published version. *(Audit update: the other previously open venues are now confirmed on Crossref: Sudowoodo is ICDE 2023, NLSHBlock is SDM 2024, BlockingPy is SoftwareX 34 (2026), Steorts is PSD 2014, and HNSW is TPAMI 42(4) 2020. See the audit log.)*
- *(Audit update)* Parameter counts are now filled in from HF safetensors metadata (mE5-base/large, gte-multilingual-base, arctic-embed-m-v2.0) or computed from config.json (MuRIL). IndicXlit's ~11M comes from its GitHub README. All are far below the 8B cap.
- Every throughput and memory figure in section 6 is an *estimate*. Benchmark on a 1M-record shard on Modal before sizing the full run.

---

## Audit log

Independent citation and license audit, 2026-09-25. Every item was treated as possibly hallucinated until it was confirmed on a primary source. Nothing was run on the laptop except metadata HTTP requests and text extraction from PDFs.

### How items were checked
- **arXiv (25 IDs)**: the arXiv export API (title, authors, comments, DOI, journal-ref) for 1905.06167, 1905.06397, 2204.08801, 2202.12521, 2304.12329, 2303.03132, 1912.03417, 2104.03986, 2404.14831, 2401.18064, 2308.01927, 2005.14326, 2503.08298, 2503.13226, 2504.04266, 1407.3191, 1702.08734, 2401.08281, 1603.09320, 2308.15136, 2004.13012, 2003.01343, 2402.03216, 2402.05672 and 2207.04122. **All 25 exist, and every listed author list matches.**
- **Crossref (DOI and title search)**: all DOIs in section 9, plus title searches that found the published versions of ICDE'23, SC-Block, MultiEM, CAGRA, Sudowoodo, Choppy, Steorts, NLSHBlock, BlockingPy, Zhou TACL, Aksharantar, LaBSE and M3. Titles, authors, volume, issue and pages were checked.
- **Full text re-read to check the quoted numbers**:
  - Sparkly: vldb.org PDF, all 13 pages.
  - DeepBlocker: vldb.org PDF, Table 6 and section 5.
  - Zeakis: vldb.org PDF, blocking and scalability sections.
  - Papadakis et al. 2016: vldb.org PDF.
  - ICDE'23 filtering benchmark: arXiv PDF v4, conclusions.
  - BLAST and GSM: vldb.org PDFs, first page.
  - RRF: author PDF (k = 60 confirmed).
  - Arampatzis SIGIR'09: author PDF (abstract).
  - AutoBlock: arXiv PDF, page 1 (Amazon affiliations confirmed).
  - ACL Anthology pages: LaBSE (83.7% Tatoeba over 112 languages; model for 109+ languages) and Aksharantar (26M pairs, 21 languages, 12 scripts).
  - PMLR page: ScaNN (PMLR 119:3887-3896).
  - ESWC 2024 accepted-papers page: SC-Block.
- **HF models (15)**: `huggingface.co/api/models/<id>` `cardData.license`, the license tag and safetensors parameter totals; config.json for mE5-small, bge-m3 and MuRIL; bge-m3 card text (MIT, 8192 tokens, 100+ languages, dense + sparse + multi-vector). **All 15 license strings in the notes are exactly right**: mit ×5, apache-2.0 ×8, cc-by-nc-4.0 and gemma. The 13 claimed compliant are MIT/Apache-2.0 and well under 8B parameters. jina-v3 and embeddinggemma are correctly excluded.
- **GitHub repos (25 + 2 extra)**: `gh api repos/<r>` for spdx license, stars, pushed_at and redirects; README checks for libpostal, BlockingPy, sparkly, sparse_dot_topn, string_grouper, MultiEM and IndicXlit; PyPI JSON for aksharamukha, Unidecode, indic-transliteration and anyascii. **Every repo license string matches**, and star counts match within rounding.

### What was changed and why
1. **DeepBlocker union claim (TL;DR 3 and section 2.5).** The recall numbers (68.7 to 83.0 and 70.5 to 85.0) are correct. "The candidate set grew only modestly" was misleading: on Walmart-Amazon2 the union has 7.9M candidates against 12.8k for DL alone, because RBB alone is 7.9M. The caveat and the paper's exact size-growth figures were added.
2. **Papadakis 2016 (section 2.3).** "Q-grams Blocking … scaling linearly" was wrong. The paper says lazy methods, QGBl included, scale slightly super-linearly, and only SuAr, ESuAr and SoNe scale linearly, at lower recall. The text was corrected. q=6 as the QGBl default is confirmed.
3. **Pruning-scheme attribution (sections 2.1 and 2.2).** ReCNP and ReWNP come from EDBT 2016 "enhanced meta-blocking", not from TKDE 2014, and the 2016 PVLDB study cites these schemes rather than defining them. The attribution was fixed. WNP's "mean weight" wording was softened to "local, automatically set threshold", which is what the re-read source says.
4. **ICDE'23 (section 2.2).** Added the nuance that SBW beat kNN-Join in the schema-agnostic setting; kNN-Join is preferred for linear candidate count and ease of configuration. Also noted that the current arXiv v5 abstract concludes that blocking workflows and string similarity joins are superior. Added DOI 10.1109/ICDE55515.2023.00389.
5. **"Three independent studies" (TL;DR 1).** Sparkly and DeepBlocker share authors, so they are not independent. Reworded.
6. **Sparkly "beating PBW, DBW, JD, Union" (section 2.4).** The paper's claim is predictability, not strict dominance (PBW can reach 100% recall at up to 4.2B pairs). Reworded. **All other Sparkly numbers were confirmed verbatim**: 92.5-100 / 96.4-100 / 98.7-100; 10M rows in under 100 min for $12.5; 26M in 130 min for $67.5 on 30 nodes; 1.3-2 GB index; 691 and 925 min; recall@50 of 40 and 85 against Sparkly's 94-100; 10 of 15 datasets and 0.7%; the idf, tf, 3-gram and BM25 k1/b ablations; SM+; 10k sample with k = 250 and Wilcoxon; up to 3 attributes; the one-sided vs two-sided top-k result.
7. **Venues upgraded from UNVERIFIED to confirmed (Crossref).**
   - Sudowoodo: ICDE 2023, pp. 1502-1515.
   - NLSHBlock: SDM 2024, pp. 887-895.
   - BlockingPy: SoftwareX 34:102583, 2026.
   - Steorts et al.: PSD 2014, LNCS, pp. 253-268.
   - HNSW: TPAMI 42(4):824-836, 2020.
   - FAISS: IEEE TBD 7(3):535-547, 2021.
8. **Metadata filled or corrected.**
   - DIAL is PVLDB 15(1):31-45 (2021), not just "VLDB 2022".
   - pBlocking is VLDBJ 30(4):537-557.
   - AutoBlock is pp. 744-752.
   - SC-Block is LNCS 14664, pp. 121-142.
   - MultiEM is ICDE 2024, pp. 3421-3434.
   - CAGRA is pp. 4236-4247.
   - Choppy is pp. 1513-1516.
   - PVLDB 2016 and BLAST DOIs were added.
   - The arXiv-vs-published title differences are now noted for Christophides (arXiv: "End-to-End ER for Big Data: A Survey") and Maciejewski (arXiv: "Progressive Entity *Resolution*…").
9. **MultiEM verified_via was imprecise.** The notes claimed "README says ICDE 2024". The README body does not mention ICDE, but the repo's GitHub About description does ("…ICDE 2024."), as re-checked in pass 2 through the GitHub API. The venue is now sourced from Crossref. The repo also has **no license**, which was added to the repo table.
10. **Repo table.**
    - `rapidsai/cuvs` now redirects to `NVIDIA/cuvs`, so the name was updated.
    - BlockingPy's last push is 2026-03-09 (it was listed only as "active").
    - dedupe's last push is 2025-07-29 (it was "n/a").
    - string_grouper now defaults to the `sp_matmul_rs` backend (Apache-2.0), not sparse_dot_topn.
    - aksharamukha: the GitHub API detects no license, but the repo root has `gpl-3.0.txt`, and PyPI says "GNU AGPL 3.0". It is copyleft either way.
    - libpostal's risk note now quotes the repo description ("Powered by statistical NLP and open geo data") and the README's training sources (OSM under ODbL, and OpenAddresses).
11. **Model table.**
    - Parameter counts that were marked "not read" are now filled in from HF safetensors: mE5-base 278.0M, mE5-large 559.9M, gte-mb 305.4M, arctic-m-v2 305.4M, LaBSE 470.9M, para-MiniLM 117.7M, jina-v3 572.3M and embeddinggemma 302.9M.
    - MuRIL is about 237M, computed from config.json.
    - IndicXlit is about 11M per its README, and its HF repo ships the `indic-en` (native-to-Roman) checkpoint the plan relies on.
    - embeddinggemma is also gated on HF.
    - No license string needed changing.

### Items confirmed with no change needed (numbers read in source)
- **Zeakis PVLDB'23**: S-GTR-T5 first; recall 0.8 at D2M, 17% below D10K; FastText 0.901 to 0.415 (-54%); +15% over DeepBlocker on 8 datasets; HNSW for Dirty ER.
- **DeepBlocker**: Autoencoder best on structured and dirty data, Hybrid on textual; FAISS under 1 min at K=100, Song-Song under 35 min; top-K cosine beats threshold cosine and LSH.
- **ICDE'23**: 14 filters and 4 baselines on 10 datasets; FAISS D2M in 1.3 h, SCANN second; k ≪ 100; LSH needs excessive candidates; syntactic beats semantic.
- **SC-Block**: 99.5% PC, 1.5-2× faster, 8× (2.5 h to 18 min), 5 min of training.
- **pBlocking**: 5× and 60%; O(n log² n).
- **FAISS**: 8.5×; 95M images in 35 min.
- **CAGRA**: 2.2-27× build; 33-77× large-batch at 90-95% recall.
- **Zhou TACL**: +16.9% top-30 recall and +7.9% end-to-end.
- **mE5**: about 1B multilingual pairs.
- **The rest**:
  - GSM: minimum training-set size.
  - DIAL: multilingual dataset; blocker and matcher trained differently.
  - Choppy: scores only, directly optimizes any metric.
  - Arampatzis: rank cut-off optimizing a given effectiveness measure.
  - BlockingPy backend list.
  - M3: MIT, 8192 tokens, 100+ languages.
  - JedAI Information Systems 93:101565.
  - All foundational Crossref metadata: Hernández-Stolfo, McCallum, Bilenko, Broder, Bayardo, Cormack, Arampatzis, Christen, Meta-blocking TKDE.

### Still UNVERIFIED after the audit (updated in pass 2)
- **The UniBlocker (arXiv 2404.14831) venue.** No published version was found. Its repo has no license.
- *Resolved in pass 2:* the Christen 2012 content summary is now confirmed from the author's PDF.
- *Resolved in pass 2:* "Autoencoder crashed on large tables" is correctly attributed to Sparkly in section 2.5. Sparkly's words are "quickly exhausts memory and crashes" before optimization. After optimization it ran in 691 min on MB 10M and 925 min on WDC 10M.
- **Content summaries for foundational papers are metadata-only.** For Hernández-Stolfo 1995, McCallum et al. 2000, Broder 1997 and Bayardo et al. 2007, only the metadata was confirmed (Crossref). Their one-line method descriptions are the standard textbook characterizations and were not re-read in the full text. Bilenko et al. 2006 was re-read in pass 2 (author PDF from cs.utexas.edu: it learns blocking functions from predicates in disjunctive and conjunctive (DNF) form).

---

### Audit pass 2: independent re-verification (2026-09-25)

The user asked for the audit to be run again. This pass re-checked every item from scratch against primary sources rather than trusting pass 1. Only lightweight HTTP metadata requests and PDF text extraction (PyMuPDF) were run locally; no models and no heavy computation.

**What was checked in pass 2**
- **arXiv export API, all 25 IDs**: titles, full author lists, comments, DOIs and abstracts. All exist, and all author lists match the notes.
  - Abstract numbers confirmed: SC-Block (99.5% PC, 1.5-2×, 8×, 2.5 h to 18 min, 5 min training); pBlocking (5×, 60%, O(n log² n)); FAISS (8.5×, 95M images in 35 min); CAGRA (2.2-27× build, 33-77× at 90-95% recall); Zhou (16.9%, 7.9%); mE5 (1B pairs); M3 (100+ languages, 8,192 tokens).
  - Also confirmed: DIAL (a multilingual dataset; blocker and matcher trained differently), UniBlocker ("comparable and complementary"), Choppy (scores only, any metric) and Auto-Config (HPO samplers plus regression transfer).
- **Crossref, 36 DOIs**: every DOI in section 9 resolves to the stated title, authors, venue, volume, issue and pages, including BlockingPy SoftwareX 34:102583 (2026), NLSHBlock SDM 2024 pp. 887-895, Sudowoodo ICDE 2023 pp. 1502-1515, DIAL PVLDB 15(1):31-45, CAGRA and MultiEM ICDE 2024 pages, Choppy pp. 1513-1516, FAISS TBD 7(3), HNSW TPAMI 42(4), and TACL 8:109-124 (10.1162/tacl_a_00303). SC-Block is in LNCS 14664 (ESWC 2024 Part I, confirmed on the Springer book page). The EDBT 2016 paper was confirmed on the OpenProceedings index page, because its DOI is not in Crossref.
- **Full-text PDFs re-read** (vldb.org, arXiv, OpenProceedings, author pages):
  - Sparkly: every number in section 2.4 was re-found, including Table 3 (SM/SA recall@50 of 94/98 on MB 10M and 100/94 on BC 2.5M, against Autoencoder's 40 and 85).
  - DeepBlocker: Table 6 and section 5.
  - Zeakis: the blocking and scalability sections.
  - Papadakis 2016: Table 3 defaults, the scalability section and refs [27]/[28].
  - ICDE'23 v4: conclusions 1-6.
  - GSM and BLAST: first pages.
  - EDBT 2016: Reciprocal Pruning.
  - RRF: k = 60.
  - Christen 2012 and Bilenko 2006: author PDFs.
  - AutoBlock page 1: Amazon.com affiliations for Wei, Sisman and Dong.
- **ACL Anthology pages**: Zhou TACL, Aksharantar (26M pairs, 21 languages, 12 scripts, IndicXlit), LaBSE (83.7% over 112 languages; 109+ languages) and M3. **PMLR page**: ScaNN, with 7 authors and pp. 3887-3896.
- **HF API, 15 models**: `cardData.license`, the license tag, safetensors totals and the gated flag. **All 15 license strings are exactly right**: mit ×5, apache-2.0 ×8, cc-by-nc-4.0 and gemma. Parameter counts match safetensors. The config-based counts for bge-m3 (567.8M) and MuRIL (237.6M) were recomputed and match. mE5-small's language tags include hi, ta, kn, mr, bn, te and fr. The IndicXlit HF repo ships both `indicxlit-indic-en-v1.0` and `indicxlit-en-indic-v1.0`, and so does GitHub release v1.0. The README states "~11M" parameters. embeddinggemma is `gated: manual`.
- **GitHub API, 26 repos plus Bergvca/sp_matmul_rs**: SPDX license, stars and pushed_at. **Every license matches**, and stars and last-push dates match the table. `rapidsai/cuvs` resolves to `NVIDIA/cuvs`. aksharamukha has no SPDX license, but `gpl-3.0.txt` is at the repo root.
- **READMEs**: libpostal (trained on OpenStreetMap and OpenAddresses; ODbL training files), string_grouper (the default backend is now sp_matmul_rs), BlockingPy (FAISS lsh/hnsw/flat, Voyager, HNSW, MLPACK, NND, Annoy; GPU Flat/IVF/IVFPQ/CAGRA; 250 commits), MultiEM and sparkly.
- **PyPI JSON**: aksharamukha "GNU AGPL 3.0"; Unidecode GPLv2+; indic-transliteration MIT; anyascii ISC.

**What pass 2 changed and why**
1. **Section 3, oracle ceiling example.** "1 of 6 gives 0.968" was an arithmetic error. 1.25·5/(1.25·5 + 0.25·1) = 6.25/6.5 = 0.962.
2. **Section 6.1, CSR memory estimate.** 3e8 non-zeros with float32 values and int32 indices is about 2.4 GB, not 3.6 GB. 3.6 GB applies with int64 indices.
3. **Section 2.1, Papadakis 2016.** Pass 1 said the paper "does not define" CEP/CNP/ReCNP/WEP/WNP/ReWNP. That is inaccurate: the paper describes all six but attributes them to [27] TKDE 2014 and [28] EDBT 2016. Reworded, and the EDBT DOI was added.
4. **Section 2.1, Christen 2012.** The content is now confirmed from the author PDF, so it was removed from UNVERIFIED. Wording aligned with the paper: "pairs completeness" and "pairs quality", string-map indexing, and twelve variations of six techniques.
5. **Section 2.10, table row 17 and log item 9, MultiEM.** Pass 1 said the original researcher's "README says ICDE 2024" was wrong. In fact the README body is silent, but the repo's About description says "ICDE 2024". Softened to "imprecise".
6. **Section 2.4, Sparkly SM tokenization.** The paper removes non-alphanumeric *tokens* after 3-gram tokenization, not non-alphanumeric characters before it. Fixed.
7. **TL;DR 2 and table row 10.** The 10M-tuples-under-100-minutes result belongs to Sparkly **Auto** (SA), not to SM. Attribution fixed, and the node type was added (m5.4xlarge, 16 cores).
8. **Section 2.5, DeepBlocker union growth.** Added the Table 6 evidence that the percentages are relative to the larger member: on Abt-Buy the union is 44.6k against RBB's 28.3k, +57.6%. The paper itself does not state the base.
9. **Section 2.2, ICDE'23.** Added that FAISS ran with an approximate IVF index in that benchmark, so its 1.3 h D2M time is ANN speed and does not compare directly with the exact flat search in section 6.2.
10. **Section 2.2, GSM.** Added the source numbers: 9 real datasets, "50 labelled instances (25 per class) suffice", and a scalability test up to 300k entities.
11. **Section 9, table row 1.** Noted the arXiv title variant and added pages 1-42.
12. **Section 9, row 42 title.** Completed to the full title, "…Data Integration and Preparation".
13. **Section 9, new row 43.** Added the EDBT 2016 enhanced meta-blocking paper, which sections 2.1 and 2.2 cite but which was missing from the table.
14. **Section 10, USearch.** Updated the repo name's capitalization as the GitHub API returns it.

**Confirmed with no change in pass 2**
- All 42 original papers exist with correct authors, years and venues. UniBlocker's venue stays UNVERIFIED.
- All 15 model license strings and compliance verdicts are right.
- All 26 repo licenses are right. The libpostal RISK flag and the copyleft flags (Unidecode GPL-2.0, aksharamukha GPL/AGPL, SparkER GPL-3.0) and no-license flags (UniBlocker, MultiEM) stand.
- No item had to be deleted as fabricated.
