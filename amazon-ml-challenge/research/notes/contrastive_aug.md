# Contrastive / metric learning, hard negatives, synthetic noise augmentation (and GBDT vs neural)

Topic notes for Amazon ML Challenge 2026, Business Entity Resolution (BER). Written 2026-09-25.
Scope: how to train small encoders (bi-encoders, char-level Siamese nets, cross-encoder teachers) for this
challenge, how to mine negatives, how to augment with synthetic noise (including for the unseen France split),
and whether any of it should replace the LightGBM-on-similarity-features core. Sibling notes cover overlapping
ground: `neural_em.md` (cross-encoders, LLMs, throughput), `models.md` (license table), `blocking.md`,
`name_address.md` (transliteration, abbreviation maps), `fmeasure.md` (decision rule).

Verification rules applied here:
- Each paper was confirmed on a primary page (arXiv abs/API record, ACL Anthology, NeurIPS proceedings, ICLR
  virtual site, Crossref DOI record of the ACM/IEEE/Springer/Elsevier version, or the authors' PDF). The URL
  is in the "Verified via" column. DBLP and OpenReview blocked automated access today (bot challenge), so they
  were not used.
- Numbers are only the ones I read in the paper text or on the page. Anything not read is marked UNVERIFIED
  or left out.
- Model licenses come from the `license:` line in each Hugging Face model card (README YAML) and match the
  HF API `cardData.license`. Repo stars, licenses and last-push dates come from the GitHub API on 2026-09-25.
- No model or heavy computation was run on the laptop. The only local work was PDF-to-text extraction and
  metadata lookups.

---

## 0. TL;DR

1. **LightGBM stays the final matcher. Neural models are features and blocking views, not replacements.**
   The EM literature is consistent here. Classic learners on similarity features match DL on structured data
   (DeepMatcher: 88.8 vs 87.9 avg F1, 1.5 min vs 5.4 h training). DL wins on *dirty* and *textual* data
   (+6.2 to 32.6 % F1 on dirty). Our data is dirty and cross-script, so neural signals should add value, but
   as inputs to the GBDT (stacking). The Foursquare Kaggle winners did the same: LightGBM filters plus
   transformer matchers. Shwartz-Ziv & Armon also find that an XGBoost plus deep ensemble beats either alone.
2. **Our data removes the main failure mode of contrastive training: false negatives.** Every S2/S3 target
   maps to at most one S1, the ground truth covers every train S1, and S1 is deduplicated
   (`DATA_ANALYSIS.md` section 3). So any non-GT candidate of an S1 anchor is a *true* negative. The retrieval
   literature has to "denoise" hard negatives: RocketQA found 70 % of top-retrieved unlabeled MS MARCO passages
   were actually positive, and naive hard negatives dropped MRR@10 from 33.3 to 26.0. We do not. **Mine the
   hardest negatives aggressively, straight from our blocking candidates.**
3. **For the same reason, self-supervised contrastive training on unlabeled S1 records is clean.** R-SupCon
   found SimCLR-style pre-training *hurt* because of false negatives, and that performance fell 55 % / 37 % F1
   without source-aware sampling. On deduplicated S1, other S1 records are guaranteed negatives. We can
   therefore pre-train on **test S1 (including France)** with "record vs. its synthetic corruption" positives.
   This transductive step is label-free and uses no external data, so it is our best France lever.
4. **Recipe for the retrieval bi-encoder.** Fine-tune `intfloat/multilingual-e5-small` (MIT, 118M; already in
   `stage_embed.py`) with symmetric InfoNCE (MultipleNegativesRanking). Use country-homogeneous and
   TF-IDF-cluster batches (Sudowoodo's k-means batching: +up to 6.3 % F1 from clustered negatives alone), plus
   1 to 3 mined hard negatives per anchor from our existing candidate files. Add one ANCE-style refresh and
   Matryoshka dims (a 128-d index keeps 10M targets in about 2.6 GB at fp16). The main expected payoff is
   **blocking recall on the cross-script, alias and pure-address slices**, not raw precision.
5. **Hard-negative types we must include (this is what F0.5 precision needs):**
   (a) same normalized name, different address (only 64 to 74 % of S1 names are unique);
   (b) same street and number, different name (co-located businesses, alias records);
   (c) same name and city, different house number (digit typos vs. real neighbours);
   (d) near-duplicate distractor targets (26 % of targets have no S1).
   WDC Products shows that "corner cases" of this kind are what break matchers.
6. **Distillation chain (RocketQA / AugSBERT / Margin-MSE).**
   - Train a cross-encoder teacher (mDeBERTa-v3-base or XLM-R-base, both MIT) on the bi-encoder's own top-k.
   - Use it (i) as a GBDT feature on the uncertain slice only, (ii) to distill into the bi-encoder with
     Margin-MSE, and (iii) to pseudo-label test France at thresholds >0.9 / <0.1.
   - Evidence: AugSBERT gives up to +6 points in-domain and +37 under domain shift. In Sudowoodo,
     pseudo-labelling was the single most valuable component (removing it cost about 10 %).
7. **Augmentation must mirror the real noise generator.** Belinkov & Bisk (ICLR 2018): models trained on one
   synthetic noise type do *not* transfer to other types, while mixed-noise training is robust to every type it
   saw. Our operator library should therefore reproduce the measured noise catalogue (`DATA_ANALYSIS.md`
   section 5), with operator rates fitted from train S1-to-target diffs. That is the Rotom idea without the
   meta-learning, which is not worth it at our label volume. Generic Ditto ops (span_del and friends) gave only
   +1.39 / +2.53 avg F1 on benchmarks with far fewer labels.
8. **France without external data.** Use language-agnostic operators:
   - Abbreviation = prefix or consonant-skeleton of *frequent tokens*, with frequencies computed from the
     unlabeled France records themselves.
   - Accent strip/inject, hyphen and case variants.
   - Movement of frequent name-edge tokens (the legal-form behaviour, discovered rather than listed).
   Combine them with transductive self-supervised training and one pseudo-label round. Validate the whole
   recipe with leave-one-country-out (US to India) before trusting it. Region-to-département swaps cannot be
   synthesized without external knowledge. Rely on name, street and number evidence, and mine co-occurrence
   from high-confidence test pairs (E-8).
9. **Char-level Siamese from scratch is a cheap, compliance-free extra view.** It has 1 to 5M parameters, no
   pretrained weights, and learns from our 551k cross-script pairs. Evidence: char-BiLSTM Siamese nets learn
   typo and extra-word invariance only when the matching augmentation is in training (Neculoiu et al.:
   extra-words accuracy 0.29 to 0.76), and a plain char-n-gram baseline is already 0.99 on typos. So expect
   gains on cross-script and extra-token noise, not on typos.
10. **Compliance.**
    - Non-compliant: `jinaai/jina-embeddings-v3` (CC-BY-NC-4.0), `google/embeddinggemma-300m` (Gemma),
      fastText pretrained vectors (CC-BY-SA-3.0; the library is MIT, so train our own), and any gazetteer-trained
      weights such as Symphonym (GeoNames/Wikidata/Getty).
    - `nlpaug` is MIT, but its synonym augmenters rely on external lexicons, so use only our own char/token
      operators.

---

## 1. Where contrastive learning actually helps under macro F0.5

- The metric is F = 1.25TP / (1.25TP + 0.25FN + FP), per S1, macro-averaged.
- A missed true pair that never reaches the candidate set is a guaranteed FN. So **blocking recall is where a
  better embedding pays off most directly**, especially for slices that TF-IDF misses:
  - cross-script India names (name token-set p10 = 11 to 14 without transliteration, `DATA_ANALYSIS.md` section 7);
  - alias-only records ("Kelojax" matched by address);
  - domain-as-name records ("laxmibombayservices.com").
- Precision is decided by the final GBDT plus the decision rule. Embeddings help precision only as *features*
  (cosine, rank of the target among the S1's candidates, margin to the second candidate). Hard-negative
  training makes those features sharper on chain and franchise cases.
- Singletons (5.6 %) score 1 only when the prediction is empty. Features such as "best dense cosine among
  candidates" and "gap to the 2nd best" feed the empty-set decision in `fmeasure.md`.

## 2. Facts about our data that change the textbook recipes

| Fact (measured, `DATA_ANALYSIS.md`) | Consequence for contrastive training |
|---|---|
| Every target belongs to at most 1 S1; GT covers every train S1; S1 is deduplicated | S1-anchored negatives (in-batch and mined) are **true negatives**. No RocketQA/NV-Retriever denoising is needed for S1 to target training. Self-supervised S1-vs-S1 negatives are also clean. |
| 26 % of targets are distractors (no S1); duplicates of a distractor entity are not linked | Target-to-target contrastive terms **can** contain false negatives (two copies of the same distractor). Mask distractor-distractor pairs, or filter with a guide model (GISTEmbed) or a positive-aware threshold (NV-Retriever TopK-PercPos 95 %). |
| 2 to 6 true targets per S1, across S2 and S3 | Multi-positive losses (SupCon) fit naturally. Pulling all copies together also helps target-target collective features. |
| Within-source copies share the same corrupted base | Target-target positives are easier than S1-target ones. Weight the S1-target loss higher so the model does not learn a source-specific shortcut. |
| S1 names only 64 % (India) to 74 % (US) unique | Plenty of "same-name-different-address" hard negatives exist for free. The address view must carry the discrimination. |
| Country label agrees on 100 % of true pairs | Build country-homogeneous batches. Cross-country negatives are useless (trivially easy). |
| France is unseen (15 % of test S1), with accents, French abbreviations and region-to-département swaps | Transductive self-supervised training on test France S1, generic operators, pseudo-labels, and LOCO validation. |
| 551k cross-script train pairs, learned native-to-Latin map (1,347 tokens) | Cross-script positives are abundant. The learned map can be *inverted* to synthesize cross-script and mixed-script positives for any Indian S1 record. |

---

## 3. Verified papers

Access date for all pages: 2026-09-25.

### 3.1 Contrastive and augmentation-based entity matching

| # | Paper | Venue | Verified via | What I read (method + numbers) | Adaptation here |
|---|---|---|---|---|---|
| 1 | Peeters, Bizer. *Supervised Contrastive Learning for Product Matching* (R-SupCon) | WWW '22 Companion, pp. 248–251, doi:10.1145/3487553.3524254 | https://arxiv.org/abs/2202.02098 (PDF read) + Crossref DOI record | SupCon pre-training of RoBERTa, then a frozen encoder with a CE head. Entity labels come from connected components of the match graph. **Source-aware sampling**: one sampling set per source, holding that source's offers plus only the labelled matches from other sources, which removes inter-source false negatives. **Without it, F1 drops 55 % (Abt-Buy) and 37 % (Amazon-Google).** Abt-Buy 94.29 F1 (+3.24), Amazon-Google 79.28 (+3.7), WDC Computers +0.8 to +8.84. The gain is large for small/medium training sets (+8.8 / +6.6) and small for large/xlarge (+0.8 to 1.8). Explicit augmentation (nlpaug: typos, swap, delete words/spans, synonyms, split) gave mixed results: +1 to 2 % on small WDC, **−4 % on Amazon-Google**. **Self-supervised (SimCLR) pre-training was worse than no pre-training** because of false negatives. Batch 1024, temperature 0.07, 200 epochs. | Our GT is complete, so the false-negative problem that forced source-aware sampling does not arise for S1-anchored batches. It *does* arise for target-target batches (distractor duplicates). Their "large data means small gain" finding says the value for us is the **retrieval view and robustness**, not an in-distribution F1 jump. |
| 2 | Brinkmann, Shraga, Bizer. *SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines* | ESWC 2024, LNCS pp. 121–142, doi:10.1007/978-3-031-60626-7_7 | https://arxiv.org/abs/2303.03132 (PDF read) + Crossref | Supervised contrastive roberta-base with source-aware sampling, plus FAISS kNN. At 99.5 % pair completeness, pipelines run **1.5 to 2x faster**. On the large WDC-B benchmark they run **8x faster (2.5 h to 18 min), for 5 min of training**. BM25 misses **16.3 % more pairs** than SC-Block on WDC-B (large vocabulary). SimCLR and Barlow Twins miss 45 % / 43 % more. Supervised SBERT was the worst dense blocker. Batch 1024, 20 epochs, lr 5e-5. | Direct precedent for a supervised contrastive *blocker*. Our vocabulary is huge (10M records, multi-script), which is exactly where they saw sparse methods fall behind. Keep TF-IDF in the union regardless (Sparkly, see `blocking.md`). |
| 3 | Wang, Li, Wang. *Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation* | ICDE 2023, pp. 1502–1515, doi:10.1109/ICDE55515.2023.00391 | https://arxiv.org/abs/2207.04122 + Crossref (PDF text read) | SimCLR plus four additions: (1) **cutoff DA** (token, feature and span cutoff, applied batch-wise); (2) **clustering-based negative sampling**, i.e. TF-IDF then k-means, filling batches cluster by cluster (Algorithm 2); (3) redundancy regularization (Barlow Twins); (4) **pseudo-labelling**. With 500 labels, **PL is the most effective (removing it costs close to 10 %)**. Clustering negatives give **up to 6.3 % F1 (Abt-Buy)**. Cutoff +3.3 % and RR +1.74 % on Walmart-Amazon. All combined give up to 25 % (WA), 11.2 % on average over plain SimCLR. Sudowoodo with 500 labels roughly equals Rotom with 750 labels (78.3 vs 78.5). Pseudo-label TPR 66 to 99.8 %, TNR 96 to 99.6 %. | Use (2) as our cheap in-batch hard-negative mechanism (k-means within country on the TF-IDF vectors we already build), and (4) for France. For S1-only self-supervision, our negatives are clean (S1 deduplicated), unlike their setting. |
| 4 | Peeters, Der, Bizer. *WDC Products: A Multi-Dimensional Entity Matching Benchmark* | EDBT 2024 (per arXiv comment) | https://arxiv.org/abs/2301.09521 (abstract read) | Three dimensions: share of corner cases, **unseen entities**, and dev-set size. "All matching systems struggle with unseen entities to varying degrees." Contrastive learning is more training-data-efficient than cross-encoders. (Per-system numbers not re-read here; see `neural_em.md`.) | Test S1 entities are all unseen and France is an unseen *domain*. Validate with S1-grouped folds and LOCO. Corner cases (near-identical negatives) are our "same-name-different-address" negatives. |
| 5 | Li, Li, Suhara, Doan, Tan. *Deep Entity Matching with Pre-Trained Language Models* (Ditto) | PVLDB 14(1), 2021, doi:10.14778/3421424.3421431 | https://arxiv.org/abs/2004.00584 (PDF text read) | DA operators (Table 2): **span_del** (delete a span of 4 or fewer tokens), **span_shuffle**, **attr_del**, **attr_shuffle**, **entry_swap** (symmetry), plus **MixDA**, a convex interpolation of LM representations of the original and augmented example to limit label corruption (2 to 3x slower). DA improves the average F1 over Ditto(DK) by **1.39 (ER-Magellan) and 2.53 (WDC)**; span_del is best on WDC. The employer case study chose attr_del for robustness to missing values and reached 96.53 F1. | Directly usable ops: attr_del (our empty addresses are 2.3 to 3.7 %), attr_shuffle (component reordering), entry_swap (symmetric models). The gains are small at benchmark scale, so they are regularizers, not the main lever. |
| 6 | Miao, Li, Wang. *Rotom: A Meta-Learned Data Augmentation Framework for Entity Matching, Data Cleaning, Text Classification, and Beyond* | SIGMOD 2021, pp. 1303–1316, doi:10.1145/3448016.3457258 | Crossref DOI record + Megagon publication page https://megagon.ai/publications/rotom-a-meta-learned-data-augmentation-framework-for-entity-matching-data-cleaning-text-classification-and-beyond/ + repo README | **InvDA**: a seq2seq (T5) model that generates natural-but-diverse augmentations. A **meta-learned policy** filters and weights examples from several DA operators; "combining" operators helps even when single operators do not. Repo: EM operators `del, drop_col, append_col, swap, ins`; EM label budgets 300 to 750. (ACM PDF was 403, so exact F1 table UNVERIFIED here.) | Designed for low-label regimes; we have 7.6M labeled pairs. Borrow the idea "learn the operator mix from data". Our version fits operator rates from observed S1-to-target diffs, with no meta-learning. P3 for InvDA. |
| 7 | Tu, Fan, Tang, Wang, Chai, Li, Fan, Du. *Domain Adaptation for Deep Entity Resolution* (DADER) | SIGMOD 2022, pp. 443–457, doi:10.1145/3514221.3517870 | Crossref + authors' PDF https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/entity-sigmod-2022.pdf (read) | Framework of Feature Extractor, Matcher and Feature Aligner (discrepancy/MMD, adversarial/GRL/InvGAN, reconstruction). DA improves on NoDA by **6.8 and 14.2 F1** (Walmart-Amazon to Abt-Buy and the reverse). Finding 2: gains are larger when source is close to target. Finding 3: **discrepancy-based (MMD) converges reliably; adversarial oscillates**. Finding 4: not all CV/NLP DA methods transfer to ER. Finding 5: DA gains depend on pre-trained LMs (RNN extractors gain little). | France is an unlabeled target domain with a close source (US/India, same schema, same generator). An MMD term between France and train embeddings is a cheap P3 option. Pseudo-labelling is simpler and better evidenced (Sudowoodo). |
| 8 | Kasai, Qian, Gurajada, Li, Popa. *Low-resource Deep Entity Resolution with Transfer and Active Learning* | ACL 2019, pp. 5851–5861, doi:10.18653/v1/P19-1586 | https://aclanthology.org/P19-1586/ | Transfer from a high-resource ER dataset, then active learning of a few informative examples; comparable or better with "an order of magnitude fewer labels". | We cannot label France. The transfer half applies (US+India to France); the active-learning half is replaced by confidence-based pseudo-labels. |
| 9 | Ge, Wang, Chen, Liu, Zheng, Gao. *CollaborEM: A Self-Supervised Entity Matching Framework Using Multi-Features Collaboration* (arXiv title: CollaborER) | IEEE TKDE 35(12), 2023, pp. 12139–12152, doi:10.1109/TKDE.2021.3134806 | Crossref DOI record + https://arxiv.org/abs/2108.08090 (PDF text read) | Zero-annotation EM in two phases. **Automatic label generation (ALG):** (i) *Reliable positive labels* (RPLG, borrowed from IKGC): the pair must be **mutually most similar**, *and* the gap from the best to the second-best candidate must exceed a threshold θ **on both sides**. The authors found that mutual-best alone admits many wrong pairs. (ii) *Negatives* (SNLG): replace one side of a positive with one of its ε-nearest neighbours in embedding space, which gives hard negatives instead of random ones. Then graph + sentence features are trained collaboratively. Generated labels: **average accuracy 99 % (pos) and 97 % (neg)**, vs 88 % / 89 % for ERGAN. It is about 23 % better on average F1 than ERGAN, and on eight benchmarks the abstract calls it "comparable or even superior" to supervised methods. | **A reliability filter for France pseudo-labels.** Our S1-to-target relation is 1-to-many, so the S1-side "mutual best" test does not apply. The *target side* does, because every target has at most one S1. Rule: accept (S1, t) when t's best S1 is this S1 **and** its margin to t's second-best S1 is above θ, with CE/GBDT > 0.9 on top. Fit θ on LOCO India (see 4.5). SNLG is the same as our embedding-kNN hard negatives. |

### 3.2 Contrastive objectives, bi-encoders, batch mechanics

| # | Paper | Venue | Verified via | What I read | Adaptation here |
|---|---|---|---|---|---|
| 10 | Khosla et al. *Supervised Contrastive Learning* (SupCon) | NeurIPS 2020 (Adv. NeurIPS 33) | https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html | Batch contrastive loss with **multiple positives per anchor** (same class). "Subsume or significantly outperform" triplet, max-margin and N-pairs losses. ResNet-200 ImageNet 81.4 % top-1 (+0.8). More robust to corruptions and hyperparameters. | The multi-positive form fits our 2 to 6 copies per S1: label = S1 id, and a batch holds an S1 plus 2 of its targets. |
| 11 | Gao, Yao, Chen. *SimCSE* | EMNLP 2021 | https://arxiv.org/abs/2104.08821 (arXiv record: "Accepted to EMNLP 2021") | Unsupervised: dropout as the only augmentation. Supervised: NLI entailment as positives, **contradictions as hard negatives**. BERT-base avg STS Spearman **76.3 (unsup) and 81.6 (sup)**, +4.2 / +2.2 over the previous best. | For France S1 self-supervision, dropout alone is a weak positive. Use our corruption operators (closer to the real noise) and keep dropout on. |
| 12 | Henderson et al. *Efficient Natural Language Response Suggestion for Smart Reply* | arXiv 1705.00652 (2017) | https://arxiv.org/abs/1705.00652 (API record) | Origin of the in-batch "multiple negatives" dual-encoder objective (MultipleNegativesRankingLoss in sentence-transformers). | Our default loss: symmetric MNRL with hard negatives appended. |
| 13 | Karpukhin et al. *Dense Passage Retrieval* (DPR) | EMNLP 2020 | https://arxiv.org/abs/2004.04906 (API record) | Dual encoder with in-batch negatives plus BM25 hard negatives; beats Lucene-BM25 by **9 to 19 % absolute top-20 accuracy**. | BM25/TF-IDF negatives = our blocking candidates. Same trick, zero extra cost. |
| 14 | Feng, Yang, Cer, Arivazhagan, Wang. *Language-agnostic BERT Sentence Embedding* (LaBSE) | ACL 2022, pp. 878–891, doi:10.18653/v1/2022.acl-long.62 | Crossref + https://arxiv.org/abs/2007.01852 | Dual-encoder translation ranking with **additive margin softmax** and in-batch negatives, on MLM/TLM pre-training. **83.7 % Tatoeba bitext retrieval over 112 languages** (vs 65.5 %). Pre-training cuts the needed parallel data by 80 %. | Candidate backbone for the India cross-script slice (Apache-2.0, 471M). The additive margin is a cheap add-on to MNRL. |
| 15 | Gao, Zhang, Han, Callan. *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup* (GradCache) | RepL4NLP 2021 | https://arxiv.org/abs/2101.06983 | Gradient caching decouples the loss from the encoder backward pass: large in-batch-negative batches at nearly constant memory. | Lets us run batch 1024 or more on a T4/L4/A10G. sentence-transformers ships `CachedMultipleNegativesRankingLoss`. |
| 16 | Kusupati et al. *Matryoshka Representation Learning* | NeurIPS 2022 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html | Nested embeddings; "up to 14x smaller embedding size" at the same ImageNet-1K accuracy; up to 14x retrieval speed-ups. | Train at 384-d with MRL heads at 64/128/256. A 128-d FAISS index for about 10M test targets is about 2.6 GB fp16 vs about 7.7 GB at 384-d. |
| 17 | Deng et al. *ArcFace: Additive Angular Margin Loss* | CVPR 2019 / TPAMI version, doi:10.1109/TPAMI.2021.3087709 | https://arxiv.org/abs/1801.07698 (API record) | Additive angular margin classification loss for embeddings. | The Foursquare-matching prize winner (item 38) used ArcFace over POI ids. For us the classes would be 2.2M S1 ids, which needs Partial FC (item 18). P3, since InfoNCE needs no classifier. |
| 18 | An et al. *Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC* | CVPR 2022 | https://arxiv.org/abs/2203.15565 (journal-ref CVPR2022) | Samples a subset of class centers per step, so million-class margin losses become feasible. | Only needed if we try ArcFace on S1 ids. |

### 3.3 Hard and semi-hard negative mining; false negatives

| # | Paper | Venue | Verified via | What I read | Adaptation here |
|---|---|---|---|---|---|
| 19 | Schroff, Kalenichenko, Philbin. *FaceNet* | CVPR 2015, doi:10.1109/CVPR.2015.7298682 | https://arxiv.org/abs/1503.03832 | Triplet loss with **online triplet mining**; the paper introduces semi-hard negatives (farther than the positive but inside the margin). LFW 99.63 %, YTF 95.12 %, 128-byte embeddings. | Semi-hard is the classic safe choice when labels are noisy. Ours are clean, so hardest-in-candidate-set is viable. Keep semi-hard as an ablation arm. |
| 20 | Wu, Manmatha, Smola, Krähenbühl. *Sampling Matters in Deep Embedding Learning* | ICCV 2017 | https://arxiv.org/abs/1706.07567 | "Selecting training examples plays an equally important role" as the loss. **Distance-weighted sampling** plus a simple margin loss beats other losses. | Sampling policy matters as much as the loss. Log which negative types are sampled. |
| 21 | Xiong et al. *ANCE* | ICLR 2021 | https://iclr.cc/virtual/2021/poster/2673 + https://arxiv.org/abs/2007.00808 | Negatives mined **globally from an ANN index of the corpus, refreshed asynchronously during training**. Nearly matches sparse-retrieval-plus-BERT-rerank accuracy with about 100x speed-up. | One or two synchronous refreshes. Epoch 0 negatives come from TF-IDF candidates plus off-the-shelf e5; re-mine with the fine-tuned model once. |
| 22 | Qu et al. *RocketQA* | NAACL 2021 | https://arxiv.org/abs/2010.08191 (PDF read) | MS MARCO MRR@10: in-batch 32.39, cross-batch 33.32, **hard negatives without denoising 26.03**, with cross-encoder denoising 36.38, plus pseudo-label augmentation 37.02. **70 % of top-retrieved unlabeled passages were actually positive.** Pseudo labels: score <0.1 counts as negative, >0.9 as positive (>90 % accurate on manual check). The cross-encoder is trained on the retriever's own top-k (important). | (i) Our GT is complete, so the denoising step is unnecessary for train. (ii) Their 4-step loop (retriever, CE on retriever negatives, retriever with hard negatives, CE pseudo-labels on unlabeled queries) maps directly onto France (unlabeled queries = test France S1). |
| 23 | Robinson, Chuang, Sra, Jegelka. *Contrastive Learning with Hard Negative Samples* | ICLR 2021 | https://arxiv.org/abs/2010.04592 (comment: ICLR 2021) | Importance-weighted sampling with a **tunable hardness** parameter; no computational overhead. | A tunable-hardness reweighting inside the in-batch softmax. Cheap to try. |
| 24 | Chuang, Robinson, Lin, Torralba, Jegelka. *Debiased Contrastive Learning* | NeurIPS 2020 | https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html | Corrects the contrastive objective for same-label points sampled as negatives, without labels. | Relevant only for target-target or unlabeled-target terms (distractor duplicates). |
| 25 | Zhou et al. *SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval* | EMNLP 2022 | https://arxiv.org/abs/2210.11773 (PDF read) | Negatives ranked *around the positive* are the most informative and least likely to be false negatives. Sampling p_i ∝ exp(−a(s(q,d_i) − s(q,d+) − b)²). Evaluated on 4 public datasets plus 1 industry dataset. | Our mined negatives include near-exact chain duplicates that may be *too* hard to learn from. Compare top-k vs SimANS-weighted sampling as an ablation. |
| 26 | Moreira et al. *NV-Retriever: Improving text embedding models with effective hard-negative mining* | arXiv 2407.15831 (2024) | https://arxiv.org/abs/2407.15831 (PDF read) | Positive-aware mining. **TopK-PercPos with a max negative score of 95 % of the positive score was best**. TopK-MarginPos best margin 0.05; TopK-Abs best threshold 0.7. **Sampling 4 negatives from the top-10 was best** (k>10 hurt). 60.9 on MTEB Retrieval, 1st on the leaderboard in July 2024. | Use for target-target terms and pseudo-labelled France queries, where false negatives are possible. Not needed for train S1-target. |
| 27 | Solatorio. *GISTEmbed: Guided In-sample Selection of Training Negatives* | arXiv 2402.16829 (2024) | https://arxiv.org/abs/2402.16829 | A guide model filters in-batch negatives that look like positives, reducing false-negative noise. | sentence-transformers has `GISTEmbedLoss`. Use it on target-target and France pseudo-label batches, with the fine-tuned model as guide. |

### 3.4 Distillation (cross-encoder to bi-encoder) and self-training

| # | Paper | Venue | Verified via | What I read | Adaptation here |
|---|---|---|---|---|---|
| 28 | Thakur, Reimers, Daxenberger, Gurevych. *Augmented SBERT* | NAACL 2021 | https://arxiv.org/abs/2010.08240 | A cross-encoder labels extra pairs for the bi-encoder. **Up to +6 points in-domain, up to +37 under domain adaptation.** "Selecting the sentence pairs is non-trivial and crucial." | Label test-France blocking candidates with the CE, then train the bi-encoder on them. The pairs to label are our own candidates (already the right distribution). |
| 29 | Hofstätter et al. *Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation* (Margin-MSE) | arXiv 2010.02666 (preprint) | https://arxiv.org/abs/2010.02666 (PDF read) | Different architectures output scores at different magnitudes. Distill the teacher's **margin** (pos minus neg score) with MSE. Improves TK, ColBERT, PreTT and BERT-dot students without slowing them down. | Distillation loss for our bi-encoder from the CE teacher on (S1, pos, hard-neg) triples. Also works with a GBDT teacher margin (P3 idea). |
| 30 | Zhuang, Zuccon. *CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers on Queries with Typos* | SIGIR 2022 | https://arxiv.org/abs/2204.00716 | Small character perturbations badly hurt WordPiece-based dense retrievers, because a typo changes the token distribution. Fix: a char-level backbone plus **self-teaching** (distill the clean-query score distribution into the typo'd query). | Self-teaching is the right loss for "S1 clean vs synthetic-corrupted S1": KL between their candidate-score distributions. It works with any backbone. |
| 31 | Xie, Luong, Hovy, Le. *Self-training with Noisy Student* | CVPR 2020 | https://arxiv.org/abs/1911.04252 | Teacher pseudo-labels unlabeled data; a noised (dropout, augmentation) student trains on labeled plus pseudo-labeled data and is iterated. Works "even when labeled data is abundant". ImageNet 88.4 %. | Template for the France round: teacher = US+India stack; student trained with our corruption operators as noise. |

### 3.5 Character-level Siamese, company names, cross-script

| # | Paper | Venue | Verified via | What I read | Adaptation here |
|---|---|---|---|---|---|
| 32 | Neculoiu, Versteegh, Rotaru. *Learning Text Similarity with Siamese Recurrent Networks* | RepL4NLP @ ACL 2016, pp. 148–157, doi:10.18653/v1/W16-1617 | https://aclanthology.org/W16-1617/ (PDF read) | Char-level stacked BiLSTM Siamese net, contrastive loss, for job-title normalization. Typo pairs made by substituting 20 % of chars and deleting 5 %, forming 10 % of the training set. Accuracy table: **n-gram baseline 0.99 on typos**; RNN base 0.95, then 0.99 with typos. Composition 0.55 to 0.83 with synonyms. **Extra words 0.29 to 0.76** only when extra-word augmentation is added. Annotations 0.69 to 0.87. | **Invariance comes from the augmentation that matches the test noise.** Char n-grams already handle typos, so a char-Siamese earns its keep only on cross-script, junk-token and alias noise. |
| 33 | Basile et al. *Disambiguation of Company names via Deep Recurrent Networks* | Expert Systems with Applications 238 (C), 2024, 122035 | https://arxiv.org/abs/2303.05391 (journal-ref) | Siamese LSTM embedding of company names. It surpasses standard string matching "when enough labelled data are available"; active learning saturates with fewer labels. | Company-name precedent. We have hundreds of thousands of name pairs, i.e. the "enough labels" regime. |
| 34 | Zhang, Zhao, LeCun. *Character-level Convolutional Networks for Text Classification* | NIPS 2015 | https://arxiv.org/abs/1509.01626 (comment cites NIPS 28) | Foundational char-CNN (no numbers used here). | Encoder option for a from-scratch char-Siamese (fast on CPU/GPU, no license issue). |
| 35 | Bojanowski, Grave, Joulin, Mikolov. *Enriching Word Vectors with Subword Information* (fastText) | TACL 2017, pp. 135–146, doi:10.1162/tacl_a_00051 | Crossref DOI record | Char n-gram subword embeddings (method only). | **Train our own** fastText on the provided records (library MIT). The official pretrained vectors are CC-BY-SA-3.0 and **non-compliant**. |
| 36 | Clark, Garrette, Turc, Wieting. *CANINE* | TACL 2022, 10:73–91, doi:10.1162/tacl_a_00448 | https://arxiv.org/abs/2103.06874 | Tokenization-free character encoder (method only). | `google/canine-s` (Apache-2.0, 132M) is a pretrained char-level backbone for a cross-script name bi-encoder. |
| 37 | Khanuja et al. *MuRIL: Multilingual Representations for Indian Languages* | arXiv 2103.10730 (2021) | https://arxiv.org/abs/2103.10730 | Pre-trained with **translated and transliterated document pairs** as cross-lingual signal, and evaluated on transliterated (native to Latin) test sets. | `google/muril-base-cased` (Apache-2.0) is a strong backbone for Devanagari/Tamil/Kannada to Latin name pairs. Ablate against e5-small on the India cross-script slice. |
| 38 | Gadd. *Symphonym: Universal Phonetic Embeddings for Cross-Script Name Matching* | arXiv 2601.06932 (2026 preprint) | https://arxiv.org/abs/2601.06932 | Teacher (articulatory/IPA features) to char-level student; 20 writing systems, 128-d. **R@1 85.2 %, MRR 90.8 %** on MEHDIE; articulatory features alone 45.0 % MRR. **Trained on GeoNames, Wikidata and Getty (67M toponyms).** | **Method only.** Any Symphonym weights embed external gazetteers, so using them is a **compliance risk (C4)**. The distill-to-char-student idea is reusable with our own teacher (cross-encoder on train pairs). |
| 39 | Madhani et al. *Aksharantar: Open Indic-language Transliteration datasets and models* | EMNLP Findings 2023 (per arXiv comment) | https://arxiv.org/abs/2205.03018 | Web-mined transliteration corpora plus IndicXlit models (details in `name_address.md`). | IndicXlit code is MIT, but its models are trained on web-mined external data. Treat as **compliance risk**. Our learned native-to-Latin map is the compliant route. |

### 3.6 Synthetic corruption and noise robustness

| # | Paper | Venue | Verified via | What I read | Adaptation here |
|---|---|---|---|---|---|
| 40 | Christen, Vatsalan. *Flexible and Extensible Generation and Corruption of Personal Data* | CIKM 2013, pp. 1165–1168, doi:10.1145/2505515.2507815 | Crossref + authors' PDF https://users.cecs.anu.edu.au/~Peter.Christen/publications/christen2013cikm.pdf (read) | Six corruptors: **missing value; character edit (insert/delete/substitute/transpose, positioned more likely toward the middle or end); keyboard; OCR (5↔S, m↔rn); phonetic (ph↔f); categorical value swap.** Also: attribute-selection probabilities, max corruptions per attribute and per record, and **the number of duplicates per original drawn from a Poisson, uniform or Zipf distribution**. Unicode-compatible. | Our S2/S3 look like output from this family of generators (several copies per entity, per-attribute corruptions). Implement the same operator families. **Fit the per-operator rates and the copies-per-entity distribution from train** instead of guessing. |
| 41 | Tran, Vatsalan, Christen. *GeCo: an online personal data generator and corruptor* | CIKM 2013 (demo), pp. 2473–2476, doi:10.1145/2505515.2508207 | Crossref DOI record | Online tool version of #40 (venue verified; content not re-read). | Reference implementation of the operator families. |
| 42 | Belinkov, Bisk. *Synthetic and Natural Noise Both Break Neural Machine Translation* | ICLR 2018 | https://arxiv.org/abs/1711.02173 (PDF read) | Noise kinds (Swap, Mid, Rand, Key, Nat) "are not mutually beneficial. Models trained on one do not perform well on the others." A model trained on a **mix of Rand, Key and Nat is robust to all kinds**: not best on any single one, but best on average. Char-CNN filters specialize per noise type. | **Core rule for our augmentation:** cover *every* measured noise type with its measured frequency, and mix them. For France, the risk is noise types we cannot see (region-to-département). Rely on LOCO to measure transfer. |

### 3.7 GBDT on similarity features vs neural: evidence

| # | Paper / item | Venue | Verified via | What I read | Implication |
|---|---|---|---|---|---|
| 43 | Mudgal et al. *Deep Learning for Entity Matching: A Design Space Exploration* (DeepMatcher) | SIGMOD 2018, pp. 19–34, doi:10.1145/3183713.3196926 | Crossref + authors' PDF (text read) | Structured: DL "competitive with Magellan (87.9 % vs 88.8 % average F1)" but **5.4 h vs 1.5 min training**. **Textual: DL +3.0 to 22.0 % F1; dirty: +6.2 to 32.6 % F1.** Training-size sweep (structured): Magellan beat Hybrid below about 50K, comparable after that, Hybrid slightly better at 200K. Hybrid more robust to label noise. Domain-specific IE plus features over DL: only +0.8 / 1.1 / 1.4 % (structured/textual/dirty). | We have dirty data and millions of labels. DL signals should add value, but as GBDT features, because GBDT on good features is the fast, strong baseline. |
| 44 | Konda et al. *Magellan: Toward Building Entity Matching Management Systems* | PVLDB 9(12), 2016, pp. 1197–1208, doi:10.14778/2994509.2994535 | Crossref DOI record (+ https://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf) | The features-plus-classic-learners reference system (py_entitymatching). | Our LightGBM pipeline is this design at scale. |
| 45 | Papadakis, Kirielle, Christen, Palpanas. *A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms* | ICDE 2024 (per sibling note; arXiv record read) | https://arxiv.org/abs/2307.01231 (+ text) | "Most of the popular datasets pose rather easy classification tasks". Defines **non-linear boost** (best non-linear minus best linear F1) and **learning-based margin**; a dataset is challenging only if both are ≥5 % (ideally 10 %). | Measure our own non-linear boost per slice (LR vs LightGBM vs plus-neural features) before investing GPU hours. |
| 46 | Grinsztajn, Oyallon, Varoquaux. *Why do tree-based models still outperform deep learning on typical tabular data?* | NeurIPS 2022 Datasets & Benchmarks | https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html | 45 datasets: trees remain SOTA on medium-sized (about 10K) tabular data. NNs need robustness to uninformative features, preservation of data orientation, and learning irregular functions. | Pair-feature tables are tabular with irregular decision boundaries (thresholds on numbers matching etc.). GBDT is the right head. |
| 47 | Shwartz-Ziv, Armon. *Tabular Data: Deep Learning is Not All You Need* | Information Fusion, 2022, pp. 84–90, doi:10.1016/j.inffus.2021.11.011 | Crossref + https://arxiv.org/abs/2106.03253 | XGBoost beats the proposed deep tabular models and needs less tuning, but **an ensemble of deep models plus XGBoost beats XGBoost alone.** | Supports stacking neural scores into the GBDT, or averaging a GBDT and a CE on the uncertain slice. |
| 48 | Zeakis, Papadakis, Skoutas, Koubarakis. *Pre-trained Embeddings for Entity Resolution: An Experimental Analysis* | PVLDB 16(9), 2023, pp. 2225–2238, doi:10.14778/3598581.3598594 | Crossref + https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf (text read) | Without fine-tuning, BERT-style models have worse blocking recall than static GloVe, and **Sentence-BERT models are best overall** (sentence-level training on 1B+ pairs). | Off-the-shelf sentence embedders (e5, MiniLM) are a sound starting point. Our fine-tuning then adapts them to the noise. |
| 49 | Foursquare Location Matching (Kaggle 2022), winners' approaches | Foursquare developer blog (company page, 2022-09-27) | https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ | Four prize-winning approaches (the blog does not give ranks): (1) Philipp Singer, "metric learning model with ArcFace loss" for candidates plus a bi-encoder NLP model; (2) Yuki Uehara: candidates by text similarity plus geo, "filtering by LightGBM", "pairwise match classification by two transformer language models", GNN post-processing (0.907 to 0.946); (3) team re:waiwai: LightGBM with limited features, then LightGBM with "similarity features (Levenshtein and Jaro-winkler distances)", then BERT models; (4) team 2:30: TF-IDF and geo blocking, "transformer-based blocking stage", LightGBM, XGBoost and BERT ensemble. | The closest public analog: multilingual POI name+address matching at millions of records. The winning shape is **cheap blocking, then a GBDT filter, then a transformer on survivors, then a graph/cluster post-process**. That is our plan. The Kaggle writeup pages themselves are JS-rendered and were not read. |

---

## 4. Synthesis for this challenge

### 4.1 GBDT vs neural: verdict

- **Keep LightGBM as the final pairwise scorer.** Evidence: #43, #44, #46, #47, #49. Our pairs are
  "structured but dirty": name and address fields with heavy noise. That is the regime where DL added
  6 to 33 % F1 *over a feature learner with generic features* (#43). But our GBDT has domain features
  (transliteration, learned abbreviation maps, numbers, competition ranks), so the neural margin will be much
  smaller. Domain features closed most of the gap in #43 as well (+0.8 to 1.4 %).
- **Neural signals enter as features**: fine-tuned bi-encoder cosines (name, addr, full), their ranks within
  the S1's candidate list, CE probability on the uncertain slice, and char-Siamese name cosine. Measure
  per-slice non-linear boost (#45): LR vs GBDT vs GBDT+neural, on the slices cross-script India, alias,
  pure-address, empty-address, US, and France-proxy (LOCO).
- A full cross-encoder over all candidates is not affordable at 1.7M S1 × about 40 candidates (see
  `neural_em.md`). Ditto measured about 467 pairs/s on a V100 at max_len 256. Restrict it to GBDT
  p ∈ [0.05, 0.95].

### 4.2 Domain-fine-tuned small encoder vs generic embeddings: verdict

- **Evidence for fine-tuning:**
  - SC-Block (#2): supervised contrastive beats generic SBERT and BM25 at large vocabularies.
  - R-SupCon (#1): contrastive pre-training helps, most at small data.
  - AugSBERT (#28): +6 in-domain, +37 under shift.
  - Neculoiu (#32): invariances only appear for trained noise types.
- **Evidence for generic:** Zeakis (#48): off-the-shelf sentence models are the best *untuned* option. The
  existing `stage_embed.py` already uses e5-small untuned, which is the right baseline.
- **Our case is extreme on the fine-tune side.** 7.6M labeled positives with a synthetic noise process the
  generic model never saw: junk prefixes, leetspeak, domain-as-name, mixed-script tokens. Generic models also
  split typo'd and leet tokens into unrelated WordPieces (#30).
- **Risk: forgetting French handling** while fine-tuning on US+India. Mitigate by mixing in transductive
  France S1 self-supervision (4.8), keeping lr small (2e-5 to 5e-5) and epochs few (1 to 3), and checking on
  LOCO.

### 4.3 Loss choice

| Loss | Fit for us | Decision |
|---|---|---|
| Symmetric InfoNCE / MNRL, in-batch plus appended hard negatives (#12, #13) | Simple, strong, scales with GradCache (#15); the standard in sentence-transformers | **Default (P0)** |
| SupCon multi-positive (#10) | Uses all 2 to 6 copies per S1 at once; also tightens target-target similarity | P1/P2 variant (needs distractor masking for target-target) |
| Triplet with semi-hard mining (#19, #20) | Superseded by batch contrastive per #10; useful only as an ablation | P3 |
| Additive margin (#14) | Cheap addition; helps bitext (cross-script) retrieval | P1 flag on MNRL |
| ArcFace over S1 ids (#17, #18) | Used by a Foursquare winner. Test entities are unseen (the metric still transfers), but it needs a 2.2M-class head (Partial FC) | P3 |
| Margin-MSE distillation from CE (#29) | Best way to push CE knowledge into the retriever | P1 after the CE exists |
| Self-teaching KL, clean vs corrupted (#30) | Directly targets the noise operators, and works label-free on test S1 | P1 (France) |

Hyperparameters to start with (as used in the verified papers, not tuned for us):
- temperature 0.05 to 0.07 (R-SupCon used 0.07);
- batch 512 to 1024 (R-SupCon and SC-Block used 1024);
- lr 2e-5 to 5e-5, max_len 48 to 64 tokens (names and addresses are short: median name 19 to 27 chars,
  address 32 to 76);
- 1 to 3 epochs over S1 anchors.

### 4.4 Hard-negative recipe (S1-anchored, clean labels)

Sources, cheapest first:

1. **Cluster batching (Sudowoodo Alg. 2, #3).** K-means on the TF-IDF vectors *within a country*, then fill
   batches cluster by cluster. Every in-batch negative becomes lexically close, at no extra encoding cost.
2. **Blocking candidates as mined negatives (DPR #13 / ANCE #21).** `cands_<tag>.parquet` minus GT gives the
   hard negatives per S1. Take 1 to 3 per anchor per epoch.
3. **Typed negatives**, logged so we can ablate them:
   - N1 same normalized name, different address (chains and franchises; the S1 name-uniqueness gap);
   - N2 same street and house number, different name (co-located businesses; alias traps);
   - N3 same name and city, house number differs by one digit (real neighbour vs digit typo; our number
     features must separate these);
   - N4 near-duplicate distractor targets (no S1). They are true negatives for every S1.
4. **Refresh once** (ANCE): after epoch 1, re-encode and re-mine top-10 with the fine-tuned model.
5. **Sampling policy:**
   - Arm A: hardest-k.
   - Arm B: NV-Retriever style, 4 sampled from the top-10 (#26).
   - Arm C: SimANS weighting around the positive score (#25).
   - Arm D: semi-hard (#19).
   Pick by blocking recall@K *and* downstream GBDT macro F0.5 on the validation fold.
6. **False-negative guards (only where labels are incomplete):**
   - Target-target terms: mask distractor-distractor pairs, or drop negatives with
     score > 0.95 × positive score (#26), or use GISTEmbed (#27).
   - France pseudo-label batches: same guard.

### 4.5 Distillation and self-training plan (RocketQA-shaped, #22)

1. **M0:** e5-small fine-tuned with in-batch plus cluster negatives (P0).
2. **CE:** train a cross-encoder (mDeBERTa-v3-base or XLM-R-base, MIT) on M0's top-k from the train folds,
   with negatives taken from *that* distribution (RocketQA stresses this).
3. **M1:** M0 plus mined hard negatives plus Margin-MSE from CE (#29).
4. **France round (Noisy Student #31 / AugSBERT #28 / Sudowoodo PL #3):**
   - Run the full US+India stack on test France.
   - Keep pairs with CE >0.9 and GBDT >0.9 as positives, and <0.1 as negatives.
   - Also require the CollaborEM (#9) *target-side* reliability test on positives: the S1 is the target's
     top-scoring S1, with margin > θ over its second-best S1. CollaborEM reports 99 % average positive-label
     accuracy with mutual-best-plus-margin, vs 88 % for a GAN labeller. Tune θ on LOCO India.
   - Fine-tune M1 on them plus France S1 self-supervision, with the noise operators as student noise.
   - Also feed the high-confidence pairs to abbreviation and region mining (E-8, `name_address.md`).
   - One round only, validated first on LOCO (India treated as unlabeled).

### 4.6 Synthetic noise operator library (the augmentation core)

Implement it once in `src/ber/augment.py` (pure Python, deterministic with a seed, unit-tested) and reuse it for:
- contrastive positives (S1 vs corrupt(S1));
- self-teaching (#30);
- France self-supervision;
- a synthetic "stress" validation set.

Fit each operator's rate from train S1-to-target diffs, per country and per source (S2 vs S3 differ). The
rates quoted below come from `DATA_ANALYSIS.md`.

| Operator family | Examples seen in data | Implementation (language-agnostic) | Rate source |
|---|---|---|---|
| Case change | S2 US names 21 % ALL-UPPER | upper / lower / title | train diff |
| Whitespace noise | doubled spaces about 11 % | double random spaces | train diff |
| Junk prefix/suffix | `--`, `<<`, `##`, `***` about 2 % | sample from the junk-token set *observed in train* | train diff |
| Bracketing | `[Inc]`, `(Ltd)` 6 to 10 % | wrap an edge token in `[]` / `()` | train diff |
| Char edits (GeCo #40) | "Hospirlg", "Madr" | insert/delete/substitute/transpose, biased to the middle/end | train diff |
| Leetspeak / OCR | `0`↔`o`, `l`↔`I` | confusion pairs *learned from train diffs* | train diff |
| Diacritic injection/removal | "Léarning", "Àmicale", "Immôbiliere" | NFD, add/remove a combining mark on a random vowel; strip all | train diff (+France: target non-ASCII 24 %) |
| Token reorder | "Center Solutions Cinderella's" | shuffle a span of 4 or fewer (Ditto span_shuffle) | train diff |
| Edge-token movement | "LLC Moncada …", "Pvt. EFS …" | move a *frequent edge token* (top-N by per-country frequency at name start/end) to the other edge | discovered per country, including France (SARL/SAS emerge from counts, not a list) |
| Generic token inject/delete | legal forms, "Services", "Group", "The", "Dr" | inject/delete from the per-country frequent-token list | train diff |
| Truncation | "Deleon", "Oo, Scott &" | keep the first k tokens | train diff |
| Alias wrapper | "X dba Y", "a/k/a", "f/k/a" | wrap with an alias marker plus a random *other* name from the same country (the positive keeps the original part) | train diff |
| Domain/handle | "laxmibombayservices.com", "@parabolalaw", "www.…" | lowercase, strip spaces and legal forms, add a suffix from the observed set (.com / www. / @) | train diff |
| Acronym | "SC", "PC" | initials of the tokens | train diff |
| Cross-script (India) | "राम मार्केटिंग प्राइवेट लिमिटेड", "Sun पावर Provision" | **invert the learned native-to-Latin map** (Latin to native for mapped tokens): full or partial (mixed-script) | train diff (India S2 names 28 % non-ASCII) |
| Address component reorder | "OH, Columbus, 5559 Orville Avenue" | split on commas and permute | train diff |
| Component drop | house number / city dropped | drop a comma-component (Ditto attr_del analogue) | train diff |
| Abbreviation (generic) | St/Street, Rd/Road, R./Rue, BD/Boulevard | for frequent address tokens: prefix(k)+optional ".", or consonant skeleton (Blvd/Bd), or first+last letter. **Learned maps apply first; generic rules cover unseen languages.** | train diff + per-country token frequency |
| State/region form | TX↔Texas, MH↔Maharashtra↔महाराष्ट्र | learned maps only (US/India); no hand-coded France regions | learned maps |
| Null tokens | "null", "NULL", "<NULL>", "N/A" 3 to 4 % | insert or replace a component | train diff |
| Landmark | "Near Fortis Hospital" (India 9 to 13 %) | prepend "Near <random frequent POI-like token span from the same country>" | train diff |
| Number noise | "00514", "5935-D", "#", "No", "H.NO", digit typos | pad zeros, append a letter/dash, prefix markers observed in train, one-digit substitution | train diff |

Guardrails (from #1, #5 and #42):
- Never corrupt so hard that the label flips (R-SupCon's "iPhone 4s" example). Cap at 2 or 3 operators per
  record and never delete *all* digits of the house number together with the name.
- MixDA-style interpolation (#5) is an option if we see label corruption.
- Report results with and without augmentation on the in-distribution validation fold (it may cost a little
  there) and on LOCO (where it should help).

### 4.7 Preparing for unseen France without external data

Compliance position: `COMPLIANCE_CHECKLIST.md` C4/C6/D1. No external lexicons. No hard-coded country logic.
Hand-written French abbreviation lists are "allowed but disfavoured" (C6). We avoid them.

1. **Transductive self-supervision on test S1, all countries including France.**
   - Anchor = S1 record; positive = corrupt(S1) using the generic operators.
   - Negatives = other S1 records of the same country, in cluster batches.
   - These negatives are clean because S1 is deduplicated. This fixes the reason SimCLR pre-training failed
     in R-SupCon (#1).
2. **Frequency-driven generic operators.** The abbreviation, edge-token and inject/delete operators all draw
   from *per-country token statistics computed on unlabeled records*. For France they automatically pick up
   Rue/Avenue/Boulevard/Chemin/SARL/SAS without a list.
3. **Self-teaching (#30)** on France S1: make the score distribution of corrupt(S1) over its TF-IDF candidates
   match that of clean S1.
4. **One pseudo-label round (4.5 step 4).** High-confidence France pairs then (a) fine-tune the encoders and
   (b) feed E-8 abbreviation and region co-occurrence mining.
5. **Validation before trusting any of it:**
   - LOCO: train on US only, treat India as the unseen country, and measure the lift from steps 1 to 4 on India.
   - Reverse direction as well.
   - Belinkov & Bisk (#42) warn that unseen noise *types* do not transfer. The France-only noise type we know of
     (region to département) is not synthesizable. Our features must degrade gracefully when the region token
     disagrees: street+number+name evidence dominates, and region is a soft feature.

### 4.8 Cross-script robustness

- **Positives:** the 551k train cross-script pairs, plus synthetic cross-script and mixed-script variants
  made by inverting the learned map (4.6). The synthetic variants also cover Indic scripts that are rare in
  train (Oriya 0.4 %, Gurmukhi 0.3 %).
- **Two options, both compliant:**
  - (a) Multilingual subword model (e5-small / MuRIL / LaBSE) fed the *raw* native-script text. It relies on
    pretraining that saw Indic scripts; MuRIL explicitly trained on transliterated pairs (#37).
  - (b) A from-scratch char-Siamese over Unicode code points (#32, #34) that learns transliteration
    correspondences from our pairs (Symphonym-style student, #38, without their gazetteer data).
- Ablate both on the India cross-script slice: name-view recall@K, and GBDT F0.5 on India S2 (where the name
  token-set p10 is 11).
- Also compare against "transliterate first, then Latin model" (already in normalization v1). The neural view
  mainly helps tokens *outside* the 1,347-token learned map (3.6 % of test native name tokens are uncovered).

### 4.9 Compute (analytic estimates, ±3x, not measured; all on Modal / Kaggle / Colab / Lightning)

| Job | Size | Estimate |
|---|---|---|
| Fine-tune e5-small (118M), fp16, max_len 48, batch 512, GradCache, 1 positive + 2 hard negatives per anchor | 2.2M anchors, about 8.8M sequences per epoch | About 30 to 60 min per epoch on A10G/L4; 1 to 3 epochs |
| Re-encode train for the ANCE refresh | 12.5M records × 1 to 3 views | About 20 to 60 min A10G |
| Encode test and FAISS 128-d search | about 11.7M records; 1.73M queries × top-50 | About 20 to 60 min GPU + CPU FAISS IVF |
| CE teacher (mDeBERTa-base) training | about 2 to 5M pairs from top-k | About 2 to 4 h A10G / about 1 h A100 |
| CE scoring of the uncertain slice only | 5 to 15M pairs | About 1 to 3 h A10G |
| Char-Siamese from scratch (1 to 5M params) | name pairs | Under 1 h GPU, or a few hours on a CPU box |

---

## 5. Models (license read from the HF model card YAML `license:` line on 2026-09-25)

| Model | License (card) | Params (safetensors count where available) | Compliant | Role here |
|---|---|---|---|---|
| intfloat/multilingual-e5-small | mit | 117.65M | Yes | **P0 bi-encoder to fine-tune** (already in `stage_embed.py`) |
| intfloat/multilingual-e5-base | mit | 278.0M | Yes | Larger bi-encoder if small saturates |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | apache-2.0 | 117.65M | Yes | Alternative small bi-encoder |
| sentence-transformers/LaBSE | apache-2.0 | 470.9M | Yes | Cross-script bitext-trained backbone (India slice) |
| google/muril-base-cased | apache-2.0 | base-size (count not listed on the API) | Yes | Indic plus transliteration-aware backbone |
| Alibaba-NLP/gte-multilingual-base | apache-2.0 | 305.4M | Yes | Alternative multilingual bi-encoder |
| BAAI/bge-m3 | mit | about 568M (not listed on the API; UNVERIFIED count) | Yes | Strong multilingual embedder; heavier |
| Qwen/Qwen3-Embedding-0.6B | apache-2.0 | 595.8M | Yes | Heavier teacher-grade embedder |
| google/canine-s | apache-2.0 | 132.1M | Yes | Pretrained char-level backbone |
| google/byt5-small | apache-2.0 | small (count not listed) | Yes | Byte-level encoder option |
| FacebookAI/xlm-roberta-base | mit | 278.9M | Yes | Cross-encoder teacher backbone |
| microsoft/mdeberta-v3-base | mit | base (count not listed) | Yes | **Cross-encoder teacher backbone (P1)** |
| BAAI/bge-reranker-v2-m3 | apache-2.0 | 567.8M | Yes | Off-the-shelf multilingual cross-encoder baseline |
| ai4bharat/IndicBERTv2-MLM-only | mit | base (count not listed) | Yes | Indic encoder option |
| google-t5/t5-base | apache-2.0 | 222.9M | Yes | Only if we try Rotom InvDA (P3) |
| jinaai/jina-embeddings-v3 | **cc-by-nc-4.0** | 572.3M | **No** | Do not use |
| google/embeddinggemma-300m | **gemma** (gated) | 302.9M | **No** | Do not use |
| facebook/fasttext-en-vectors | **cc-by-sa-3.0** | n/a | **No** (not MIT/Apache) | Train our own fastText on the provided data instead |

All compliant candidates are far below the 8B cap. Pin the commit sha (as `stage_embed.py` already does).
Models trained from scratch on the provided data (char-Siamese, own fastText, LightGBM) carry no pretrained
license.

## 6. Repos (GitHub API, 2026-09-25)

| Repo | License | Stars | Last push | Use |
|---|---|---|---|---|
| huggingface/sentence-transformers | Apache-2.0 | 19,118 | 2026-09-24 | Training library: MNRL, CachedMNRL (GradCache), GISTEmbed, MatryoshkaLoss, MarginMSE, hard-negative mining utilities |
| luyug/GradCache | Apache-2.0 | 448 | 2024-03-26 | Reference GradCache |
| princeton-nlp/SimCSE | MIT | 3,652 | 2024-10-16 | Reference SimCSE code |
| microsoft/ANCE | MIT | 389 | 2026-01-06 | Reference ANN-refresh mining |
| microsoft/SimXNS (SimANS) | MIT | 115 | 2024-01-09 (archived) | SimANS sampling reference |
| megagonlabs/ditto | Apache-2.0 | 318 | 2024-04-17 | DA ops reference |
| megagonlabs/rotom | BSD-3-Clause | 23 | 2022-05-31 | InvDA / meta-DA reference (code license is not a model license) |
| megagonlabs/sudowoodo | BSD-3-Clause | 19 | 2023-05-24 | Cluster batching, cutoff DA, pseudo-labelling |
| wbsg-uni-mannheim/contrastive-product-matching | BSD-3-Clause | 38 | 2022-02-11 | R-SupCon plus source-aware sampling |
| anhaidgroup/deepmatcher | BSD-3-Clause | 624 | 2024-06-18 | Historical DL EM baseline |
| anhaidgroup/py_entitymatching | BSD-3-Clause | 195 | 2024-05-29 | Magellan feature generation reference |
| FlagOpen/FlagEmbedding | MIT | 12,191 | 2026-08-24 | bge-m3 / reranker fine-tuning scripts |
| makcedward/nlpaug | MIT | 4,667 | 2026-09-08 | Char/keyboard/OCR augmenters. **Do not use the synonym/word-embedding augmenters** (external lexicons/embeddings, risk C4) |
| dhwajraj/deep-siamese-text-similarity | MIT | 1,413 | 2020-05-06 | Char-level Siamese LSTM reference (TensorFlow, stale) |
| TheoViel/kaggle_foursquare | **no license file** | 14 | 2022-07-08 | Foursquare 4th-place pipeline (2-level boosting). **Method reference only; no license, so do not copy code.** |
| lightgbm-org/LightGBM | MIT | 18,811 | 2026-09-24 | Final matcher |
| catboost/catboost | Apache-2.0 | 9,116 | 2026-09-24 | Alternative GBDT |
| rapidfuzz/RapidFuzz | MIT | 4,138 | 2026-09-12 | String features |
| AI4Bharat/IndicXlit | MIT (code) | 142 | 2023-10-13 | **Model weights trained on web-mined data, so compliance risk.** Not recommended. |

## 7. Compliance flags specific to this topic

- **OK:**
  - fine-tuning MIT/Apache weights on the provided train data;
  - self-supervised training on unlabeled *provided* test records (transductive; document it as label-free,
    F3 in the checklist);
  - synthetic corruption whose operator rates are fitted from provided data.
- **Risk / avoid:**
  - pretrained fastText vectors (CC-BY-SA);
  - jina-v3 (CC-BY-NC) and EmbeddingGemma (Gemma);
  - Symphonym or IndicXlit-style weights trained on external gazetteers or web-mined corpora;
  - nlpaug synonym augmenters (WordNet/PPDB);
  - any hand-coded French region-to-département table (external knowledge; C6/D1).
- Rotom InvDA would use t5-base (Apache-2.0). Acceptable, but P3.
- Library licenses (BSD/MIT/Apache) are separate from model licenses. Record both in the final README.

---

## 8. What we should do

Experiment IDs refer to `EXPERIMENT_PLAN.md`; new IDs are proposed as E-C*.

### P0 (do first)

- **E-C1 (extends E-9): fine-tuned e5-small bi-encoder for blocking and features.**
  - Symmetric MNRL, temperature 0.05, batch 512 to 1024 via CachedMNRL, 1 to 2 epochs on train folds.
  - Batches are country-homogeneous, 50 % TF-IDF k-means cluster batches.
  - Add 1 to 2 hard negatives per anchor from `cands_<tag>.parquet` minus GT.
  - Three views: name, addr, full.
  - Metrics: dense-view recall@K and union recall vs untuned e5 and TF-IDF (E-B2 grid); then GBDT macro F0.5
    with the new cosine and rank features.
  - Expected: most of the lift on cross-script, alias and pure-address slices.
- **E-C2: `src/ber/augment.py` noise-operator library** (table 4.6), with rates fitted from train diffs per
  country and source. Unit tests: each operator is deterministic under a seed, and the label-preservation caps
  hold. It is needed by E-C1 (optional positive augmentation), E-C5 and E-C6.
- **E-C3: LOCO evaluation for every neural component** (train US only, evaluate India; and the reverse).
  Report the gap closure, not just in-distribution F0.5.
- **E-C4: non-linear-boost audit** (#45): LR vs LightGBM vs LightGBM+neural features, per slice, before any
  bigger GPU spend.

### P1

- **E-C5: transductive self-supervision on test S1 (all countries, France included).**
  - S1 vs corrupt(S1) positives, clean S1 negatives, mixed as 20 to 30 % of E-C1 batches.
  - Add self-teaching KL (#30).
  - Validate by the LOCO analogue (unlabeled India S1 included, India labels hidden).
- **E-C6 (with E-10): cross-encoder teacher.**
  - mDeBERTa-v3-base or XLM-R-base, trained on E-C1's top-k (RocketQA step 2).
  - Uses: a GBDT feature on p ∈ [0.05, 0.95]; Margin-MSE distillation into the bi-encoder (M1).
- **E-C7 (with E-8): France pseudo-label round** (Noisy Student / AugSBERT).
  - Thresholds CE and GBDT >0.9 / <0.1.
  - Fine-tune M1 with operator noise.
  - Feed abbreviation and region co-occurrence mining.
  - Guard pseudo-label batches with NV-Retriever 95 % or GISTEmbed.
- **E-C8: hard-negative type and sampling ablation.**
  - Types N1 to N4.
  - Sampling arms: hardest-k, sampled-from-top-10, SimANS, semi-hard.
  - Report blocking recall and precision-side F0.5 separately.
- **E-C9: Matryoshka (64/128/256/384)** to cut the FAISS index for about 10M targets to 128-d, if recall
  holds within 0.2 pp.

### P2

- **E-C10: from-scratch char-Siamese name encoder** over Unicode code points (char-CNN or trigram-hash
  two-tower, 1 to 5M params), trained on name pairs including cross-script and augmented pairs. Serves as a
  feature and an extra blocking view for names. Compare with CANINE-s fine-tuned.
- **E-C11: backbone ablation on the India cross-script slice:** e5-small vs LaBSE vs MuRIL vs gte-multilingual.
- **E-C12: SupCon multi-positive with target-target terms** (distractor-distractor masked), to sharpen
  collective (target-cluster) features (E-6/E-14).
- **E-C13: one ANCE refresh** (re-mine with the fine-tuned model) if E-C8 shows the mined negatives matter.

### P3

- ArcFace with Partial FC over S1 ids (Foursquare-winner style), only if MNRL plateaus.
- Rotom InvDA (t5-base seq2seq augmentation): low value at 7.6M labels.
- DADER-style MMD alignment between France and train embeddings (#7). Pseudo-labelling is simpler and better
  evidenced.
- Barlow Twins / redundancy regularization (Sudowoodo RR, +1.74 % on one dataset).

---

## 9. Open items / not verified

- Rotom's exact F1 tables (the ACM PDF returned 403, and the Megagon page gives no numbers). Only the method
  description and the repo README were read.
- CollaborEM (#9): read on 2026-09-25 from the arXiv PDF (the CollaborER preprint). The ALG method and the
  label-accuracy numbers are recorded, but not the per-dataset F1 tables.
- WDC Products per-system unseen-entity numbers: not re-read here; see `neural_em.md`.
- Foursquare final ranks: the blog does not rank the four winners, and the Kaggle writeups are JS-rendered and
  were not read.
- Parameter counts for MuRIL, ByT5-small, mDeBERTa-v3-base, IndicBERTv2 and bge-m3 are not listed in the HF API
  safetensors metadata. All are base-size encoders, well under 8B.
- Every compute number in 4.9 is an analytic estimate and must be replaced by measured Modal timings.
