# Multi-source ER, clustering, collective and transitive inference: research notes

Topic owner: research workflow, topic "graph". Written 2026-09-25.
Scope: how to turn pairwise S1-S2 / S1-S3 scores (plus S2-S2, S3-S3, S2-S3 evidence) into per-S1 match sets that score well under per-entity macro F0.5.

Verification rule used here: every paper below was confirmed on a primary page (arXiv abs/HTML, VLDB/PVLDB PDF, OpenProceedings PDF, AAAI OJS, NeurIPS proceedings, publisher/institutional repository page, or the Crossref DOI registry record). The page or record I read is in the `verified via` column. Numbers are only the ones I read on that page. Anything I could not confirm is marked UNVERIFIED. No model or heavy computation was run on the laptop. The only local computation was PDF-to-text extraction and a tiny F0.5 arithmetic table (section 1.3).

---

## 0. TL;DR

1. **Never use plain transitive closure (connected components) on a low-threshold graph.** Every comparative study found it has the worst precision: Hassanzadeh et al. PVLDB 2009, Saeedi et al. ADBIS 2017 / ESWC 2018, Papadakis et al. EDBT 2022 / VLDBJ 2023, Draisbach et al. JDIQ 2019, and GraLMatch EDBT 2025. In GraLMatch, a pairwise matcher with F1 97.66 collapses to near-zero group-level precision once transitive matches are added. The cause is a few false edges that bridge two groups. Under F0.5 this is the worst failure mode.
2. **Our data is exactly the "multi-source clean/dirty" (MSCD) setting** of Lerm, Saeedi & Rahm (BTW 2021) and Saeedi, David & Rahm (KEOD 2021). S1 is clean (duplicate-free), while S2 and S3 are dirty (several copies per entity). Their finding: enforcing "at most one clean-source record per cluster" raises precision a lot at the same recall. MSCD-HAC single-linkage gives the best F on mixed datasets, complete-linkage the best precision, and average-linkage wins when clusters are large. Splink (MIT) now ships this constraint directly: `cluster_using_single_best_links(duplicate_free_datasets=[...])`, which can be restricted to a subset of the sources.
3. **Our measured GT makes the constraints hard facts, not assumptions.** No target is linked to two S1 records (many-to-one). Per S1 there are ≤5 S2 matches and ≤6 S3 matches. Within-source copies share a corrupted base. So: (i) each target joins at most one S1 cluster, (ii) each cluster holds exactly one S1 or none (distractor groups), and (iii) cluster-size caps are available as over-merge detectors.
4. **Recommended architecture: B, then a constrained, anchored version of C used as repair.**
   - (A) independent scoring plus soft many-to-one exclusivity is the floor.
   - (B) collective "support" features in a second stacked stage (out-of-fold). This is where most of the precision and recall gain should come from, and the project already has a v1 in `stage_collective.py`.
   - (C) global clustering, but anchored and constrained: S1 records are fixed cluster seeds. Each target is assigned to at most one seed or to "none". Use average or complete linkage, drop weak links (CLIP link strength), apply per-source size caps, and gate every addition on a floor for the direct S1-target probability. Use it as a repair and veto step, not as the primary decision.
5. **Make the decision layer group-aware.** Copies of one entity have strongly correlated labels. The current expected-F DP assumes independent Bernoulli labels, so it underestimates the risk of accepting a wrong copy group. At n=4, adding a wrong group of 2 to a perfect answer drops F0.5 from 1.0 to 0.714. Treat each copy group as one Bernoulli variable weighted by its size (section 7.4).
6. **Failure modes to design against:**
   - chains: A~B~C with A≁C;
   - franchises and same-name S1s (26-36% of S1 names are non-unique);
   - shared addresses (malls, office blocks, "Near X" landmarks in India);
   - hub names (acronyms "SC"/"PC", generic names, website names);
   - distractor copy groups (≈26% of targets in train, and test is denser);
   - error propagation from early merges;
   - stacking leakage;
   - density shift to test.

---

## 1. Problem framing: the graph view of our data

### 1.1 Nodes, edges, constraints (all measured in DATA_ANALYSIS.md)

- Nodes (test): S1 = 1.73M (clean), S2 = 4.9M and S3 = 5.1M (dirty, multi-copy). About 12M nodes per split.
- Edge types:
  - `S1-T`: stage-1 pair probabilities p1(q,t) over blocked candidates (≈124 candidates per S1 before pruning in the benchmark).
  - `T-T`: target-target similarity or dedup probability. Within-source edges (S2-S2, S3-S3) and cross-source edges (S2-S3).
  - `S1-S1`: never needed as match edges (S1 is clean). S1-S1 name or address collisions are useful as *ambiguity* features (franchise indicator).
- Hard constraints from the GT:
  - C1 **many-to-one**: every matched target belongs to exactly one S1. Target multiplicity = 1 for all 7.64M matched targets.
  - C2 **source consistency for S1**: a cluster contains at most one S1 record (S1 deduplicated).
  - C3 **per-source caps**: S2 matches per S1 are in 0-5 (mode 1) and S3 matches in 0-6 (mode 1). The total is ≤ 11 (max observed 11, then 10 at 534 S1s). This is a soft prior: test may differ, and it must be validated before being used as a hard cap.
  - C4 **country consistency**: 100% of true pairs have the same country, so all graph work runs inside country partitions (label-agnostic, works for France).
- Soft structure:
  - Within-source copies share a corrupted base, so their within-source similarity is high. Cross-source agreement (S2 copy vs S3 copy) is the more *independent* evidence.
  - Distractors: ≈26% of S2/S3 records match no S1, and test has ~24% more targets per S1.

### 1.2 What the metric rewards

Per-S1 F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP), macro-averaged over all S1s. A singleton S1 (5.58% of train) scores 1 only if we predict nothing.

Consequences for graph inference:
- **Group errors are expensive.** Merging one wrong copy group into an S1 cluster adds g false positives at once.
- **Transitive recall gains are cheap but small.** Adding one missed copy when 3 of 4 are already correct raises F0.5 from 0.938 to 1.0 (+0.062). Adding one wrong copy drops it to 0.75 (-0.188).
- Singletons: any accepted target costs the full 1.0 for that S1. A copy group that "floats" near a singleton S1 (for example a distractor entity with the same name) must be rejected as a unit.

### 1.3 Break-even posteriors (computed exactly from the formula; no data involved)

Adding one more candidate to an S1 whose true count is n and that already has k correct and 0 false predictions:

| n | k | F now | ΔF if TP | ΔF if FP | break-even P(true) |
|---|---|---|---|---|---|
| 2 | 1 | 0.833 | +0.167 | -0.333 | 0.667 |
| 3 | 1 | 0.714 | +0.195 | -0.260 | 0.571 |
| 3 | 2 | 0.909 | +0.091 | -0.242 | 0.727 |
| 4 | 1 | 0.625 | +0.208 | -0.208 | 0.500 |
| 4 | 2 | 0.833 | +0.104 | -0.208 | 0.667 |
| 4 | 3 | 0.938 | +0.062 | -0.188 | 0.750 |
| 6 | 3 | 0.833 | +0.076 | -0.152 | 0.667 |
| 6 | 5 | 0.962 | +0.038 | -0.128 | 0.769 |

Adding a whole copy group of size g that is either all-true or all-false:

| n | k | g | F now | F if right | F if wrong | break-even P(group true) |
|---|---|---|---|---|---|---|
| 4 | 2 | 2 | 0.833 | 1.000 | 0.500 | 0.667 |
| 3 | 1 | 2 | 0.714 | 1.000 | 0.333 | 0.571 |
| 6 | 3 | 3 | 0.833 | 1.000 | 0.500 | 0.667 |
| 4 | 4 | 2 | 1.000 | n/a | 0.714 | never add |

Reading: graph evidence (siblings, transitivity) should only add a target when its calibrated posterior clears roughly 0.6-0.77. The bar depends on how complete the current set already is. The expected-F DP in `src/ber/decide.py` already handles this when posteriors are well calibrated. The graph layer's job is to (1) make posteriors sharper and better calibrated, and (2) supply the correlation structure (groups) to the decision rule.

---

## 2. Literature by theme (verified; full table in section 3)

### 2.1 Comparative studies of ER clustering algorithms

- **Hassanzadeh, Chiang, Lee, Miller, PVLDB 2(1) 2009** (Stringer). The foundational comparison of "unconstrained" clustering on a thresholded similarity graph. It covers Partitioning (transitive closure), CENTER, MERGE-CENTER, Star, Ricochet (SR/BSR/CR/OCR), Correlation Clustering (Cautious approximation), MCL, Cut clustering and Articulation Point.
  - Their conclusion: partitioning "results in poor quality of duplicate groups", even compared with equally efficient algorithms. MCL is among the most accurate and also efficient. The sophisticated Cut and Correlation clustering gave lower accuracy than some single-pass algorithms.
  - Table 3 (medium-error data, F1 at threshold θ=0.2, a dense graph): Partitioning 0.177, CENTER 0.666, MERGE-CENTER 0.389, Star 0.644, SR 0.917, CCL 0.659, MCL 0.712. At θ=0.4: Partitioning 0.850, MCL 0.906. So transitive closure is only acceptable when the threshold is already near-optimal, and it falls apart on dense graphs.
  - MCL is least sensitive to the threshold, and not pruning low-weight edges "does not result in clusters with low precision".
  - **Adapt:** our S1-T plus T-T graph will be dense at useful recall levels. Use MCL, average-linkage or center-style algorithms, never partitioning.
- **Draisbach, Christen, Naumann, JDIQ 12(1) 2019.** Transitive clusters can be inconsistent (not all members are similar). They propose three structure-based algorithms. Extended Maximum Clique Clustering (EMCC) and MCL are best for larger clusters. EMCC has the best precision and works without edge weights.
  - **Adapt:** clique-like consistency (every pair of copies in a group is mutually similar) is a good *group validity* check before a group is attached to an S1.
- **Saeedi, Peukert, Rahm, ADBIS 2017 (LNCS, pp. 278-293).** Distributed (Flink) versions of six schemes for the multi-source case in FAMER: Connected Components, CCPivot, Center, Merge-Center, Star-1 and Star-2.
  - As summarised in their ESWC 2018 follow-up, Connected Components and Merge-Center have high recall and the poorest precision. Star variants produce overlapping clusters.

### 2.2 Exploiting clean sources: source-consistent and one-to-one clustering

- **CLIP / RLIP, Saeedi, Peukert, Rahm, ESWC 2018 (best research paper).** Clustering based on LInk Priority for duplicate-free sources.
  - Link *strength*: a link is **strong** if it is the maximum link from both endpoints (mutual best per source pair), **normal** if it is the maximum from one side only, and **weak** otherwise.
  - Link *degree* = min degree of the two endpoints. Priority order: higher similarity, then stronger, then lower degree.
  - Phase 1: connected components over strong links only, keeping "complete" clusters (one entity per source).
  - Phase 2: drop weak links and take components over strong+normal links. Source-inconsistent components are split by iterative max-priority merging that never violates source consistency.
  - RLIP repairs other algorithms' clusters (overlap resolution by association degree, then CLIP).
  - Reported: CLIP "outperforms all previous algorithms in terms of precision and F-measure for all three datasets". On the dirty music dataset DS2, F-measure is 82% vs 65-75% for the others. Recall comes from strong and normal links; the splitting mostly helps precision. Ignoring weak links keeps precision high at low thresholds.
  - **Adapt:** S1 plays the clean source, so "mutual best" between an S1 and a target is well defined (the target's best S1 and the S1's best targets in that source). S2 and S3 are *not* clean, so CLIP's one-per-source rule must not be applied to them. Use MSCD variants instead.
- **MSCD-AP, Lerm, Saeedi, Rahm, BTW 2021.** First treatment of a mix of clean and dirty sources. Affinity propagation gets a clean-source consistency constraint. A hierarchical version assigns clean-source entities to exemplars with the Hungarian algorithm (1:1) and dirty-source entities to their most similar exemplar.
  - Results: best F-measure on all mixed clean/dirty camera datasets, "excellent precision", comparatively low recall, and F stable across similarity thresholds.
  - The authors also report that first deduplicating the dirty sources and then running a clean-source clustering "performed worse than matching dirty sources" in a challenge they took part in.
- **MSCD-HAC, Saeedi, David, Rahm, KEOD 2021 (pp. 40-50).** Agglomerative clustering where a merge is skipped if it would put two records of the same clean source in one cluster. Optionally, weak links between clean sources are removed first.
  - Results: MSCD raises precision "dramatically while keeping the same recall" on all mixed datasets. MSCD single-linkage gives the best F on mixed data and matches CLIP on clean-only data. Complete-linkage gives the best precision (lower recall). Average-linkage beats single-linkage on the larger-cluster person dataset, because single-linkage and CLIP fill clusters to the maximum size and pick up false positives.
  - **Adapt:** this is almost exactly our Option C. Our clusters are small (≤11 targets), which favours S-LINK/A-LINK. Under F0.5 we should lean towards A-LINK or C-LINK and tune on validation.
- **Nentwig, Groß, Rahm, ICDMW 2016.** Holistic clustering of existing links from many sources. It identifies wrong links and finds additional ones.
- **Papadakis, Efthymiou, Thanos, Hassanzadeh, EDBT 2022 (arXiv 2112.14030); extended with Christen in VLDBJ 32:1369-1400, 2023.** 8 bipartite (1:1) matching algorithms on 10 datasets and more than 700 similarity graphs: CNC, RSR, RCA, BAH, BMC, EXC, KRC and UMC.
  - Conclusions (arXiv v3 section 7): UMC (greedy unique mapping, sort edges descending, accept if both ends are free) "is the best choice for balanced entity collections". EXC (mutual best) is the best choice when both effectiveness and speed matter. KRC (Király stable-marriage approximation) has very high or the highest effectiveness at higher cost. CNC (connected components restricted to 2-node clusters) has the highest precision and low recall. BAH is stochastic and unreliable.
  - CLIP reduces to UMC in the two-clean-source case.
  - **Adapt:** our S1-to-copy-group assignment is 1:many on the target side and at most 1 S1 per group, so UMC/EXC logic applies to *groups vs S1*. The classic Hungarian algorithm is cubic and was excluded there for scale. It is fine per connected component (small).
- **Gemmell, Rubinstein, Chandra, arXiv 1108.6016 (MSR-TR-2011-100).** Global one-to-one constraints between list sites give "very strong precision and recall", even when the sites contain some duplicates. They found max-weight matching "performs poorly in some situations", and report the method was used in a major search engine's back end.
  - **Adapt:** do not blindly maximise total weight. Greedy or mutual-best assignment with thresholds is more robust.
- **Splink `cluster_using_single_best_links`** (MIT; code read in `splink/internals/linker_components/clustering.py`). It "will include a record into a cluster if it is mutually the best match for the record and for the cluster". Clusters never contain more than one record from each dataset listed in `duplicate_free_datasets`, which can be a subset of the sources. It supports `ties_method` "drop" or "lowest_id". Splink also computes `node_degree`, `node_centralisation` and edge `is_bridge` graph metrics.
  - **Adapt:** a ready-made, MIT-licensed baseline for Option C with `duplicate_free_datasets=["s1"]`. Runs on DuckDB or Spark on a CPU box.

### 2.3 Correlation clustering: theory and scale

- **Bansal, Blum, Chawla, Machine Learning 56(1-3):89-113, 2004.** Defines correlation clustering: maximise agreements or minimise disagreements with +/- edge labels, with no k needed. Constant-factor approximation for min-disagree, PTAS for max-agree. Extends to real-valued edge weights in [-1,+1].
- **Ailon, Charikar, Newman, JACM 55(5) 2008** (Crossref record). Aggregating inconsistent information. The pivot algorithm (KwikCluster) that Pan et al. describe as a 3-approximation that "serially clusters neighborhoods of vertices".
- **Chierichetti, Dalvi, Kumar, KDD 2014, pp. 641-650** (Crossref record). Correlation clustering in MapReduce. This is the CCPivot used in FAMER. The (3+ε)-approximation detail is from a search snippet only and is UNVERIFIED on the primary page.
- **Pan, Papailiopoulos, Oymak, Recht, Ramchandran, Jordan, NeurIPS 2015.** C4 (serialisable, 3-approximation) and ClusterWild! (coordination-free). The abstract reports clustering "billion-edge graphs in under 5 seconds on 32 cores" and a 15x speedup.
- **Shi, Dhulipala, Eisenstat, Łącki, Mirrokni, PVLDB 14(11) 2021.** Parallel LambdaCC framework, which unifies modularity and correlation clustering. Billions of edges; up to 28.44x speedup over their own sequential baseline on a 30-core machine. Code: `jeshi96/parallel-correlation-clustering`, which has **no license** and so cannot be reused (reimplement if needed).
- **Gruenheid, Dong, Srivastava, PVLDB 7(9) 2014 (Incremental record linkage).** The motivating example is *business records*. Correlation clustering with cohesion penalty (1 - sim) inside clusters and correlation penalty (sim) across clusters. A greedy merge / split / move-records-between-clusters procedure can fix earlier linkage errors using new evidence.
  - **Adapt:** their merge/split/move moves are the natural local-search repair for our anchored clusters (move a target to a different S1 seed when its group evidence says so).
- **van Dongen, SIAM J. Matrix Anal. Appl. 30(1):121-141, 2008.** MCL: alternating expansion and inflation of a stochastic matrix.
  - The reference `micans/mcl` binary is **GPL-3.0**. `GuyAllard/markov_clustering` (Python) is MIT.

Take-away for us: correlation clustering is the principled objective for turning calibrated pair probabilities into clusters (+ weight = log-odds). Pivot-style algorithms scale. But with only ≤11 targets per true cluster and a strong anchor (S1), an anchored, constrained agglomerative method is simpler and easier to control for precision.

### 2.4 Collective ER and record-to-cluster matching

- **Bhattacharya & Getoor, TKDD 1(1) 2007.** Collective relational clustering: the similarity between clusters combines attribute similarity with neighbourhood (relational) similarity (weight α). Greedy agglomerative merging with a priority queue. High-precision bootstrap initialisation, because "our algorithm cannot undo any of these initial merge operations".
  - Table I (best F1 over thresholds), BioBase (high ambiguity): attribute-only 0.568; attribute plus **transitive closure 0.559 (closure hurts)**; naive relational 0.710; with closure 0.753; collective CR 0.819.
  - The authors note transitive closure "may result in false identifications in datasets with higher ambiguity".
  - **Adapt:** our "ambiguity" is S1 name non-uniqueness (26-36%) and shared addresses. Collective evidence helps most exactly there. Seed with high-precision merges only.
- **Rastogi, Dalvi, Garofalakis, PVLDB 4(4) 2011.** A framework that scales any collective EM algorithm by running it on small neighbourhoods and passing messages between them, with guarantees.
  - **Adapt:** our country partitions plus connected components of the thresholded graph are natural neighbourhoods.
- **Kirielle, Christen, Ranbaduge, TKDD 17(1) 2023 (RELATER).** Unsupervised graph-based ER for complex entities. It propagates constraints and attribute values, exploits ambiguity, and ends with a dynamic cluster refinement step.
- **Sen, Namata, Bilgic, Getoor, Gallagher, Eliassi-Rad, AI Magazine 29(3) 2008.** Survey and comparison of collective classification inference: iterative classification (ICA), Gibbs sampling, loopy belief propagation, mean-field.
  - **Adapt:** our stage-2 stacking with neighbourhood aggregates of stage-1 probabilities is one ICA iteration. A second iteration is cheap to test.
- **Benjelloun et al., Swoosh, VLDB J. 18(1):255-276, 2009.** Generic match-and-merge ER. Records are merged into composite records that are then matched again (record-to-cluster). They give properties under which merge-based ER is efficient and order-independent.
  - **Adapt:** a "group profile" built from S2 and S3 copies (majority tokens, union of house numbers) can be matched against S1 as one record. It is noise-robust when the group is correct, and toxic when the group is wrong.
- **Lacoste-Julien et al., SiGMa, KDD 2013, pp. 572-580** (Crossref; arXiv 1207.4525). Greedy 1:1 matching of large knowledge bases. Score updates propagate through the relationship graph after each accepted match. The abstract claims millions of entities at high precision.
- **Efthymiou, Papadakis, Stefanidis, Christophides, MinoanER, EDBT 2019 (arXiv 1905.06170).** Non-iterative, massively parallel matching rules that combine content, name and *neighbour* evidence. Apache-2.0 code.

### 2.5 Transitivity: when it helps, how it hurts, and how to clean it

- **GraLMatch, De Meer Pardo et al., EDBT 2025 (OpenProceedings, doi:10.48786/edbt.2025.01).** Multi-source *company* and security records: entity group matching. Predicted pairwise matches are turned into groups by connected components, so one false edge between two groups creates many false transitive matches (their Crowdstrike/Crowdstreet example).
  - Graph Cleanup (Algorithm 1): while the largest component exceeds γ, remove Minimum Edge Cuts. Above threshold μ, switch to Edge Betweenness Centrality, which removes fewer true edges but is slower. γ is set to the number of sources, because each group has at most one record per source.
  - A pre-cleanup drops token-overlap-blocked matches inside components larger than 50.
  - Numbers, synthetic companies with DistilBERT(128)-ALL: pairwise P/R/F1 99.28/96.09/97.66 (EDBT PDF). From the arXiv HTML: group-level F1 before cleanup ≈ 0.00, after cleanup 60.03. Real companies: pairwise 99.73, pre-cleanup 56.92, post-cleanup 91.64.
  - The paper: "precision becomes the deciding factor in the entity group matching of large volumes of records."
  - Repo: `FernandoDeMeer/GraLMatch`, **no license**.
- **TransClean, De Meer Pardo, Hadji Misheva, Braschler, Stockinger, arXiv 2506.04006 (2025).** *Transitive consistency*: does the pairwise model also predict "match" for the implied pairs? Components with more negative than positive transitive predictions are cut with minimum edge cuts. It needs a labelling budget (manual or LLM pseudo-labels) and ends with an edge-recovery phase.
  - Reported: +24.42 average F1 over pairwise matching in the multi-source setting. On synthetic companies, F1 goes from 0.04 to 69.47, removing 96.07% of false positives while losing 36.35% of true positives (DistilBERT, 10K labels).
  - **Adapt:** we have full train GT, so no labelling loop is needed. Use transitive consistency as a *feature* (fraction of implied S1-t / t-t' pairs the model agrees with) and as a trigger for component cleanup.
- **Christen, Obraczka, Hofer, Franke, Rahm, JDIQ 17(2) 2025 (arXiv 2401.14992, "GraphCR").** Cluster repair by classifying edges with graph-metric features:
  - node-level: PageRank, closeness, betweenness, clustering coefficient;
  - edge-level: similarity, link category, bridge flag, edge betweenness;
  - cluster-level: completeness ratio |E| / (|V|(|V|-1)/2).
  - Active learning (LLM as oracle in the journal version). It works for clean *and* dirty sources and beats CLIP-style repair when sources have duplicates.
  - **Adapt:** these exact metrics are cheap second-stage features on our per-S1 candidate subgraphs.
- **Primpeli & Bizer, ALMSER, ISWC 2021, LNCS 12922 pp. 182-199.** Graph-boosted active learning for multi-source ER. It uses the correspondence graph both to pick informative pairs and to derive extra training labels from graph structure. Repo `wbsg-uni-mannheim/ALMSER-GB` has **no license**.
- **Wang, Li, Kraska, Franklin, Feng, SIGMOD 2013, pp. 229-240** (arXiv 1408.6916). Deduces matches via transitivity (and non-matches via "match plus non-match implies non-match") to cut labelling cost. The logical rules are useful for consistency features.
- **Aberbach, Kejriwal, Shen, AAAI 2024 student abstract, 38(21):23434-23435.** Multipartite ER with k > 2 sources. Evaluating k-tuples directly gives substantially different results from aggregating pairwise evaluations.
  - **Adapt:** our metric is per-S1 set-based, so always evaluate clustering variants with the exact per-S1 macro F0.5 (`src/ber/metric.py`), never with pairwise P/R.

### 2.6 GNN and learned graph clustering

- **GraphER, Li et al., AAAI 2020, 34(05):8172-8179.** A token-centric ER-GCN on a record-token graph. It is pairwise matching, not clustering, and only an analogy for us.
- **HierGAT, Yao, Gu, Cong, Jin, Lv, SIGMOD 2022, pp. 429-442** (Crossref). A hierarchical heterogeneous graph (entity, attribute and token nodes) that models the interdependence between ER decisions. MIT code.
- **Geo-ER, Balsebre, Yao, Cong, Hai, WWW 2022, pp. 3061-3070.** POI entity resolution with *neighbourhood attention* over nearby entities with similar names. The motivation is chains, for example several distinct Starbucks POIs in one district: "a small business with a unique name is more likely to match with another with the same name".
  - **Adapt:** we have no coordinates, but the analogue is local name density: how many S1s or targets in the same country or city share this normalised name or this address. That is a chain/franchise indicator feature.
- **Yang et al., CVPR 2020, pp. 13366-13375** (face clustering; GCN-V vertex confidence and GCN-E edge connectivity on kNN affinity graphs) and **Xing et al., Hi-LANDER, ICCV 2021, pp. 3447-3457.** Hi-LANDER is a hierarchical GNN that merges predicted connected components level by level. Its abstract reports a 54% F-score gain over earlier GNN clustering methods. These are supervised, scalable, learned clustering of millions of nodes into an unknown number of clusters.
  - **Adapt:** they show how to learn T-T grouping at our scale. Code: `yl-1993/learn-to-cluster` (MIT) and a DGL example (DGL Apache-2.0). The pretrained face weights are irrelevant: we would train from scratch on train GT.
- **Kaggle Foursquare Location Matching (2022).** POI (business) matching with a per-record averaged IoU metric, structurally close to ours. Per Foursquare's own recap blog (2022-09-27), one top solution (Yuki Uehara) used candidate generation, a LightGBM filter, transformer pair models and then **GNN node-classification post-processing, which moved the score from 0.907 to 0.946**. This is a competition writeup, not peer-reviewed. Final-rank attributions differ between sources, so I do not claim a rank.
- **LLM-CER, Fu et al., PACMMOD 3(4) 2025 (SIGMOD 2026; arXiv 2506.02509).** LLM-based in-context set clustering: "up to 150% higher accuracy, 10% increase in the F-measure, and reducing API calls by up to 5 times". Not practical at 1.7M S1 with ≤8B open models. Listed for completeness.
- **Alper, Wang, Yang, Zheng, Ke, arXiv 2605.25814 (2026).** Iterative probabilistic label propagation over an evolving graph, with budgeted LLM queries. It argues that static blocking, then matching, then clustering pipelines propagate errors. (Also listed in neural_em.md.)

---

## 3. Paper table

| # | Paper | Authors | Year | Venue | Verified via | Main result read on source | How it adapts here |
|---|---|---|---|---|---|---|---|
| 1 | Framework for evaluating clustering algorithms in duplicate detection | Hassanzadeh, Chiang, Lee, Miller | 2009 | PVLDB 2(1):1282-1293 | http://www.vldb.org/pvldb/vol2/vldb09-1025.pdf ; Crossref 10.14778/1687627.1687771 | Partitioning (transitive closure) is poor even vs equally fast algorithms. MCL among the most accurate and robust. At θ=0.2, F1: Part. 0.177 vs MCL 0.712 | Ban connected components on dense graphs. MCL, center or average-linkage for T-T groups |
| 2 | Comparative evaluation of distributed clustering schemes for multi-source ER | Saeedi, Peukert, Rahm | 2017 | ADBIS, LNCS pp. 278-293 | Crossref 10.1007/978-3-319-66917-5_19 ; https://dbs.uni-leipzig.de/research/projects/famer | 6 Flink clustering schemes. CC and Merge-Center have high recall and poorest precision (as summarised in #3) | Baseline list; confirms CC is precision-poor in multi-source |
| 3 | Using link features for entity clustering in knowledge graphs (CLIP/RLIP) | Saeedi, Peukert, Rahm | 2018 | ESWC, LNCS pp. 576-592 | https://dbs.uni-leipzig.de/files/research/publications/2018-6/pdf/eswc.pdf ; Crossref 10.1007/978-3-319-93417-4_37 | Strong/normal/weak links, link degree, source-consistent clusters. Best P and F on 3 datasets; DS2 F 82% vs 65-75% | Mutual-best (strong) S1-target links as seeds; drop weak links; link degree for hubs |
| 4 | Extended affinity propagation clustering for multi-source ER (MSCD-AP) | Lerm, Saeedi, Rahm | 2021 | BTW 2021, doi:10.18420/btw2021-11 | https://dl.gi.de/server/api/core/bitstreams/d4020b5b-3b7e-4091-ae65-26fb26dce895/content | First mixed clean+dirty scheme; best F on mixed data through excellent precision; Hungarian 1:1 for clean sources | Exactly our S1-clean / S2,S3-dirty setting |
| 5 | Matching entities from multiple sources with hierarchical agglomerative clustering (MSCD-HAC) | Saeedi, David, Rahm | 2021 | KEOD (IC3K) pp. 40-50 | https://dbs.uni-leipzig.de/files/research/publications/2021-10/pdf/MSCD_HAC_KEOD.pdf ; Crossref 10.5220/0010649600003064 | MSCD raises precision "dramatically" at the same recall. S-LINK best F on mixed data; C-LINK best precision; A-LINK better for large clusters | Option C algorithm: HAC with ≤1 S1 per cluster, weak-link removal, A-/C-LINK for F0.5 |
| 6 | Bipartite graph matching algorithms for clean-clean ER: an empirical evaluation | Papadakis, Efthymiou, Thanos, Hassanzadeh | 2022 | EDBT 2022 (arXiv 2112.14030) | https://arxiv.org/abs/2112.14030 (v3 PDF read) | UMC best for balanced collections; EXC best effectiveness-to-speed; KRC top F at higher cost; CNC highest precision, low recall | UMC/EXC logic for assigning copy groups to S1 |
| 7 | An analysis of one-to-one matching algorithms for entity resolution | Papadakis, Efthymiou, Thanos, Hassanzadeh, Christen | 2023 | VLDB J. 32:1369-1400 | https://link.springer.com/content/pdf/10.1007/s00778-023-00791-3.pdf (header/abstract read) | Journal version of #6 (8 algorithms, 10 datasets, >700 graphs) | Same as #6 |
| 8 | Improving entity resolution with global constraints | Gemmell, Rubinstein, Chandra | 2011 | arXiv 1108.6016 (MSR-TR-2011-100) | https://arxiv.org/abs/1108.6016 | Global 1:1 constraints give very strong P/R even with some duplicates; max-weight matching performs poorly in some situations | Prefer greedy or mutual-best over max-weight assignment |
| 9 | Correlation clustering | Bansal, Blum, Chawla | 2004 | Machine Learning 56(1-3):89-113 | https://research.tue.nl/en/publications/correlation-clustering-2/ | Defines CC; constant-factor min-disagree; PTAS max-agree | Principled objective with log-odds weights |
| 10 | Aggregating inconsistent information: ranking and clustering | Ailon, Charikar, Newman | 2008 | JACM 55(5) | Crossref 10.1145/1411509.1411513 (ACM page blocked) | Pivot / KwikCluster, a 3-approximation (as described in #12's abstract) | Pivot = S1 seeds in anchored clustering |
| 11 | Correlation clustering in MapReduce | Chierichetti, Dalvi, Kumar | 2014 | KDD pp. 641-650 | Crossref 10.1145/2623330.2623743 | Parallel pivot CC (CCPivot in FAMER); approximation detail UNVERIFIED | Scalable CC if needed |
| 12 | Parallel correlation clustering on big graphs | Pan, Papailiopoulos, Oymak, Recht, Ramchandran, Jordan | 2015 | NeurIPS 28 | https://proceedings.neurips.cc/paper_files/paper/2015/hash/b53b3a3d6ab90ce0268229151c9bde11-Abstract.html | C4 / ClusterWild!; billion-edge graphs in <5 s on 32 cores; 15x speedup | Proof that global CC at our scale (≈1e8 edges) is cheap |
| 13 | Scalable community detection via parallel correlation clustering | Shi, Dhulipala, Eisenstat, Łącki, Mirrokni | 2021 | PVLDB 14(11):2305-2313 | https://www.vldb.org/pvldb/vol14/p2305-shi.pdf | LambdaCC parallel; billions of edges; up to 28.44x over own sequential | Alternative global solver (reimplement: repo has no license) |
| 14 | Incremental record linkage | Gruenheid, Dong, Srivastava | 2014 | PVLDB 7(9):697-708 | http://www.vldb.org/pvldb/vol7/p697-gruenheid.pdf | CC with cohesion (1 - sim) and correlation (sim) penalties; greedy merge/split/move fixes earlier errors; business-record example | Local-search repair moves for anchored clusters |
| 15 | Graph clustering via a discrete uncoupling process (MCL) | van Dongen | 2008 | SIAM J. Matrix Anal. Appl. 30(1):121-141 | Crossref 10.1137/040608635 ; micans/mcl README | MCL expansion/inflation process | T-T group formation alternative |
| 16 | Transforming pairwise duplicates to entity clusters for high-quality duplicate detection | Draisbach, Christen, Naumann | 2019 | JDIQ 12(1) art. 3 | https://researchportalplus.anu.edu.au/en/publications/transforming-pairwise-duplicates-to-entity-clusters-for-high-qual ; Crossref 10.1145/3352591 | EMCC and MCL best for larger clusters; EMCC best precision, works without weights | Clique-consistency check for copy groups |
| 17 | Collective entity resolution in relational data | Bhattacharya, Getoor | 2007 | TKDD 1(1) art. 5 | https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf ; Crossref 10.1145/1217299.1217304 | BioBase F1: attr 0.568, attr+closure 0.559, collective 0.819; closure harms under ambiguity; bootstrap precision crucial | Collective evidence for same-name S1s; high-precision seeding |
| 18 | Large-scale collective entity matching | Rastogi, Dalvi, Garofalakis | 2011 | PVLDB 4(4):208-218 | https://arxiv.org/abs/1103.2410 | Scale any collective EM via neighbourhoods plus message passing, with guarantees | Country / component partitioning of collective inference |
| 19 | Unsupervised graph-based ER for complex entities (RELATER) | Kirielle, Christen, Ranbaduge | 2023 | TKDD 17(1) | Crossref 10.1145/3533016 (abstract via search only) | Constraint and ambiguity propagation plus cluster refinement | Ambiguity-aware refinement ideas |
| 20 | Collective classification in network data | Sen, Namata, Bilgic, Getoor, Gallagher, Eliassi-Rad | 2008 | AI Magazine 29(3):93 | https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157 | Compares ICA, Gibbs, loopy BP, mean-field | Our stacking = ICA; test a 2nd iteration |
| 21 | Swoosh: a generic approach to entity resolution | Benjelloun, Garcia-Molina, Menestrina, Su, Whang, Widom | 2009 | VLDB J. 18(1):255-276 | https://pure.kaist.ac.kr/en/publications/swoosh-a-generic-approach-to-entity-resolution/ | Match plus merge (record-to-cluster) with efficiency properties | Group-profile matching vs S1 |
| 22 | SiGMa: simple greedy matching for aligning large knowledge bases | Lacoste-Julien, Palla, Davies, Kasneci, Graepel, Ghahramani | 2013 | KDD pp. 572-580 | https://arxiv.org/abs/1207.4525 ; Crossref 10.1145/2487575.2487592 | Greedy 1:1 with score propagation from neighbours; millions of entities at high precision | Greedy anchored assignment with support updates |
| 23 | MinoanER | Efthymiou, Papadakis, Stefanidis, Christophides | 2019 | EDBT 2019 | https://arxiv.org/abs/1905.06170 | Non-iterative parallel rules with neighbour evidence | Cheap rule-based support signals |
| 24 | GraLMatch: matching groups of entities with graphs and language models | De Meer Pardo, Lehmann, Gehrig, Nagy, Nicoli, Hadji Misheva, Braschler, Stockinger | 2025 | EDBT 2025 (doi:10.48786/edbt.2025.01) | https://openproceedings.org/2025/conf/edbt/paper-10.pdf ; https://arxiv.org/html/2406.15015 | Pairwise F1 97.66 becomes ≈0 group-level via transitivity; min-cut / betweenness cleanup gives 60.03; real companies 56.92 to 91.64 | Size-triggered component cleanup; precision is decisive |
| 25 | TransClean | De Meer Pardo, Hadji Misheva, Braschler, Stockinger | 2025 | arXiv 2506.04006 | https://arxiv.org/abs/2506.04006 ; https://arxiv.org/html/2506.04006 | Transitive consistency; +24.42 avg F1; synthetic companies 0.04 to 69.47 | Transitive-consistency features and cut triggers |
| 26 | Graph metrics-driven record cluster repair meets LLM-based active learning (GraphCR) | V. Christen, Obraczka, Hofer, Franke, Rahm | 2025 | JDIQ 17(2) | https://arxiv.org/abs/2401.14992 ; https://arxiv.org/html/2401.14992 ; Crossref 10.1145/3735511 | Edge classifier on graph metrics beats prior repair, including for dirty sources | Graph-metric features for stage 2 |
| 27 | Graph-boosted active learning for multi-source ER (ALMSER) | Primpeli, Bizer | 2021 | ISWC, LNCS 12922 pp. 182-199 | https://www.uni-mannheim.de/en/news/paper-accepted-at-iswc-2021/ ; Crossref 10.1007/978-3-030-88361-4_11 | Correspondence-graph signals improve multi-source AL | Graph-derived labels and features |
| 28 | Leveraging transitive relations for crowdsourced joins | Wang, Li, Kraska, Franklin, Feng | 2013 | SIGMOD pp. 229-240 | https://arxiv.org/abs/1408.6916 ; Crossref 10.1145/2463676.2465280 | Deduce matches and non-matches by transitivity | Consistency rules as features |
| 29 | Multipartite entity resolution: motivating a k-tuple perspective | Aberbach, Kejriwal, Shen | 2024 | AAAI 38(21):23434-23435 (student abstract) | https://ojs.aaai.org/index.php/AAAI/article/view/30417 | k-tuple evaluation differs substantially from aggregated pairwise | Evaluate with the true per-S1 metric only |
| 30 | Holistic entity clustering for linked data | Nentwig, Groß, Rahm | 2016 | ICDMW pp. 194-201 | https://dbs.uni-leipzig.de/research/publications/holistic-entity-clustering-for-linked-data | Holistic multi-source clustering finds wrong and missing links | Background for clean-source clustering |
| 31 | GraphER: token-centric ER with GCNs | Li, Wang, Sun, Zhang, Ali, Wang | 2020 | AAAI 34(05):8172-8179 | https://ojs.aaai.org/index.php/AAAI/article/view/6330 | Token-level ER-GCN for pair matching | Analogy only; not a clustering method |
| 32 | Entity resolution with hierarchical graph attention networks (HierGAT) | Yao, Gu, Cong, Jin, Lv | 2022 | SIGMOD pp. 429-442 | Crossref 10.1145/3514221.3517872 ; GitHub CGCL-codes/HierGAT | Models interdependence of ER decisions on a heterogeneous graph | P3 GNN option |
| 33 | Geospatial entity resolution (Geo-ER) | Balsebre, Yao, Cong, Hai | 2022 | WWW pp. 3061-3070 | https://pasqualeturin.github.io/files/Geospatial%20Entity%20Resolution.pdf ; Crossref 10.1145/3485447.3512026 | Neighbourhood attention handles chains (same name, nearby) and shared buildings | Name-density / chain features |
| 34 | Learning to cluster faces via confidence and connectivity estimation | Yang, Chen, Zhan, Zhao, Loy, Lin | 2020 | CVPR pp. 13366-13375 | https://arxiv.org/abs/2004.00445 ; Crossref 10.1109/cvpr42600.2020.01338 | GCN-V / GCN-E learned clustering on kNN graphs, much faster than earlier supervised methods | Learned T-T grouping at scale |
| 35 | Learning hierarchical GNNs for image clustering (Hi-LANDER) | Xing, He, Xiao, Wang, Xiong, Xia, Wipf, Zhang, Soatto | 2021 | ICCV pp. 3447-3457 | https://arxiv.org/abs/2107.01319 ; Crossref 10.1109/iccv48922.2021.00345 | Hierarchical GNN merges components; +54% F-score vs GNN baselines | P3 learned grouping |
| 36 | In-context clustering-based ER with LLMs (LLM-CER) | Fu, Tang, Khan, Mehrotra, Ke, Gao | 2025 | PACMMOD 3(4) (SIGMOD 2026) | https://arxiv.org/abs/2506.02509 ; Crossref 10.1145/3749170 | Up to 150% higher accuracy, 10% F increase, 5x fewer API calls | Not at our scale; reference only |
| 37 | Adaptive graph refinement and label propagation with LLMs (Alper) | Wang, Yang, Zheng, Ke | 2026 | arXiv 2605.25814 | https://arxiv.org/abs/2605.25814 | Iterative label propagation on an evolving graph beats cascaded pipelines | Idea: iterate blocking and graph refinement |
| 38 | Finding the right (POI) match (Foursquare Kaggle recap) | Foursquare (blog) | 2022 | Company blog, not peer-reviewed | https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ | A top solution's GNN post-processing: 0.907 to 0.946 (per-record IoU metric) | Evidence that graph post-processing pays off in business matching |

Also seen, low priority: Ebeid, Talburt, Siddique, "Graph-based hierarchical record clustering for unsupervised ER" (ITNG 2022, arXiv 2112.06331): transitive closure followed by modularity refinement. Kannangara et al., MERAI (arXiv 2508.03767): enterprise ER; Dedupe failed beyond 2M records.

---

## 4. Repositories and libraries

GitHub API read 2026-09-25 (stars and last push as reported). These are **software library licences, which are separate from model licences.** Copyleft libraries are not a model-rule violation, but prefer permissive ones so the final code can be released cleanly. None of these ship embedded external datasets or gazetteers.

| Repo | Licence | Stars | Last push | Use here | Compliance |
|---|---|---|---|---|---|
| moj-analytical-services/splink | MIT | 2,428 | 2026-09-22 | `cluster_using_single_best_links(duplicate_free_datasets=[s1])`, `compute_graph_metrics` (degree, centralisation, is_bridge); DuckDB/Spark backends | OK |
| AI-team-UoA/pyJedAI | Apache-2.0 | 101 | 2026-03-22 | Reference implementations: UniqueMapping, Exact, BestMatch, KiralyMSM, RowColumn, Center, MergeCenter, Correlation, Cut, Markov, RicochetSR, ConnectedComponents clustering | OK |
| FAMER (git.informatik.uni-leipzig.de/dbs/FAMER) | Apache-2.0 (GitLab API) | 0 | 2023-03-31 | Java/Flink reference for CLIP, RLIP, MSCD-AP, MSCD-HAC | OK (read the logic, reimplement in Python) |
| dedupeio/dedupe | MIT | 4,514 | 2025-07-29 | Hierarchical (centroid) clustering reference | OK |
| zinggAI/zingg | AGPL-3.0 | 1,251 | 2026-09-13 | not recommended | copyleft: avoid |
| scipy/scipy | BSD-3-Clause | 15,038 | 2026-09-24 | `csgraph.connected_components`, `linear_sum_assignment` per component | OK |
| networkx/networkx | BSD-3-Clause (LICENSE.txt) | 17,281 | 2026-09-24 | min-cut, edge betweenness, bridges on small components | OK |
| networkit/networkit | MIT | 878 | 2026-09-24 | Parallel CC and community detection on 1e8-edge graphs | OK |
| rapidsai/cugraph | Apache-2.0 | 2,237 | 2026-09-21 | GPU connected components, Louvain/Leiden if a GPU is on hand | OK |
| pola-rs/polars, duckdb/duckdb | MIT, MIT | 39,855; 41,686 | 2026-09-24 | Group-by support features over 1e8 pairs | OK |
| GuyAllard/markov_clustering | MIT | 175 | 2021-12-15 | MCL in Python (small components) | OK |
| micans/mcl | GPL-3.0 (COPYING) | 116 | 2025-09-19 | reference MCL binary | copyleft: avoid shipping |
| igraph/python-igraph | GPL-2.0 | 1,464 | 2026-05-14 | not needed | copyleft: avoid |
| vtraag/leidenalg | GPL-3.0 | 800 | 2026-09-14 | not needed | copyleft: avoid |
| jeshi96/parallel-correlation-clustering | no licence | 9 | 2022-11-09 | ideas only | cannot reuse code |
| FernandoDeMeer/GraLMatch | no licence | 1 | 2024-06-27 | ideas only | cannot reuse code |
| wbsg-uni-mannheim/ALMSER-GB | no licence | 4 | 2021-10-01 | ideas only | cannot reuse code |
| PasqualeTurin/Geo-ER | no licence | 21 | 2023-04-27 | ideas only | cannot reuse code |
| vefthym/MinoanER | Apache-2.0 | 18 | 2020-11-18 | neighbour-evidence rules | OK |
| CGCL-codes/HierGAT | MIT | 24 | 2023-08-29 | P3 GNN reference | OK |
| yl-1993/learn-to-cluster | MIT | 721 | 2021-12-27 | GCN-V / GCN-E learned clustering reference | OK (train from scratch) |
| dmlc/dgl | Apache-2.0 | 14,285 | 2025-07-31 | Hi-LANDER example; GNN training | OK |
| pyg-team/pytorch_geometric | MIT | 24,097 | 2026-09-01 | GNN edge classification (P3) | OK |

Models: this topic needs **no pretrained model weights**. Any GNN would be trained from scratch on train GT, so there is no model-licence exposure. Benchmark datasets that come with these papers (Leipzig MSC/MSCD, WDC, Foursquare, MusicBrainz) are **external data and must not be used for training or augmentation**.

---

## 5. The design question: options A, B, C for our data

### Option A: independent S1-S2 and S1-S3 scoring

Pipeline: blocking, then a pair model p(q,t), then a per-S1 decision (threshold or expected-F DP), optionally with target exclusivity.

- Precision levers:
  - **Many-to-one exclusivity.** Keep t only for its best S1. This is valid by C1 and helps exactly on same-name S1s (franchises), which are 26-36% of S1 names.
  - Literature support: the bipartite matching studies (#6, #7) show unique-mapping and mutual-best constraints beat unconstrained thresholding in clean-clean settings.
  - Caveat under F0.5: when p(t,q1) ≈ p(t,q2), hard exclusivity keeps one side, and the right move may be to drop both. Prefer a *soft* version: feed the margin to the second-best S1 into stage 2 (already done as `p1_tmargin`) and let the model or the DP decide.
- Blind spots:
  - Each copy is judged alone. A hard copy (cross-script name, empty address, pure alias, domain-as-name) is missed although its siblings are certain.
  - A lookalike distractor copy near a good S1 match is accepted, although it does not agree with the accepted siblings.
- Cost: lowest. This is the floor and the fallback.

### Option B: joint evidence as "support" features (collective, stacked)

Stage 2 re-scores each (q,t) with aggregates over (i) the other candidates of q and (ii) t's own target-side neighbourhood, computed from out-of-fold stage-1 probabilities. It is one iteration of iterative collective classification (Sen et al. 2008). It is the same principle as Bhattacharya-Getoor's relational similarity, GraphCR's graph-metric features, ALMSER's graph signals, and TransClean's transitive consistency, but learned by a GBDT instead of hand rules.

- Recall gain: a weak direct match with strong, cross-source-consistent siblings is lifted. Examples: a Devanagari-name copy whose address equals the accepted S3 copy's address, or a "dba" alias copy.
- Precision gain: an "isolated" candidate is pushed down. That is one that scores well against q, but q has ≥2 confident candidates and t agrees with none of them, or t's own near-duplicates belong to another S1 or to no S1 (a distractor group).
- Why it is safest under F0.5:
  - it never forces transitive merges; it only changes calibrated posteriors, and the expected-F DP then makes the risk-aware call (section 1.3);
  - it is trained on the exact data distribution;
  - it degrades gracefully when support is absent (1-match S1s, 1 copy per source).
- The project already has v1: `stage_collective.py` / `ber.features.collective_batch` provides the anchor similarity, the counts of confident siblings with address or name ≥ 90, same-source flags, `p1_tmargin`, `p1_qrank`, `p1_qgap` and `p1_qsum`. Section 7.1 lists v2 additions.

### Option C: global graph clustering with constraints

Graph = S1 seeds plus targets. Edges are S1-T probabilities plus T-T probabilities. Constraints: at most one S1 per cluster (MSCD clean-source constraint) and each target in exactly one cluster. That gives one S1 per cluster, or none for a distractor group.

- What it adds beyond B:
  - globally consistent assignments (a copy group cannot be split across two S1s, and cannot be assigned to one S1 while its siblings go to another);
  - explicit distractor groups;
  - a natural place for size caps and component cleanup.
- Literature: MSCD-HAC and MSCD-AP show large precision gains from the clean-source constraint, with HAC S-LINK or A-LINK the best-F choices and C-LINK or AP the high-precision choices. CLIP shows that dropping weak links is what keeps precision at low thresholds. Splink's single-best-links implements anchored mutual-best growth with `duplicate_free_datasets`.
- Risks (the reason not to make it the primary decision):
  - Transitive chain FPs (GraLMatch: pairwise 97.66 F1 collapses to ≈0 group precision without cleanup).
  - Early wrong merges are irreversible in greedy agglomeration (Bhattacharya-Getoor).
  - Single-linkage fills clusters to the cap and adds FPs (KEOD 2021, person dataset).
  - T-T edges can be much denser than S1-T edges in same-address or same-name hubs.
- Verdict: use C in an **anchored, constrained, precision-first** form (section 7.3), mainly as repair:
  - veto targets whose group is anchored at a different S1 or at none;
  - add a borderline target only when its group is anchored at q *and* p(q,t) clears a floor;
  - feed cluster-level statistics back as stage-2 or stage-3 features.

### Which options give precision gains under F0.5?

| Mechanism | Precision | Recall | Main risk | Evidence |
|---|---|---|---|---|
| Hard many-to-one exclusivity | up (same-name S1s) | slightly down when the wrong S1 wins | near-ties between franchise branches | #6-#8; GT C1 |
| Soft exclusivity (margin feature + DP) | up | neutral | none major | stage-2 design |
| Mutual-best (strong-link) seeding | up | down if used alone | misses secondary copies | #3, #6 (EXC), Splink |
| Weak-link removal | up | slight down | none major | #3, #5 |
| Sibling / support features (B) | up (isolation penalty) | up (sibling lift) | leakage if not OOF; shared-base correlation | #17, #20, #26, #27 |
| Transitive-consistency feature | up | neutral | compute on large components | #25 |
| Anchored MSCD clustering, A-/C-LINK | up | modest up | chains if single-linkage | #4, #5 |
| Plain connected components | **down, sharply** | up | giant components | #1, #16, #24 |
| Per-source size caps and component cleanup | up | small down | caps may differ on test | #24; GT C3 |
| GNN post-processing | up and up | | training cost, leakage | #34, #35, #38 |

---

## 6. Failure modes and mitigations (specific to this data)

1. **Chains / transitive closure.** A~B (shared address), B~C (shared name), but A≁C. Mitigation:
   - never do CC on low-threshold T-T graphs;
   - use average or complete linkage within groups, or the EMCC clique check;
   - trigger min-cut or edge-betweenness cleanup when a component exceeds the caps (>5 S2, >6 S3, or more than one S1-anchored mutual-best link);
   - keep a floor on the direct S1-T probability for any target added through the graph.
2. **Franchises / same-name S1s.** 26-36% of S1 names are non-unique ("Shree Trading Private Limited", "Primary Care Group"). Name-only T-T edges merge branches. Mitigation:
   - T-T edges must need address agreement (house number, street core) *or* a learned T-T model trained with hard negatives drawn from same-name, different-S1 pairs;
   - add "number of S1s in country sharing this normalised name" and "number of S1 candidates of t with p1 > 0.3" (a Geo-ER-style density signal);
   - resolve by exclusivity on address-driven scores.
3. **Shared addresses.** Malls, office towers, Indian commercial complexes, "Near Fortis Hospital" landmarks, PO boxes. Address-only edges merge distinct businesses. Mitigation:
   - require name agreement (after the cross-script bridge) for T-T edges;
   - address-frequency (IDF of the address core) as a feature;
   - for pure-alias copies matched by address only, accept only with cross-source sibling support.
4. **Hubs.** Short acronyms ("SC", "PC"), generic names, website names, empty addresses. High-degree nodes create bridges. Mitigation:
   - CLIP link-degree priority (prefer low degree);
   - Splink/GraphCR `is_bridge`, node degree and betweenness features;
   - drop or down-weight T-T edges from records whose name and address are both uninformative.
5. **Distractor copy groups.** ≈26% of targets in train have no S1, and test is denser. A distractor group with the same name as some S1 in the same city is the classic FP, and costs a whole singleton (1.0 per S1). Mitigation:
   - group-level decision: attach a group only if the group-level posterior clears the section 1.3 break-even;
   - features comparing the group's consensus address with the S1 address;
   - "does the group have a better anchor elsewhere" (margin);
   - the S1-level match-exists model π (already in `stage_train.py`) should see group features.
6. **Shared corrupted base inside a source.** Two S3 copies agree with each other because they share an error, not because they are right. Mitigation: count cross-source agreement (S2 copy vs S3 copy) separately from same-source agreement, and do not treat same-source siblings as independent votes.
7. **Error propagation.** Greedy agglomeration cannot undo merges. Mitigation: seed only with strong (mutual-best, high-p) links; allow Gruenheid-style move and split local search as a final pass; measure over-merge diagnostics.
8. **Stacking leakage.** Support features computed from in-sample stage-1 probabilities inflate validation. The project already uses cross-fitted OOF p1 (keep it). Any second iteration needs its own OOF.
9. **Distribution shift to test (France, density).** Absolute degree counts shift with density. Prefer relative features (ranks, margins, fractions of neighbours, normalised by candidate count). Validate on the density-matched split. Graph features are language-agnostic, so they should transfer to France better than text features.
10. **Correlated labels in the decision DP.** The independent-Bernoulli expected-F underestimates the variance of accepting a copy group. Mitigation: a group-aware DP (section 7.4).

---

## 7. Concrete algorithms for this challenge

All steps run inside country partitions on a cloud CPU box (Modal, 32-64 cores, 128-256 GB). Nothing runs on the laptop.

### 7.1 Support features v2 (extend `collective_batch`; OOF p1 as input)

For pair (q,t), let C(q) = candidates of q with OOF p1, and let s(t,t') be a T-T similarity (v1: RapidFuzz token-set on normalised name and address; v2: the learned T-T probability from 7.2).

- Soft support: `sup_w = Σ_{t'≠t} p1(q,t') · s(t,t')`, split into same-source and cross-source sums. Also the max over t' of `p1(q,t')·s(t,t')`.
- Cross-source confirmation: max s(t,t') over confident t' (p1 ≥ 0.5) *from the other source*. Same for the same source.
- Isolation: flag = q has ≥2 confident candidates and max_{confident t'} s(t,t') < τ_low.
- Transitive consistency (TransClean-style): mean over confident t' of an indicator [s(t,t') ≥ τ] vs the implied expectation. That is the fraction of implied pairs that "agree".
- Target-side neighbour vote. Let N(t) = t's top-k T-T neighbours from the dedup kNN (not restricted to C(q)). Features:
  - fraction of N(t) whose best-p1 S1 is q;
  - fraction whose best-p1 S1 is some other q';
  - fraction with no S1 candidate above 0.3 (distractor-group signal);
  - max p1(q,t') over t' in N(t).
- Group statistics after grouping C(q) (7.3, local version): group size per source, group cohesion (min, mean pairwise s), group's max and mean p1, whether t is in the top group.
- Graph metrics on the local subgraph {q} ∪ C(q) at a fixed threshold (Splink/GraphCR style): node degree of t, is_bridge for edge (q,t) or for t's strongest T-T edge, component completeness ratio.
- Ambiguity: count of S1s in the country with the same normalised name (and same name+city), and count of S1 candidates of t with p1 ≥ 0.3.

Cost: C(q) is pruned to the top ~20-40 by p1 before pairwise T-T scoring. That is ≈1.7M × (20²/2) ≈ 3.4e8 string comparisons at k=20, which fits in polars plus RapidFuzz on 64 cores in well under an hour. With k=10 it is 4x smaller.

### 7.2 A T-T copy model, trained for free from train GT

- Positives: pairs of targets that share the same S1 in GT (both within-source and cross-source).
- Hard negatives:
  - targets of *different* S1s with the same normalised name (franchise negatives);
  - same address core (shared-address negatives);
  - blocking neighbours that belong to different S1s.
- Distractor targets have no group labels, so leave them out of training, or treat them as negatives only against matched targets of other entities with a clearly different address.
- Features: the same family as the S1-T model (name, address, number, cross-script bridge), plus a same-source flag, because same-source copies share a corrupted base.
- Use: s(t,t') in 7.1, group formation in 7.3, and the neighbour vote.

### 7.3 Anchored, constrained clustering (Option C as repair)

Input: stage-2 posteriors p2(q,t), T-T probabilities r(t,t'), and country partitions.

1. **Seeds.** For each q, the strong links: t with q = argmax_{q'} p2(q',t) *and* t among q's top candidates within its source (CLIP strong link / EXC mutual best), with p2 ≥ τ_seed (high, for example the value giving about 99% precision on validation).
2. **Weak-link removal.** Drop S1-T edges that are the max from neither side (CLIP weak). Drop T-T edges below τ_tt.
3. **Anchored agglomeration** (MSCD-HAC with average or complete linkage):
   - start from seed clusters plus singleton targets;
   - merge the pair of clusters with the highest linkage similarity (combining S1-T and T-T edge probabilities as log-odds);
   - forbid merges that would put two S1s in one cluster or exceed the per-source caps (5 S2 / 6 S3; validate as soft);
   - stop below τ_merge.
   - Clusters without an S1 are distractor groups.
4. **Cleanup.** Any component that violates the caps before clustering, or contains more than one mutual-best S1 link, is cut with networkx min-edge-cut or edge-betweenness (GraLMatch Algorithm 1 logic) before step 3.
5. **Repair moves** (Gruenheid-style local search, one or two passes). Move t to the S1 cluster that maximises the correlation-clustering objective change (sum of log-odds to members), or eject it to "none".
6. **Output use (not a final decision):**
   - (a) veto: drop (q,t) from the final set if t's cluster is anchored at q' ≠ q, or is a distractor group, *and* p2(q,t) is below a high bar;
   - (b) rescue: allow (q,t) into the DP candidate set with an adjusted posterior if t's cluster is anchored at q and p2(q,t) ≥ floor;
   - (c) features: cluster-level statistics as stage-3 inputs.
7. **Baseline to compare:** Splink `cluster_using_single_best_links(duplicate_free_datasets=["s1"])` on the same edges (MIT; DuckDB backend on the CPU box).

### 7.4 Group-aware expected-F decision

The current rule is `decide.expected_f` (independent Bernoulli, Poisson-binomial DP, top-k optimal). Proposed version:
- Within each q, partition candidates into copy groups (7.3 local, or connected components of r ≥ τ_high inside C(q)).
- Model each group G as one Bernoulli with probability π_G (for example the mean or max of p2 in G, recalibrated on validation) and integer weight |G|.
- The DP over groups gives the distribution of weighted TP counts. Choose the subset of groups (sorted by π_G) that maximises E[F0.5], including the empty set.
- Allow within-group trimming: drop members whose p2 is far below the group's.
- Compare on validation against the independent DP. Expect gains on S1s where one group is uncertain (distractor lookalikes) and on singletons.

### 7.5 Validation and ablation protocol (per-S1 macro F0.5 only)

Report on the density-matched validation split, per country and per slice: singleton S1s, same-name S1s, cross-script targets, empty-address targets, n=1 S1s. Include bootstrap SE, as `stage_train` already does.

Ablations, in order:
1. A: threshold / expected-F
2. A + hard exclusivity
3. A + soft exclusivity
4. B v1 (current)
5. B v2 (7.1)
6. B v2 + group-aware DP (7.4)
7. B v2 + anchored C veto/rescue (7.3)
8. Splink single-best-links baseline

Diagnostics:
- over-merge rate: predicted S2 > 5 or S3 > 6 per q; clusters with more than one S1;
- the size distribution of the largest component;
- the singleton false-positive rate;
- the fraction of FPs that are "whole wrong group" vs "stray copy".

---

## 8. What we should do (priorities)

### P0 (cheap, highest expected value; do now)
1. **Keep OOF stacking and extend support features to v2** (7.1):
   - soft weighted support;
   - cross-source vs same-source confirmation;
   - an isolation flag;
   - the target-side neighbour vote;
   - an ambiguity (same-name S1 count) feature.

   Hypothesis: gains concentrate on same-name S1s, distractor lookalikes (precision) and cross-script / empty-address copies (recall).
2. **Soft exclusivity by default; keep hard exclusivity as an ablation.** Hard exclusivity is a valid constraint (C1), but near-ties between franchise branches are better dropped than guessed under F0.5.
3. **Group-aware expected-F DP** (7.4). It corrects the independence assumption for correlated copies and directly targets singleton and wrong-group FPs.
4. **Over-merge and singleton diagnostics** in `stage_analyze.py`: caps, component sizes, singleton-FP rate, and the whole-group vs stray-copy FP split.

### P1 (the main graph win; after P0 is measured)
5. **Train the T-T copy model from GT** (7.2) with franchise and shared-address hard negatives. Use it for support features and grouping.
6. **Anchored MSCD clustering as veto/rescue** (7.3):
   - A-LINK or C-LINK;
   - ≤1 S1 per cluster;
   - weak-link removal;
   - soft per-source caps;
   - a floor on the direct probability.

   Compare against the Splink single-best-links baseline. Adopt only if the per-S1 macro F0.5 gain exceeds the bootstrap SE.
7. **Second collective iteration** (ICA round 2, with its own OOF) only if round 1 shows a clear gain.

### P2 (hardening)
8. GraLMatch-style **component cleanup** (min-cut or edge betweenness) triggered by cap violations. Use the TransClean transitive-consistency score as a feature.
9. **Group-profile (record-to-cluster) features**: S1 vs consensus of the copy group (majority tokens across sources, union of house numbers), Swoosh-style.
10. GraphCR / Splink **graph-metric features** (bridge, betweenness, completeness ratio) on local subgraphs.

### P3 (optional / research)
11. **GNN edge classifier** on the heterogeneous S1-T-T graph (PyG MIT or DGL Apache-2.0, trained from scratch on an A100/L4 via Modal). This is the Foursquare-style post-processing. Only if B and C gains plateau.
12. **MCL or learned clustering** (GCN-E / Hi-LANDER style) for T-T grouping as an alternative to HAC. Use MIT `markov_clustering` or a reimplementation; do not ship GPL `mcl`.
13. **Global parallel correlation clustering** (C4 / ClusterWild! / LambdaCC-style reimplementation) as a research comparison. It is not needed at ≤11-target clusters.

Do not do:
- plain connected components on thresholded graphs;
- CLIP's one-per-source rule on S2 or S3 (they are dirty);
- max-weight global matching as the final assignment;
- LLM cluster-level resolution at 1.7M S1 scale;
- any use of external benchmark datasets for training.

---

## 9. Open items and unverified claims

- Chierichetti et al. KDD 2014: existence, venue and pages confirmed via Crossref. The (3+ε) approximation and pass-count details come from a search snippet only (UNVERIFIED).
- Kirielle et al. TKDD 2023: metadata confirmed via Crossref. The precision/recall gain figures in search snippets were not read on a primary page, so they are not reported.
- HierGAT SIGMOD 2022: metadata confirmed via Crossref and GitHub. Performance numbers seen only in search snippets are not reported.
- GraLMatch group-level numbers (0.00 and 60.03 synthetic; 56.92 and 91.64 real) were read on the arXiv HTML. The EDBT PDF confirms the pairwise 97.66 and 99.73 values. Table layout in the PDF extraction made the group-level columns hard to align, so treat the exact group-level figures as arXiv-version numbers.
- Foursquare 0.907 to 0.946: read on Foursquare's blog, which is not peer-reviewed. The team's final rank is not asserted.
- Per-source caps (≤5 S2, ≤6 S3) come from train GT. Test may differ. Use them as soft caps and diagnostics until they are validated on the density-matched split.
