# F-beta optimisation, calibration, thresholding, singleton / abstention

*Amazon ML Challenge 2026: Business Entity Resolution. Research notes, 2026-09-25.*

**Scope.** How to turn per-candidate match scores into the per-S1 output set that maximises the official
macro F0.5. That covers the decision rule, probability calibration, the singleton ("no match exists")
decision, abstention, and robustness to the test shift (France unseen, denser test).

**Compute note.** Nothing was trained or run on the laptop. The only local computation was a few lines of
exact arithmetic: brute-force enumeration over at most 6 Bernoulli variables, used to check the worked
examples in section 1.8. All pipeline work below is meant for Modal, Kaggle, Colab or Lightning.

**Measured facts used here** (from `DATA_ANALYSIS.md`, train):
- Singletons are 5.58% of S1.
- Matches per S1: 0: 5.58%, 1: 5.40%, 2: 17.00%, 3: 24.05%, 4: 21.94%, 5: 14.59%, 6: 7.47%, 7: 2.90%,
  8: 0.85%, 9+: 0.22%. Mean 3.46.
- Singleton rate and cardinality are the same for the US and India.
- Every matched target is linked to **exactly one** S1, so a many-to-one constraint holds.
- About 26% of S2/S3 records are distractors.
- Test has about 24% more targets per S1 than train.
- France is 15% of test S1 and is unseen in training.
- S1 names are highly non-unique: only 64-74% are distinct.

---

## 0. TL;DR

1. **The metric decomposes per S1.** The Bayes-optimal decision is, for each S1 separately,
   `argmax_S E[F0.5(S, Y) | x]`. This is the "instance-wise F-measure" of Waegeman et al. 2014 (§7) and
   the decision-theoretic approach (DTA) of Ye et al. 2012. No single global pair threshold is optimal:
   the implied threshold adapts to each S1's candidate list and cardinality.
2. **Under conditional independence the optimal non-empty set is the top-k by probability for some k**
   (Lewis 1995; Ye et al. 2012, Thm 9). Exact E[F0.5] for every k costs O(n²) to O(n³) per S1 with a
   Poisson-binomial dynamic program (DP). With n ≤ 50 candidates and 1.73M S1, this is well under a
   minute with numba on a multi-core cloud box.
3. **The empty set is worth exactly P(Y=∅).** Model it with a **separate calibrated S1-level
   "match-exists" model π(x)**. Convert pair probabilities to conditional ones, `q_i = p_i/π` (a hurdle
   model). Predict ∅ iff `π < 1/(1 + max_k E_q[F_k])`. That cut-off is about 0.53 when the best set is
   good (E about 0.9) and about 0.67 when it is weak (E about 0.5). Never use `Π(1-p_i)` as P(∅): our
   labels are strongly dependent.
4. **Dependence is real here.** It comes from multi-copy entities, exactly one entity per S1, and
   many-to-one targets. Waegeman et al. show the F-maximiser depends mostly on the **distribution of
   the number of positives**. The fix is the General F-measure Maximizer (GFM), fed by a
   **cardinality model P(|Y|=s | x)**, which also absorbs blocking misses for free (§1.5).
5. **Rule of thumb for large sets:** include an item iff its calibrated p ≥ **F\*/(1+β²) = 0.8·F\*
   for F0.5**. For F1 the cut-off is F\*/2 (Lipton et al. 2014, Cor. 1; the β form is derived in §1.7).
   With F\* ≈ 0.9 that gives a cut-off of about 0.72, far stricter than F1's 0.45.
6. **Calibration quality sets the ceiling.** Calibrate on out-of-fold predictions (GroupKFold by S1),
   using isotonic regression or a GBDT calibrator with list-wise features. Check reliability per
   country, source, script and rank, and compare the predicted macro-F with the realised one.
7. **Shift.** Test is denser (more distractors) and includes France. Train-calibrated probabilities
   will be over-confident on test, which inflates the output sets, and F0.5 punishes that 4x. Three
   remedies:
   - Simulate the test density in validation.
   - Estimate the test prior per country with Saerens EM (SLD).
   - Run leave-one-country-out (LOCO) checks.
8. **Singleton value is bounded.** Getting every singleton right is worth at most about 0.056 macro-F.
   But each non-singleton wrongly given ∅ costs a full 1.0 for that S1. The π model must therefore be
   well calibrated in the 0.3–0.8 band.

---

## 1. The metric as a decision problem

### 1.1 Definitions and decomposition

For one S1 `q`, let C_q be its candidate list (from blocking), Y ⊆ S2∪S3 the true set and S the
predicted set:

```
F(S,Y) = 1.25·|S∩Y| / (|S| + 0.25·|Y|)      if S ∪ Y ≠ ∅
F(∅,∅) = 1
```

This is the same as `1.25TP/(1.25TP + 0.25FN + FP)` in `src/ber/metric.py`, since `|S| = TP+FP` and
`|Y| = TP+FN`.

The macro score is `(1/N) Σ_q F(S_q, Y_q)`. Expectation is linear, and S_q appears only in its own term.
So **each S1 is optimised independently**: choose S_q = argmax_S E[F(S, Y_q) | x_q].

In the language of Dembczyński et al. (ICML 2017), this is the expected-test-utility setting on tiny
"test sets": one per S1, each with only n ≈ 5–50 candidates. That is exactly the regime where Ye et
al. (2012) found DTA beats empirical utility maximisation (EUM, i.e. picking a threshold on training
data): rare classes on small test sets (their N_ts = 100 experiment), **provided the probability model
is accurate**. Under severe model misspecification they found EUM more robust.

**Why a global pair threshold cannot be optimal** (exact arithmetic):

| true \|Y\| | prediction | F0.5 |
|---|---|---|
| 4 | 3 TP, 0 FP (1 FN) | 0.9375 |
| 4 | 4 TP, 1 FP | 0.833 |
| 1 | 1 TP, 1 FP | 0.556 |
| 6 | 6 TP, 1 FP | 0.882 |

The cost of one extra FP depends on the S1's cardinality and on its other candidates. The implied
threshold therefore has to adapt per S1.

### 1.2 Top-k optimality (the probability ranking principle for F)

**Lewis (1995), as stated by Ye et al. (2012, Thm 9):** if labels are independent given x and
s\* maximises E[F_β], then `min{p_i : s_i=1} ≥ max{p_i : s_i=0}`. The optimum is the top-k for some
k ∈ {0..n}, which cuts the search from 2ⁿ sets to n+1.

**The caveat, from Waegeman et al. (JMLR 2014, Thm 4.5):** without independence, thresholding the
sorted marginals has worst-case regret of at least `max(0, 1/6 − 2/(m+4))`. They also give a 12-label
example where the F-maximiser does not include the label with the highest marginal. PRP is safe only
when (conditional) independence approximately holds; §1.4–1.6 show how to restore it.

### 1.3 Exact expected F0.5 for top-k (independent case)

Sort the candidates so that p_1 ≥ … ≥ p_n. Define:
- A_k = number of positives among the top k (Poisson-binomial over p_1..p_k).
- B_k = number of positives in ranks k+1..n.
- M = number of true matches that blocking missed (an independent count with its own distribution).

Then:

```
E_k = Σ_{a=1..k} Σ_{b≥0} P(A_k=a)·P(B_k+M=b) · 1.25·a / (k + 0.25·(a+b))     (k ≥ 1)
E_0 = P(A_n=0)·P(M=0)          (independent model; §1.4 replaces this with 1-π)
```

**Cost.** The prefix and suffix PMFs are O(n²). The naive double sum over all k is O(n³).
- Ye et al. give an **O(n²)** algorithm when β² is rational. Here β² = 1/4, i.e. q=1, r=4 in their
  Algorithm 1.
- Per Ye et al., Chai (SIGIR 2005) is O(n³) and Jansche (ACL 2007) is O(n⁴).
- At n = 50, O(n³) is about 2·10⁴ flops per S1, or about 4·10¹⁰ for 1.73M S1. That takes seconds to
  a minute with numba on a 32–64-core cloud CPU box.

**Missed-by-blocking mass.** Append "virtual" Bernoulli items after the real candidates. They are
never selectable (k ≤ n_real), and their probabilities come from a blocking-miss model: expected missed
matches as a function of S1 features, fitted on train where the ground truth (GT) is known. This lets
Ye's algorithm run unchanged.

Because FN weight is β² = 0.25, M barely moves the optimal k (example D, §1.8). M matters mainly
through E_0: if a match exists but blocking missed it, predicting ∅ scores 0 anyway.

**Empty-set convention.** Ye et al.'s F_β is 0/0 when S = Y = ∅, so handle k = 0 separately. Waegeman
et al.'s GFM Algorithm 1 does exactly that (`E[F(Y,0)] = P(y=0)`), which matches this challenge's
singleton rule.

Sketch (numba, to run on cloud):

```python
@njit(cache=True)
def best_set_indep(q, m_pmf, pi):
    """q: conditional match probs sorted desc (given 'a match exists');
       m_pmf: PMF of #matches missed by blocking; pi: P(some match exists).
       Returns k* and E[F0.5] for k=0..n (hurdle model, section 1.4)."""
    n, L = q.shape[0], m_pmf.shape[0]
    pre = np.zeros((n + 1, n + 1)); pre[0, 0] = 1.0          # PMF of TP among top-k
    for k in range(1, n + 1):
        for a in range(k + 1):
            pre[k, a] = pre[k-1, a] * (1 - q[k-1]) + (pre[k-1, a-1] * q[k-1] if a > 0 else 0.0)
    suf = np.zeros((n + 1, n + L)); suf[n, :L] = m_pmf       # PMF of positives outside top-k (+M)
    for k in range(n - 1, -1, -1):
        for b in range(n - k + L):
            suf[k, b] = suf[k+1, b] * (1 - q[k]) + (suf[k+1, b-1] * q[k] if b > 0 else 0.0)
    E = np.zeros(n + 1)
    for k in range(1, n + 1):
        s = 0.0
        for a in range(1, k + 1):
            if pre[k, a] < 1e-12: continue
            for b in range(n - k + L):
                s += pre[k, a] * suf[k, b] * 1.25 * a / (k + 0.25 * (a + b))
        E[k] = pi * s                 # non-empty sets score 0 when no match exists
    E[0] = 1.0 - pi                   # empty scores 1 iff Y is empty
    return np.argmax(E), E
```

### 1.4 Singletons as a separate calibrated model (the hurdle model)

Let Z = 1[Y ≠ ∅] and π(x) = P(Z=1 | x_S1, candidate list). Since y_i = 1 implies Z = 1:

```
p_i = P(y_i=1) = π · q_i ,    q_i := P(y_i = 1 | Z=1)
E[F(S)] = π · E_q[F(S)]   for S ≠ ∅     (F = 0 whenever Y = ∅ and S ≠ ∅)
E[F(∅)] = 1 − π
```

- **Non-empty choice.** Maximising over non-empty S is the same as maximising under q. If the q_i are
  conditionally independent, PRP gives top-k by q, equivalently by p.
- **Empty choice.** Predict ∅ iff `1 − π > π·max_k E_q[F_k]`, i.e. **π < 1/(1 + E_q\*)**.
  - E_q\* = 0.9 gives ∅ iff π < 0.526.
  - E_q\* = 0.5 gives ∅ iff π < 0.667.

  The more convincing the best candidate set, the more evidence against a match is needed before
  predicting empty.
- **Coherence.** Use q_i = p_i/π and enforce π ≥ max_i p_i. Training π on S1-level labels (singleton
  or not) gives a model calibrated for the event that actually decides the empty prediction.
  `Π(1−p_i)` is not calibrated for it, because true matches come in correlated clusters.
- **Features for π.** Max, top-2 and top-3 p per source; number of candidates above several levels;
  Σ p_i; margin between top-1 and top-2; S1 name frequency in S1 (non-unique names); country; name
  script; address emptiness; blocking diagnostics (candidate count, best block key).
- **Value bound.** With 5.58% singletons, the empty decision can add at most about 0.056. The
  "never predict ∅" baseline loses exactly the singleton share. Evaluate π with a risk–coverage curve:
  the share of S1 predicted ∅ against the precision of those ∅ predictions.

### 1.5 Dependence and cardinality: GFM (Dembczyński et al. 2011; Waegeman et al. 2014)

**GFM needs no independence assumption.** It needs `P(y=0)` plus the n×n matrix
`p_is = P(y_i=1, s_y=s)`. For F_β (our generalisation of their F1 derivation):

```
E[F_β(h_k)] = Σ_i h_i Δ_ik ,   Δ_ik = Σ_s (1+β²)·p_is / (k + β²·s)
→ for each k take the k items with the largest Δ_ik; compare with E[F(∅)] = P(y=0).
```

Here s counts **all** true matches, including those blocking missed, so the missed mass is handled
automatically.

**Complexity and trade-offs.**
- Given Δ, GFM is O(n²) (Waegeman et al., Thm 5.2). Building Δ = P·W is O(n³) naive.
- The authors observe the maximiser is "more affected by the number of 1s" than by pairwise
  dependence. Marginals that account for the co-occurrence count are enough.
- In their independent-label simulations, GFM matched the independence-based DP. Under strong
  dependence it won clearly.

**Cheap cardinality-aware approximation (our derivation, not from a paper; validate before use).**
Suppose each item's positive probability scales with the cardinality: P(y_i=1 | s) ≈ p_i·s/E[s].
Then `p_is ≈ p_i · w_s` with the size-biased weights `w_s = s·P(s|x)/E[s|x]`. This gives:

```
Δ_ik = p_i · c_k ,   c_k = Σ_{s≥1} 1.25·w_s/(k + 0.25·s)
E_k  = c_k · Σ_{i≤k} p_i ,       E_0 = P(s=0 | x)
```

Top-k by p still holds, and each S1 costs O(n·S_max).

The only extra model is a **per-S1 multiclass cardinality model** P(|Y|=s | x) for s = 0..8+. Train it
with GBDT on S1-level features plus candidate-score summaries per source. It is a natural sibling of π,
because P(s=0) is exactly 1−π.

The train prior (mode at 3, mean 3.46, 93.4% of S1 have 1–7 matches) is informative. For example, if 3
very confident candidates are found, s ≥ 4 is still likely, so more mass sits in the FN term.

**Full plug-in version (Dembczyński et al. ICML 2013).** Fit a per-pair multinomial model over
{y_i=0} ∪ {(y_i=1, s) : s=1..S_max}. That directly estimates p_is. The authors show this plug-in
rule is consistent, whereas structured SVM (SSVM) approaches are not. It costs more to train (about
10 classes over tens of millions of pairs), which is feasible on cloud GBDT. Treat it as P2.

### 1.6 Exploiting entity structure: mutually exclusive copy-clusters

The candidates of one S1 often fall into clusters of near-identical S2/S3 copies. Only one real entity
is linked to each S1, and each target belongs to at most one S1. If the candidates are grouped into
clusters G_1..G_g, the true Y is (approximately) one of {∅, G_1, …, G_g}: a categorical variable, not
independent Bernoullis.

Decision: take the few hypotheses h with probabilities P(h), and enumerate a handful of outputs O:
∅, each G_j, unions of the top-2 clusters, and the top-k by marginal. Pick
`argmax_O Σ_h P(h)·F(O,h)`. This is exact under the model and cheap, since g is small.

Hedging by union can win when two clusters are equally likely (example F in §1.8), but F0.5 makes it
rarely worthwhile. Imperfect clusters can be handled with a Monte Carlo over sampled Y from any joint
model, scoring the same candidate outputs by sample-average F. This P2 idea needs S2/S3 intra-source
similarity (dedup edges), which the blocking stage may already compute.

### 1.7 The threshold result and its F-beta analogue

**Population / large-set view** (Lipton, Elkan & Narayanaswamy 2014, Thm 1 / Cor. 1, for β=1).
Write `F_β = (1+β²)TP / (TP + FP + β²P)`, where P = total positives, a fixed quantity.

Adding one item with calibrated probability p changes:
- the numerator by (1+β²)p;
- the denominator by exactly 1, because TP+FP grows by 1 whatever the label.

So add the item iff `(N + (1+β²)p)/(D+1) > N/D`, i.e. **p > F/(1+β²)**:
- β = 1 (F1): p\* = F\*/2. This is Lipton et al.'s Cor. 1; they note in a footnote that the results
  generalise to F_β. The algebra above is our own.
- β = 0.5 (F0.5): p\* = F\*/1.25 = **0.8·F\***.

Zhao, Edakunni, Pocock & Brown (JMLR 2013) derive optimal thresholds for F-score and BER, and Lipton
et al. cite them for a similar corollary. Koyejo et al. (NeurIPS 2014) and Narasimhan et al.
(NeurIPS 2014) prove that thresholded class-probability ("plug-in") classifiers are optimal and
consistent for this whole family of metrics. Puthiya Parambath et al. (NeurIPS 2014) reduce
F-maximisation to cost-sensitive classification: F is pseudo-linear in the error rates.

**What changes for F0.5 compared with F1.** At F\* ≈ 0.9, the F1 threshold is 0.45 but the F0.5
threshold is 0.72. Two consequences:
- Under F0.5 the second- and third-ranked "maybe" copies are usually excluded unless strongly supported.
- Recall-oriented tricks (adding every plausible copy) hurt.

**Per-S1 caution.** The rule uses the S1's own current F, so each S1 has its own implied threshold:
strong S1s get a high one, weak S1s a lower one (example B). For tiny sets it is only a heuristic; use
the exact DP or GFM from §1.3–1.5.

**Special case.** With exactly one candidate and no missed mass, E_1 = p and E_0 = 1−p. The cut-off is
0.5 whatever β is (example C).

### 1.8 Worked examples (exact enumeration, checked numerically)

| # | setting | E[F0.5] by k | optimum |
|---|---|---|---|
| A | p = [.95,.93,.90,.85,.30,.10], independent | k0 ≈0, 1 .593, 2 .781, 3 .865, **4 .903**, 5 .814, 6 .715 | k=4 (drops p=.30; 0.8·F\* ≈ 0.72) |
| B | p = [.60,.55,.50,.20] | k0 .072, 1 .488, 2 .563, **3 .579**, 4 .506 | k=3 (includes p=.50; 0.8·F\* ≈ 0.46) |
| C | single candidate p | E_0 = 1−p, E_1 = p | include iff p > 0.5 |
| D | A + 20% chance of 1 match missed by blocking | 4: .903 → .895, k\* unchanged | FN is cheap under β²=.25 |
| E | hurdle π=0.7, q = [.9,.85,.8,.2] | k0 .302\*, 1 .465, 2 .565, **3 .600**, 4 .509 | k=3 |
| F | two exclusive clusters of 3 copies | π1=π2=.5: G1 .500, G1∪G2 **.556**; π1=.6/.4: **G1 .600**, union .556 | hedging only when ambiguous |

\*Audit note (2026-09-25): all rows were re-enumerated independently and match to 3 d.p. The one
exception is example E, k=0. With the §1.4 hurdle formula, E[F(∅)] = 1−π = **0.300**. The value .302
comes from treating the q_i as unconditional independent Bernoullis inside the Z=1 branch:
1−π+π·Π(1−q_i) = 0.3017. That is slightly inconsistent with §1.4, where q is conditional on at least
one match. The optimum (k=3) does not change.

### 1.9 Practical precedent: the Kaggle Instacart "None" problem

Kaggle Instacart Market Basket Analysis (2017) scored mean per-order F1, with "None" as a valid
answer. A widely used public kernel by "Faron", "F1-Score Expectation Maximization in O(n²)" (title
confirmed on the Kaggle page https://www.kaggle.com/code/mmueller/f1-score-expectation-maximization-in-o-n;
the author display name and the notebook licence did not render there), implements Ye et al.'s algorithm. It solves
`argmax_(0≤k≤n,[None]) E[F1(P,k,[None])]` and by default sets `pNone = Π(1−p_i)`. The docstring,
the Ye et al. citation, the argmax formula and the `pNone` default were read in a GitHub copy
(`asagar60/.../f1optimization_faron.py`, header `@author: Faron`). The phrase "slightly expanded to
handle the None-singularity" is **not** in that copy; it appeared only in a search-engine snippet of
the Kaggle page (audit, 2026-09-25). The copy's MIT LICENSE was granted by the repo owner (2021), not
by Faron, so its licence status is uncertain (see §6).

Two differences for us:
- In Instacart, None could be added **alongside** items, which is a hedge. Here ∅ is exclusive.
- pNone must come from a dedicated calibrated model (§1.4), not from Π(1−p_i).

---

## 2. Calibration

### 2.1 Calibrator menu

| method | params | when | caveats for us |
|---|---|---|---|
| Platt / sigmoid (Platt 1999/2000; improved solver in Lin, Lin & Weng 2007) | 2 | NN/SVM-like scores; small calibration sets | assumes sigmoid-shaped distortion |
| Temperature scaling (Guo et al. 2017) | 1 | cross-encoder or NN logits; keeps ranking | single global knob; the natural thing to tune under shift |
| Isotonic (Zadrozny & Elkan 2002) | non-parametric | lots of data (we have tens of millions of OOF pairs) | step function creates ties, so break ties with the raw score; prone to overfit when data are scarce (Niculescu-Mizil & Caruana 2005) |
| Beta calibration (Kull et al. 2017) | 3 | skewed score distributions (boosting, NB) | can represent the identity, so it won't damage an already-calibrated model |
| Venn-Abers / IVAP (Vovk & Petej 2014; Vovk et al. 2015) | isotonic-based | calibration validity under exchangeability; returns an interval [p0, p1] | no guarantee for France (not exchangeable). Interval width is a useful uncertainty feature |
| Bayesian binning (BBQ) (Naeini et al. 2015) | binning ensemble | reliability plus the ECE/MCE measures | mostly for measurement |
| Scaling-binning (Kumar et al. 2019) | parametric + bins | calibration error you can actually measure; debiased ECE estimator | use its estimator for reporting |
| GBDT calibrator with list-wise features | many | re-ranks and calibrates at once (features: raw score, rank, gap to top-1, #candidates, per-source counts, country, script) | must be fit OOF; monotone constraint on the raw score |

Niculescu-Mizil & Caruana (ICML 2005) found:
- Max-margin methods (boosted trees and stumps, SVMs) push probabilities away from 0/1, giving a
  sigmoid-shaped distortion.
- Naive Bayes pushes probabilities toward 0/1.
- Platt scaling is best when the distortion is sigmoid-shaped. Isotonic can correct any monotone
  distortion but overfits and does worse than Platt when data are scarce.

**Pitfall: negative subsampling.** If negatives were subsampled at rate r when training the pair
model, first undo it: `p = r·p_s / (r·p_s + 1 − p_s)`. Then calibrate.

**Pitfall: ranking objectives.** LambdaMART/pairwise objectives do not output probabilities. They
**must** be calibrated before the DP.

### 2.2 Calibration protocol (to run on cloud)

1. **Out-of-fold scores** with GroupKFold **by S1 id**, keeping all of an S1's candidates in the same
   fold. Fit calibrators only on OOF scores; in-sample GBDT outputs are over-confident.
2. **Same pipeline as test.** Calibration and validation data must come from the same blocking
   pipeline and, ideally, the same **candidate density** as test (§2.4).
3. **Report**, by the buckets (country, source S2/S3, name script Latin / native-Indic / mixed, rank
   1 / 2–3 / 4+, candidate-count bucket, address-empty):
   - log-loss;
   - reliability diagram;
   - debiased ECE (Kumar et al. 2019).
4. **Metric-level check.** Compare `mean_q max_k E_k` (the model's predicted macro-F0.5) with the
   realised macro-F0.5. A positive gap means over-confidence in the region that matters for decisions.
5. **π model.** Check reliability on the 0.3–0.8 band, and the singleton risk–coverage curve.

### 2.3 Group-wise thresholds and their overfitting risk

With the DTA layer there is no threshold left to tune; group effects enter through the calibrator.
Separate per-group thresholds or offsets add degrees of freedom:
- **Seen, large groups** (US 1.32M S1, India 0.88M S1): statistically harmless. The standard error of
  macro-F0.5 is about σ/√N; if σ ≈ 0.3 (**assumed; measure it**), N = 200k gives ±0.0007.
- **Small cells** (a rare script, tiny candidate lists): N = 5k gives ±0.004. Picking the best of many
  knobs inflates optimism (winner's curse).
- **France has no labels.** Any France-specific knob is a guess. Country must stay an open set.

**Recommendation.** Put country, source and script *into the calibrator as features* (a GBDT
calibrator is a cheap stand-in for multicalibration, Hébert-Johnson et al. 2018: calibrated on every
identifiable subgroup), and verify per-group reliability. Shrink any residual group offset toward the
global value, and accept it only if it helps in leave-one-country-out validation.

### 2.4 Distribution shift: denser test and unseen France

**The problem.** Test has about 24% more targets per S1. If the extra targets are distractors
(inferred in `DATA_ANALYSIS.md`), the positive rate among candidates falls. Train-calibrated p then
over-estimates, sets get too large, and FPs pile up. Under F0.5 this is the costliest direction.

**Remedies.**
- **(a) Simulate the density in validation.** Drop S1 records, or inject extra distractor targets, so
  targets per S1 match test ratios per country. This matches `VALIDATION_STRATEGY` §density.
- **(b) Prior-shift correction on test**, per country, on the binary pair label (or on Z):
  - Saerens-Latinne-Decaestecker EM (Neural Computation 2002).
  - Esuli et al. (TOIS 2020) found SLD helps "only when the number of classes … is very small and the
    classifier is calibrated". Our binary, calibrated case is the favourable regime.
  - Alternative: Black Box Shift Estimation (BBSE; Lipton, Wang & Smola 2018). It needs only an
    invertible confusion matrix.
  - Then rescale the odds: `p' = w·p / (w·p + (1−p))`, with `w = (π_test/π_train)·((1−π_train)/(1−π_test))`.
- **(c) Prefer self-normalising features** (ranks, margins, counts) in the calibrator. They move less
  under density shift than raw scores do.
- **(d) Ensembles.** Ovadia et al. (NeurIPS 2019) report that traditional post-hoc calibration
  "does indeed fall short" under dataset shift, while methods that marginalise over models
  (ensembles) give "surprisingly strong results" (abstract wording). Average several pair models
  (seeds or folds) before calibrating.
- **(e) LOCO.** Train on US and evaluate on India, then the reverse. This measures how calibration and
  the chosen knobs transfer to an unseen country, which is a proxy for France. Pick the knobs that
  maximise the *worst* LOCO country.
- **(f) Unsupervised sanity check on France:** is the predicted singleton rate on French S1 near
  5.6%? Is the mean predicted cardinality near 3.46? The generator probably keeps these constant, as
  it does for the US and India. A large deviation signals miscalibration.
  - ZeroER (Wu et al. SIGMOD 2020) fits a Gaussian-mixture generative model of match / non-match
    similarity vectors without labels. It is a possible way to estimate the French match prior
    independently (P3).

### 2.5 The many-to-one constraint (each target ↔ at most one S1)

This constraint was measured: every matched target has exactly one S1. Pair models scored
independently break `Σ_a P(r↔a) ≤ 1`, especially for the 26–36% of S1 names that are duplicated.

Fixes, cheapest first:
1. **Competition features in the second-stage calibrator:** the rank of S1 `a` among all S1s
   competing for target `r`; `p(r,a) − max_{b≠a} p(r,b)`; the number of competing S1s above 0.5.
2. **Projection:** `p'(r,a) = p(r,a) / max(1, Σ_b p(r,b))`.
3. **Hard one-sided assignment:** give each target to its best S1 only.

Papadakis et al. (VLDB J 2023) benchmark bipartite 1-1 matching algorithms for Clean-Clean ER. Our
setting is one-to-many from S1 and many-to-one from the targets, so only the target-side uniqueness
transfers.

---

## 3. Abstention and selective prediction

- **The only abstention available is ∅**, and it is rewarded only for true singletons. There is no
  "possible link" or clerical-review region:
  - Fellegi & Sunter (1969) define link / possible link / non-link regions by two thresholds on the
    match weight. Their middle region has no counterpart in this metric, so we must commit.
  - Chow's reject rule (1970) becomes, here: predict ∅ iff P(Y=∅) > max_k E_k (§1.4).
- **Loss-derived decisions:** Sadinle (JASA 2017) derives Bayes estimates of bipartite matchings from
  explicit loss functions, including **partial estimates that leave uncertain links unresolved**
  (rejection option). It is a precedent for deriving decisions from the target loss rather than fixed
  thresholds. Its one-to-one assumption does not fit our one-to-many setting.
- **Selective classification** (Geifman & El-Yaniv NeurIPS 2017) and **learning with rejection**
  (Cortes, DeSalvo & Mohri ALT 2016) give risk–coverage tools. Use them to evaluate π and the
  ∅ decision, not to change the objective.
- **Partial abstention** in multi-label classification (Nguyen & Hüllermeier JAIR 2021) gives the
  Bayes-optimal prediction when abstaining on individual labels is allowed under label independence.
  Our metric gives no credit for partial abstention, so this is reference only.
- **Conformal risk control** (Angelopoulos et al. ICLR 2024) chooses a threshold that controls any
  monotone expected loss (e.g. FNR; the paper's examples include token-level F1) with finite-sample
  guarantees under exchangeability. Our goal is to maximise expected F, not to meet a constraint, and
  test is shifted. Use it at most as a guard-rail (P3).

---

## 4. EUM on top of DTA: the few knobs worth tuning

Ye et al. (2012, Thm 8) show that EUM-optimal and DTA-optimal classifiers are asymptotically
equivalent on large i.i.d. test sets, given the true model and provided that instances sharing the
same conditional probability carry negligible mass. Empirically,
EUM was more robust to model misspecification, and DTA was better for rare classes, small data and a
common domain-adaptation setting.

Practical hybrid: use the exact DTA layer plus **at most 3 global scalars**, tuned by direct
macro-F0.5 on validation (with simulated density and LOCO):
1. T_pair: a temperature on the pair logit.
2. A bias or temperature on the π logit.
3. α: the prior-shift odds multiplier for the test density.

Compare variants with a **paired bootstrap over S1** (same S1 resamples for A and B). Treat
differences below about 2 standard errors (about 0.0013 at N = 200k, i.e. 2·0.3/√200000, if σ ≈ 0.3) as
noise. The paired-difference σ is usually smaller than a single system's σ, so measure it rather than
assume it.

---

## 5. Papers (verified)

"Verified via" gives the primary or registry page that was actually fetched. Numbers quoted are only
those read on those pages. Items marked *(content)* are our own reading of the PDF.

| # | paper | venue / year | what it gives us | verified via |
|---|---|---|---|---|
| 1 | D. D. Lewis, *Evaluating and optimizing autonomous text classification systems* | SIGIR 1995, pp. 246–254 | PRP for expected F: optimal set = top-k under independence (as stated in Ye et al. 2012, Thm 9) | Crossref DOI 10.1145/215206.215366; statement read in Ye et al. PDF |
| 2 | M. Jansche, *Maximum expected F-measure training of logistic regression models* | HLT-EMNLP 2005, pp. 692–699 | trains LR by maximising (an approximation of) expected F | https://aclanthology.org/H05-1087/ |
| 3 | K. M. A. Chai, *Expectation of F-measures: tractable exact computation and some empirical observations of its properties* | SIGIR 2005 (poster), pp. 593–594 | exact expected F; O(n³) optimal prediction (per Ye et al.) | Crossref DOI 10.1145/1076034.1076144 |
| 4 | M. Jansche, *A maximum expected utility framework for binary sequence labeling* | ACL 2007, pp. 736–743 | exact expected-F decoding, O(n⁴) (per Ye et al.) | https://aclanthology.org/P07-1093/ |
| 5 | K. Dembczyński, W. Waegeman, W. Cheng, E. Hüllermeier, *An exact algorithm for F-measure maximization* | NeurIPS 2011 | GFM: exact for any distribution with a quadratic number of parameters | https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html |
| 6 | N. Ye, K. M. A. Chai, W. S. Lee, H. L. Chieu, *Optimizing F-measures: A tale of two approaches* | ICML 2012 | EUM vs DTA; PRP (Thm 9); O(n²) algorithm for rational β²; given a good model, DTA better for rare classes on small data and a domain-adaptation scenario (arXiv listing title is singular: "Optimizing F-measure: …") | https://arxiv.org/abs/1206.4625 (PDF read) |
| 7 | M.-J. Zhao, N. Edakunni, A. Pocock, G. Brown, *Beyond Fano's inequality: bounds on the optimal F-score, BER, and cost-sensitive risk and their implications* | JMLR 14 (2013) 1033–1090 | optimal thresholds for F-score and BER | https://www.jmlr.org/papers/v14/zhao13a.html |
| 8 | K. Dembczyński, A. Jachnik, W. Kotłowski, W. Waegeman, E. Hüllermeier, *Optimizing the F-measure in multi-label classification: plug-in rule approach versus structured loss minimization* | ICML 2013, PMLR 28(3):1130–1138 | multinomial-regression plug-in for GFM; consistent, while SSVMs are not | https://proceedings.mlr.press/v28/dembczynski13.html |
| 9 | W. Waegeman, K. Dembczyński, A. Jachnik, W. Cheng, E. Hüllermeier, *On the Bayes-optimality of F-measure maximizers* | JMLR 15 (2014) 3513–3568 (arXiv lists 3333–3388) | regret of thresholding under dependence (Thm 4.5); GFM Alg. 1 with E[F(Y,0)]=P(y=0); instance-wise F; cardinality matters most *(content)* | https://www.jmlr.org/papers/v15/waegeman14a.html + arXiv 1310.4849 PDF |
| 10 | Z. C. Lipton, C. Elkan, B. Naryanaswamy, *Optimal thresholding of classifiers to maximize F1 measure* | ECML PKDD 2014, LNCS pp. 225–239 (arXiv v2 title: *Thresholding classifiers to maximize F1 score*) | Thm 1 / Cor. 1: calibrated optimal threshold = F\*/2; footnote says it generalises to F_β | arXiv 1402.1892 (PDF read) + Crossref DOI 10.1007/978-3-662-44851-9_15 |
| 11 | O. Koyejo, N. Natarajan, P. Ravikumar, I. Dhillon, *Consistent binary classification with generalized performance metrics* | NeurIPS 2014 | optimal classifier = thresholded conditional probability, with a metric-dependent threshold | https://proceedings.neurips.cc/paper_files/paper/2014/hash/98053046e0dce5c7d946c67b96a85e18-Abstract.html |
| 12 | H. Narasimhan, R. Vaish, S. Agarwal, *On the statistical consistency of plug-in classifiers for non-decomposable performance measures* | NeurIPS 2014 | consistency of plug-in (estimate + empirical threshold) for F | https://proceedings.neurips.cc/paper_files/paper/2014/hash/3644e33a5161ec5f3997a6acb98d4447-Abstract.html |
| 13 | S. Puthiya Parambath, N. Usunier, Y. Grandvalet, *Optimizing F-measures by cost-sensitive classification* | NeurIPS 2014 | F is pseudo-linear, so it reduces to cost-sensitive classification | https://proceedings.neurips.cc/paper_files/paper/2014/hash/5c0314ec1b57fcd36bbb013f3f025868-Abstract.html |
| 14 | K. Jasinska, K. Dembczyński, R. Busa-Fekete, K. Pfannschmidt, T. Klerx, E. Hüllermeier, *Extreme F-measure maximization using sparse probability estimates* | ICML 2016, PMLR 48:1435–1444 | thresholding at scale with sparse (top-labels-only) probability estimates, analogous to candidate lists | https://proceedings.mlr.press/v48/jasinska16.html |
| 15 | K. Dembczyński, W. Kotłowski, O. Koyejo, N. Natarajan, *Consistency analysis for binary classification revisited* | ICML 2017, PMLR 70:961–969 | the two settings for non-decomposable metrics (population vs expected test utility) | https://proceedings.mlr.press/v70/dembczynski17a.html |
| 16 | S. Decubber, T. Mortier, K. Dembczyński, W. Waegeman, *Deep F-measure maximization in multi-label classification: a comparative study* | ECML PKDD 2018, Part I (LNCS 11051, 2019), pp. 290–305 | compares thresholding (EUM) against decision-theoretic F_β inference on (C)NNs. DT inference is "worth the investment"; a DT algorithm based on **proportional-odds models** was best. That is directly relevant to an ordinal cardinality model P(\|Y\|=s) | Crossref DOI 10.1007/978-3-030-10925-7_18 + Ghent repository abstract https://biblio.ugent.be/publication/8574368 |
| 17 | J. Platt, *Probabilistic outputs for SVMs and comparisons to regularized likelihood methods* | in A. Smola, P. Bartlett, B. Schölkopf, D. Schuurmans (eds.), *Advances in Large Margin Classifiers*, MIT Press, 2000 (often cited as 1999) | Platt scaling | **Primary page UNVERIFIED** (MIT Press page returned 403; the chapter has no DOI). Existence confirmed by (a) the reference list of Lin, Lin & Weng, *A note on Platt's probabilistic outputs for support vector machines*, Machine Learning 68(3):267–276, 2007, DOI 10.1007/s10994-007-5018-6 (PDF text read in the audit), and (b) OpenAlex record W1618905105. Page numbers (often given as 61–74) are not verified. Note that the DOI above is Lin et al.'s, not Platt's. Title wording varies: Lin et al.'s reference list prints "comparison" (singular) and gives the year 2000; OpenAlex prints "Comparisons" and gives 1999 |
| 18 | B. Zadrozny, C. Elkan, *Transforming classifier scores into accurate multiclass probability estimates* | KDD 2002, pp. 694–699 | isotonic calibration | Crossref DOI 10.1145/775047.775151 |
| 19 | A. Niculescu-Mizil, R. Caruana, *Predicting good probabilities with supervised learning* | ICML 2005, pp. 625–632 | which learners distort how; Platt vs isotonic and data size | Crossref DOI 10.1145/1102351.1102430 + author PDF (abstract read) |
| 20 | M. P. Naeini, G. Cooper, M. Hauskrecht, *Obtaining well calibrated probabilities using Bayesian binning* | AAAI 2015, 29(1) | BBQ; ECE/MCE | Crossref DOI 10.1609/aaai.v29i1.9602 + PMC full text https://pmc.ncbi.nlm.nih.gov/articles/PMC4410090/ (ECE/MCE definitions read) |
| 21 | C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, *On calibration of modern neural networks* | ICML 2017, PMLR 70:1321–1330 | temperature scaling ("a single-parameter variant of Platt Scaling … surprisingly effective", abstract) | https://arxiv.org/abs/1706.04599 + https://proceedings.mlr.press/v70/guo17a.html |
| 22 | M. Kull, T. Silva Filho, P. Flach, *Beta calibration* | AISTATS 2017, PMLR 54:623–631 | 3-parameter calibration for skewed scores | https://proceedings.mlr.press/v54/kull17a.html |
| 23 | V. Vovk, I. Petej, *Venn-Abers predictors* | UAI 2014 | calibration with validity guarantees, isotonic-based | https://arxiv.org/abs/1211.0025 + AUAI proceedings PDF https://www.auai.org/uai2014/proceedings/individuals/166.pdf |
| 24 | V. Vovk, I. Petej, V. Fedorova, *Large-scale probabilistic predictors with and without guarantees of validity* | NeurIPS (NIPS) 2015 | IVAP/CVAP at scale | https://arxiv.org/abs/1511.00213 + https://papers.nips.cc/paper_files/paper/2015/hash/a9a1d5317a33ae8cef33961c34144f84-Abstract.html |
| 25 | A. Kumar, P. Liang, T. Ma, *Verified uncertainty calibration* | NeurIPS 2019 | scaling-binning; debiased ECE estimator | https://proceedings.neurips.cc/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html |
| 26 | U. Hébert-Johnson, M. Kim, O. Reingold, G. Rothblum, *Multicalibration* | ICML 2018, PMLR 80:1939–1948 | calibrated on every identifiable subgroup | https://proceedings.mlr.press/v80/hebert-johnson18a.html |
| 27 | Y. Ovadia et al., *Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift* | NeurIPS 2019 | abstract: traditional post-hoc calibration "does indeed fall short" under shift; methods that marginalise over models (ensembles) give "surprisingly strong results" | https://arxiv.org/abs/1906.02530 + https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html |
| 28 | M. Saerens, P. Latinne, C. Decaestecker, *Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure* | Neural Computation 2002, pp. 21–41 | EM prior-shift estimation (SLD) | Crossref DOI 10.1162/089976602753284446 |
| 29 | A. Esuli, A. Molinari, F. Sebastiani, *A critical reassessment of the SLD algorithm for posterior probability adjustment* | ACM TOIS 39(2), 2021, pp. 1–34 (online Dec 2020) | SLD helps only with few classes and a calibrated classifier | Crossref DOI 10.1145/3433164 + Zenodo abstract https://zenodo.org/records/4468009 |
| 30 | Z. C. Lipton, Y.-X. Wang, A. Smola, *Detecting and correcting for label shift with black box predictors* | ICML 2018, PMLR 80:3122–3130 | BBSE label-shift estimation; works even with uncalibrated predictors if the confusion matrix is invertible (abstract) | https://arxiv.org/abs/1802.03916 + https://proceedings.mlr.press/v80/lipton18a.html |
| 31 | C. K. Chow, *On optimum recognition error and reject tradeoff* | IEEE Trans. IT 1970, pp. 41–46 | the reject rule | Crossref DOI 10.1109/tit.1970.1054406 |
| 32 | Y. Geifman, R. El-Yaniv, *Selective classification for deep neural networks* | NeurIPS 2017 | risk–coverage, selective prediction | https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks (redirects to https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |
| 33 | C. Cortes, G. DeSalvo, M. Mohri, *Learning with rejection* | ALT 2016, LNCS pp. 67–82 | jointly learning a predictor and a rejector | Crossref DOI 10.1007/978-3-319-46379-7_5 |
| 34 | V.-L. Nguyen, E. Hüllermeier, *Multilabel classification with partial abstention: Bayes-optimal prediction under label independence* | JAIR 2021, pp. 613–665 | decision-theoretic partial abstention | Crossref DOI 10.1613/jair.1.12610 (abstract read) |
| 35 | A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, *Conformal risk control* | ICLR 2024 (proceedings pp. 55198–55218) | controls any monotone expected loss | https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html |
| 36 | I. P. Fellegi, A. B. Sunter, *A theory for record linkage* | JASA 64(328), 1969, pp. 1183–1210 | link / possible / non-link decision regions | Crossref DOI 10.1080/01621459.1969.10501049 (metadata only; region rule from standard knowledge) |
| 37 | M. Sadinle, *Bayesian estimation of bipartite matchings for record linkage* | JASA 2017, pp. 600–612 | loss-based Bayes estimates; partial estimates / rejection | https://arxiv.org/abs/1601.06630 + Crossref DOI 10.1080/01621459.2016.1148612 |
| 38 | D. Hand, P. Christen, *A note on using the F-measure for evaluating record linkage algorithms* | Statistics and Computing 28(3), 2018, pp. 539–547 | F's implicit precision/recall weighting depends on the linkage method | Crossref DOI 10.1007/s11222-017-9746-6 (metadata; content from the search abstract) |
| 39 | P. Christen, D. J. Hand, N. Kirielle, *A review of the F-measure: its history, properties, criticism, and alternatives* | ACM Computing Surveys 56(3):1–24 (issue dated 2024; online Oct 2023) | survey | Crossref DOI 10.1145/3606367 (abstract read) |
| 40 | G. Papadakis, V. Efthymiou, E. Thanos, O. Hassanzadeh, P. Christen, *An analysis of one-to-one matching algorithms for entity resolution* | VLDB Journal 2023, pp. 1369–1400 | 8 bipartite matching algorithms over 700+ similarity graphs | Crossref DOI 10.1007/s00778-023-00791-3 (abstract read) |
| 41 | R. Wu, S. Chaba, S. Sawlani, X. Chu, S. Thirumuruganathan, *ZeroER: entity resolution using zero labeled examples* | SIGMOD 2020, pp. 1149–1164 | GMM generative match / non-match model without labels | https://arxiv.org/abs/1908.06049 + Crossref DOI 10.1145/3318464.3389743 |
| 42 | I. Kamsteeg, J. Cardenas-Cartagena, F. van Beers, T. M. Tashu, M. Valdenegro-Toro, *Confidence calibration in large language model-based entity matching* | Proc. 2nd Workshop on Uncertainty-Aware NLP (UncertaiNLP 2025, co-located with EMNLP), pp. 120–137 (workshop) | (modified) RoBERTa entity matching: slight over-confidence, ECE 0.0043–0.0552 across datasets; temperature scaling cut ECE by up to 23.83% | https://aclanthology.org/2025.uncertainlp-main.12/ + https://arxiv.org/abs/2509.19557. Author lists differ: arXiv v2 adds a sixth author (Gineke ten Holt) that the ACL Anthology record omits; the list here follows the Anthology |
| 43 | M. H. Moslemi, M. Milani, *Threshold-independent fair matching through score calibration* | GUIDE-AI '24 workshop (co-located with SIGMOD/PODS 2024), pp. 40–44 (peer-reviewed workshop, not just a preprint) | group-wise score calibration in entity matching (Wasserstein barycentres) | https://arxiv.org/abs/2405.20051 + Crossref DOI 10.1145/3665601.3669845 |

---

## 6. Repos and libraries

Stars and last-push dates were read from the GitHub API on 2026-09-25. These are **software licenses**,
not model licenses, and all of these are libraries, not pretrained models.

| repo | license | stars | last push | use for us |
|---|---|---|---|---|
| scikit-learn/scikit-learn | BSD-3-Clause | ~67.4k | 2026-09-24 | `CalibratedClassifierCV` (sigmoid / isotonic / temperature, ensemble or `cross_val_predict` modes); `TunedThresholdClassifierCV` accepts an `fbeta(β=0.5)` scorer. It tunes a *global* threshold, so it is only a baseline, not the final rule |
| lightgbm-org/LightGBM (formerly microsoft/LightGBM) | MIT | ~18.8k | 2026-09-25 | pair calibrator with list-wise features; π model; cardinality multiclass model |
| dmlc/xgboost | Apache-2.0 | ~28.8k | 2026-09-24 | alternative GBDT |
| numba/numba | BSD-2-Clause | ~11.2k | 2026-09-24 | JIT for the expected-F DP / GFM over 1.73M S1 |
| ip200/venn-abers | MIT | ~207 | 2026-09-02 | IVAP/CVAP calibration; interval width as a feature |
| EFS-OpenSource/calibration-framework (netcal) | Apache-2.0 | ~380 | 2026-04-16 | reliability diagrams, ECE, BBQ, beta / histogram calibration |
| betacal/python | MIT | ~32 | 2024-02-12 | beta calibration |
| gpleiss/temperature_scaling | MIT | ~1.18k | 2025-07-26 | reference temperature scaling (PyTorch). **Caveat:** its README opens with "WARNING: REPO UNMAINTAINED" (written for PyTorch 0.3). Read it as a reference only; for a maintained implementation use scikit-learn's `CalibratedClassifierCV(method='temperature')` (≥ 1.8) |
| p-lambda/verified_calibration | MIT | ~155 | 2022-11-10 | scaling-binning; debiased ECE |
| scikit-learn-contrib/MAPIE | BSD-3-Clause | ~1.59k | 2026-09-08 | conformal / risk control (P3 guard-rail) |
| aangelopoulos/conformal-risk | MIT | ~85 | 2023-01-23 | reference conformal risk control code |
| henrikbostrom/crepes | BSD-3-Clause | ~584 | 2026-07-08 | conformal predictors (optional) |
| chu-data-lab/zeroer | Apache-2.0 | ~34 | 2024-06-29 | unsupervised match-prior estimation idea for France (P3). **Compliance note:** the repo bundles a public benchmark dataset (`datasets/fodors_zagats`, `fodors_zagats_single`) and uses Magellan for blocking. Reuse only the GMM idea or code; never train or augment on the bundled external data |
| moj-analytical-services/splink | MIT | ~2.43k | 2026-09-22 | Fellegi-Sunter match weights → probabilities (reference only) |
| asagar60/Instacart-Market-Basket-Analysis (`f1optimization_faron.py`) | MIT (repo LICENSE © 2021 Arun Sagar). **Caveat:** the file header says `@author: Faron`; it copies a third-party Kaggle kernel, so the repo owner's MIT grant may not cover it | ~8 | 2021-04-05 | copy of the Kaggle O(n²) expected-F1 + None kernel (Ye et al.). **Reimplement** for F0.5 with an exclusive ∅; do not copy |
| sdcubber/f-measure | **no license** (no LICENSE file; README has only a title) | 4 | 2018-07-05 | GFM / Ye et al. / thresholding / CNN code by sdcubber (Stijn Decubber). The repo's GitHub "About" description states it is the "Accompanying code for 'Deep F-measure Maximization in Multi-Label Classification' - ECML 2018 proceedings". **Do not copy** (all rights reserved); reimplement from the paper |
| msadinle/BRL | **GPL-3**, declared in the R package `DESCRIPTION` (`License: GPL-3`). There is no LICENSE file, so GitHub shows none. **Copyleft: never copy into an MIT/Apache deliverable** | 7 | 2020-01-11 | Sadinle's R package *Beta Record Linkage* (implements Sadinle 2017): reference only. Its README also says BRL should not be used with datafiles that contain duplicates, and our S2/S3 hold several copies per entity, so it does not fit our data anyway |

**Models.** None are needed for this topic. The decision layer and calibrators are small models fit
on the provided training data only: no external data, and no pretrained weights involved.

---

## 7. What we should do

### P0 (do first; biggest expected gain per hour)

1. **Build the exact per-S1 expected-F0.5 decision layer** (§1.3 DP, with the hurdle from §1.4) in
   numba. Unit-test it against brute-force enumeration for n ≤ 8, reusing the §1.8 examples as fixtures.
   - Replaces any global pair threshold.
   - Expected impact: typically worth more than a global threshold because it adapts per S1. Measure
     the gain on validation.
2. **OOF calibration of the pair model.**
   - GroupKFold by S1.
   - Undo negative subsampling first.
   - Isotonic, or a monotone GBDT calibrator with list-wise features (rank, gap to top-1, per-source
     counts, candidate count, country, script).
   - Report reliability per bucket, plus predicted vs realised macro-F.
3. **The "match-exists" model π** (S1-level GBDT, calibrated). Set `q_i = p_i/π` with π ≥ max p_i.
   Predict ∅ iff π < 1/(1+E_q\*). Target: the singleton share (5.58%) at high ∅-precision. Never use
   Π(1−p_i).
4. **Density-matched validation plus LOCO (US↔India).** Every threshold or temperature decision is
   made on these folds. This protects the 15% France share and the denser test.

### P1

5. **Cardinality model P(|Y|=s | x)** (multiclass GBDT; train prior mode 3, mean 3.46). Feed it into
   the size-biased GFM approximation (§1.5). Compare with the independent DP on validation. It also
   handles blocking misses implicitly.
6. **Many-to-one competition features and projection** (§2.5), especially for duplicated S1 names
   (26–36% non-unique).
7. **Test prior-shift correction** per country: SLD EM on the calibrated binary pair or Z label, and
   check it with BBSE. Apply it as an odds multiplier. Check that the predicted French singleton rate
   and mean cardinality are plausible (about 5.6% and about 3.5, if the generator is consistent).
8. **At most 3 global knobs** (T_pair, π bias, α), tuned by direct macro-F0.5 on density-matched
   validation. Choose them to maximise the worst LOCO country. Use a paired bootstrap over S1 for
   every comparison.

### P2

9. **Cluster-aware decision** (§1.6): group the candidates into near-duplicate copy clusters, then
   enumerate the hypotheses {∅, G_j} and the outputs {∅, G_j, G_j∪G_l, top-k}. Fall back to Monte
   Carlo over a joint model.
10. **Full GFM plug-in** (Dembczyński et al. 2013): a per-pair multinomial over (y_i, s).
11. **Venn-Abers interval width and ensemble variance as calibrator features.** Ensembles
    (model marginalisation) held up well under shift in Ovadia et al. 2019.

### P3

12. **Conformal risk control guard-rail** for the ∅ rate or precision. Its guarantees do not hold
    under the France shift, so treat it as a diagnostic only.
13. **ZeroER-style unsupervised GMM** for an independent estimate of the French match prior.
14. **Per-group offsets** (country, script), only with shrinkage and LOCO evidence. Never a hard-coded
    French value.

## 8. Open questions (measure on cloud)

- **Per-S1 variability.** What is the standard deviation of per-S1 F0.5 on validation? It sets the
  bootstrap and noise floor (0.3 is assumed above).
- **Candidate depth.** How many candidates does blocking return per S1 (n distribution)? And what is
  the blocking-miss rate by country and script (the missed-mass model)?
- **Independence check.** Is the conditional-independence DP close to GFM on validation? Waegeman et
  al. suggest that cardinality-aware marginals capture most of the gap.
- **Density shift.** How much does calibration drift between train density and simulated test
  density? What does SLD estimate for the test positive rate per country?

---

## Audit log

*Independent citation and license audit, 2026-09-25. Every item was treated as possibly
hallucinated until confirmed on a primary source. No model was run; the only local computation was
metadata parsing, PDF text extraction and a brute-force enumeration of at most 6 Bernoullis to
re-check §1.8.*

### Method

- **Crossref REST API** (`api.crossref.org/works/<doi>`) for all 19 DOIs: title, authors,
  container, volume, issue, pages and date. Abstracts were used where Crossref carries them (Esuli,
  Saerens, Naeini, Nguyen, Christen, Papadakis).
- **arXiv export API** for all 13 arXiv IDs: title, authors, journal-ref, comments, DOI, abstract.
- **Venue pages** via WebFetch:
  - ACL Anthology: H05-1087, P07-1093.
  - NeurIPS proceedings: 2011 GFM; 2014 Koyejo, Narasimhan and Puthiya Parambath; 2019 Kumar;
    2017 Geifman.
  - PMLR: v28 Dembczyński, v48 Jasinska, v54 Kull, v70 Dembczyński, v80 Hébert-Johnson.
  - JMLR: v14 Zhao, v15 Waegeman.
  - ICLR 2024: CRC proceedings page.
  - Other: Zenodo 4468009 (Esuli), Imperial Spiral (Hand & Christen abstract), Ghent biblio
    8574368 (Decubber abstract), AAAI OJS 9602 (Naeini).
- **PDF text extraction** (PyMuPDF) to check quoted theorem-level claims:
  - Ye et al. 2012: Thm 9; O(n³) and O(n²) for rational β² with q/r the reduced fraction; Chai
    O(n³); Jansche O(n⁴); EUM vs DTA conclusions.
  - Lipton et al. 2014: Thm 1; Cor. 1 (s ≥ F/2); footnote on F_β; similar result credited to Zhao
    et al. [20].
  - Waegeman et al. 2014: Thm 4.5 regret ≥ max(0, 1/6 − 2/(m+4)); the m=12 example; Thm 5.2 O(m²);
    Alg. 1 with E[F(Y,0)] = P(y=0); "more affected by the number of 1s"; instance-wise F in §7;
    FM ≈ GFM under independence.
  - Dembczyński et al. 2017: PU vs ETU terminology.
  - Niculescu-Mizil & Caruana 2005: sigmoid distortion; isotonic overfits when data are scarce.
  - Naeini et al. 2015: ECE and MCE defined.
  - Lin, Lin & Weng 2007: Platt reference string.
- **GitHub API** via authenticated `gh api repos/<r>` for SPDX license, stars, `pushed_at` and
  archived flag, for all 17 repos (plus `microsoft/LightGBM`, which redirects to
  `lightgbm-org/LightGBM`). Root file listings and README/DESCRIPTION checks for the 3 repos with no
  detected licence. README feature spot-checks for venn-abers, netcal, verified_calibration, zeroer,
  conformal-risk and betacal.
- **scikit-learn docs:** `CalibratedClassifierCV(method='temperature')` exists (added in 1.8), and
  `ensemble=False` uses `cross_val_predict`. `TunedThresholdClassifierCV` accepts
  `make_scorer(fbeta_score, beta=0.5)` and tunes a single global `best_threshold_`.

### Results

**Papers: 43 checked, all 43 exist with correct core metadata. None was fabricated or deleted.**

- 42 were confirmed on a primary or registry page.
- Platt (row 17) stays **UNVERIFIED on a primary page**. Its existence is confirmed only through
  secondary routes (the Lin et al. reference list and OpenAlex).

All quoted numbers were found in their sources:
- Kamsteeg et al.: ECE 0.0043–0.0552 and "up to 23.83%", both in the arXiv abstract.
- Waegeman bound and complexity (see Method).
- Ye et al. complexities (see Method).
- Lipton F\*/2 (see Method).

**Repos: 17 checked.** Licences, star counts and last-push dates all match the GitHub API on
2026-09-25, with one licence error (BRL) fixed below.

**Models: none listed**, so there was nothing to check. No pretrained weights are proposed for this
topic.

### Changes made

1. **msadinle/BRL licence: "no license" became GPL-3.** The R `DESCRIPTION` file declares
   `License: GPL-3`. There is no LICENSE file, so GitHub's detector reports none. It is flagged as
   copyleft, reference only.
2. **asagar60/... (Faron kernel copy):**
   - Added a provenance caveat. The MIT LICENSE is © 2021 by the repo owner, but the file is
     `@author: Faron` (third-party Kaggle code), so the MIT grant may not cover it.
   - The §1.9 quote "slightly expanded to handle the None-singularity" was said to be verified in
     the GitHub copy, but it is **absent** there. It was re-attributed to a search snippet of the
     Kaggle page; the Kaggle page title itself was confirmed.
3. **sdcubber/f-measure:** confirmed no licence. Noted that attributing it to Decubber et al. is an
   inference from the owner and contents, not stated in the README. *(Superseded in pass 2: the
   repo's GitHub "About" description does state it is the ECML 2018 paper's code.)*
4. **Moslemi & Milani:** "arXiv 2024 (preprint)" became the GUIDE-AI '24 workshop at SIGMOD/PODS
   2024, pp. 40–44, DOI 10.1145/3665601.3669845 (the arXiv record carries this DOI; confirmed via
   Crossref).
5. **Esuli et al.:** "TOIS 2020" became TOIS 39(2), 2021, pp. 1–34 (online Dec 2020).
6. **Christen, Hand & Kirielle:** added CSUR 56(3):1–24, issue dated 2024 (online Oct 2023).
7. **Decubber et al.:**
   - Venue detail made precise (Part I, LNCS 11051).
   - The vague "GFM-style inference on NNs" was replaced with the verified abstract content: DT
     inference beats thresholding with (C)NNs, and a proportional-odds DT algorithm was best.
     That is a useful pointer for an ordinal cardinality model.
8. **Platt:**
   - Venue made precise (eds. Smola, Bartlett, Schölkopf, Schuurmans; MIT Press 2000).
   - Stated explicitly that the cited DOI is Lin et al.'s, not Platt's. The summary JSON's `url`
     field for Platt points to Lin et al.
   - Page numbers marked unverified.
9. **Ovadia et al.:** the paraphrase was tightened to the abstract's actual wording ("does indeed
   fall short"; model-marginalising methods give "surprisingly strong results"). The same change
   was made in §2.4(d) and P2 item 11.
10. **Jansche 2005:** removed the unverified qualifier "(non-convex)".
11. **Ye et al. 2012:** noted that the arXiv listing title is singular ("Optimizing F-measure").
    The ICML paper's running head is "Optimizing F-Measures". The DTA advantage is qualified with
    "given a good model", as in the paper.
12. **Lipton et al. 2014:** noted that the arXiv v2 title ("Thresholding Classifiers to Maximize F1
    Score") differs from the published LNCS title. The co-author surname is printed
    "Naryanaswamy" in the paper and Crossref, but "Narayanaswamy" in arXiv metadata; §1.7 uses the
    latter.
13. **§1.8 example E:** added a footnote. k=0 = .302 is 1−π+π·Π(1−q_i), not the §1.4 formula
    1−π = .300. All other worked-example numbers reproduce exactly (A, B, D, E k≥1, F), as do the
    §1.1 table values (0.9375, 0.833, 0.556, 0.882) and the §1.4 cut-offs (0.526, 0.667).

### Confirmed without change (selected)

- **Waegeman et al.:** JMLR page 15(103):3513–3568; arXiv journal-ref lists 3333–3388. The notes
  already flag this.
- **Chai 2005:** Crossref subtitle confirms the full title.
- **Zhao et al.:** JMLR 14(32):1033–1090; the abstract states "optimal thresholds for F-score and
  BER".
- **Dembczyński 2013:** PMLR 28(3):1130–1138; plug-in consistent, SSVMs not.
- **Koyejo, Narasimhan, Puthiya Parambath:** abstract claims as stated.
- **Jasinska 2016:** PMLR 48:1435–1444.
- **Dembczyński 2017:** PMLR 70:961–969; PU/ETU confirmed in the PDF.
- **Kull:** PMLR 54:623–631; the identity map is representable.
- **Hébert-Johnson:** PMLR 80:1939–1948.
- **Kumar:** scaling-binning and debiased estimator.
- **Guo:** temperature scaling.
- **Vovk & Petej:** UAI 2014.
- **Vovk, Petej & Fedorova:** NIPS 2015.
- **Saerens:** Neural Computation 14(1):21–41.
- **BBSE:** works with uncalibrated predictors if the confusion matrix is invertible.
- **Chow:** IEEE TIT 16(1):41–46.
- **Geifman & El-Yaniv:** NIPS 2017.
- **Cortes et al.:** ALT 2016, pp. 67–82.
- **Nguyen & Hüllermeier:** JAIR 72:613–665.
- **Angelopoulos et al.:** ICLR 2024; token-level F1 example.
- **Fellegi & Sunter:** JASA 64(328):1183–1210.
- **Sadinle:** JASA 112(518):600–612; partial estimates.
- **Hand & Christen:** 28(3):539–547; the abstract confirms the method-dependent weighting of
  precision and recall.
- **Papadakis et al.:** VLDB J 32(6):1369–1400; 8 algorithms, 700+ graphs.
- **ZeroER:** SIGMOD 2020, pp. 1149–1164; GMM.
- **Lewis:** SIGIR '95, pp. 246–254.
- **Zadrozny & Elkan:** KDD 2002, pp. 694–699.
- **Naeini et al.:** AAAI 29(1).

### Could not verify

- Platt's primary publisher page returned 403, and no DOI exists. Retained, marked UNVERIFIED.
- The Kaggle kernel's own licence could not be read because the page did not render. Kaggle
  notebooks are commonly Apache-2.0 by default, but this was **not verified**. Reimplement rather
  than copy.
- DBLP was behind a bot-check and was not used; this was not bypassed.

### Pass 3: independent re-audit (2026-09-25, re-run)

Every item was re-checked from scratch, without relying on the earlier passes. Nothing ran on the
laptop beyond metadata requests, PDF text extraction and a brute-force re-enumeration of §1.8.

**How each item was checked**
- **Crossref API:** all **20** DOIs in the notes. The earlier log said 19 because Moslemi & Milani was
  added later. Title, authors, container, volume, issue, pages and dates all match the §5 table. The
  Esuli et al. abstract has the exact clause quoted in §2.4(b).
- **arXiv export API:** all 13 arXiv IDs (titles, authors, comments, DOIs, abstracts). Checks:
  - Ovadia: both quoted phrases.
  - Kamsteeg: 0.0043–0.0552 and 23.83%.
  - BBSE: "uncalibrated … confusion matrices are invertible".
  - Sadinle: partial Bayes estimates.
  - ZeroER: GMM.
  - Moslemi: Wasserstein barycenters.
- **Venue pages** (HTML `citation_*` metadata plus abstract):
  - NeurIPS: 2011 GFM, 2014 ×3, 2019 Kumar, 2017 Geifman, 2015 Vovk et al., 2019 Ovadia.
  - PMLR: v28, v48, v54, v70 (Dembczyński; Guo), v80 (Hébert-Johnson; Lipton BBSE).
  - JMLR: v14 Zhao ("we derive similar optimal thresholds for F-score and BER"), v15 Waegeman.
  - ICLR 2024 CRC (token-level F1 example).
  - ACL Anthology: H05-1087 (pp. 692–699), P07-1093 (pp. 736–743), 2025.uncertainlp-main.12.
  - AUAI UAI-2014 PDF (Vovk & Petej).
  - Ghent biblio 8574368 (Decubber: "worth the investment", proportional odds).
  - Imperial Spiral (Hand & Christen weighting sentence).
  - PMC4410090 (Naeini ECE/MCE).
  - GUIDE-AI @ SIGMOD 2024 co-location (Technion / SIGMOD workshop pages).
- **PDF text (PyMuPDF):**
  - Ye et al. 2012: Thm 8 (with the negligible-mass condition), Thm 9 (Lewis PRP), O(n³) → O(n²)
    for rational β² with q/r, Chai O(n³), Jansche O(n⁴), the N_ts = 100 rare-class result, the
    domain-adaptation result and EUM robustness.
  - Lipton et al. 2014: Thm 1, Cor. 1 (s ≥ F/2), the F_β footnote, and ref. [20] = Zhao et al.
  - Waegeman et al. 2014:
    - Thm 4.5 bound max(0, 1/6 − 2/(m+4)); the m = 12 example.
    - Thm 5.2 O(m²); E[F(y,0)] = P(y=0).
    - "more affected by the number of 1s"; instance-wise F in §7.
    - FM ≈ GFM under independence (§6.1); GFM "a clear winner" under strong dependence (§6.2).
  - Dembczyński et al. 2017: PU / ETU.
  - Jansche 2005: "(approximately) maximizing the expected F-measure".
  - Jansche 2007: exact and inexact methods.
  - Niculescu-Mizil & Caruana: sigmoid distortion for SVMs and boosted trees/stumps; NB toward 0/1;
    Platt best for sigmoid distortion; isotonic overfits "when data is scarce".
  - Cortes et al. 2016: classifier and rejection function learned simultaneously.
  - Lin, Lin & Weng: Platt reference string.
- **Platt:** OpenAlex W1618905105 re-read (title, 1999, John C. Platt). The MIT Press / direct.mit.edu
  pages returned 403 again, the Microsoft Research page returned 404, and the DBLP API was
  bot-blocked (not bypassed). It stays **UNVERIFIED on a primary page**.
- **GitHub API** (`gh api repos/<r>`) for all 17 repos:
  - SPDX licence, stars, `pushed_at` and archived flag: all match §6.
  - BRL `DESCRIPTION` says `License: GPL-3`, and its README says it is only for duplicate-free files.
  - sdcubber/f-measure: root holds only `README.md` and `src/`, so no licence. Its "About" text names
    the ECML 2018 paper.
  - asagar60: LICENSE is "Copyright (c) 2021 Arun Sagar"; the `f1optimization_faron.py` header reads
    `@author: Faron` and defaults `pNone` to Π(1−p).
  - gpleiss: README opens "WARNING: REPO UNMAINTAINED".
  - zeroer: bundles `datasets/fodors_zagats*`.
  - README feature claims hold for venn-abers (IVAP/CVAP), netcal (BBQ, beta, histogram, ECE,
    reliability), verified_calibration (scaling-binning, debiased), splink (Fellegi-Sunter) and
    MAPIE (risk control).
- **scikit-learn docs (1.9.1):**
  - `CalibratedClassifierCV` has `method='temperature'` ("Changed in version 1.8"), and
    `ensemble=False` uses `cross_val_predict`.
  - `TunedThresholdClassifierCV` (added in 1.5) accepts callable scorers and returns one global
    `best_threshold_`.
- **Internal numbers:**
  - Checked against `DATA_ANALYSIS.md`: 5.58% singletons; the cardinality percentages recomputed from
    the raw counts (N = 2,206,821); 93.4% with 1–7 matches; mean 3.46; France 15.0%; +24% targets
    per S1 (2.82/2.28 and 2.93/2.40); 64–74% distinct S1 names.
  - Also confirmed: `VALIDATION_STRATEGY.md` P-dense, and the F0.5 formula in `src/ber/metric.py`.
- **Arithmetic:**
  - Re-enumerated: §1.1 table, §1.4 cut-offs (0.526 and 0.667), §1.7 thresholds (0.72 and 0.45),
    §1.8 examples A, B, D, E and F. All reproduce to 3 d.p., including the existing E k=0 footnote
    (1−π+π·Π(1−q) = 0.3017 against 1−π = 0.300).
  - §1.3's "about 2·10⁴ per S1 at n = 50" matches the inner-loop count Σ_k k(n−k) ≈ n³/6.

**Results.** Of 43 papers, 42 are confirmed on a primary or registry page and 1 (Platt) is still
UNVERIFIED on a primary page. None is fabricated and none was deleted. All 17 repos are confirmed,
with licence strings exact. There are no models. No licence was wrong in this pass.

**Changes made in pass 3**
1. **§4 noise floor:** "about 0.0015" became **about 0.0013**, since 2·0.3/√200000 = 0.00134. Added a
   note that the paired-difference σ must be measured.
2. **Kamsteeg et al. (row 42):**
   - Added the ACL Anthology primary page and pp. 120–137.
   - Replaced "et al." with the Anthology's five authors.
   - Flagged that arXiv v2 lists a sixth author (Gineke ten Holt).
   - Clarified that the ECE range is for the paper's *modified* RoBERTa model.
3. **Primary proceedings pages added** where only arXiv had been cited:
   - Guo (PMLR 70:1321–1330).
   - Lipton/Wang/Smola BBSE (PMLR 80:3122–3130).
   - Ovadia (NeurIPS 2019).
   - Vovk et al. (NIPS 2015).
   - Vovk & Petej (AUAI UAI-2014 PDF).
4. **Geifman & El-Yaniv:** recorded the canonical URL that the legacy link redirects to.
5. **Naeini et al.:** added the PMC full-text page on which the ECE/MCE definitions were read.
6. **Angelopoulos et al.:** added the initial "N." and the ICLR proceedings pages (55198–55218).

**Open points (unchanged):**
- Platt's primary page is unavailable.
- The Kaggle kernel page renders only its title (no author, no licence), so its licence is still
  unverified. Reimplement from Ye et al. rather than copy.
