*Working note for the ROGII Wellbore Geology Prediction competition — team LeeDongHyuk (odyssey189), public LB 6.986.*

---

## Abstract

This note is organized around a negative result that we believe is the most physically meaningful fact about this competition: after a strong base model has consumed the spatial and log-alignment signal, **roughly 60% of the remaining squared error is a per-well constant offset (a "datum" error) that is fundamentally unobservable from the legal, leak-free data** — not merely hard to estimate, but structurally outside the information available at inference time. We support this with seven independent measurement gates (each attacking the offset from a different information channel, each returning correlation ≈ 0), with oracle experiments that quantify exactly how much score is locked behind the wall (up to −4.58 ft RMSE), and with corroborating theory from inertial navigation and from the geosteering literature, whose state-of-the-art systems output *posteriors over the offset*, never point estimates.

The constructive half of the note describes what *does* help once you accept unobservability: (1) *selective* application of a robust polynomial re-projection, gated by a data-driven overlap test (public −0.045 where the naive version *hurt* by +0.030); (2) a **bimodal-datum midpoint hedge** — detecting when the GR alignment cost has two near-tied minima and outputting the variance-optimal midpoint instead of committing to either (public −0.16, our single largest verified gain); (3) L1-family training objectives and objective diversity, justified by the heavy-tailed residual distribution (OOF −0.156); and (4) forensic overlap analysis establishing that the training set contains zero internal duplicates and that safe re-identification matching has a hard, small ceiling (−0.027). We close with a measured claim: for this base-model family, leak-free post-processing headroom is ≈ 0, and we explain what kind of *information* (not estimator) would be needed to break the ~7 ft floor.

**Map to the evaluation criteria** — (1) *Breadth & depth*: §3 and §4 document 20+ genuinely distinct approaches, each with motivation, validation numbers, and cause of success/failure; negative results are first-class throughout. (2) *Data & well insights*: §1–§2 and §4.1/§4.2 (including how per-well method decisions were gated). (3) *Physical meaningfulness*: §3.2 and §6 — including where we draw the line between physics and metric optimization. (4) *Individual contributions*: the accumulated public-LB ledger opening §4, every delta measured by a dedicated submission. (5) *Uncertainty estimation*: §5.

---

## 1. Problem structure: the task is secretly two tasks

Each well provides a horizontal-well log — measured depth (MD), coordinates (X, Y, Z), gamma ray (GR: natural radioactivity of the rock, a lithology fingerprint), and the target TVT (true vertical thickness position within the stratigraphy) known only on the "heel" prefix (~first quarter of the lateral) — plus a *typewell*: a reference vertical log of GR versus TVT for the area. The task: predict TVT along the remaining ~74% of the lateral ("tail"), scored by RMSE in feet, on ~200 hidden wells (773 wells for training).

The first thing worth measuring is a rigidity check. For every training well:

```
TVT + Z − surf = const_well        (per-well constant)
```

where `surf` is the local formation-surface elevation. Measured within-well standard deviation of this quantity: **0.0065 ft** — six orders of magnitude below the target's dynamic range. The stratigraphy is a rigid layer cake translated vertically per well.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2Fa38cbb6a82da045b142c256aca08ee10%2Flayer-cake-identity.png?generation=1783087021989300&alt=media)
*Figure 1 — The layer-cake identity (geometry schematic; inset real data): across 544 training wells the within-well std of TVT + Z − surf collapses at a median of 0.0065 ft — the stratigraphy is rigid, and only the per-well constant is unknown.*

This identity decomposes the task exactly into:

1. **A per-well absolute datum** (`const_well`): a single scalar per well. Its spread *across* wells is huge — std ≈ 644 ft.
2. **The shared surface shape** along the lateral: smooth structural dip plus high-frequency bedding wiggle, common to nearby wells.

The decomposition matters because these two sub-problems have completely different information structures — the shape is spatially correlated and richly observable; the datum, as we will show, is not observable at all once the obvious sources are drained.

## 2. Base model and error anatomy

Our base is a two-engine pipeline (built on the public kernel lineage for this competition, which we re-trained from scratch on all 773 wells): each engine is a GBM ensemble (LightGBM ×3 + CatBoost ×2, ridge-stacked, grouped 5-fold by well) over ~195 features, blended 0.55/0.45, with a particle filter that tracks the lateral's GR against the typewell. Feature-importance auditing tells a clean story: **spatially interpolated surface features dominate** (the top feature, a dense cross-well surface interpolation, carries ~19% of gain; the dense family together ~38%), the particle-filter alignment deltas come second (~12%), and — strikingly — **the entire raw-GR feature family is dead weight**: 63 raw-GR/texture features can be pruned from both engines for a combined loss of < 0.3% of gain share. The particle filter absorbs everything the raw log has to say; the GBM only wants the filter's *conclusions*, not the log itself.

We then dissected the out-of-fold residual `r = true − blend` per well:

| Residual component | Share of MSE |
|---|---|
| Per-well DC offset (datum error) | **~60%** |
| Linear drift (slope) | ~20% |
| Smooth curvature | ~20% |
| High-frequency wiggle | **~0%** |

The wiggle — the part that looks hardest — is fully captured; the base model tracks bedding oscillations essentially perfectly. What remains is *smooth*: residual lag-1 autocorrelation is 1.000 (99.7% of residual variance survives a smoothing filter), sign consistency along a well is 0.778, and mean |r| grows from 1.9 ft near the heel to 8.6 ft at the toe. The error is also violently heavy-tailed: **the worst 8% of wells contribute 56% of the total squared error**, and per-well RMSE correlates +0.618 with the well's true drift magnitude (versus +0.028 with label jaggedness — this is not label noise; it is drift).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2Ff44c2c28ee89aa097a22c73d6dcdb661%2Fresidual-decomposition.png?generation=1783086814996359&alt=media)
*Figure 2 — Residual anatomy, recomputed from the 550-well OOF blend: ~61% per-well datum, ~20% linear drift, ~19% smooth curvature, 0.8% wiggle+noise (left); the residual traces of three hard wells are smooth per-well drift, not bedding wiggle (right).*



![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2F52859b23295e2cc462af169c84e7a88b%2Ferror-concentration.png?generation=1783087060587775&alt=media)
*Figure 3 — Hard wells are the ones that truly drift (r ≈ +0.6; +0.59 on this blend recomputation, +0.62 on the champion-base measurement), and the error is violently concentrated: the worst 8% of wells carry ~53% of SSE here (56% on the champion base).*

So the entire game, after a competent base, is: *can you learn each well's datum/drift from anything you are allowed to see?*

## 3. The central negative result: the datum is unobservable

We attacked the per-well offset from every information channel available at inference, one dedicated gate per channel, each with the same protocol: leak-free features only (heel truth, tail inputs, typewell, neighbors' *training* data under leave-one-well-out), scored by correlation with the true per-well residual and by deployed RMSE delta. The results:

| # | Information channel | Gate | Result |
|---|---|---|---|
| 1 | The well's own heel residual / GR shape | wiggle-prior | corr ≈ **0** (the tail drift is a fresh random walk; the heel carries no signal about it) |
| 2 | Typewell *absolute* matching (align tail GR to typewell GR on the absolute TVT axis) | typewell-absolute | recovers ~15% at best; the lateral samples only **~21.7 ft** of vertical section, and the GR column is quasi-periodic → the NCC minimum aliases, typically **14 ft** off |
| 3 | Spatial neighbors | neighbor-typewell | after dense cross-well surface interpolation (already in the base), the leftover offset is spatially **white**: LOWO inverse-distance prediction corr = **0.002** |
| 4 | Geology / formation labels | regime & multi-reference gates | explains **0.1%** of residual variance; an *oracle* given perfect formation positions still fails to predict the residual |
| 5 | TVT trajectory motifs across wells | motif gate | **0%** recovery |
| 6 | Conditional GR regime (where the PF "lost lock") | pf-regime | corr = **0.012** |
| 7 | Structural dip direction | dip-deploy | the one non-zero signal: corr = **+0.142** — but deployable monetization = **−0.02 ft**, inside noise |

A direct meta-test confirms the pattern: a GroupKFold LightGBM given *every* leak-free channel at once (distance, fraction along lateral, engine disagreement, GR statistics/gradients, MD, Z) and asked to predict the signed residual achieves corr = **−0.018**; applying its "correction" makes RMSE *worse* by +0.89 ft. The residual's *magnitude* is predictable (Section 5); its *direction* is not — and only direction converts to RMSE.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2F7916521cc38a53a1ced56fb73960d1a5%2Fseven-gates-datum.png?generation=1783087085784366&alt=media)
*Figure 4 — Seven information channels, one wall: every gate's |correlation| with the true per-well residual collapses near zero except structural dip (+0.142), which monetizes to only −0.02 ft; even the all-channel meta-model (gray) deploys at +0.89 ft — worse than doing nothing.*

### 3.1 Oracle ceilings: measuring the height of the wall

To be sure we were fighting for something real, we measured what *perfect* per-well knowledge would buy (these use tail truth — diagnostics only, never deployable):

| Oracle | RMSE gain |
|---|---|
| Perfect per-well datum | **−3.6 ft** (9.71 → 6.09) |
| Perfect per-well engine-blend weight | −1.5 ft |
| Perfect per-well drift-scale multiplier | **−4.58 ft** |


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2F67e1fb8047a3a29232444b8a0516f9aa%2Foracle-ladder.png?generation=1783087113364879&alt=media)
*Figure 5 — Oracle ladder (recomputed on the 550-well OOF blend: 9.71 → 6.09 → 4.29 → 3.40; drift-scale oracle 5.13): up to −6.3 ft of RMSE is locked behind per-well knowledge that the legal, leak-free inputs do not contain.*

The prize is enormous — and the seven gates say the key does not exist in the legal data. A subtle confirmation comes from empirical-Bayes theory: we implemented James–Stein shrinkage of per-well heel-fitted offset/slope/curvature toward the population mean, with the shrinkage factor set by the measured between-well/within-well variance ratio. The data-driven optimum came out at **α\* = 1.0 exactly — shrink fraction 0.000**: the estimator itself reports that the heel-fit carries zero exploitable information about the tail, i.e. the base blend is already risk-optimal along the shrinkage axis. When your James–Stein estimator refuses to shrink, you are not being timid; you are being told the SNR is zero.

### 3.2 Why this is physics, not a modeling failure

Three disjoint literatures predict exactly this wall:

- **Inertial navigation.** A wellbore trajectory integrated from measurements is a dead-reckoning process. Observability analyses of strapdown INS alignment (Silva et al. 2017, *Sensors*) show constant bias states are "individually unobservable" from increments alone — the absolute datum is an external *input*, never recoverable from the path itself.
- **Change-point theory.** A datum jump or slope break inside the unseen 74% is provably undetectable from the seen 26% (no method predicts *future* change-point locations; arXiv:2405.09485).
- **Geosteering state of the art.** The most rigorous industrial formalizations — the MVS stratigraphic-misfit patents (US11946360B2: SVD-likelihood over offsets, *requires external offset-well data*) and the NORCE particle-filter line — output a **posterior distribution over the vertical offset**, never a point. The field's best practice is to *not answer* the question we are being scored on.

Adding a constant to `surf` and subtracting it from `const_well` produces an observationally identical world: this is a structural null space. Every one of the ten additional post-hoc method families we audited in a systematic literature sweep (test-time adaptation, feature-weighted stacking, fractional/long-memory forecasting, particle-Gibbs smoothers, distributional recalibration, …) reduces, on inspection, to needing a leak-free signal correlated with residual *direction* — which we measured at −0.018. The wall is one wall, wearing nineteen costumes.

## 4. What works under uncertainty

Accepting unobservability does not mean doing nothing. It means changing the objective: stop trying to *identify* the offset, and instead **minimize risk under a posterior you cannot collapse**. Everything that worked for us is a variance argument, not a bias argument.

How the pieces accumulated on the public leaderboard (each row = one submitted change; deltas are measured, not estimated):

| Stage | Public LB | Δ |
|---|---|---|
| Public two-engine baseline (fork) | 7.590 | — |
| + deg-4 robust projection applied to the *final* blend (§4.1) | 7.504 | −0.086 |
| + GOLD prefix-backtest layer | 7.247 | −0.257 |
| GOLD *without* our projection (A/B isolation — the two operators conflict) | 7.217 | (+0.030 from stacking) |
| + **selective** IRLS, gated per well (§4.1) | 7.172 | −0.045 |
| From-scratch base retrain + PF seed stabilization (reproducibility, not score) | 7.173 | ≈0 |
| + bimodal-datum midpoint hedge (§4.2) | 7.013 | **−0.160** |
| + safe re-identification copy, strict guard (§4.4) | **6.986** | −0.027 |

### 4.1 Selective IRLS: apply corrections only where the model of the error holds

Our first verified contribution is embarrassingly simple in hindsight. A robust degree-4 re-projection (IRLS: iteratively reweighted least squares — fit the heel-anchored trajectory with a polynomial, down-weighting outliers, then blend the prediction β toward the fit) reliably improved honest out-of-fold wells by −0.134 ft: it regularizes exactly the smooth drift that dominates the residual. Yet stacked on the strongest public post-processing layer it *hurt* the public score by +0.030. The resolution: a subset of public wells are near-duplicates of training wells, where the base is already nearly exact — there, a polynomial "correction" only adds error. The fix:

```
for each test well w:
    if overlap_test(w):        # strict re-ID: prefix RMSE vs any train well < guard
        skip IRLS               # base is already (nearly) truth; don't touch it
    else:
        pred[w] = (1−β)·pred[w] + β·IRLS_deg4_fit(pred[w])   # β = 0.6
```

Public: 7.217 → **7.172 (−0.045)**, while keeping the full −0.134 OOF benefit on non-overlap wells (which is what the private set is made of). Two transferable lessons: **(a)** a post-processor is a *model of the error*, and should be gated by a test of that model's applicability per well; **(b)** independently validated operators do **not** compose — a stronger correction layer that consumes the same drift signal turned our previously positive IRLS (−0.086 on a weaker base) into a negative, so every *combination* must be re-gated.

### 4.2 The bimodal-datum midpoint hedge (our largest genuine gain)

Scanning a constant datum shift Δ ∈ [−40, +40] ft and scoring the misfit `J(Δ) = mean[(GR_tail − GR_typewell(pred + Δ))²]` reveals that on a meaningful minority of hard wells the cost curve has **two near-tied local minima ~14 ft apart** — the GR column's quasi-cyclic bedding (Milankovitch-band cyclicity is the standard geological reading of such regular stacking) makes two vertical positions almost equally consistent with the observed log. The visible prefix genuinely cannot tell them apart.

The naive responses both fail, instructively:

- *Hard-committing* to the better minimum: catastrophic (+5.18 ft on triggered wells) — the decoy wins often enough to blow up the tail;
- *Ignoring* the ambiguity: leaves the base straddling risk on wells where it silently commits to the wrong branch.

The decision-theoretically correct move under a symmetric two-point posterior is the **midpoint** — it is the point prediction that minimizes expected squared error when you honestly cannot pick a side:

```
J = shift_scan(GR_tail, typewell, pred)          # Δ ∈ [−40, 40] ft
Δa, Δb = two_best_separated_local_minima(J)      # 8–20 ft apart
if J(Δb) ≤ 1.15 · J(Δa) and pf_vs_beam_disagreement(w) is high:
    target = pred + (Δa + Δb)/2                  # variance-optimal midpoint
    pred   = (1−α)·pred + α·target               # gentle: α = 0.2, triggered wells only
```

Deployed: public **7.173 → 7.013 (−0.16)** — our single largest verified leak-free gain, and (as far as we can tell from public code) not present in any shared notebook. A later diagnostic gate explained *why* the gentle form is right: the two minima are usually an alias pair *straddling* the truth (the base blend is already within ~5 ft), so the hedge works as a weak variance-reduction pull toward the straddle center — not as a datum *corrector*. Consistently, our attempt to make it "smarter" (tighter trigger 1.15→1.05, stronger α = 0.3) improved OOF by −0.049 and **worsened** the public score by +0.14: the public subset is drift-biased, the loose trigger's broad coverage was exactly where the gain lived, and OOF cannot be trusted to tune a trigger whose benefit is subset-dependent. We report this failed tightening as a first-class result — it is the cleanest demonstration we have of public-subset/validation mismatch, and it echoes the M5 "magic multiplier" post-mortem: an un-anchored level correction tuned to a validation slice is a mirage; *ensembling the correction* (hedging) is the response that survives.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2F63eedb865672099fab8db89c689e90ac%2Fbimodal-cost-curve.png?generation=1783087150582844&alt=media)
*Figure 6 — The GR shift-scan cost J(Δ) on two real training wells: an easy well with one confident minimum (left) vs a hard well with two near-tied minima 14 ft apart, J ratio 1.00 (right) — hard-committing costs +5.18 ft on triggered wells; the gentle α = 0.2 midpoint hedge gained −0.16 public.*

### 4.3 Robust objectives: respect the tail you measured

Section 2's heavy tail (top 8% of wells = 56% of SSE; residual kurtosis far above Gaussian) has a direct training-side consequence: the conditional *median* is a better point target than the conditional mean when the noise is heavy-tailed, even under an RMSE metric, because L2 training lets the un-fixable drift wells bend the whole regression. Measured on identical folds and features:

| Objective | OOF RMSE |
|---|---|
| L2 | 10.564 |
| Huber(δ=40) | 10.527 |
| Quantile(0.5) | 10.545 |
| **L1** | **10.521 (−0.043)** |

Better still, the L1 model is the most *decorrelated* cheap ensemble member we found: residual ρ(L2, L1) = 0.958, and blending the L1 member with the full champion blend gives **−0.156 OOF** (NNLS weight 51% — it earns half the stack), beating both alone. One hyperparameter change buys more diversity than an entire second model family (a histogram-GBM variant we tested at ρ = 0.981 earned a NNLS weight of exactly 0.000). Interestingly, hard-well *re-weighting* — the seemingly obvious response to the tail — was catastrophic (+0.2 to +1.2): you cannot fix unobservable drift by shouting at the loss function; you can only stop it from distorting the observable part. Same wall, seen from the training side.

### 4.4 Overlap forensics: how much of the leaderboard is re-identification?

Public discussion suggested large gains from matching test wells back to training wells. We audited this quantitatively rather than rhetorically, with a self-excluded train-vs-train duplicate detector (each training well's heel is used to search all other wells; since we hold its true tail, we can measure the exact copy-error cost of every guard threshold — a full precision/recall curve for re-identification):

- **The training set contains zero internal duplicates.** Under-thresholds tight enough to be safe, no train well matches another.
- Deployed on test with a strict guard (prefix RMSE < 0.02 ft, GR-consistency check, zero false accepts by construction), full-copying matched wells gained exactly **−0.027** public — a real but *small* number of true re-identifications exist in the public subset.
- Loosening the guard to 1 ft added **nothing** (−0.000): there is no second shell of "near-duplicates" to harvest; the matcher is either exact or wrong.

This bounds the phenomenon: safe re-identification is worth ~0.03 ft on public, and cannot explain multi-foot gaps by itself. It also yielded the gating signal that makes Section 4.1's selective IRLS possible — forensics and modeling paid for each other.

## 5. Uncertainty estimation: knowing *that* you'll fail without knowing *how*

The competition's most interesting epistemic situation is a clean dissociation between *difficulty detection* and *error correction*:

- **Detection works.** The particle filter's seed-to-seed spread (we re-ran the stochastic filter across seeds and scale settings) correlates **+0.33** with per-well RMSE; a classifier on leak-free uncertainty features (PF spread, effective sample size, engine A/B disagreement, spatial isolation) identifies which wells will land in the worst tail at **AUC 0.69**. We can rank wells by expected pain rather well.
- **Actuation fails.** The same feature set predicting the *signed* residual achieves R² = **−0.16** (worse than predicting zero). We know *where* we are wrong, with no idea *which way*.

This asymmetry dictates the entire correct strategy. A direction-blind uncertainty signal cannot fund a correction (every "shrink toward an anchor on uncertain wells" variant we gated made things worse, because the flat anchor is a worse prior than the model's own drift estimate). What it *can* fund:

1. **Hedging** — the bimodal midpoint (§4.2) is precisely an uncertainty-triggered variance reduction: fire only where the posterior is measurably two-horned, and move to its mean.
2. **Down-weighting inside the model** — the GBMs learn on their own to use PF-spread features to discount the filter on hard wells (these features rank in the top gain decile); uncertainty enters as an *input*, not as a post-hoc knob.
3. **Honest evaluation** — we measured the leaderboard's own noise floor by submitting a byte-identical kernel twice: the stochastic PF contributes ±0.03–0.06 ft of pure run-variance. Every "improvement" smaller than that, including several of our own (−0.045 selective-IRLS included), must be argued from OOF logic and mechanism, not from a single public score. Seed-averaging (K=5 deterministic PF) removed this variance entirely and is, we would argue, a competitive-integrity improvement worth more than its 0.00 ft score delta.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F28549212%2F9cac06bf34e0c854cbef928e82bdee5c%2Fdetection-vs-actuation.png?generation=1783087176271743&alt=media)
*Figure 7 — "We can see the fog; we cannot see through it": PF seed-spread ranks per-well failure (left; r = +0.40 on the 100-well re-run subset, +0.33 conservative session estimate; worst-tail classifier AUC 0.69, 5-fold CV) while the same leak-free features cannot predict the signed residual (right; R² < 0 — worse than predicting zero).*

## 6. Honest conclusion

**What we claim.** For the GBM + particle-filter family of solutions — which is, as far as public evidence shows, the dominant paradigm here — the per-well datum term that dominates the remaining error is unobservable from the competition's legal inputs. We showed this seven independent ways empirically, once more via a self-diagnosing shrinkage estimator, and it is the *expected* answer from navigation observability theory and from the geosteering industry's own posterior-only formalizations. Post-hoc processing on such a base has measured headroom ≈ 0: our two systematic literature sweeps (19 method families) found nothing that does not reduce to needing the direction signal we measured at |corr| < 0.02. Consequently our leak-free score sits at what we believe is this paradigm's floor, ~7 ft.

**What we do not claim.** That 7 ft is the floor of the *task*. Top teams report substantially better scores, and public analyses attribute them to genuinely better *shape/dip modeling within fields* — i.e., to a better base, not better post-processing. Our own gates are consistent with this reading in one specific sense: the oracle ladder shows the score is information-limited, so any real breakthrough must inject information we did not extract — plausibly simultaneous multi-well inversion (treating all laterals in a field as one structural model, so each well's datum is constrained by the *others'* full trajectories rather than by its own heel), which is exactly the "external offset-well data" that the MVS patent identifies as the missing observability rank. That is where we would spend the next month: not on a nineteenth estimator, but on a formulation in which the unobservable state becomes shared, and therefore observable.

**The one-sentence summary.** The most valuable thing we learned in this competition is the difference between a hard estimation problem and a missing observation — and that once you can prove which one you have, the winning moves change from *correcting* to *hedging*: skip the wells where your error model doesn't apply, split the difference where the data honestly cannot decide, and let a median-family objective keep the unknowable tail from bending the knowable middle.

---

## Appendix A. Key measurements referenced (single table)

| Quantity | Value |
|---|---|
| Within-well std of (TVT + Z − surf) | 0.0065 ft |
| Cross-well datum std | ~644 ft (base recovers 99.99%; leftover DC std 7.2 ft) |
| Residual decomposition (DC / slope / curvature / wiggle) | 60% / 20% / 20% / ~0% |
| Residual lag-1 autocorrelation / sign consistency | 1.000 / 0.778 |
| Worst 8% wells share of SSE | 56% |
| corr(per-well RMSE, true drift) | +0.618 |
| Seven datum gates (heel / typewell-abs / spatial / geology / motif / regime / dip) | ~0 / alias 14 ft / 0.002 / 0.1% var / 0 / 0.012 / +0.142 (→ −0.02 ft) |
| Lateral's vertical GR window | ~21.7 ft |
| All-channel direction model | corr −0.018; deployed +0.89 ft (worse) |
| Oracle ceilings (datum / blend weight / drift scale) | −3.6 / −1.5 / −4.58 ft |
| James–Stein optimal shrink fraction | 0.000 |
| Selective IRLS | OOF −0.134 (non-overlap); public −0.045; naive stacking +0.030 |
| Bimodal hedge (α=0.2, loose trigger) | public −0.16; hard-commit +5.18; tightened variant OOF −0.049 → public +0.14 |
| L1 objective / champ+L1 blend | −0.043 OOF / −0.156 OOF (ρ = 0.958 vs L2) |
| Train internal duplicates / strict test re-ID / loose re-ID | 0 / −0.027 / −0.000 |
| PF spread vs well RMSE / failure AUC / direction R² | +0.33 / 0.69 / −0.16 |
| Public-LB stochastic run-variance (identical resubmission) | ±0.03–0.06 ft |

## Appendix B. Attribution

Our base pipeline builds on the public two-engine kernel lineage for this competition and on the public prefix-backtest calibration layer; both are gratefully acknowledged. The contributions specific to this note are: the layer-cake decomposition and residual anatomy; the seven-gate unobservability proof and oracle ladder; selective (overlap-gated) IRLS; the bimodal-datum midpoint hedge and its failure-mode analysis; the L1-diversity result; the duplicate forensics; and the uncertainty detection-vs-actuation dissociation. All OOF numbers use grouped 5-fold cross-validation by well on from-scratch retrained models; all gates are leak-free by construction (heel truth and tail inputs only) and were audited for oracle contamination.
