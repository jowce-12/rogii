# The Wiggle Is Free, the Trend Is the Wall

### A Rigorous Negative-Result Study of Wellbore Stratigraphic-Position (TVT) Prediction

**ROGII Wellbore Geology Prediction — Working Note**
**Team:** Malyshev Danil

---

> **One-paragraph summary.** We predict the true vertical thickness (TVT, stratigraphic position) of the
> unlabeled toe of a horizontal well. We prove by an exact decomposition that the *high-frequency* part of
> the target is **free** (it equals the exactly-known trajectory depth −Z), so the whole task collapses to
> one *smooth per-well surface trend*. We then show — from twelve independent measurements — that this trend
> is set by sub-seismic faults whose locations are **not encoded in the gamma-ray log**, making the residual
> error *broadband and physically irreducible*. This single fact explains why the trivial "flat" baseline
> is so strong (15.9 ft), why a weak-GR/strong-continuity prior beats explicit log-matching, and why **60+
> experiments and 22 neural architectures** — MDN, transformers, 2D misfit-SDF, synthetic pretraining —
> all cap at the isolated particle-filter level (~11 ft), while the tuned classical stack reaches ~7 ft.
> Our first transferable competition gain came not from new modeling but from **variance reduction**: a
> decorrelated particle-filter ensemble (7.230 → 7.096) and a more-seeds robustness variant (7.091),
> engineered to fit the 9-hour runtime limit via parallelized candidate generation. A second, larger gain
> then came from **stacking a decorrelated neural corrector with a robust physics post-process** on top of
> that pipeline, then further refined by transferring a cross-validated hyperparameter back onto the real
> submission: real-LB-confirmed **7.080 → 6.836 → 6.794** (−4.0% relative), our new best submission. The same
> real-leaderboard test also produced a controlled negative result of independent interest: a third,
> individually cross-fit-validated correction (a second particle filter, `pf_z`) **breaks this combination
> regardless of where it sits in the pipeline** (7.446–7.515), and a simple shrink-toward-last-known-TVT
> hedge fails in both placements we tested (7.252 / 7.304) — direct real-LB evidence that corrections do
> not compose indefinitely, even when each one is honestly validated on its own.

---

## Contributions at a glance

- **A trustworthy validation protocol** for a task whose public leaderboard is dominated by seed noise: a
  whole-well holdout whose flat baseline (15.1) matches the LB flat baseline (15.9), so model rankings
  transfer. We demonstrate the seed-noise problem directly (byte-identical pipeline → 7.096 / 7.135 / 7.091
  on separate runs).
- **A real, transferable gain by variance reduction**, not new modeling: a decorrelated particle-filter
  ensemble (7.230 → **7.096**, below the pipeline's own seed-noise floor of 7.168) and a more-seeds
  robustness variant (**7.091**) — made feasible within the 9-hour, no-internet Code Requirements by
  **parallelizing the candidate-generation stage** (joblib), i.e. buying back the compute the ensemble and
  extra seeds consume.
- **A second, larger real-LB gain from stacking a decorrelated neural corrector with a physics
  post-process**: blending the WARP neural net into the pipeline together with a robust surface
  post-process pushes the real leaderboard from **7.080 → 6.836**, then a cross-validated WARP-weight
  correction pushes it further to **6.794** (−4.0% relative overall), our new best submission — confirmed
  by direct resubmission, not proxy validation.
- **A controlled real-LB negative result on over-stacking corrections.** Adding a third, independently
  cross-fit-validated lever (`pf_z`, a second particle filter) on top of the winning two-stage combination
  *breaks* it — **6.836 → 7.446–7.515**, regardless of where in the pipeline it is inserted. A simple
  shrink-toward-last-known-TVT hedge fails the same way in both placements (7.252 / 7.304). Both facts were
  established with a controlled, placement-crossed real-LB experiment (§7.2) — a rare case where "does an
  offline-validated idea transfer" was answered with real leaderboard evidence in both directions, not
  assumed.
- **The new best survives harsher, spatially-honest validation.** A Leave-Spatial-blocks-Out (LSO) check —
  holding out whole geographic regions by K-means on well coordinates, not just random well groups — gives
  the same gain (+2.244 vs +2.235, positive in 6/6 blocks), and isolated wells (weakest baseline) benefit
  *more* from the new combo, not less (§7.3): the gain is not a neighbour-leakage artifact. A separate
  cross-fit test explains *why* the hedge/pf_z stacking failures above happen: hedge-style corrections are
  base-dependent, and their cross-fit-optimal weight collapses to exactly zero once WARP+physics-pp have
  already spent that variance-reduction budget (§7.2).
- **An exact problem decomposition** proving the high-frequency error component is *free* (it equals the
  known −Z) and isolating the true difficulty to a single low-frequency per-well surface trend
  (oracle 3.9 ft).
- **A quantified impossibility argument:** the gamma-ray log's localization error is *broadband and
  irreducible* (measured from twelve independent angles, including a signal-processing measurement of the
  acquisition instrument itself), which explains why a weak-GR/strong-continuity prior
  beats explicit alignment and why every neural architecture caps at the isolated-PF level.
- **A full experiment catalogue** — 60+ rigorous experiments across baselines, GR-matching, particle-filter
  internals, post-processing, GBM blending, denoising, human-markup reverse-engineering, and a
  **22-architecture neural-network study** — each with its failure mechanism, a map of what does not work
  and precisely why.

---

## Abstract

We present a systematic study of the ROGII wellbore-geology task: predicting TVT (stratigraphic position)
of the unlabeled eval-zone tail of a horizontal well from its gamma-ray (GR) log, trajectory, and a
reference typewell. Our contribution is fourfold. **(1) Methodology:** a whole-well holdout whose trivial
baseline matches the public leaderboard (15.1 vs 15.9), a trustworthy stand-in for the hidden test — unlike
the public LB, which for the dominant particle-filter (PF) solution family is dominated by *seed variance*.
**(2) A positive result:** a *decorrelated PF ensemble* improving the public pipeline from 7.230 to
**7.096**, below the pipeline's own seed-noise floor (7.168) — a real gain, plus a more-seeds variant at
**7.091**. **(3) A second, larger positive result:** stacking a decorrelated neural corrector (WARP) with a
robust physics post-process on top of that pipeline, real-LB-confirmed **7.080 → 6.836**, then a further
real gain from transferring a cross-validated hyperparameter back onto the submission, **6.836 → 6.794**
(our new best) — paired with a controlled real-LB negative result showing this stack does *not* extend to a
third, individually-validated correction (§7.2). **(4) A rigorous negative-result analysis** — the bulk of
this note — mapping *why* the residual error is hard: an exact decomposition shows the exploitable error is
a low-frequency per-well *surface trend*; GR localization error is **broadband and irreducible** due to log
self-similarity; and a weak-GR / strong-continuity prior provably beats explicit GR matching. **(5) A
complete experiment catalogue** of 60+ attempts and 22 neural architectures, each with its failure mode. We
believe these quantified, explained negative results are the most useful thing we can offer the community,
and they reframe what a "good score" means on this task.

---

## Contents

1. Task, Data, and Metric
2. Related Work
3. Validation Methodology — our most reusable contribution
4. The Exact Decomposition: the wiggle is free
5. The Wall: the surface trend is not recoverable
6. Why the GR Log Cannot Localize
7. A Positive Result: the Decorrelated PF Ensemble (7.1 Three offline stacking positives · 7.2 Real-LB
   confirmation: a new best, and a stacking limit · 7.3 Spatial-block holdout and isolation check)
8. Full Experiment Catalogue (60+ experiments)
9. The Neural-Network Study: 22 Architectures (9.1 Synthetic-transfer gate · 9.2 WARP-blend follow-up)
10. Leaderboard Context
11. Uncertainty Estimation
12. Physical Meaningfulness · 13. Lessons · 14. Reproducibility · 15. Limitations & Future Work · 16. Conclusion

---

## 1. Task, Data, and Metric

**The task.** Each example is one horizontal well. We are given its gamma-ray (GR) log along measured depth
(MD), its trajectory (`X, Y, Z, MD`), a set of known-zone stratigraphic labels `TVT_input` near the heel,
and a nearby vertical **typewell** (a reference `TVT` vs `GR` vs `Geology` profile). The eval zone is the
toe-side tail of the well, where `TVT_input` is withheld. We must predict `TVT` there — the well's vertical
position within the stratigraphic column, in feet.

**The data.** The training set has **773 wells** (we use all of them in a 613/160 whole-well split). A
typical horizontal well has ~5,000 MD samples, of which ~3,800 fall in the eval zone; the typewell has
~1,300 samples. There are only ~69 geologically-distinct typewells shared across the field, which matters
for the synthetic-data discussion in §15.

**The metric: pooled RMSE (ft).** The competition concatenates *every* eval point of *every* test well into
one vector and takes a single RMSE. Two consequences shaped every decision we made:

- **Pool, don't average per well.** Averaging per-well RMSEs gives an optimistic ~10; pooling gives ~15.9
  for the trivial baseline. A handful of long, badly-faulted wells dominate the pooled score (Fig. 12), so
  the objective rewards *not blowing up on the hard wells* far more than shaving the easy ones. All numbers
  in this note are pooled RMSE.
- **The trivial "flat" baseline is strong.** Predicting `TVT = last known TVT` (hold the last heel-side
  label flat through the toe) scores **15.883** on the LB and **15.1** on our holdout. As the catalogue in
  §8 shows, *almost every "smarter" idea we tried did worse than flat.* Understanding why is the whole
  study.

## 2. Related Work

Automated geosteering / log-to-typewell correlation has a substantial literature. Sequential Bayesian
methods — particle filters, HMM/Viterbi over stratigraphic position — are the classical backbone and
underpin the public pipeline we build on. Recent deep-learning work frames the problem as image-like
matching or multi-modal inversion: convolutional "geosteering-image" networks that consume a 2D
misfit/heatmap between the horizontal and vertical logs; Temporal Convolutional Networks (TCN) with
cross-attention to the typewell; and Mixture-Density Networks (MDN) for multi-modal depth inversion of
geophysical logs (e.g. Alyaev et al., *Direct Multi-Modal Inversion of Geophysical Logs Using Deep
Learning*), which model the ±cycle bedding ambiguity as a Gaussian mixture rather than collapsing to a
mean. We implemented representatives of each family (§9). Our finding is not that these methods are wrong —
leading teams reach 5.2–5.4 with them — but that, at this dataset's scale (773 wells) and signal
(self-similar GR), they cap at the isolated-PF level *unless* paired with realistic synthetic pretraining,
which is the decisive and hard-to-reproduce ingredient.

## 3. Validation Methodology (our most reusable contribution)

The public LB is deceptive: the PF-based cluster (the ~7.2 fork family shared by all three public notebooks,
which are byte-identical descendants of one "LB7295 Public Rebuild") is dominated by **seed variance** —
byte-identical reruns scatter between ~7.10 and ~7.5. Tuning on the public LB is tuning on noise. We
verified this directly by resubmitting an unchanged pipeline three times: **7.096, 7.135, 7.091**.

We therefore built a **whole-well holdout**: 160 unseen wells (grouped split, spatially shuffled via
StratifiedGroupKFold so validation wells are not neighbours of training wells), pooled RMSE on their eval
zones only. Its flat baseline = **15.1**, versus the LB's 15.883 — the scale matches, so *relative* model
rankings transfer. On top of this we cached, for all 773 wells, the PF outputs at several process-noise
scales, the seed spread, and the beam/flat/linear baselines, so most experiments run in seconds rather than
hours.

**Rule adopted:** submit to Kaggle only a model confidently better than the incumbent on this holdout *and*
stable across seeds. This rule saved us from a "4-way ensemble" that looked best on a multi-slice holdout
(9.49) but was worst on the real LB (7.752), because all slices shared one isolated-component blind spot
(§7, §8.3).

## 4. The Exact Decomposition: the wiggle is free, the trend is the wall

The single most important fact about this task is a geometric identity. TVT and the trajectory depth Z are
related through the *stratigraphic surface* the well is drilling within:

$$\text{TVT}(MD) = \text{surface}(MD) - Z(MD)$$

where `surface = TVT + Z`. **Z is measured and therefore known exactly in the eval zone.** So all we ever
need to predict is `surface`, and — this is the crux — the surface is almost perfectly smooth.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F134b067a625caf81dbcd984ea295365d%2Ffig01_decomposition.png?generation=1783380193420644&alt=media)

**Figure 1 (real well `000d7d20`).** *Left:* a degree-2 polynomial fits the true surface with **R² = 0.9925**.
*Right:* the entire high-frequency "wiggle" of TVT is tracked by −Z (demeaned), which is *known exactly at
eval time*. On the holdout, an oracle that knows only the *smooth* surface (a degree-2 fit of the true
surface) and subtracts the known Z reaches **RMSE 3.0–3.9 ft** — below the per-well-line oracle (6.6) and
below the competition leaders (5.2–5.4).

The interpretation is decisive: **the entire high-frequency detail of the target is carried by −Z, for
free; the only hard part is the smooth, low-frequency per-well surface trend.** Every wiggle you might hope
a fancy model captures is already handed to you. What remains is a single slowly-varying curve per well —
and that curve is the wall.

This oracle argument places the whole competition on one ruler:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F091d86dda31cf61de7127a7e614a9ef9%2Ffig03_oracle_ladder.png?generation=1783380379678214&alt=media)

**Figure 3.** The ceiling ladder. From the top: the trivial baselines (flat 15.9, beam 15.8) and the
isolated PF / best NN (~11); our variance-reduction submission (7.08); our best submission (6.79); the
best-linear-eval-trend oracle (6.6); the competition leader (5.26); and the smooth-surface-minus-known-Z
oracle (3.9). The narrow green band between 3.9 and 6.6 is the entire "learnable" headroom — a reminder of
how compressed this problem is, and why sub-foot improvements are noise.

## 5. The Wall: the surface trend is not recoverable

That smooth trend is exactly what cannot be recovered, because it **changes** between the known and eval
zones. The well crosses sub-seismic faults where the formation dip shifts; the eval-zone slope is therefore
*not* the known-zone slope.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F2c31e0964b590fade55e57409f70a5aa%2Ffig02_wall_slopes.png?generation=1783380444828200&alt=media)

**Figure 2 (real, ~400 wells).** *Left:* per-well surface slope in the known zone vs the eval zone. If the
trend persisted, all points would lie on the dashed diagonal; instead they scatter substantially off it.
*Right:* the distribution of the slope *change* across the eval boundary has σ ≈ 0.022 ft/ft. That looks
small — but multiplied by the long eval lever arm (thousands of feet), it is exactly what turns a tiny
trend error into feet of TVT error.

**Making that multiplication explicit: an identifiability argument, not just an empirical one.** The
per-well slope change is, structurally, an unknown constant of integration: any single-well estimate of the
eval-zone dip carries some error ε (ft/ft), and because TVT is the *integral* of dip along MD, that error
does not stay ε — it accumulates linearly with distance past the anchor. For an error growing as `ε · x`
over an eval zone of length `L`, the resulting RMS error is `ε · L / √3` (the RMS of a linear ramp over
`[0, L]`). Plugging in our own measured numbers — the slope-change σ = 0.0216 ft/ft from the panel above,
and the measured median eval-zone length of 4,905 ft — predicts an RMS error of **≈ 61 ft**, matching our
directly-measured linear-extrapolation experiment (43–73 ft, below) almost exactly, from a two-line
derivation using only quantities we had already measured independently. This is the same "the wall" fact
as the panel above, stated as an identifiability argument instead of a histogram: the eval-zone dip is not
a small perturbation of the known-zone dip, it is effectively a **new, unobserved constant** whose error
integrates without bound over the lever arm — which is also why *only* locally-anchored trackers (never
extrapolating far past their last correction) are well-conditioned on this task.

This is why trend extrapolation is catastrophic and why the problem is **ill-conditioned globally but
well-conditioned locally**:

- Linear extrapolation of the known surface into the eval zone: **43–73 ft** (far worse than flat).
- A GBM predicting the *global* surface coefficients: **22 ft** — reconstructing TVT from a few global
  numbers amplifies any slope error over the lever arm.
- Only *local* tracking (a sequential filter that never strays far from the last anchor) is well-conditioned
  — which is exactly what the PF and our best neural net do.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F184a6123d3494dacf516a2f07242e2a4%2Ffig12_error_vs_geom.png?generation=1783380485489821&alt=media)

**Figure 12 (real, 773 wells).** *Left:* per-well PF error grows with eval-zone length — the extrapolation
lever arm. *Right:* per-well error is heavy-tailed; a minority of wells with large fault jumps dominate the
*pooled* RMSE. This is why the metric rewards robustness on the tail over precision on the bulk.

**Quantifying that concentration, and why it explains CV↔LB swings.** We measured directly, on our own
773-well proxy: the **worst 10 wells (1.3% of the set) contribute 25.4% of total pooled squared error**, and
the **worst 40 (5.2%) contribute 49.8%** — on par with a similar concentration measurement independently
reported for this competition (≈29% / ≈47%). This concentration has a clean derivation. If each well's
dominant error term is a roughly-constant per-well offset `b_i` (drawn i.i.d. with variance σ²) observed
over `n_i` eval points, pooled MSE = `(1/N) Σᵢ nᵢbᵢ²`, so `Var(pooled MSE) ∝ σ⁴ · Σᵢ nᵢ² / N²` — i.e. the
*sampling variance* of the pooled metric across different train/test splits is itself dominated by the same
handful of long, badly-biased wells, not spread evenly across the dataset. On our data this weight
concentration is even sharper than the error concentration above (top-10 wells carry **75%** of this
variance term, top-40 carry **91%**), which is a rigorous reason — not just an empirical one — for why our
own resubmission variance (§3, the seed-noise floor) and any CV-vs-LB gap on this task can swing by several
tenths from ordinary sampling noise, with no real change in the underlying model: whichever handful of
long, hard wells happen to land in a given split dominates both the score and its variance.

## 6. Why the GR Log Cannot Localize

The natural hope is that the GR log lets us re-register the well against the typewell and thus recover the
trend. It does not, because the GR log is **self-similar**: the same bedding pattern repeats up and down
the column, so many depths look alike.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2Ffc400544a27ff6182256920f2fcf0f10%2Ffig04_gr_selfsim.png?generation=1783380510407643&alt=media)

**Figure 4 (real).** *Left:* the typewell GR is quasi-periodic — many candidate depths share almost the
same signature. *Right:* the horizontal GR autocorrelation is broadband with no single sharp peak, so there
is no unique lag at which the two logs "click" into place.

We made this quantitative from many angles. A learned contrastive GR↔typewell matcher localizes to only
**179 ft** (versus a ~31 ft full-sequence oracle); decomposing the aligner's error shows that removing an
oracle constant (31), an oracle linear trend (31), or the low frequencies (28) barely helps.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F9cf7a58a40dd5c60ebc17215e7a228a2%2Ffig10_align_decomp.png?generation=1783380523455429&alt=media)

**Figure 10.** The GR-matching error has roughly equal power at all frequencies: you cannot filter, low-pass,
anchor, or continuity-decode your way past it. Removing any one band leaves ~30 ft. This is the deepest form
of the wall — GR corrupts localization at *every scale*.

The clearest single picture is the 2D misfit heatmap that "geosteering-image" networks consume:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2Feb0b22e77961b1c5fbcc5f42b0de11c2%2Ffig11_2d_heatmap.png?generation=1783380534675061&alt=media)

**Figure 11 (real well `000d7d20`).** The misfit `M[z,x] = |typewell_GR[z] − horizontal_GR[x]|` (dark =
good match) with the true TVT path in cyan. Note the *many* parallel dark ridges: for almost every MD there
are several equally-plausible TVT depths. A 2D receptive field or a Viterbi continuity decode can smooth the
path, but there is no signal that tells it *which* ridge is the true one. This is precisely why the 2D-SDF
approach (§9) caps at ~30 ft.

### 6.1 The wall, confirmed from twelve independent angles

| # | Probe | Result | Meaning |
|---|---|---|---|
| 1 | Flat baseline | 15.1 | strong; drift has small predictable mean |
| 2 | Linear / dip / surface extrapolation | 37–85 | trends do not persist → worse than flat |
| 3 | Inter-well drift corr. with nearest neighbour | −0.08 | neighbours don't give the direction |
| 4 | Bimodal tie-break (±15 ft Eagle Ford) | r=0.054, mode correct 48.8% | a coin flip |
| 5 | PF residual vs any observable | all \|corr\| < 0.15 | we know *where* error is, not *which way* |
| 6 | Surface breakpoints vs GR shift | 0.082 | fault locations not in GR |
| 7 | Breakpoints vs formation-column curvature | 0.293 (columns removed in test) | weak & unavailable |
| 8 | Oracle GR window registration | ~31 ft | pure GR matching is weak |
| 9 | Learned contrastive GR↔typewell matcher | 179 ft localization | GR alone localizes nowhere |
| 10 | Aligner-error frequency decomposition | −const 31 / −linear 31 / −LF 28 | error is broadband, irreducible |
| 11 | James–Stein shrinkage of a heel-dip-extrapolation offset estimator | data-optimal shrink a\*=0.025 (≈full shrink to zero) | the estimator itself reports zero exploitable signal |
| 12 | Gated, gentle bimodal-midpoint hedge (loose two-minima trigger AND high PF/beam disagreement, α=0.2) | pooled 10.83 → 10.83–10.97 (flat to worse) | doesn't transfer to our pipeline even with the correct joint gate |

**The recurring principle.** Across every experiment, **a weak-GR / strong-continuity prior beats explicit
GR matching.** The PF (and our best NN) anchor to the last known position and only lightly correct with GR
→ ~7–11 ft. Approaches that *trust* GR matching (alignment, 2D contour) → ~30 ft. The exploitable signal is
continuity, not GR.

### 6.2 The acquisition physics: an independently-measured coherence spectrum

Angle 8–10 show the GR-matching error is broadband from the *statistics* side. We additionally measured
*why*, from the *instrument* side, using a signal-processing diagnostic: for every training well we pair
the horizontal GR with the typewell GR looked up at the *true* depth (available because we hold ground
truth on train), then compute the magnitude-squared coherence between the two curves (Welch, 1-ft sampling,
256-sample segments, 50% overlap) as a function of GR wavelength — separately for the known zone ("heel")
and the eval zone ("toe").

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F0fe0de37d42a4b6cd2fdf8d9f5262f27%2Ffig15_coherence_spectrum.png?generation=1783380550497211&alt=media)

**Figure 15 (real, 773 wells, our own measurement).** Correlation between GR and typewell-at-true-depth:
0.756 full lateral, 0.817 heel, 0.668 toe. The coherence spectrum tells the sharper story: on the heel,
coherence in the 5–32 ft band (0.16–0.50) is respectable — enough to disambiguate a bedding cycle. On the
**toe — every row the competition scores** — coherence in that exact band collapses to 0.02–0.04. This is
the *instrument*, not just the geology, removing the fine-cyclicity information that would break the
bimodal tie: the recorded GR at usable fidelity simply does not carry the 5–32 ft signal on the section that
is graded. This gives a physical, acquisition-side mechanism for why heel-calibration (angle 8) and every
matcher we tried (angles 8–10) hit the same wall specifically on the toe.

## 7. A Positive Result: the Decorrelated Particle-Filter Ensemble

The public pipeline is a full classical stack: a 128-seed particle filter over stratigraphic position +
beam-search DP/Viterbi continuity + spatial FormationPlaneKNN/IDW over (X,Y) + a LightGBM+CatBoost→Ridge
residual stack + a "gold" visible-prefix calibration.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F11c4701ee8722c41451f85eeabdd6b86%2Ffig13_pipeline.png?generation=1783380564660489&alt=media)

**Figure 13.** The pipeline, with our contribution highlighted in red. Everything upstream is the public
fork; our genuine additions are the decorrelation+seeds layer and the parallelization that makes it fit the
runtime budget.

Our positive results are all forms of **variance reduction over a Bayesian sequential filter**, not new
physics:

1. **Reverting an over-aggressive GR denoise.** A rotation-band notch denoise looked +2.8% on an isolated PF
   test but scored −0.24 on the LB (7.475 → 7.230). This was our first hard evidence that isolated-component
   tests mislead — the theme of §13.
2. **A diverse, decorrelated PF ensemble.** Blend the base PF with a *stiffer* configuration (lower process
   noise) at 0.65/0.35. The two make *independent* errors, so averaging cancels variance.
   → **7.230 → 7.096**, below the seed-noise floor (7.168): a real improvement, not luck.
3. **Variance reduction by more seeds** (128 → 160 → 192): converges toward the true ensemble mean, less
   luck-dependent → **7.091**, a more robust submission for the private split.
4. **An engineering enabler under the Code Requirements — parallelized candidate generation.** The
   competition caps runtime at 9 h with no internet, which normally makes variance reduction (a decorrelated
   partner *and* more seeds) infeasible — it would blow the budget. We parallelized the per-well SP45
   candidate-generation stage (`build_sp45_candidate`) across CPU workers via **joblib**, roughly recovering
   the extra wall-clock the ensemble and higher seed count consume. This is small but load-bearing: on a
   time-boxed code competition, *the runtime budget is part of the model*, and buying back compute is a
   legitimate lever that directly enabled our first transferable gain (a second, larger one follows in
   §7.2).

**The honest climb (real public-LB deltas, one component at a time):**

| Step | Component | Public LB | Δ |
|---|---|---|---|
| 0 | SP45-only physics pipeline, no companion datasets | 8.270 | — |
| 1 | + companion datasets (fleongg pretrained + ravaghi artifacts) | 7.475 | −0.795 |
| 2 | − revert over-aggressive GR denoise (isolated test misled us) | **7.230** | −0.245 |
| 3 | + decorrelated PF ensemble (lo-noise partner, 0.65/0.35) | **7.096** | −0.134 |
| 4 | + more seeds (128→160) | 7.091 | −0.005 (≈ seed-noise floor) |
| 5 | + more seeds (160→192) | 7.080 | −0.011 (≈ seed-noise floor) |
| 6 | + WARP-blend (decorrelated neural corrector, §9.2) | 6.882 | **−0.198** |
| 7 | + physics post-process on top of WARP-blend (§7.1a) | 6.881 | −0.001 |
| 8 | steps 6–7, both moved to run *after* gold-calibration (§7.2) | 6.846 / **6.836** | −0.036 / −0.045 |
| 9 | + WARP weight raised 0.15→0.30, the value 5-fold CV independently selected in every fold (§7.2) | **6.794** | −0.042 |

Steps 4–5 sit inside our own measured seed-noise floor (±0.07, §3) — we report them as *directional and
seed-checked* rather than precise. Step 9's −0.042 is itself smaller than that same noise floor, so we hold
it to the same standard: directionally consistent with what the cross-validation predicted, not claimed as
a precise, certain delta. Steps 6–9 are real, resubmitted leaderboard numbers (not proxy), detailed in
§7.2, and **6.794 is our best submission to date.** Steps we tried and *rejected* after they hurt the real
pipeline (GBM blend +0.35 to +0.84, the 4-way decorrelation over-dilution +0.52, twjit/gs partners within
noise of step 3, and — critically — a third correction stacked on top of step 8 that reverses all of this
gain, §7.2) are catalogued in full in §7.2, §8.3 and §8.5 — the same "isolated tests mislead" lesson that
step 2 first taught us, now confirmed at the level of *whole pipeline stages*, not just features.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F7fccdce7867edd1ddc1dbf554351208a%2Ffig06_pf_spread.png?generation=1783380578145262&alt=media)

**Figure 6 (real, 3 longest-eval wells).** The PF ensemble members (pf3/5/8/12, i.e. different process-noise
scales) bracket the true TVT; each is a noisy estimate of the same latent path. Averaging decorrelated
members and adding seeds is what pushes below the single-run noise floor.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2Fcf356a425c65fdb86bc2db56c95ec2d1%2Ffig07_seed_noise.png?generation=1783380626641469&alt=media)

**Figure 7 (real, 773 wells).** The stochastic seed spread averages ~5 ft per well. A *single* PF run
therefore carries several feet of purely-stochastic error on top of its bias. Variance reduction removes
this component — it is the free lunch, and it is exactly what our submissions cashed in.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F4dab6a9d19b7bfcabb60bbf42d8bcb66%2Ffig08_submissions.png?generation=1783380636626812&alt=media)

**Figure 8 (real LB scores).** Our full submission ladder. Decorrelation helped (7.230 → 7.096);
over-dilution (the 4-way ensemble) hurt (7.752); more seeds gave a robust 7.080. Stacking WARP-blend with
physics post-process gave 6.836 (§7.2), and transferring the cross-validated WARP-weight (0.30) back onto
the real submission pushed it further to our current best, **6.794** — while wall-hedge and the 3-way stack
(adding `pf_z`) sit above 7.08 in both pipeline placements we tested, confirming they are genuinely harmful
rather than a placement artifact.

**Where decorrelation stopped.** A 3rd/4th partner (different mechanisms) or an off-base weight always hurt
on the LB. The single-partner ensemble was the sweet spot; more dilution of the tuned base only hurt. The
multi-slice-holdout story that led us to over-dilute, and the lesson it taught, is detailed in §8.3.

### 7.1 Three more offline-validated stacking levers

We independently implemented and honestly cross-fit-validated three further levers on our own proxy and
data. **Update (§7.2 reports the real-LB outcome):** lever (a), combined with the WARP-blend of §9.2, is
the basis of our real best submission (6.836, later refined to **6.794**, §7.2); lever (d) is now a
confirmed real-LB negative when stacked as a third correction on top of (a). Lever (b) remains offline-only
— it was not part of the real-LB test in §7.2.

**(a) Physics post-process: robust projection + warm-up damping + smoothing.** A Tukey-robust (IRLS,
4 iterations) degree-4 polynomial fit of the structural surface `U = TVT + Z`, blended back toward the raw
prediction with a *warm-up ramp* (full trust in the raw track right after the known-zone anchor, ramping to
the smooth fit further into the eval zone), followed by light Savitzky–Golay smoothing. Cross-fit on an
80/80 well split: **both halves independently selected the identical optimal hyperparameters**
(β=0.75, 500 ft warm-up, 51-point smoothing window) and both showed a real out-of-sample gain; on the full
773-well proxy, pooled RMSE **10.826 → 10.522**. We also tested a *gated/selective* variant (apply the
correction only on high-disagreement wells, to protect any well where the pipeline is already near-exact)
— gating **strictly hurt** here, monotonically with how much we restricted it (§15's override audit
explains why: on our proxy we have no such overlap subset to protect, so uniform application is optimal
for us). **This lever, combined with WARP-blend, is now real-LB confirmed as the basis of our best
submission (6.836, later refined to 6.794 by transferring a cross-validated hyperparameter, §7.2).**

**(b) L1-objective diversity in the residual corrector.** Training a LightGBM residual corrector on the
same 16 features with an **L1** (MAE) objective instead of the standard L2, then blending it back in at
weight 0.9. Despite the L1 and L2 correctors being highly correlated (as expected — same features, same
target), cross-fit on the same 80/80 split **independently agreed on the same optimal blend weight (0.9)
in both directions** and both showed a real gain (pooled 10.83 → ~10.58 on the full set). This is a
different mechanism from the L2 GBM blend we tried earlier (§8.5) and confirmed hurts on the real LB —
the loss function, not just the feature set, is the source of useful diversity here.

**(c) The three levers stack.** Applying (a) then (b) in sequence gives pooled RMSE **10.826 → 10.522 →
10.278** — a combined **−0.55**, each stage adding on top of the other rather than being redundant. Adding
the WARP blend (§9.2) on top of (a) *further* improves over WARP alone at the same blend weight (on the
160-well WARP-holdout subset: physics-pp + WARP(0.3) = **9.69** vs WARP(0.3) alone = 9.82) — three
independently-motivated corrections (classical smoothing, tabular residual diversity, and a decorrelated
neural net) compounding rather than competing. As with §9.2, we originally flagged this as
**offline-validated, LB-pending**: gains of this size on the proxy do not guarantee an equal move on the
real 7.08 pipeline. **§7.2 reports what actually happened on resubmission: the *two*-lever combination
(WARP + physics-pp) transferred and gave our new best (6.836), but extending the same stack with a third
lever did not — a real, controlled limit on how far "corrections stack" generalizes**, sharpening rather
than overturning the finding here.

**(d) A fourth finding, upgraded from a partial negative to a real, previously-unexploited positive:
`pf_z`, a Z-velocity-coupled second particle filter that our own inherited pipeline already computes.**
Every decorrelated PF partner we had previously built ourselves (§8.3: process-noise scale, GR-sensitivity,
typewell-jitter) reused the *same* near-constant-dip motion model with a different noise/sensitivity
setting, and all ended up correlated 0.82–0.94 with the base — too similar to buy much variance reduction.
We first prototyped an independent second motion model whose proposed dip rate is pulled toward
`β·dZ/dMD + intercept` (fit per well on the known prefix) rather than pure momentum persistence, so its
prior comes from the trajectory's own curvature — a genuinely different physical source. Our own quick
prototype was too numerically unstable to be useful alone (correlation 0.14 with the base, but its own
pooled RMSE was 39.6 against the base's 10.3). While tracing through the *inherited* pipeline's own code to
understand what it already computes, we discovered it **already implements exactly this idea** — a mature,
tuned, numba-compiled `pf_z` (soft Bayesian velocity-consistency likelihood, dual raw/smoothed-GR emission)
— but its output was **only ever wired into the largely-inactive GBM stack as a feature** (`pf_z`,
`pf_z_delta`, `pf_vs_z`), never blended directly into the SP45 prediction. Running the real, verbatim
`pf_z` on a 160-well holdout: error correlation with the main tracker (`pf_ancc`) is **0.47** — still far
more decorrelated than any partner we engineered ourselves — and, critically, `pf_z` **alone is a
perfectly usable tracker** (single-seed pooled RMSE 14.7 vs the main tracker's 13.5, the same order of
magnitude, not a divergent mess). Blending gives a real single-seed gain: pooled **13.49 → 12.05** at blend
weight 0.4 (**−10.7% relative**). We then validated the blend weight properly: seed-averaging `pf_z` (8
seeds) on 300 proxy wells and blending against the real `sp45` output (not just the isolated tracker),
cross-fit on a 50/50 split — **both halves independently landed in the same 0.10–0.15 optimal range**
(H1→H2: 9.60 → 9.44; H2→H1: 11.59 → 11.39, both real gains), matching the full-sweep optimum on all 300
wells (10.66 → 10.47 at a≈0.10–0.15, a near-flat minimum). We built a submission stage that seed-averages
`pf_z` (16 seeds in the real submission, to tame its own stochastic noise the same way the main ensemble
does, §7) and blends it in at **weight 0.15** — squarely inside the cross-fit-validated optimum — after the
WARP blend and physics post-process. On the 160-well WARP-holdout subset, all three stages together
(physics-pp + WARP(0.3) + `pf_z`(0.15)) reach **9.53**, down from the sp45 baseline's 10.62 (**−10.3%**) —
the strongest combined offline result of this study. This is, by a wide margin, the most decorrelated *and*
individually-usable lever surfaced in this entire study — an existing capability in the inherited pipeline
that nobody (including its original authors, as far as its wiring shows) had connected to the final
prediction. **Real-LB update (§7.2): despite this careful offline validation, adding `pf_z` as a third
stacked correction on top of the already-real-LB-confirmed WARP+physics-pp combination reversed the entire
gain** (6.836 → 7.446–7.515, in both pipeline placements we tested) — the sharpest demonstration in this
whole study that offline/proxy transfer can hold for two stacked corrections and still fail for a third.

### 7.2 Real-LB confirmation: a new best, and a hard limit on stacking

Rather than trust proxy numbers, we resubmitted every lever above as an actual Kaggle submission, in a
controlled design: four components — **WARP-blend alone**; **WARP-blend + physics post-process**; the
**wall-hedge** shrink of §9.2; and the **full three-lever stack** (WARP + physics-pp + `pf_z`) — each
submitted twice: once in its original position (before the pipeline's `guarded_contact_override` and gold
visible-prefix calibration), and once moved to run *after* gold-calibration completes, the placement fix
first motivated by the wall-hedge regression reported in §9.2.

| Component | Before gold-calibration | After gold-calibration (placement fix) |
|---|---|---|
| WARP-blend alone | 6.882 | 6.846 |
| WARP-blend + physics post-process | 6.881 | **6.836 — new best** |
| wall-hedge (shrink toward last-known TVT) | 7.252 | 7.304 |
| full stack: WARP + physics-pp + `pf_z` | 7.515 | 7.446 |

*(incumbent baseline: 7.080)*

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F73fb509cc6d8ef8705a46bc98c3d6d94%2Ffig16_real_stacking.png?generation=1783380653035312&alt=media)

**Figure 16 (real, all 8 numbers are actual Kaggle public-LB resubmissions, not proxy).** The two good
components (WARP alone, WARP+physics-pp) beat the baseline in both placements and improve further with the
placement fix; the two harmful components (wall-hedge, the 3-lever stack) stay worse than baseline in both
placements.

Two findings, both established by direct resubmission rather than proxy validation:

**1. A new best submission.** WARP-blend and physics post-process together, placed after gold-calibration,
score **6.836** — a genuine, real-LB, −3.4% relative improvement over the prior best of 7.080. WARP-blend
alone is close behind at 6.846, so physics-pp contributes a further real gain of only −0.010 on top of
WARP — smaller than, but the same direction as, its offline-measured contribution in §7.1a.

**2. Placement helps, but it is not a cure — some components are simply harmful.** The after-gold placement
gives a small, consistent additional improvement to the two components that are already good (WARP alone:
−0.036; WARP+physics-pp: −0.045), which is consistent with our §9.2 mechanism — that inserting a correction
*upstream* of gold-calibration disturbs the anchor its per-well backtest relies on. But the same placement
fix does **not** rescue the two components that are inherently harmful: wall-hedge is *worse* after the fix
than before (7.304 vs 7.252 — no positive effect from reordering, if anything slightly worse), and the
three-lever full stack, though it improves somewhat with the fix (7.515 → 7.446), remains far worse than
either the two-lever combination or the unmodified baseline. **This means pipeline placement was never the
primary cause of the wall-hedge or full-stack failures — both components are harmful in their own right on
the real hidden test, independent of where in the pipeline they run.** We state this plainly as a
correction to the stronger causal claim we made in §9.2 (see the note there): placement is a real, small,
second-order effect, but the dominant cause of these two failures is the components themselves.

**Is the new best (6.836) overfit? A proper 5-fold grouped cross-validation, not just the earlier 80/80
split.** The WARP-blend weight and physics-pp hyperparameters (β, warm-up, smoothing window) reported in
§7.1a/§9.2 were originally selected via a single 80/80 cross-fit. To rule out that this was a lucky split,
we ran a full 5-fold grouped cross-validation on **all 773 wells**: for each fold, the WARP-blend weight
and physics-pp grid were selected using *only* the other four folds, then evaluated on the held-out fold
with those frozen hyperparameters (no refitting on the test fold).

| Fold | wells | baseline (sp45 only) | best combo (held-out) | gain |
|---|---|---|---|---|
| 0 | 155 | 10.697 | 8.225 | +2.47 |
| 1 | 155 | 13.871 | 11.271 | +2.60 |
| 2 | 155 | 9.242 | 7.181 | +2.06 |
| 3 | 154 | 9.414 | 7.363 | +2.05 |
| 4 | 154 | 10.165 | 8.175 | +1.99 |

The gain is positive in **5/5 folds** (+1.99 to +2.60), and — the strongest evidence against overfitting —
**all five folds independently selected the identical hyperparameters** (WARP weight 0.30, physics-pp
β=0.75, 500 ft warm-up, 51-point smoothing), matching what the original 80/80 split had already found. If
the combination were chasing split-specific noise, different folds would select different "optimal"
settings; instead every fold agrees exactly. We flag the obvious caveat plainly: these are proxy pooled-RMSE
numbers (baseline ≈10.8, not the real pipeline's 7.08 scale) and the *magnitude* of the gain here (−20.7%)
is far larger than the real-LB gain (−3.4%) — consistent with our standing finding that the proxy predicts
*direction*, not *magnitude* (§13). What the K-fold test targets specifically is the overfitting question,
and on that question the answer is unambiguous: this is a stable, cross-validated effect, not a fitted
artifact of one split.

**The cross-validated weight transfers: a further real-LB gain (6.836 → 6.794).** The 5-fold CV above made a
falsifiable prediction, not just a robustness check: every fold selected WARP-blend weight 0.30, not the
more conservative 0.15 we had actually shipped (§9.2). If that specific value — not merely its direction —
transfers to the real hidden test, raising the shipped weight from 0.15 to 0.30 (physics-pp and everything
else unchanged) should give a further real gain. We built and submitted exactly that change and it scored
**6.794** — a further **−0.042** move in the predicted direction, our best submission to date (7.080 →
6.836 → 6.794, **−4.0%** relative overall). We hold this single step to the same honesty standard as steps
4–5 above: −0.042 is itself smaller than our measured seed-noise floor (±0.07, §3), so we report it as
*directionally consistent with the cross-validation's prediction*, not as a precise, certain delta on its
own. Taken together with the 5/5-fold agreement, though, it is a clean instance of a cross-validated
hyperparameter transferring in more than just sign.

**Why this belongs next to §7.1's stacking positives.** The over-stacking result is the same lesson as the
GBM-blend cautionary tale (§8.5) and the 4-way ensemble (§8.3), now demonstrated one level higher: not just
*isolated-component* tests mislead, but even a *pair* of independently cross-fit-validated, real-LB-confirmed
corrections can fail to extend to a third. Two genuinely different corrections (a decorrelated neural net,
a robust physics smoother) compounded cleanly; a third (a second particle filter, itself individually sound
and cross-fit-validated on 300 wells, §7.1d) did not.

**A mechanistic account, tested directly rather than assumed.** We had originally reported no full
mechanistic explanation here. We since tested a specific hypothesis: that variance-reduction / hedge-style
corrections are *base-dependent* — they pay off only where the base pipeline has not already spent that
same variance-reduction budget through some other means. We cross-fit (5-fold grouped) the simple
wall-hedge shrink both on bare `sp45` and on top of the already-tuned WARP+physics-pp combo, selecting the
hedge weight fresh in each case:

| Hedge applied to | mean gain (5-fold cross-fit) | folds where hedge helps |
|---|---|---|
| bare `sp45` (no WARP/physics-pp) | +0.102 | 4/5 |
| WARP+physics-pp combo (already tuned) | **+0.000 — grid search selects weight 0 in every fold** | 0/5 |

The hedge is a real, if modest, gain on the untouched base — consistent with our original §7 finding — but
once WARP-blend and physics-pp have already been applied, the cross-fit-optimal hedge weight collapses to
*exactly* zero in all five folds, not just a small value. This is a clean, testable confirmation that
hedge-style corrections are base-dependent, not universal: they draw on the same limited variance-reduction
budget as WARP's own continuity anchor and physics-pp's own robust smoothing, and once that budget is
already spent, there is nothing left for a further shrink to correctly reallocate. This is also our best
available account of why `pf_z`, stacked as a *third* correction, broke the combination above: by the third
stage there is very little of the original, uncorrected pipeline output left for another correction to
usefully act on.

### 7.3 Does the new best survive harsher, spatially-honest validation?

Our whole-well holdout (§3) groups by well and shuffles spatially, but a held-out well can still keep
*nearby* trained wells in the training set — the concern (independently raised in this competition's
literature) is that grouped-well cross-validation cannot see this kind of neighbour leakage, because it
never removes a whole geographic *region* at once. We tested this directly with our own **Leave-Spatial-
blocks-Out (LSO)** check: cluster all 773 wells into 6 spatial blocks by (X, Y) with K-means, then hold out
one whole block at a time — training-fold hyperparameters (the frozen WARP-weight / physics-pp settings
from §7.2) never touch the held-out block's wells *or any of their neighbours*, since the entire block is
geographically removed.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F8a4ebb013cfc37656dc124e56c2ae101%2Ffig17_lso_isolation.png?generation=1783380670213019&alt=media)

**Figure 17 (real, our own 773-well proxy).** *Left:* the WARP+physics-pp gain under ordinary random
grouped 5-fold (+2.235) versus true spatial-block LSO (+2.244) — statistically indistinguishable, and
**positive in 6/6 spatial blocks**. The gain is not a neighbour-leakage artifact of our validation scheme.
*Right:* splitting wells by nearest-neighbour distance (bottom vs top quartile), isolated wells run higher
baseline error (9.81 vs 6.62 ft) — an independent replication of the general "isolated wells are harder"
pattern — but **the combo's gain is larger on isolated wells** (+2.14 vs +1.26), not smaller. If our new
best submission were secretly relying on neighbour information, we would expect the opposite: bigger gains
on dense wells, and gains vanishing under LSO. Neither happened, which is the reassuring result for private-
leaderboard transfer.

## 8. Full Experiment Catalogue (60+ experiments)

This is the heart of the study: every serious thing we tried, grouped by family, with the mechanism of its
failure. Legend: ✅ helped · ❌ hurt/null · 🟡 partial/mirage.

### 8.1 Baselines and framing (3)

| Experiment | Result | Why |
|---|---|---|
| ❌ Flat (hold last TVT) | 15.9 | strong; drift's mean is small and predictable |
| ❌ Linear trend extrapolation | ~32 | trend does not persist across faults (§5) |
| ✅ Methodological fix: pooled (not per-well) RMSE | — | per-well averaging gave a false ~10 and would have mis-ranked everything |

### 8.2 GR matching — the leaders' documented lever (9)

This family is where the "obvious" signal lives, and where the wall bites hardest.

| Experiment | Result | Why it fails |
|---|---|---|
| ❌ Naive Pearson window correlation | ~8% localized | self-similarity (§6) |
| ❌ Banded / anchored correlation tracker | worse | drifts onto a look-alike ridge |
| ❌ Fixed-anchor correlation | 62 | locks onto the wrong lag early |
| ✅ Heel affine GR calibration | 77–80% localized | *diagnostic* only — calibrates scale, not position |
| ❌ NCC windowed likelihood inside PF | 12.4 → 23.3 | MD/TVT axis mismatch corrupts the filter |
| ❌ Shape / derivative matching | 24% vs value 50% | derivative amplifies GR noise |
| ❌ Affine calibration inside beam search | worse | breaks the tuned emission scale `es` |
| ❌ Full bidirectional DP alignment | ~18 | weaker than PF; negligible blend value |
| ❌ Trajectory-informed DP transition | 69 (catastrophe) | locks onto a wrong but self-consistent trend |

**Takeaway.** Every attempt to make GR *lead* the estimate lost to the continuity prior. The one "success"
(heel calibration) only fixes the GR *scale*; it does nothing for *position*, which is the wall.

### 8.3 Particle-filter internals and decorrelation (10)

The PF is the workhorse, so we probed it hard. The decorrelation study below is also the origin of our best
submission *and* our worst.

| Experiment | Result | Why |
|---|---|---|
| ❌ Affine heel calibration in PF likelihood | 10.35 → 14.13 | over-trusts miscalibrated GR |
| ❌ Resample-threshold tuning | noise | below seed variance |
| ❌ Adaptive scale by seed spread | no gain | spread ≠ headroom |
| ❌ Adaptive scale by known-zone ambiguity | r=−0.23 | signal too weak to act on |
| ✅ **Diverse-config PF ensemble (lo-noise blend)** | **LB 7.096** | independent errors → variance cancels |
| 🟡 gs-decorrelation (GR-sensitivity `gs_mul=0.6`) | mirage on slice A | brilliant on one slice, worse on another |
| ❌ Heavy-tail process noise (Student-t df=3) | slice-A mirage | A −0.22 but B +0.77; corr 0.90 with base → weak partner |
| 🟡 Typewell-jitter (bootstrap the reference, `tw_jit=0.15`) | real but redundant | decorrelates (corr 0.82–0.92) yet doesn't beat v3 |
| ✅→❌ **4-way kitchen-sink** (base+lo+gs+twjit) | holdout 9.486 **best**, LB **7.752 worst** | shared isolated-component blind spot |
| — Runtime guard: split seeds across partners | <9 h | engineering, so the ensemble fits the budget |

**The three-slice decorrelation study (real numbers).** We evaluated candidate partners on three
independent 80-well slices to avoid a single-slice mirage:

| slice | default | v1 lo-blend | gsT0.30 | triple d0.6+lo0.2+gsT0.2 |
|---|---|---|---|---|
| A (last 80) | 11.281 | 11.133 | 10.736 | **10.692** |
| B (mid) | 9.347 | **8.705** | 9.257 | 8.872 |
| C (early) | 9.576 | 9.304 | 9.311 | 9.155 |
| **mean** | 9.735 | 9.714 | 9.768 | **9.573** |

The lesson is written into the numbers: `gsT` alone looks best on slice A (10.736) but is *worse than v1* on
slice B — a slice mirage. The triple blend is best on average and beats pure PF on all three slices, but the
margin is thin and distribution-dependent. **The 4-way extension then over-fit the holdout itself and lost
on the real LB — the sharpest possible demonstration that even a multi-slice holdout has a shared blind
spot.** This is why we shipped the *single-partner* ensemble.

### 8.4 Smoothing and post-processing (5)

| Experiment | Result | Why |
|---|---|---|
| ❌ Savgol / robust polynomial smoothing | null | the error is *drift*, not high-frequency noise |
| ❌ deg-2 vs deg-4 projection | noise | difference vanishes at 60 wells |
| ❌ Piecewise-linear projection (2–8 segments) | null | deg-4 already optimal |
| ❌ Physical clipping (typewell range, delta bounds) | null | predictions already in range |
| 🟡 Bimodal posterior-mean hedge | oracle-gated only | works only if you already know the mode (you don't) |

**Takeaway.** You cannot post-process your way out. Because the residual is a low-frequency *drift* (not
noise), smoothing removes signal you already have and leaves the drift untouched.

### 8.5 GBM / classical ML blending (8)

| Experiment | Result | Why |
|---|---|---|
| Rich-feature GBM standalone | 14.55 | weaker than PF but decorrelated (corr 0.44) |
| ✅ PF + GBM blend (isolated) | 10.56 (−6%) | our only isolated-component success |
| ❌ PF + GBM blend on real LB (w=0.15 / 0.25) | 7.582 / 8.066 | **the isolated win did not transfer** |
| ❌ Meta-stacker (pipeline signals + features) | 10.79 vs 10.56 | no gain over a simple blend |
| 🟡 Q-3D tortuosity feature | −0.048 isolated, +0.065 in context | redundant with PF signals |
| 🟡 Signed-azimuth feature | −0.36 isolated, +0.035 in context | powerful alone, redundant in context |
| ❌ Adaptive/confidence blend by seed_std | null | spread predicts magnitude, not direction |
| ❌ Blend-weight sweep SP45/GBM | 0.20 optimal | confounded by a weak GBM |

**Takeaway and warning.** The GBM blend is the cautionary tale of the whole competition: **−6% in isolation,
then +5% *worse* on the real LB.** The isolated PF is not the full pipeline; a decorrelator that helps a weak
component can hurt the tuned whole. This single result reshaped our entire methodology (§3, §13).

### 8.6 Denoising and external knowledge (3)

| Experiment | Result | Why |
|---|---|---|
| ❌ GR rotation-band notch denoise | +2.8% isolated → 7.475 vs 7.230 LB | isolated test ≠ full pipeline |
| ✅ Mined `rogii-geosteering-toolkit` (tortuosity, dcor, signed azimuth, StratifiedGroupKFold) | adopted the *split*, rejected the features | spatially-shuffled validation matters; features redundant |
| ✅ Established the three public notebooks share one lineage | stopped chasing forks | they are byte-identical descendants |

### 8.7 Reverse-engineering the human markup (a fresh angle) (3)

TVT was *drawn by geologists*, not measured by nature — so we asked whether the drawing conventions leak
signal:

- **64.5% of eval points have ~zero curvature** — the labels are long piecewise-linear segments.
- **Dips are quantized:** `dTVT/dMD` clusters at round values {0, 0.01, 0.02}, median exactly **0.0200**.
- **Breakpoints are structural:** corr(|d²TVT|, |d²surface|) = **0.80** — the kinks sit at real dip changes.

Three leads followed: (1) snap predicted dips to the round grid; (2) piecewise-linear reconstruction with
kink detection; (3) place kinks at formation intersections. **All failed to help**, for the same reason as
everything else: the *breakpoint locations are the faults*, and §6 shows the faults are not recoverable from
GR. Snapping dips to the round grid merely discretizes the error (10.35 → 10.4+). Even knowing the labels are
human-drawn does not rescue us, because the hard quantity — *where* the geologist put a kink — is set by the
unobservable geology.

## 9. The Neural-Network Study: 22 Architectures

We built a full research harness (773-well feature cache, whole-well holdout, augmentation, EMA, a GPU
training stand) and swept a broad architecture space. **Every from-scratch model floors near ~11 = the
isolated-PF level.**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F963e792e461818df5842aefa50e73f85%2Ffig05_nn_bars.png?generation=1783380693559699&alt=media)

**Figure 5 (real holdout RMSE).** Twenty-two architectures, coloured by family. The reference lines mark flat
(15.1), our best NN (WARP, 11.3), our PF submission (7.09), and the leaders (5.26). Two clusters are worth
naming: the *continuity-anchored* nets (orange, WARP family) that reach ~11, and the *GR-trusting* aligners
(dark red, 2D-SDF / soft-argmax / surface-space) that blow up to ~30.

| Architecture | Holdout RMSE | Failure mode |
|---|---|---|
| flat baseline | 15.1 | — |
| Point-wise TCN (regress TVT) | 15.0 | no per-point signal → ≈flat |
| Neural matcher (static anchor) | 14.95 | collapses to flat |
| Neural matcher (iterated) | 101 | diverges (positive feedback) |
| GRU sequential filter | 14.7 | tiny safe steps ≈ flat |
| Contrastive GR↔typewell matcher | 179 ft loc. | GR self-similarity (§6) |
| Neural grid Bayesian filter (PF) | 24.5 | learned GR emission net-harmful |
| **WARP (dTVT integration + cross-attn)** | **11.28** | ★ best; continuity-anchored |
| WARP + multi-scale GR | 11.19 | marginal |
| WARP + multi-scale + EMA | 11.34 | stable |
| Surface-space (−Z) regression | 30 | drift; ill-conditioned |
| Low-DOF / spline head | 12 | — |
| Alignment (soft-argmax over typewell) | 30 | GR wall |
| Hybrid (WARP + low-passed align anchor) | 11.5 | align adds nothing |
| Mixture-Density Network (K=3) + Viterbi | 14 | fixes mean-collapse but weak |
| Synthetic pretrain (naive) | 13.2 | too-clean GR → no transfer |
| Joint real + synthetic augmentation | 11.57 | ≈WARP, less overfit |
| 2D misfit-heatmap + SDF U-Net (per-column) | ~29 | GR wall (Fig. 11) |
| 2D-SDF + Viterbi continuity decode | ~32 | continuity can't pick the true contour |
| Transformer (global self + cross attention) | 13.4 | overfits 773 wells |
| Learned stack / distillation of PF signals | 10.7 | = isolated-PF (signals correlated) |
| Pure classification over typewell bins (softmax, argmax decode) | 29.8 | no continuity anchor → GR wall |
| Pure classification, top-3-expectation decode | 28.6 | softens bimodal collapse, still hits the wall |

**Why WARP (11) is best.** It predicts a per-step derivative `dTVT`, integrates it by cumulative sum
anchored at the last known TVT, and treats GR only as a *weak* corrector via cross-attention to the typewell.
This is the weak-GR/strong-continuity principle of §6 made into an architecture: the output is
well-conditioned (it cannot drift far from the anchor) precisely because it never trusts GR to lead.

**Why the neural matcher *diverges* to 101.** When we let the matcher iterate — re-anchoring on its own GR
match each step — it forms a positive-feedback loop onto a self-similar look-alike and runs away. The static
version simply collapses to flat instead. Both are the wall in different clothes.

**Why the learned PF (neuro-grid Bayesian filter) is *worse* than the classical PF (24.5 vs 11).** Making
the GR emission likelihood learnable let the network *increase* its reliance on GR, which is exactly wrong:
the classical PF's hand-set, deliberately-weak emission is better than any likelihood the net learns,
because the informative move is to distrust GR.

**Why transformers do worse (13.4).** With only 773 wells, global attention overfits; the CNN's locality is
the correct inductive bias. Flexibility is a liability when the data are few and the signal is a fragile
low-frequency trend.

**Why 2D-SDF (the geosteering-image approach) caps at ~30.** The misfit image (Fig. 11) has *many*
plausible continuous zero-contours; neither a 2D receptive field nor a Viterbi decode can pick the true one,
because there is no disambiguating signal. Continuity smooths the wrong contour just as happily as the right
one.

**Why pure classification over typewell bins fails (28.6–29.8), and why this is an independent
confirmation.** A separate research thread proposed a literal reading of "the geosteering-image
approach": discretize the typewell axis into bins and predict each step's position via softmax +
cross-entropy, decoding by argmax (or a top-3-probability-weighted expectation, to handle bimodality
without the mean-collapse of a soft-argmax). We built and trained this from scratch, on the same
encoder/cross-attention scaffold as WARP, for a clean architecture-only comparison. It reaches only
28.6 (top-3) / 29.8 (argmax) — squarely in the alignment/2D-SDF cluster (~29–32), *not* near WARP's
11. Top-3-expectation decoding softens the naive mean-collapse a little but does not touch the
underlying problem: without a continuity anchor, the model has no reason to prefer the true bin over
any of the other equally-plausible ones in Figure 11's misfit ridges. This is the fourth independent
architecture family (after alignment, 2D-SDF, and the neural grid filter) to hit the same wall for the
same reason, which is why we treat "weak-GR/strong-continuity beats explicit matching" as a structural
property of this task rather than an artifact of one architecture.

**Why naive synthetic pretraining fails (13.2).** A generator with clean `GR = typewell_GR(TVT)` is
*trivially invertible*, so the pretrained model learns a direct inversion that collapses to flat on real,
self-similar GR. The joint real+synthetic variant regularizes a little (11.57) but does not transfer new
skill. **Realistic** synthetic — matching the real ambiguity — is required. The obvious next question is
*which* kind of realism matters, which we isolate in the controlled study below.

### 9.1 A controlled synthetic-pretraining transfer study: it is the forward model, not the noise

Because realistic synthetic pretraining is widely believed to be the decisive ingredient for the leaders'
scores, we ran a controlled experiment to find *what* about the synthetic is the bottleneck. We first
verified the premise of the whole "inverse-problem" family with a forward-model diagnostic on 160 wells:
the calibrated horizontal GR at the *true* TVT correlates 0.70 with the typewell prediction, the forward
model is locally discriminative (a ±15–40 ft shift raises the misfit by 3.8× on 98% of wells), and — key
for the diffusion hypothesis — the residual `observed − typewell(TVT)` is *structured*, with an
autocorrelation length of ~50 samples (i.e. not white noise; a naive Gaussian is the wrong noise model).

We then built a synthetic generator whose TVT paths use our own human-markup findings (§8.7): piecewise-
linear, near-quantized dips, with rare fault jumps. Onto the clean forward-model GR we added noise two ways:
**(a)** naive white Gaussian, and **(b)** the *empirically bootstrapped real residual* — i.e. the true,
autocorrelated geological noise resampled from real wells. Bootstrapping the real residual is the
best-possible noise realism, an upper bound on what any diffusion noise model could achieve. We evaluated
transfer to the *real* holdout under two protocols: synthetic-only, and the leaders' recipe of
synthetic-pretrain → real-finetune, against a real-only WARP anchor.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F5c52d2cb5d7c70dbff5735f6701e0b77%2Ffig14_synth_transfer.png?generation=1783380716588423&alt=media)

**Figure 14 (real holdout RMSE).** The result is unambiguous and, at first, counter-intuitive:

| Condition | Real holdout RMSE |
|---|---|
| real-only WARP (anchor) | **11.03** |
| synth-pretrain → real-finetune (the recipe) | 11.47 |
| synthetic-only, naive Gaussian noise | 14.86 (≈ flat) |
| synthetic-only, realistic bootstrap noise | 16.69 |

Two things stand out. First, **the more realistic noise transferred *worse*** (16.69 vs 14.86): the naive
model safely collapsed toward flat, whereas the realistic-noise model confidently learned synthetic-specific
moves that are wrong on real wells. Second, and decisively, **pretraining then finetuning slightly *hurt*
the real-only anchor** (11.47 vs 11.03). Feeding the model the *exact empirical noise distribution* — better
than any diffusion could reproduce — did not help. We therefore conclude that the synthetic bottleneck is
**not the noise model** but the **forward model**: a single typewell plus synthetic trajectories does not
reproduce the real joint GR↔TVT structure (lateral geological variation, typewell non-representativeness).
Closing that gap requires a genuine physical formation simulator (multi-typewell, faulted, laterally
heterogeneous), not a better noise generator — a substantially larger and less-defined undertaking than the
"learn the noise with a diffusion model" recipe suggests. This is, to our knowledge, a novel and useful
narrowing of *why* synthetic pretraining is hard on this task.

### 9.2 Blending WARP into the classical pipeline (offline-validated, since real-LB confirmed — §7.2)

WARP alone (11.0) is far weaker than the classical pipeline (~7.1), so we asked whether it is at least
*differently wrong* — i.e. whether its errors are decorrelated enough from the pipeline's to be worth
blending, the same logic that produced our one confirmed transferable gain (§7). We ran WARP inference
on the same 160 held-out wells used by our full-pipeline proxy (§7) and measured the error correlation
between WARP and the tuned `sp45`/GBM pipeline output: **only 0.52** — meaningfully decorrelated,
consistent with WARP's design (it deliberately does not trust GR the way the classical stack's GR
emission and alignment stages do). Blending `(1 − a)·sp45 + a·WARP` was evaluated with a proper
cross-fit (optimal weight fit on one half of the 160 wells, applied to the untouched other half, and
vice versa): the blend **improved pooled RMSE in both directions of the split** (proxy: sp45-only
10.62 → blended ≈ 9.8–10.2 depending on weight; never worse in either half). This is a materially larger
offline effect than the wall-hedge finding below.

We also tested combining this blend with a simpler "shrink toward last known TVT" hedge on
low-confidence wells, and found the hedge became **redundant once WARP is blended in** — the
jointly-optimal hedge weight collapsed to zero on both cross-fit directions, because WARP's own
continuity anchor already supplies that effect. The two should not be stacked.

**Caveat, stated plainly:** this result is measured on our full-pipeline *proxy*, not the actual
7.08 submission pipeline, so the magnitude of any real leaderboard gain is unknown — the proxy is a
weaker approximation of the real pipeline, and prior experience on this task (§13, lesson 1) is that
isolated/proxy wins do not always transfer. We built a submission that blends a WARP checkpoint into
the real pipeline at a conservative weight (0.15, versus the proxy-optimal 0.25–0.30). **This submission
has since been made and scored 6.882 (before the placement fix) / 6.846 (after) — a real gain over the
7.080 baseline** (§7.2). We report the offline finding above as originally written because the
*methodology* — measuring error correlation before blending, and cross-fit-validating the weight rather
than reading it off a single split — is itself a reusable contribution, independent of the fact that it
happened to also transfer here.

**A real-LB test of pipeline *placement*, and a correction to our first read of it.** We first placed this
WARP blend (and the physics post-process and `pf_z` blend of §7.1) after the pipeline's final blend but
*before* its train-contact override and its gold visible-prefix calibration stage — the same position we
had used for the simple wall-hedge (shrink toward last-known TVT on low-confidence wells, described just
above). We then submitted the wall-hedge variant on its own and it scored **worse than the unmodified
pipeline** on the real leaderboard (7.252), despite the hedge being honestly cross-fit validated as a real
gain on our proxy (§7, five independent splits agreed). Our first hypothesis was pipeline *placement*: the
gold-calibration stage runs *last* and makes its own per-well decision — by its own design comment, it
backtests a candidate correction on the visible prefix and only overrides the current submission when that
backtest says the candidate wins, treating "the current submission" as its anchor — so modifying that
anchor *before* gold runs, with a lever gold was never calibrated against, could plausibly confuse those
per-well decisions. We moved all three of our new stages — WARP blend, physics post-process, and `pf_z`
blend — to run **after** gold-calibration completes, and resubmitted every variant in both placements to
test this directly (§7.2).

**The placement-crossed real-LB test (§7.2) only partly bears this out.** WARP-blend and WARP+physics-pp
*did* improve slightly under the after-gold placement (6.882→6.846, 6.881→6.836) — consistent with the
placement mechanism above, and this is where our best submission (later refined to 6.794, §7.2) comes from.
But wall-hedge, tested in both placements, is *not* rescued by the fix — it scores 7.252 before and 7.304
after, i.e. no positive effect from reordering, if anything the reverse. **We therefore retract the strong
form of the claim above** ("this can flip a cross-fit-validated gain into a real regression"): placement is
a real, small, second-order lever for components that are already net-positive, but it was never the
dominant cause of the wall-hedge failure. The plainer explanation is that the wall-hedge mechanism itself —
an unconditional shrink toward the last known TVT — is harmful on the real hidden test regardless of where
it runs, a genuine proxy-to-real transfer failure independent of pipeline ordering. See §7.2 for the full
placement-crossed table and the parallel finding for the 3-lever stack.

We separately tested a more sophisticated alternative to the simple wall-hedge: a LightGBM meta-model
trained (via grouped cross-validation, no leakage) to predict the pipeline's absolute error from 16
GR-misfit and geometric features, then using its *out-of-fold* prediction to gate a per-point shrink
toward last-known TVT. This underperformed the plain constant-weight hedge (11.12 vs 11.10 on the same
honest split) — its correlation with the true error was only 0.24, weaker than the PF seed-spread signal
(+0.48, §11) already in hand. We record this as a clean negative result: added model complexity did not
buy a better uncertainty signal than the one we already had.

## 10. Leaderboard Context

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F34630893%2F4f930b03fe3995d0b2443ccff58d73ec%2Ffig09_lb_dist.png?generation=1783380737988329&alt=media)

**Figure 9 (real public LB, 4,065 teams).** The full public leaderboard. The flat baseline (15.9) sits near
the bulk's right shoulder; the median team scores ~9.9; **only 7 teams beat 6.0**, and the best is 5.26. Our
best submission, 6.794 (§7.2, up from an earlier 7.09), sits at the edge of that top group. The shape of
this distribution is itself evidence of the wall: the mass of the field is compressed into a narrow band a
little better than flat, and the leaders' separation is small in absolute feet — consistent with a task
whose irreducible floor is only ~1.5 ft below the best public score.

## 11. Uncertainty Estimation

Three usable uncertainty signals emerged. **(a)** The PF seed-spread correlates **+0.48** with the actual
error magnitude (Fig. 7) — we can predict *where* the model is uncertain (though not *which way* to correct,
§6). **(b)** The MDN head yields a full posterior per point (mixture means/variances/weights); its variance
flags the bimodal (±15 ft Eagle Ford) mode-ambiguity zones. Crucially, the bimodal tie-break is a coin flip
(mode correct 48.8%, r=0.054), so the honest uncertainty statement is that a large share of the error is
*aleatoric* — irreducible from the observations — which is itself a calibrated, useful conclusion. **(c)**
We trained a LightGBM classifier (5-fold CV) to detect, from leak-free per-well uncertainty features (PF
seed-spread aggregates, PF/beam disagreement, eval-zone length), whether a well will land in the worst
quintile by pooled error. It reaches **AUC 0.65** — real, ordered failure detection, on par with a similar
detection-vs-actuation measurement independently reported for this competition (AUC 0.69) — while, exactly
as in that measurement, the *sign* of the error stays unpredictable (§6, angle 5). This is the same
detection/actuation split stated more sharply: **we can rank wells by expected pain; we cannot tell which
way they will fail.**

## 12. Physical Meaningfulness

Our best components are physically grounded: the decomposition `TVT = surface − Z` is the geosteering datum
relation; the winning ensemble is variance reduction over a Bayesian sequential filter; the WARP model reads
the typewell as a "ruler" via cross-attention — the computational analogue of a geologist correlating logs.
The central negative result is itself physical: the eval-zone dip changes at sub-seismic faults whose
locations are not encoded in the observable GR, so the residual error is a property of the geology and the
measurement, not of modeling effort.

The identifiability argument in §5 (dip error as a constant of integration, growing unbounded over the
eval-zone lever arm) is not a coincidence specific to this dataset — it is a structural fact familiar from
navigation and inertial-observability theory: a *constant bias* state is unobservable from *rate/increment*
measurements alone (a classic dead-reckoning result), and only becomes observable given an external absolute
reference. Here, the GR log gives increments of rock character, not an absolute depth reference, so the
per-well datum offset plays exactly the role of an unobservable bias state — consistent with why every
matching approach we tried (§6, §8.2) can reduce *local* ambiguity but not resolve the *global* offset, and
why the only well-conditioned strategy is to never let that unobservable quantity accumulate error unchecked
(i.e., stay locally anchored, §5).

## 13. Lessons for the Community

1. **Isolated-component tests lie.** Denoise (+2.8% isolated → −0.24 LB) and the GBM blend (−6% isolated →
   +5% worse LB) and a 4-way ensemble (best holdout → worst LB) all looked good in isolation and lost on the
   real objective. Only full-pipeline / whole-well-holdout evidence transferred.
2. **Public LB ≈ seed noise** for the PF cluster; treat any change < ~0.07 as noise, and prefer
   variance-reduced (more-seed) submissions for private-split robustness.
3. **Know *where* the error is vs *which way* to fix it.** Seed-spread predicts error magnitude (+0.48);
   nothing predicts its sign. That distinction is the whole game.
4. **Variance reduction beat cleverness.** Our two robust real-LB gains both came from decorrelating the
   estimator (a diverse PF partner, then a decorrelated neural corrector) — not from new physics, features,
   or architectures.
5. **On a time-boxed code competition, the runtime budget is part of the model.** Parallelizing candidate
   generation (joblib) bought back the compute that a decorrelated ensemble and more seeds consume, which is
   what let our variance-reduction gain fit inside the 9-hour limit — an engineering lever as decisive as any
   modeling choice.
6. **On small geosteering datasets, weak-GR + strong-continuity beats explicit alignment**, and simple
   inductive biases (CNN + anchor) beat flexible ones (transformer).
7. **Corrections do not compose indefinitely, even when each is honestly validated.** Two independently
   cross-fit-validated, real-LB-confirmed corrections (a decorrelated neural net, a physics post-process)
   stacked cleanly (7.080 → 6.836); a third, equally well-validated correction (`pf_z`) broke the
   combination outright (→ 7.446–7.515), in a controlled test crossed against pipeline placement (§7.2) that
   ruled out ordering as the explanation. Validate the *whole stack* you intend to ship, not just its latest
   addition.

## 14. Reproducibility

Every number is from the whole-well holdout or the real public LB, never per-well averaging. Every figure in
this note is generated from the competition data, our 773-well method cache, or the public leaderboard CSV
by a single script (`make_figs.py`), and the caption states its source. The pipeline notebooks, the full
experiment log (`TESTS_LOG.md`), the NN research harness and roadmap (`NN_RESEARCH_PLAN.md`,
`NN_PLAN_V2.md`), the final NN analysis (`NN_FINAL_ANALYSIS.md`), and per-run results (`RESULTS.jsonl`) are
included. We build on the public PF pipeline and the diagnostic framing of the two traps (CV→LB mirage +
seed/refork variance) and the field-grouped "wall test" due to Georgy Mamarin, independently reproduced by
another competitor (wharekawa); our findings converge with theirs.

## 15. Limitations and Future Work

**Limitations.** (i) Our best submission builds on a public pipeline; our own contribution is the
variance-reduction ensemble and the analysis, not a new end-to-end model. (ii) The neural study, though
broad, was compute-bounded (a single modest GPU) — larger models / longer schedules might shift absolute
numbers, though the *relative* ceiling (~isolated-PF) was consistent across scales and architectures. (iii)
We did not achieve a working synthetic-pretraining pipeline; §9.1 shows *why* — the bottleneck is the
forward model, not the noise generator. (iv) **A claim we made, had to retract, and have since traced to
the mechanism level.** The inherited public pipeline contains a train-contact override that resolves a well
near-exactly (~0.01 RMSE) when its identifier also appears in the training set. Reasoning from the gate's
code alone (`if wid not in train_wells: continue`) and our understanding that hidden test wells should have
no ID overlap with training wells except through the shared typewells, we initially concluded this
mechanism must be inert on the graded set. **A real competition submission run then showed otherwise**:
its own strict audit log reports `guarded_contact_override` as an active postprocessor, with the override's
own diagnostic text explicitly describing coverage gained "on hidden overlap wells."

We have since gone further and traced *why* it is so precise, rather than leaving it as an unexplained
number. The override's physical-sounding reconstruction is `phys = ref_tvt − (Z − ref_col)` (a geological-
marker-plus-trajectory estimate that never touches any true label), but it is then shifted by
`offset = mean(TVT_true,train − phys)`, computed directly from the **train copy's own true TVT column** —
so the output is `phys + offset = phys + mean(TVT_true,train − phys)`, which cancels almost the entire
reconstruction error whenever that error is close to a per-well constant. This is why it resolves to
~0.01 ft: not because the physical reconstruction is that precise on its own, but because the correction
term is calibrated directly against the true answer for what is functionally the same wellbore. We also
found this is not a single, toggleable code path: the same substitution is implemented **redundantly in
three independent places** in the inherited pipeline — a standalone postprocessor, a candidate inside the
gold-calibration stage's own backtest pool, and a third function named (in the original code)
`_gold_reapply_guarded_contact_override`, called unconditionally after gold-calibration for every blend
profile. Disabling any one of the three leaves the other two intact; we had to patch all three to get a
clean reading. We measured the magnitude directly on the three wells in our local development stub (which
are verbatim copies of specific train wells, used only for local testing): with all three layers active,
pooled RMSE on those wells is **0.0053 ft**; with all three disabled, the same wells scored by our genuine
model stack (PF + beam + WARP-blend + physics post-process) give **6.47 ft** — a three-orders-of-magnitude
gap that leaves no ambiguity about what the mechanism does when it fires.

What this does *not* change: because the override's behavior has been identical and unmodified underneath
every submission we have made, it cannot explain the *relative* differences between our own variants
(denoise, the decorrelated ensemble, more seeds, wall-hedge, WARP-blend, physics-pp) — those deltas are
still honest, since the override runs constantly across all of them. What it *does* mean: our absolute LB
numbers are not purely attributable to the components we deliberately engineered, we do not know what
fraction of the real hidden test set (public or private) carries this ID overlap, and — unlike our first,
narrower framing — this is not merely "a mechanism of unclear magnitude," it is a train/test answer
substitution whose effect we have now quantified directly. We flag this as prominently as we know how,
rather than quietly softening the earlier text, because catching our own overclaim with real evidence —
twice now — is exactly the standard we are holding the rest of this note to.

**Future work (the path to the leaders' 5.2–5.4).** Synthetic pretraining remains the most likely route, but
§9.1 sharpens the target: a better *noise* model (e.g. a 1D diffusion / neural-SDE fit to the residual) is
**not** the missing piece — we showed that even the exact empirical noise fails to transfer. What is needed
is a better *forward* model: a physical formation simulator with multiple typewells, explicit faults, and
lateral heterogeneity, so that synthetic GR reproduces the real joint GR↔TVT ambiguity rather than a clean,
single-typewell inversion. Only then should one layer on realistic noise, bimodal ±15 ft jumps, and
miscalibration, and fine-tune on the 773 real wells behind a transfer gate (a synth-trained model must reach
~11 on the *real* holdout, not ~0 on a synthetic one). Complementary directions: distilling the *full*
7-pipeline (not the isolated PF) into a compact model; and combining the MDN head with a 2D misfit-SDF
backbone once realistic priors are in place. Our analysis suggests these are necessary because the raw GR
signal is broadband-unreliable — the gains must come from a better *prior*, learned from a genuine synthetic
*geology*, rather than from the observation likelihood or the noise distribution.

## 16. Conclusion

The ROGII task decomposes cleanly: the high-frequency TVT wiggle is free (it equals −Z, which is known at
eval time), and the entire difficulty is a smooth per-well surface trend whose slope-changes occur at
sub-seismic faults that the self-similar GR log cannot localize (broadband-irreducible error). This is why
the trivial flat baseline is so strong, why a weak-GR/strong-continuity prior wins, and why 60+ experiments
and 22 neural architectures — MDN, synthetic pretraining, 2D misfit-SDF, transformers — all cap at the
isolated-PF level (~11) while the tuned classical pipeline reaches ~7 through its full stack. Our first
concrete, transferable gain came not from new modeling but from *variance reduction* (a decorrelated PF
ensemble, 7.230 → 7.091). A second, larger real-LB gain then came from stacking a decorrelated neural
corrector with a robust physics post-process, **7.080 → 6.836**, further refined to **6.794** — our best
submission — by transferring a cross-validated hyperparameter back onto the real pipeline, paired with a
controlled real-LB negative result showing this stack does not extend to a third, individually-validated
correction (§7.2), and a correction to our own earlier causal claim about *why* the first such attempt
(wall-hedge) failed. We hope the map of *where the signal is and is not* is useful: on this task, measure
twice on a whole-well holdout, distrust the public LB, separate the recoverable (the −Z wiggle) from the
physically unrecoverable (the cross-field surface trend) — and even once a gain is real-LB-confirmed, test
whether it still holds before stacking one more thing on top of it.

---

*Acknowledgements:* public PF pipeline and the two-traps / wall-test diagnostic framing by Georgy Mamarin;
independent reproduction by wharekawa; leaderboard context from Rishikesh Jani and Tucker Arrants.

---

### Appendix A. Experiment index

For quick reference, the full list of experiments referenced above, by section:

- **§8.1 Baselines (3):** flat; linear extrapolation; pooled-RMSE methodology fix.
- **§8.2 GR matching (9):** Pearson window; banded tracker; fixed-anchor; heel affine calib; NCC-in-PF;
  shape/derivative; affine-in-beam; full bidirectional DP; trajectory-informed DP.
- **§8.3 Particle filter (10):** affine-in-likelihood; resample threshold; adaptive scale ×2; lo-noise
  ensemble ✅; gs-decorrelation; Student-t noise; typewell-jitter; 4-way kitchen-sink; runtime-guard split.
- **§8.4 Post-processing (5):** Savgol/poly; deg-2/deg-4; piecewise-linear; physical clipping; bimodal hedge.
- **§8.5 GBM/ML (8):** GBM standalone; PF+GBM isolated ✅; PF+GBM on LB ❌; meta-stacker; Q-3D tortuosity;
  signed azimuth; confidence blend; blend-weight sweep.
- **§8.6 Denoise/external (3):** rotation-band denoise; toolkit mining; lineage analysis.
- **§8.7 Human markup (3):** dip-snapping; PL reconstruction; formation-intersection kinks.
- **§7.2 Real-LB stacking/placement confirmation (2):** placement-crossed WARP+physics-pp resubmission ✅
  (new best 6.836); placement-crossed wall-hedge and 3-lever full-stack resubmission ❌ (both harmful
  regardless of placement).
- **§7.2/§7.3 Overfitting and validation-honesty checks (5):** 5-fold grouped CV of the new best (5/5 folds
  positive, identical hyperparameters every fold) ✅; real-LB transfer of the CV-selected WARP-weight
  (0.15→0.30), a further gain to our current best 6.794 ✅; hedge-on-bare-base vs hedge-on-tuned-combo
  cross-fit, confirming the hedge collapses to weight 0 once WARP+physics-pp are applied ✅; LSO spatial
  K-means-block holdout, confirming the gain is not neighbour leakage ✅; isolation-quartile check,
  confirming isolated wells benefit *more*, not less, from the new combo ✅.
- **§9 Neural nets (20):** see the architecture table.

**Total: 60+ distinct experiments.**
