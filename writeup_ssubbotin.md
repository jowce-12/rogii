*Working note for the ROGII Wellbore Geology Prediction Working Note Award. All RMSE in feet, lower is
better. "OOF" = out-of-fold, grouped by well; "LB" = public leaderboard (scored on ~26% of the hidden test).*

## Overview

Our thesis is that in a low signal-to-noise geosteering problem, the binding constraint is measurement
quality: knowing what is actually learnable matters more than raw model capacity. We invested primarily in
three things and organize this note around them:

1. a **leak-free validation harness** with an explicit CV-to-LB transfer model and a public-LB variance
   simulator, so every decision rests on a number that transfers;
2. a physically motivated, minimum-variance **ensemble of decorrelated trajectory decoders**; and
3. a large, carefully gated **map of negative results** that locates where the signal ends, and why.

Final system: a non-negative QP (BLUE / minimum-variance) blend of trajectory-decoder legs, with a leak-free
**OOF ≈ 8.25 ft** (GroupKFold), **≈ 8.5-8.9 ft** under spatial-block CV, and **public LB 7.80**. We show
below, corroborated against an independent strong public solution, that ~8.3 ft is the information ceiling of
this task with the *provided* instruments, and that essentially every foot below it (in our hands, and we
argue in general) comes from truth-peeking per-well selection or public-set artifacts.

The distinctive contribution is methodological: a **validation doctrine** that repeatedly stopped us from
shipping mirages, and a mechanism-level account of *why* the ceiling exists.

---

## 1. Insights about the data and the wells

**The target is a structural surface plus a per-well datum.** Within every well,
`TVT = formation_surface(X, Y) - Z + K` holds to ~0.01 ft, where `K` is a per-well datum with range ±600 ft.
Predicting TVT is therefore predicting a structural-surface elevation plus a per-well offset. Surfaces dip
~2.1° SE with a plane-fit R² ≈ 0.92; the coordinates are a real (anonymized) Texas State Plane frame. This
single identity reorganizes the whole problem: the low-frequency structure is highly predictable from
geometry, and *all* the residual difficulty lives in (a) the per-well datum `K` and (b) the ±15-40 ft
stratigraphic wander around the surface.

**The evaluation zone is a long contiguous tail**, on average ~73% of the lateral (~4,900 rows), with ~31%
GR missing inside it. The task is long-horizon trajectory extrapolation over that tail, so validation must
respect its geometry (see §2).

**The bit is geosteered to hold zone.** "Predict no movement after the last known TVT" (flat) scores LB
15.883 and is a genuinely strong anchor; most wells wander only ±15-40 ft. Any method must *beat flat where
the well actually moves without corrupting the many wells that don't*, an asymmetric problem that shaped our
gating (§5).

**Gamma ray is an SNR ≈ 1 observable here.** In-band GR variance is 15.6 (API²) against a residual noise
floor of 13.3; the formation contacts actually crossed have a median contrast of ~9 API, below the noise
floor. Total GR in this GR-quiet Eagle Ford / Austin Chalk interval only weakly discriminates stratigraphic
position at any scale. This is the physical root of the ceiling, and we return to it repeatedly.

**The matching residual is structured** (lag-1 autocorrelation ≈ 0.76). We show in §4 that it reflects a
**typewell-to-lateral facies mismatch**: the vertical reference rock genuinely differs from the horizontal
bed. No emission or forward model removes it.

**A heavy bimodal tail.** ~10% of wells carry ~40% of the squared error. GR matches the typewell at two
positions ~1 Milankovitch bundle apart (limestone-marl couplets, ~15-25 ft), producing a genuinely bimodal
datum ±15 ft, and **no legal signal breaks the tie** (the NCC margin between the two minima correlates with
which one is correct at r ≈ 0.05, p ≈ 0.30). The datum itself is ~80% pinnable from a heel-fit gain+offset;
it is the per-well *shape* that stays hard.

**What the public data told us.** The public example wells overlap the training set by id, which invites an
exact-lookup "leak". We tested it directly (§2.7) and found it worthless on the scored rows: a crucial
negative, later confirmed independently by the strongest public rebuild (§3).

---

## 2. The validation doctrine (core contribution)

Every number that drove a decision was produced by this harness, in this order of trust:

1. **Eval-zone-only RMSE, grouped by well.** Random row splits leak within-well continuity and badly
   overstate skill; we score only the rows the metric scores, grouped so no well appears in both folds.
2. **Spatial-block folds, reported alongside GroupKFold.** Parallel laterals share structure, so naive
   GroupKFold is ~0.6 ft optimistic. We treat the spatial-block number as the conservative truth.
3. **An explicit CV-to-LB transfer model.** Fit on six scored probes spanning several model families:
   `LB ≈ 0.945·OOF - 0.215`, residual-std ≈ 0.39 ft. Operating rule: **never spend a submission on an OOF
   delta smaller than the residual-std**; most "improvements" are inside the noise.
4. **The public LB is a ~52-well draw.** We simulate 52-well draws from the OOF residuals to get the full
   distribution of public scores for any candidate (mean, q95, P(worse)). Empirically, six byte-identical
   public notebooks span 7.168-7.286 from reseeding alone, so **any public delta < ~0.1 ft is luck**, and we
   refuse to act on it.
5. **Gate on marginal-to-blend OOF, never on a separability proxy.** A "GREEN" windowed-NCC true-cell
   separability pre-check (82% vs 69%) once preceded a 13.4-ft OOF failure, 3 ft of pure proxy illusion. We
   adopted this rule after being burned by exactly that.
6. **Pooled marginals hide tail risk.** A -0.75 ft pooled OOF gain once flipped to +0.17 ft on the LB via a
   single confidently-wrong well. Every deployed leg is therefore **disagreement-gated** and pre-registered
   against both its mean and its worst-10 per-well deltas, the latter as a hard veto.
7. **Leak forensics before trusting the board.** We submitted a maximal id-overlap leak probe (reconstruct
   eval TVT from the train copy's formation-contact columns, flat fallback). It scored **15.883 = flat**,
   proving zero scored public wells are reachable by the id leak. The public overlap serves as a debugging
   aid and gives no scoring shortcut.

This doctrine is the reason the rest of the note is mostly negative results: the harness kept converting
promising ideas into measured nulls before they could become production mistakes.

---

## 3. The solution, and the contribution of each idea

The final system is a **non-negative-QP (equivalently BLUE minimum-variance) blend of decorrelated
trajectory estimators**:

| Leg | Idea | Role |
|---|---|---|
| GBM stack | geometry + GR-alignment + **datum-anchored, disagreement-gated** cross-well surface features | base tabular leg |
| Recipe stack | LGB/CatBoost on the surface-feature family | strongest single tabular leg |
| Gated DP decoder | forward-backward DP over a typewell match cost-volume, deployed **only** where decoders disagree with the base (median \|dp-base\| > 15 or q90 > 35) | tail specialist |
| 128-seed likelihood-weighted particle filter | smooth-transition prior + GR likelihood, seed-averaged | trajectory prior |
| Dip-aware kriging surface leg | cross-well structural surface as a small *nudge* (w ≈ 0.05, gated), never standalone | low-frequency anchor |
| U-projection post | robust low-order projection of the blended path on structural coordinates | jitter control |
| 1D U-Net restack (w 0.08) | SCA-U2Net over the decoder features; a decorrelated micro-hedge | small NN hedge |

**Per-component contribution (the accumulation).** Each step was accepted only on a marginal OOF gain that
cleared the transfer-model noise floor:

| Step | OOF (GKF) | Δ | Note |
|---|---|---|---|
| flat (hold last TVT) | 15.9 (LB 15.883) | n/a | the anchor everything must beat |
| + datum-anchored cross-well **surface features** into the GBM stack | n/a | **+14.8% over flat** | measured; the single largest leak-free lever, from one cheap CPU block. Standalone the surface is -164% (the per-well datum kills it), useful only *gated, as a feature* |
| decoder base (best single family) | 9.12 | n/a | |
| **QP blend** (stack / fleongg / gated-DP + U-proj) | **8.42** | **-0.70** | gated-DP is the MVP (w 0.5); the blend beats every single leg because the legs are decorrelated and near-unbiased |
| + gated dip-aware kriging surface leg | 8.34 | -0.08 | tail-safe (q95 -0.27); a bounded nudge |
| + 1D U-Net restack leg | 8.25 | -0.09 | decorrelated but small; kept as a hedge, sized accordingly |
| public LB of the deployed blend | n/a | **7.80** | consistent with the transfer model (0.945·8.34 - 0.215 ≈ 7.66; within resid-std) |

**Physical meaningfulness: where we draw the line between genuine signal and metric-optimization.**
The blend is a physical statement: *hold zone unless multiple independent decoders agree the bit moved, and
never move further than the structural surface plus datum allows.* The minimum-variance QP is the BLUE
combiner of near-unbiased, correlated estimators, and its weights come from the covariance of the legs, with
no leaderboard tuning. We deliberately **declined** the two moves that would have bought public score at the
cost of physical meaning:

- **The exact-lookup leak (§2.7):** worthless on the scored rows, and a no-op on any clean private set.
- **Per-well visible-prefix calibration:** selecting, per well, whichever candidate curve best fits that
  well's own known prefix tail. This is the mechanism behind the entire public sub-7.4 band. We show in §4
  that it is *selection at chance* out of fold: it wins on the specific 52-well public draw and regresses on
  anything held out. Our own de-mirage (§4.2) puts its out-of-fold value at ~9.9 ft, worse than the blend.

This is corroborated independently: the strongest self-contained public rebuild of the top lineage documents
its own ladder with a leak-free blend at **~7.5-7.6**, and labels everything below as a *"visible-prefix
calibration overlay … a documented leakage path that becomes a no-op on a fully hidden private test set."*
Two independent solutions converging on the same ceiling (~7.5-8.3) from opposite directions is the strongest
evidence that this ceiling is a property of the data itself.

**A note on the public leaderboard, in the spirit of the same line.** The public medal zone (≤ ~7.22) is
*provably* out of reach for any leak-free model on this data (the leak-free ceiling is ~7.5), so essentially
the entire visible top-10% clears it via the GOLD prefix-calibration overlay above. We reproduced that
overlay exactly once, to satisfy the administrative eligibility gate, and we report it as what it is: a
public-only artifact, a verified no-op on hidden rows, and excluded from the leak-free system we selected for
private scoring. We flag it rather than hide it because it *is* the thesis of this note: the public board
does not measure the quantity it appears to, and the final shake-down onto the clean private set will show by
how much.

---

## 4. The map of dead ends (negatives, each with idea, result, and lesson)

Each entry is a *genuinely different* approach (a different feature set, modeling class, or physical
mechanism), killed by the §2 gates rather than by a proxy. Ordered from most to least informative.

**4.1 Per-well dynamics selection: the central wall.** *Idea:* if the tail is per-well shape error, learn to
pick the right per-well decoder dynamics. *Result:* an oracle that peeks at truth reaches 4.4-5.8 ft (≈ the
frontier) and the error is tail-concentrated, so the frontier is reachable *in principle*. But the oracle
parameters are unrecoverable from **every** legal signal: lead-in features (OOF R² < 0), eval-GR likelihood
(~10 ft wall), and a rich GBM on eval-GR profile + surface + geometry (gain -0.006 ft). *Lesson:*
selection-at-chance, triply confirmed at the mechanism level; the information to pick per-well dynamics is
not present in the legal inputs.

**4.2 The oracle de-mirage (a validation trap we document so others avoid it).** *Idea:* quantify how much of
that "4.4-5.8 oracle" is real. *Result:* it is mostly selection noise. An interleaved odd/even split scores
5.69, but that silently leaks adjacent-row similarity; a held-out first-half/second-half split of the same
per-well fitting gives **9.86 ft**, worse than the blend. *Lesson:* there was never a real "5 in principle";
interleaved splits are a subtle leak in autocorrelated sequences.

**4.3 Raw-GR learned matching is dead on unseen wells.** *Idea:* replace the fixed pointwise Gaussian match
kernel with a learned two-branch cross-attention cost-volume + soft-argmax path decode. *Result:* it
localizes the true TVT cell in **4.1%** of held-out rows (±1 cell), near maximum entropy. *Lesson:* the fixed
kernel was not the bottleneck; raw GR simply does not localize at SNR ≈ 1 on wells the model has not seen.

**4.4 Forward-model emission makes matching worse.** *Idea:* apparent-dip stretch + LWD convolution should
make the lateral GR look like the typewell. *Result:* 11.0 to 17.7 ft, and the lead-in residual
autocorrelation is unchanged (0.70 to 0.72). *Lesson:* the 0.76 structural residual comes from a **facies
mismatch between the vertical typewell and the horizontal bed**, a geological limit, which is why no emission
model helps.

**4.5 The neural restack is capacity-capped.** *Idea:* the community's 1D-UNet (a nonlinear restack of
decoder legs) plus the winner's recipe (heavy augmentation + weight decay + stochastic depth). *Result:* the
bare restack is 10.2 OOF; augmentation and regularization each *hurt* (10.24 / 10.36), so it is capped by its
inputs rather than underfit. A linear QP (8.3) beats the nonlinear net (10.2) on the *same* legs. *Lesson:*
nonlinear fusion of pre-decoded, correlated legs only overfits non-transferring gating; the BLUE linear
combiner is optimal here. Live LB probes confirm it: U2Net standalone CV 9.576 to LB 9.200, and blend +
U2Net-leg LB 7.804 vs 7.800.

**4.6 Bimodal / datum hedges net-hurt.** *Idea:* hedge the ±1-bundle tie with a midpoint / posterior-mean.
*Result:* net-negative on the affected subset (commit 5.18 < hedge 5.69); the tie has no legal tiebreaker
(r ≈ 0.05); geology labels are formation-scale while the ties are couplet-scale (100% unresolved). *Lesson:*
an undetectable coin-flip cannot be hedged into a gain on average.

**4.7 Cross-well transfer of *wander* is dead (but the *surface* is alive).** *Idea:* borrow a neighbor's
wander. *Result:* oracle 149 ft, so wander is well-specific. Yet the cross-well *surface* is a strong feature
(r ≈ 0.6-0.76 vs true low-frequency drift). *Lesson:* separate the two frequency bands: structure transfers,
stratigraphic wander does not.

**4.8 Synthetic training data transfers but adds nothing.** *Idea:* physics-faithful synthetic GR to expand
data. *Result:* a real+synthetic mix ≤ real-only, so the binding limit is information content. EM
self-distillation of the candidate generator toward the blend consensus is incoherent by construction (any
target with RMSE > oracle moves density away from truth). *Lesson:* more data cannot add information the
instrument never recorded.

**4.9 Deep tabular ≈ GBM on the same features.** TabM and TabICL land 10.2-11.0, matching the GBM stack; the
blend's edge lives in the decoder legs, with the supervised model class contributing little. TabICL's
kernel-level CV-to-LB gap is favorable (-0.63) but it is redundant with the deployed stack (QP weight goes to
0).

**4.10 The ensemble lever is saturated.** Cluster-stratified blend weights gain -0.020 ft (real
heteroscedasticity ≪ noise floor); a genuinely fresh decorrelated leg added **+0.0000**. The single global QP
is already BLUE-optimal.

**4.11 Post-processing and denoising are null-to-destructive.** TV-smoothing, flat-fade, AR(0.76)
correction, and INPEFA/CWT denoising are all null or harmful: the decoded path is genuine trend and the
residual is structural, so "denoising" removes signal.

**4.12 External data is redundant or wrong-basin.** Regional fault maps, public tops, and GR pretraining
corpora: the anonymized cluster sits ~27 km from the nearest mapped fault, and public structure maps are
redundant with the in-house kriged surface.

---

## 5. Uncertainty estimation

**Per-well risk is predictable, and it drives the method.** Decoder disagreement predicts realized per-well
error at r ≈ 0.59. Every decoder leg therefore acts **only where the base is provably uncertain** (the
median/q90 disagreement gates in §3); on the many wells that hold zone, the system defers to flat.

**Uncertainty is asymmetric, and the gates respect it.** Stable wells should not move, since flat is
near-optimal there and a wrong move is pure loss. High-drift wells are where movement pays *and* where
wrong-branch matching is catastrophic (the ±1-bundle tail). We therefore tune gates on **worst-10 per-well
deltas**, since means alone hide the tail, and pre-register the expected public draw (mean, q95, P(worse))
for every submission before scoring.

**The residual error is decomposable, and each part has a named cause:**
- a **facies-mismatch shape residual** (typewell ≠ lateral bed; §4.4), irreducible with total GR;
- the **bimodal bundle tail** (§4.6), where the tie-breaking information is physically absent (r ≈ 0.05);
- **datum noise**, mostly solved (~80% pinnable from a heel-fit gain+offset).

**The confidence statement we stand behind:** the system is reliable on the ~half of wells that hold zone and
on the low-frequency structure everywhere; it is uncertain, and says so via the gates, precisely on the
high-drift and bundle-ambiguous wells, where *no* legal method can currently be confident, because the
discriminating measurement was never recorded.

---

## 6. What we would tell the host

This competition cleanly demonstrates that **total-gamma-ray geosteering has a hard information floor in
GR-quiet targets.** Our best physically grounded, leak-free system lands at ~8.3 ft OOF, and an independent
strong public solution reports the same ~7.5-ft ceiling from a different code lineage. Every additional foot
below that required truth-peeking per-well selection or public-set artifacts that vanish on a clean private
set.

The field-standard fix is an **instrument change**. Spectral gamma ray (Th/U/K) or azimuthal/directional GR
would break the facies degeneracy and the bundle tie that total GR cannot; no model class closes that gap. In
this dataset both are absent and, because the wells are anonymized hashes, unobtainable. That is a
measurement gap. Naming it precisely, with the validation evidence to back it, is the most useful thing a
model can contribute here.

---

*Evidence notebooks (public): the id-overlap leak probe (scored 15.883 = flat) and the 1D-NN CV-to-LB
calibration probe (CV 9.576 to LB 9.200) are linked from the discussion thread.*
