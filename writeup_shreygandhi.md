# Working Note: Scoring Sequences, using joint probability

ROGII - Wellbore Geology Prediction · Public LB: ≈5.7 RMSE
Author: Shrey Gandhi

**Contents**

1.1 A joint distribution over `(h_len, ΔTVT)`, the backbone model
1.2 Learning the likelihood term: what neural encoders taught me about the ambiguity
1.3 Independent evidence: combining two mathematically different estimators
1.4 A structural reference from geology: a real oracle, and why it stays out of reach
1.5 Shorter negative results
2. What I learned about the data and the wells
3. Physical meaningfulness: every claim, its reasoning, and its evidence
4. What each piece contributed
5. Uncertainty: what the model knows about itself
6. Closing
7. Terms used in this note

Every well here comes with the same three things: a measured-depth path (how far the drill bit has traveled along the wellbore, as opposed to true vertical depth), a gamma-ray (GR) log along that path (a reading of natural rock radioactivity that varies with rock type, so its shape acts as a fingerprint of what depth you're passing through), and a vertical typewell nearby that already pairs GR against true depth, since it was drilled and logged earlier and its depths are fully known. Past a known point, the true depth, TVT, stops being given, and the job is to keep predicting it along the **lateral**, the long horizontal section of the well that runs out from that known point. The natural way to do that is one piece at a time: a length along the path, `h_len`, and the change in TVT over that length, `ΔTVT`. Chain enough `(h_len, ΔTVT)` pairs together from the last known point and you have a full candidate trajectory. Which sequence of strides is the real one, given only the GR log and the typewell, is the question this whole note answers. I call the overall system STRIDE, short for Segment-level TRajectory Inference with Decorrelated Evidence, since every approach below is really an attempt at that same question from a different angle.

Every number below is a pooled RMSE of dTVT in feet, the Kaggle metric, on the 773 training wells decoded leak-free as hidden laterals, unless I say otherwise. A second check, holding out whole geographic blocks of wells at once rather than one well at a time, caught more than one win that didn't survive outside its own neighborhood. Where a number is an **oracle** rather than a real prediction, I say so, and use it only to measure how much headroom a direction has, never as a result on its own.

## 1. Approaches I explored

### 1.1 A joint distribution over `(h_len, ΔTVT)` (the backbone)

**Idea.** Matching a window of GR and moving on, one segment at a time, throws away exactly the evidence that resolves a hard case: what the rest of the sequence looks like. So I don't pick segments one at a time. I treat the whole sequence `(h_len_1..n, ΔTVT_1..n)` as a draw from one joint distribution, conditioned on both logs:

```
P(h_len_1..n, ΔTVT_1..n | GR_lateral, GR_typewell)
   ∝  p(h_len_1, ΔTVT_1) · Π[i=2..n] p(h_len_i, ΔTVT_i | h_len_i-1, ΔTVT_i-1)   <- prior
      × Π[i=1..n] L(GR_lateral,i | h_len_1..i, ΔTVT_1..i, GR_typewell)         <- likelihood
```

Three ingredients decide it. A prior on segment length, fit as a log-normal shape from how real wells actually behave. A persistence prior, which says each segment's `ΔTVT`, once you scale for how long it ran, stays close to the one before it, because a well doesn't change direction on a dime. And a likelihood: how well a candidate's implied GR (read off the typewell at that candidate's depth) matches the GR actually observed, scored with a heavy-tailed Cauchy distance rather than squared error, so one stretch of genuinely good agreement can't be out-argued by a repeated bed, a rock layer that recurs at a different depth and looks the same on the GR log, happening to fit just as well somewhere else.

One more ingredient ties the whole thing to a trustworthy absolute depth, not just a locally consistent shape: a **typewell reference term**. A typewell's GR readings and this well's own GR readings can sit at slightly different levels even for identical rock, because each logging run has its own instrument offset. Before decoding, I nudge the typewell's GR-versus-depth curve toward this well's own observed GR, using only the known section where both are available at the same true depths. Without that adjustment, the likelihood above would be comparing this well's real log against a reference that might be running systematically high or low, which can drag a candidate's whole depth register off even when its shape matches well.

I find the most likely sequence with a search over the full lattice of `(h_len, ΔTVT)` pairs at every step, scoring each extension by the prior times the likelihood and keeping the best survivors.

**Validation.** Figure 1 is what that joint distribution actually looks like for one real step on one real well, `8f201368`: the real fitted prior and the real GR likelihood, over the whole `(h_len, ΔTVT)` lattice.

**Figure 1.** *The joint posterior over `(h_len, ΔTVT)`, one real step out of the known section.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2Fc37c9437cf84f94cdbd43916f4695684%2Ffig01_joint_posterior.png?generation=1783364782594764&alt=media)![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F47ce99a11d234c9be918bc5fb2fee32c%2Ffig02_well_case_study.png?generation=1783364824464720&alt=media)

The bright band near `ΔTVT=0` is the prior voting with no committed direction yet. Ground truth (dashed) climbs steadily away from that band as `h_len` grows, and the GR evidence at those longer segments isn't strong enough alone to pull the distribution back toward it. At short `h_len` the truth and the wrong, flatter path sit inside the same high-probability region, genuinely overlapping, not two clean separate peaks; the two only pull apart once `h_len` is large enough that the model has already had to commit. The single best point in the whole lattice sits at a short segment near flat: a locally safe, wrong-direction pick, and the seed of a **cycle-skip**, where the decode locks onto a lookalike, repeated rock layer instead of the true one and stays offset by roughly that same amount for a long stretch afterward.

It doesn't stay wrong, because the object being optimized is the whole sequence, not this one step. Figure 2 shows the same well decoded end to end: the early miss at 51.19 ft self-corrects to 3.58 ft, as later segments' evidence and the same prior, now acting on an already-decoded path, pull the sequence back toward the truth. The figure's middle line, labeled **consensus**, is a simple checkpoint: an intermediate reading taken by reconciling a few internal variants of the decode with each other partway through, before the rest of the sequence is even finished. It sits between the early miss and the final answer, which is exactly what it's for, a sanity check on the way to a decision, not a decision itself.

**Figure 2.** *The same well, decoded end to end: an early miss self-correcting over the full sequence.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F9ec52556b8152a0b2bb98ec132501f2b%2Ffig02_well_case_study.png?generation=1783364835301181&alt=media)

How much of what's left is even there to find, rather than lost to the kind of ambiguity Figure 1 shows? A deliberate oracle answers this: fit a low-order piecewise curve directly to the true per-row residual, something only possible with the ground truth in hand. It reaches 0.80 ft pooled RMSE, well below anything realizable without the answer, and that gap is the whole point: it proves the remaining error has real, structured shape, even though a distribution built only from GR and priors can't resolve it from what the log carries.

**Figure 3.** *An oracle check: is the remaining error structured, or just noise?*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F1fa5f2581c0313e26be64d8f856b9027%2Ffig03_oracle_locatability.png?generation=1783364901247346&alt=media)

**Lesson.** A prior built from how `ΔTVT` actually behaves segment to segment, plus a decode that scores whole sequences instead of single segments, does most of the work here. What's left concentrates in one kind of decision, which `(h_len, ΔTVT)` to commit to at a hard transition, and the 0.80 ft oracle says that decision is structured but not resolvable from the log alone. That sent me looking for a sharper likelihood term (1.2), then for genuinely independent evidence (1.3).

### 1.2 Learning the likelihood term directly: what neural encoders taught me

**Idea.** The likelihood term in 1.1 is a closed-form Cauchy distance. A network scoring the same `(h_len, ΔTVT)` candidates might learn a sharper version, one that catches subtle rock character the closed form misses. I tried four framings, each a direct response to what the last one taught me.

A contrastive embedding (InfoNCE) first: pull the representation of a matching lateral window and typewell window together, push random negatives apart. This under-delivered for a specific, informative reason. Random negatives are almost always trivially different rock character, while the one negative that actually matters, a repeated bed producing the second option Figure 1 shows, essentially never appears in training. The representation learned broad texture, not the specific ambiguity that costs points, because it was never shown that ambiguity.

So, second, I stripped the contrastive framing out and regressed `ΔTVT` directly from a single window. This was viable but no more informative than the likelihood it was meant to sharpen. That told me the negative-sampling framing was the weak link, not regression itself.

Third, I matched 1.1's structure exactly: instead of an embedding distance, predict the endpoint error of a specific `(h_len, ΔTVT)` candidate on the same lattice the search already scores. This version tracks true TVT closely on well-conditioned stretches of log, and drifts precisely where the log loses local character.

**Figure 4.** *The error-regression network tracking true TVT closely, then drifting where the log loses local character.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2Faccd46fe39cb753d86787e40a3cdec2e%2Ffig04_nn_encoder_track.png?generation=1783364916227460&alt=media)

Fourth, a DAgger-style loop: roll out the current search on training wells, harvest the candidates it visits and scores badly, and retrain on exactly those, so the training data matches the ambiguous cases in Figure 1 instead of a uniform sample over the lattice.

None of the four ever beat the closed-form Cauchy likelihood from 1.1 once evaluated honestly across many wells rather than a handful of diagnostic cases.

**Lesson.** Each step moved the learned function closer to the likelihood term it was meant to replace, not toward "understanding more geology." A sharper scorer cannot manufacture certainty about a candidate the GR genuinely does not distinguish from the truth: that is a statement about how much information the log carries, not about model capacity. I kept the closed-form likelihood and put further effort into the priors instead of a fifth architecture.

### 1.3 Independent evidence: combining two mathematically different estimators

**Idea.** If sharpening the likelihood term cannot resolve genuine ambiguity (1.1, 1.2), the mathematics of combining forecasts points to a different lever. Two estimators with error correlation `ρ` can be blended to a lower error than either alone, and the achievable gain grows as `ρ` falls toward zero. So rather than build a better version of the same reasoning, I looked for a second estimator that reasons about the log through genuinely different machinery.

`track6` is that estimator. It runs its own particle filter, a swarm of hundreds of candidate trajectories propagated forward and reweighted by GR likelihood at every step, resampling as evidence accumulates, alongside independent correlation and nearest-neighbor features, all feeding a small gradient-boosted ensemble (a blend of decision-tree models trained one after another, each correcting the last one's mistakes). Physically and mathematically its particle filter is a different inference process from 1.1's discrete lattice search. A particle filter fails by degeneracy, its swarm collapsing onto too few surviving hypotheses; a lattice search fails by committing early to the wrong discrete bin. Different failure mechanisms are exactly what a low error correlation requires.

**Validation.** That prediction held. `track6` alone scores worse than the backbone (mean-per-well RMSE 8.69 vs. 7.28, leave-one-well-out, meaning each well is scored using a model trained on every other well), which is expected of a second, differently-built estimator, but its errors correlate with the backbone's at only 0.32, confirming the two really do fail differently rather than agreeing on the same wrong candidate. Blending them with a fixed weight already recovers real ground. Two further checks show how much more the correlation structure allows: a blend weight chosen with hindsight, separately for each well, reaches an oracle of 5.11 ft; adding a third, differently-built estimator (`cvt1d`, a compact convolutional model that was never finished into a trained production model) pushes a three-way oracle to 4.76 ft.

**Figure 5.** *Realized performance against the ensemble's oracle ceiling.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F380ecb05612b6f64d6c5931477668d5f%2Ffig05_oracle_ensemble.png?generation=1783364932508029&alt=media)

**Lesson.** A single fixed blend weight is the right answer only if every well shared the same error behavior. They don't, which is exactly why a hindsight, per-well weight beats a fixed one, and exactly why the gain between 5.11 and 4.76 shrinks quickly: a third estimator built from the same GR log runs into the same underlying ambiguity, so independence has a ceiling too. `track6`'s own particle filter and correlation stack are expensive to run in full, so the deployed pipeline now leans on a leaner successor, `track8`, which keeps the same gradient-boosted trainer but swaps a self-built particle filter for the output of `sp45`, a separate, dedicated particle-filter-and-beam engine, delivering the same kind of independent evidence at lower cost. Closing the gap between a fixed weight and the per-well oracle needs a real, computable per-well signal, which section 5 investigates directly.

### 1.4 A structural reference from geology: a real oracle, and why it stays out of reach

**Idea.** Every training well also carries its named formation tops, the depths where the well crosses from one named rock layer into the next. If a candidate trajectory's stratigraphic position could be tied to the formation it actually sits in, that tie would supply a completely different kind of evidence from 1.1's likelihood term: not a better GR match, but an independent physical anchor that does not care how the log happens to look. Before building anything, I measured what that anchor could be worth.

**Validation.** As an oracle, correcting each well's answer using its true formation identity takes the backbone from 10.68 to 7.36 ft pooled on the wells this was tested on, real, substantial headroom. But three separate checks show that headroom is not reachable from what a hidden lateral actually provides. Formation offset barely correlates with the backbone's own bias (r ≈ -0.015). The lateral's own GR matches its true target formation only 17% of the time, even though the backbone's decode already gets the formation right 99% of the time by a completely different route, structural position, not GR character, meaning GR shape and formation identity are only loosely linked at this resolution. The physical reason underneath both checks: the hidden lateral typically spans a median of 26 ft of TVT, while a single formation runs 40 to 116 ft thick. The lateral almost never leaves the formation it started in, so formation identity is nearly constant exactly where the error is not, and carries no information at the scale the residual actually varies on.

**Lesson.** A second kind of structural information can be real, and still not be the fix, if it resolves at the wrong scale. This is 1.1's ambiguity check from a different angle: information that would break a tie has to vary at the same resolution as the thing being predicted, and formation-level geology does not.

### 1.5 Shorter negative results

Routing each well to a different pipeline configuration by simple observable properties (lateral length, geometry, log character) landed near chance, for the same underlying reason discrete candidate selection failed in 1.2: nothing available before decoding distinguishes a well that needs one configuration from a well that needs another, because the distinguishing information would have to come from the same ambiguous log both configurations already see.

## 2. What I learned about the data and the wells

The GR log gives with one hand and takes with the other. It pins depth to well under a foot when the beds are distinct, but Figure 1's second, off-truth option is common enough across wells that a squared-error likelihood would chase it routinely. That fact governs the rest of the problem: it is the reason 1.1 needs a robust likelihood and a strong prior at all, rather than trusting GR matches at face value.

Error does not blow up where a single segment goes wrong; it propagates. How far a segment's own `(h_len, ΔTVT)` choice misses the truth barely affects that segment's own error, but it adds to a running total that predicts trouble much further along the well (Spearman 0.45 between cumulative upstream depth-change error and a later segment's RMSE). Squared-error mass follows the same pattern, growing steadily from heel (near the known point) to toe (the far end of the lateral), because a wrong early commitment compounds forward rather than being corrected in place.

**Figure 6.** *Depth-change error upstream predicts trouble downstream; error mass grows toward the toe.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2Ff74c024612e4b18445e17183c80b8827%2Ffig06_propagation.png?generation=1783364955708566&alt=media)

What actually makes a well hard, checked directly rather than assumed? I tested the two most obvious candidates against the real outcome for all 773 wells. A well whose rate of depth change shifts abruptly between adjacent segments, the kind of transition that seeds a cycle-skip in 1.1, is a real but modest signal: wells in the roughest fifth by their sharpest such shift finish at 6.52 ft on average, against 4.77 ft for the calmest fifth, though the relationship is noisy well to well (Spearman 0.08) because one rough segment in an otherwise easy well doesn't always decide the outcome. A well whose own GR simply doesn't resemble the typewell, checked the only place it's verifiable, the known section, where I can directly correlate this well's observed GR against the typewell's GR at the same true depths, turned out **not** to predict difficulty at all (Spearman -0.01, essentially flat across the whole range of overlap quality). That second result surprised me, and it is worth taking seriously rather than explaining away: it says the failure mode in this note is a specific aliased twin, not a generally weak resemblance, exactly what 1.1's finding (99.5% of wells have some other path that fits the GR at least as well as the truth) predicts. A well can correlate well with the typewell overall and still fail, because the one candidate it fails on didn't need a weak overall correlation, just one convincing lookalike.

**Figure 7.** *Left: wells with abrupt rate-of-change shifts skew harder, weakly. Right: overall GR-to-typewell overlap quality, checked directly, barely matters.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F10ab08ba645f45b5c0e4f7c196611667%2Ffig07_hard_wells.png?generation=1783364974139074&alt=media)

Not every well responds to this reasoning, and it is worth being direct about the ones that do not. A genuine bad tail survives every lever in this note: well `86454a6f` starts at 55.62 ft and only reaches 40.16 ft; `bb682ebd` goes from 39.21 to 23.05; `2fd68f7b` from 38.11 to 23.75; `91db7070` from 42.77 to 33.55. None of these resolve to something I would call solved, and section 5 explains why in advance: no leak-free signal distinguishes a well that will actually improve from one that will not, only whether it is contested. Figure 8 shows `86454a6f`'s actual trajectory: truth pulls away early and every variant of the decode, consensus included, stays locked onto the same wrong shape together, which is exactly the signature section 5 says a disagreement signal cannot catch, since nothing here is contested, it is just wrong.

**Figure 8.** *Well `86454a6f`, a bad-tail well: truth diverges early and every decode variant tracks the same wrong shape.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F5b92c75c229e786a6ed6562bbc8c2593%2Ffig08_bad_tail_trajectory.png?generation=1783364987931440&alt=media)

The same mechanism that rescues the bad tail also costs some good wells, and that trade is not free. Blending in the decorrelated estimator occasionally pulls an already-accurate backbone answer worse: well `5f4d2a52` goes from 3.13 to 30.89 ft, `efe96181` from 8.89 to 25.22 ft. I accept this deliberately, not by accident. The pooled metric weights squared error, so rescuing wells stuck at 30-plus feet is worth far more than the occasional cost of denting a well that was already at 3 feet, but the blend is not a pure improvement well by well, and I do not pretend otherwise. Figure 9 shows `5f4d2a52`: the raw decode alone was already close to the truth, and each further correction pulls it steadily further away, the direct picture of that cost.

**Figure 9.** *Well `5f4d2a52`, a harmed well: the raw decode was close; each further correction pulls it further from the truth.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F7c6bb074e9d27aec17bc68174f9f83cb%2Ffig09_harmed_trajectory.png?generation=1783365001896365&alt=media)

I only trust a lever once it survives two checks, not one. Anything that leans on nearby wells looks strong under ordinary or leave-one-well-out cross-validation and collapses under a holdout that withholds whole geographic blocks at once, because a held-out well normally keeps its trained neighbors close by even when it shouldn't. This caught more than one false win before I trusted it, and every lever in this note that touches neighboring wells is reported against both checks.

Of the 773 training wells, only 3 overlap with the locally visible test wells and get an exact shortcut from known structure; every other prediction, and all 200 hidden Kaggle test wells, goes through the full reasoning in 1.1 to 1.4.

## 3. Physical meaningfulness: every claim, its reasoning, and its evidence

Every choice in this note follows the same pattern: state a physical claim before measuring anything, then check what it is worth against real evidence, an oracle where the true answer is not otherwise available, an ablation where it is.

| Physical claim | Why it should be true | Evidence from the data |
|---|---|---|
| Depth trajectories don't turn discontinuously | a drill bit changes direction gradually, not instantly | removing the persistence prior (1.1) costs 2.30 ft pooled |
| A logging tool's absolute reading drifts between wells, even over identical rock | each logging run carries its own instrument calibration, so raw GR level is not directly comparable well to well | removing the typewell reference term (1.1), which recalibrates the typewell's GR to this well's own known-section GR, costs 1.26 ft pooled |
| A well's own drilled path carries more truth than a straight-line guess | a geosteered well is deliberately steered in real time toward a geological target, not drilled straight | centering the persistence prior on the drilled trend is worth 0.93 ft on a 30-well spatial check |
| Repeated beds create GR aliasing that must be discounted, not chased | sedimentary rock layers repeat similar character at different depths | only 0.5% of wells have their true path as the single best-fitting GR path (1.1); the Cauchy likelihood is built specifically not to chase the alias |
| A named formation is not internally uniform | rock deposited over geological time varies continuously in grain size, cementation, and radioactivity within one formation; the named boundary between formations is just the largest such variation, not the only one | a perfect geology-based correction oracle reaches only 7.36 ft from 10.68, but real geology signals (r &asymp; -0.015 bias correlation, 17% GR-to-formation match) never approach it, because a lateral geosteered to stay inside one formation only samples that internal variation, not the boundary (1.4) |
| Depth is a running total, not a point measurement | TVT accumulates along the path from the local depth-change at every step, so a wrong step doesn't stay local, it becomes part of the baseline every later step is measured from | cumulative upstream depth-change error correlates 0.45 with a much-later segment's RMSE, while a segment's own error barely reflects its own choice (2) |
| The remaining error has real shape, not pure noise | rock properties vary smoothly and coherently along a lateral, not row by row at random | a truth-cheating piecewise fit of the true residual reaches 0.80 ft pooled, far below anything realizable (1.1) |

None of these rows are about how a model performed; each is a claim about the rock, the drilling, or the instrument, checked against the data. The blending in 1.3 is different in kind, and does not belong in this table, and I would rather say so plainly than blur the line: it exists to reduce pooled RMSE, justified by the mathematics of combining correlated estimators, not because two models disagreeing is itself a physical fact about the rock. I kept that surface small, one disagreement-based weight, checked against both the pooled and the block-holdout numbers, rather than letting it grow into a second system next to the first.

## 4. What each piece contributed

**Pipeline stages, in the order they were added.** Each row changes what got measured, so the metric column is load-bearing: read a row only against its own metric, not against the row above it.

| Stage | Metric | Result |
|---|---|---|
| Backbone alone (1.1), no blending | pooled, all 773 wells, leak-free | 9.99 ft |
| + `track6` blend (1.3) | mean-per-well, leave-one-well-out | 7.28 &rarr; 6.21 ft |
| + per-well confidence weighting (5) | same scheme as the row above | closes part of the remaining gap toward the 5.11 ft two-way oracle |
| Deployed pipeline, everything above included | public leaderboard | &asymp; 5.7 |

**Figure 5** (section 1.3) already shows the second and third rows against their oracle ceiling; **Figure 3** (section 1.1) already shows the first row against the deployed score and the locatable-error oracle.

**What's inside the backbone itself.** No single piece carries the result alone; each number below is the cost of removing (or the value of adding) exactly one piece, holding everything else fixed.

| Component | Physical role | Pooled RMSE if removed or added |
|---|---|---|
| Persistence prior | gives the search a reason to prefer a continuous trajectory over an aliased jump | costs 2.30 ft if removed |
| Typewell reference term | ties a candidate's absolute depth level back to a trustworthy datum instead of letting it float | costs 1.26 ft if removed |
| Drilled-trend centering, vs. a straight-line extrapolation | recovers directional information a straight line discards | worth 0.93 ft, on a 30-well spatially stratified check |
| *(for scale)* locatable-error oracle | upper bound on what's reachable at all, from 1.1 | 0.80 ft |

**Figure 10.** *What each backbone component is worth alone, against the 0.80 ft locatable-error ceiling for scale.*
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F7200429%2F8ca37f566b0f2f453368ad298623f01d%2Ffig10_backbone_ablations.png?generation=1783365034897760&alt=media)

None of these three levers individually closes the gap to that 0.80 ft ceiling, and because ablations aren't strictly additive, summing them doesn't either; together they still leave real headroom the oracle says exists but the available evidence does not resolve.

## 5. Uncertainty: what the model knows about itself

Section 1.3's oracle logic applies directly to `track8`, the backbone's deployed decorrelated partner: a per-well blend weight beats a fixed one, so the question is what that weight can actually be based on at test time. The first candidate signal was self-consistency: how much the backbone's own prior-tightness variants, the same underlying reasoning regularized more or less strongly, agree with each other. Tested directly against true error, that signal carries essentially no information (Spearman correlation approximately 0.00). Variants built from the same evidence can still converge confidently on the same wrong candidate under the exact ambiguity described in 1.1; agreeing with a close relative of your own reasoning says nothing about being right.

A second candidate signal does carry information: how far the backbone's answer moves when weighed against other decorrelating blends already present in the pipeline, ones that draw on genuinely different evidence rather than just a retuned version of the same prior. That measure correlates 0.58 with a well's true error, and related measures reach as high as 0.66. The distinction that matters is not self versus other exactly, it is same-evidence versus different-evidence: a disagreement signal is only informative when what it compares against actually looked at something the base reasoning did not.

And the honest boundary. This kind of disagreement says a well is contested, not that either answer is wrong; an oracle that could see the true error directly and weight `track8` perfectly, well by well, would still do a further 0.21 ft pooled better than any leak-free signal I found, because "contested" and "wrong" are genuinely different properties. The failure mode this misses entirely is a well that both the backbone and its decorrelated partner reach through independent reasoning and both get wrong in the same direction, biased along its whole length, indistinguishable on any disagreement signal from a well both get right. I use uncertainty two ways only because of this: as evidence inside the blend weight, and as a relative reliability statement, this well is contested, not this well's answer is trustworthy. Never as an after-the-fact filter on correctness, because every attempt at that made validation worse.

## 6. Closing

The GR log pins depth well most of the time, and the joint distribution in 1.1 is honest about the rest: a second candidate sits close behind the true one often enough that no amount of sharpening the likelihood term (1.2) resolves it alone. The physical answer was to add a second, mathematically independent way of reasoning about the same log, `track6`'s particle filter (and `track8`'s, built on `sp45`, in the deployed pipeline) alongside the backbone's discrete search, because two different inference processes fail differently in a way one sharpened version of either cannot. The statistical answer was to be honest about what a combination of correlated estimators can and cannot buy, using oracles to measure headroom before spending effort chasing it, and to ground every confidence signal in genuine independence rather than a model's comfort with its own assumptions. The negative results, four architectures deep into a likelihood term that was never the real bottleneck, cost real time, but they are why I can say the remaining error lives in one decision at a time, at each hard transition, structured enough to reach 0.80 ft with the answer in hand, not reachable from the log alone.

## 7. Terms used in this note

| Term | What it is | What it uses |
|---|---|---|
| measured depth (MD) | distance traveled along the wellbore path, not straight-line vertical depth | -- |
| the lateral | the long horizontal section of the well past the known point, the part being predicted | -- |
| typewell | a nearby vertical well, drilled and logged earlier, with fully known GR-versus-depth pairing | -- |
| the backbone | the core decoder (1.1): a discrete beam search over `(h_len, ΔTVT)` segments | a segment-length prior, a persistence prior, a Cauchy GR likelihood, a typewell reference term |
| consensus | an intermediate checkpoint reading, reconciling a few internal decode variants with each other partway through (1.1, Figure 2) | not a final answer, just a sanity check along the way |
| `h_len` | how far a segment runs along the measured-depth path | -- |
| `ΔTVT` | how much true stratigraphic depth changes over a segment | -- |
| typewell reference term | a small recalibration of the typewell's GR curve toward this well's own GR level, using the known section (1.1) | this well's own known-section GR, compared against the raw typewell GR at the same true depths |
| cycle-skip | the decode locking onto a repeated, lookalike rock layer instead of the true one, staying offset by roughly a constant amount afterward | -- |
| formation / formation tops | a named rock layer and the depths where the well crosses into the next one | -- |
| geosteering | steering a well's path in real time toward a geological target while drilling, instead of drilling it straight | -- |
| `track6` | a decorrelated second estimator with its own, independently-built particle filter and correlation features (1.3) | its own particle filter, multi-scale correlation, formation-plane nearest neighbors, a small gradient-boosted ensemble |
| `track8` | `track6`'s leaner, deployed successor (1.3) | the same gradient-boosted trainer as `track6`, fed `sp45`'s features instead of a self-built particle filter |
| `sp45` | a standalone particle-filter and beam-search engine that `track8` is built on (1.3) | many independently-seeded particle swarms, a multi-config beam search, a per-well selector |
| `cvt1d` | a compact convolutional model tried as a third decorrelated partner (1.3) | never finished into a trained model; only checkpoints survive |
| nn_encoders | the four-architecture research line that tried to learn the likelihood term (1.2) | contrastive embeddings (InfoNCE), direct regression, error regression on the search lattice, and search-mined training data (DAgger-style) |
| oracle | a number computed using information only available with the ground truth in hand | never a real prediction; used only to measure headroom |
| pooled RMSE | the Kaggle metric: root-mean-squared dTVT error over every predicted row at once | -- |
| mean-per-well (LOWO) | error averaged per well, computed with leave-one-well-out cross-validation | -- |

---

**Author**
Shrey Gandhi