This working note maps where the ROGII wellbore surface is recoverable from the data you are handed and where it is not, and ships the measurement harness I used as a reusable tool. The full analysis, code, and figures live in the notebook this writeup attaches: Fork the ruler, not the model.
**Notebook:** [Fork the ruler, not the model](https://www.kaggle.com/code/georgymamarin/fork-the-ruler-not-the-model)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9407128%2F3d3c965bad963e020a4beff109d41545%2Fopener_wrong_depth.png?generation=1783324537853163&alt=media)

## The map in one paragraph
The gap runs from the carry-last-TVT baseline (15.883 public) down to the board heads (~5.3). It splits in two. The **recoverable** part is a per-well offset plus a piecewise dip, read by matching the horizontal gamma-ray to the typewell; the easier wells reach roughly 3-5 ft per-well this way, and the public forks (~7.2) have captured the offset and the dominant slope. The **irreducible** part is a bimodal datum on the ~10% of wells that carry ~40% of the squared error: the Eagle Ford is rhythmically bedded, so the GR lines up with the typewell at two stratigraphic positions about one bundle apart (~15 ft), and on a truly ambiguous well which mode is real is close to a coin-flip. The single "wall" dissolves into a distribution.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9407128%2F06b0d7e3410fbfceb96b749a7cdcf975%2Fcard_oracle_ladder.png?generation=1783324565502618&alt=media)

## Two corrections I publish rather than bury
The first version of this notebook got two things wrong, and I state both plainly. There is **no leak**: the local data/test/ folder is example data, replaced by the real hidden test at scoring. And **~10 ft is not the task floor**: it was my own selector's ceiling under a deliberately hard leave-whole-field-out split; the board heads and my own smooth oracle (~3.0) both sit far below it.

## The result that separates this from the "unobservable datum" consensus
Five other working notes reached the same irreducible tail from methods I did not have (acquisition physics, observability theory, an identifiability lemma). Where this note adds something new: I measure the **calibration degeneracy** directly with a vertical-shift scan (oracle calibration localizes 82% of wells within 2 ft; a naive legal fit collapses to ~8%, shuffle level), and then I show it is **not the wall**. A legal heel-calibration, fit the GR gain and offset on the known heel and carry it to the tail, recovers ~80% localization, essentially the oracle's 82%. The datum is legally pinnable; the open legal problem moves from the datum to the per-well slope.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9407128%2F98a232dd684fca684b8662f50c90a0c7%2Fcalibration_degeneracy.png?generation=1783324556842227&alt=media)

## The deliverable is a ruler, not a model
Forking the top pipeline for a score 0.03 better buys the particle filter's lucky seed, not an edge (byte-identical notebooks reseed across a 7.168-7.286 public band). So the notebook ships three helpers that read *your* regression, not mine: oracle_ceiling (your version of the floor ladder), tail_concentration (your worst-10% error share), and a wall_test that leaves a whole group out with a shuffle-label control, so you learn whether a feature is real structure or a leak. None of it is wellbore-specific; a well is just a group.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9407128%2F876314873354cc87baecbdd2801b9e4d%2Fseed_band.png?generation=1783324581948909&alt=media)

## How this maps to the criteria
- **Breadth and depth**: the recoverable/irreducible decomposition, the aligner tournament, the leave-field-out negative, the bimodal-hedge test, each with its validation and its lesson.
- **Data and wells**: the one-DOF formation columns, the 92%-linear surface, and the specific ~10% of wells where a different model earns its keep.
- **Physical meaningfulness**: the whole note is about where GR-to-typewell matching is physically informative versus where the bedding makes it ambiguous, and where I draw the line against metric-chasing.
- **Contribution of individual ideas**: the oracle ladder and the calibration scan quantify each rung; the harness makes the accounting reproducible on any pipeline.
- **Uncertainty estimation**: the seed band as the LB noise floor, and the posterior-mean-over-modes decision theory on the bimodal tail (predict p*a + (1-p)*b, not a committed mode).

Full detail, all figures, and the runnable harness are in the attached notebook. Credits to the community whose public work this builds on are in its section 11.
