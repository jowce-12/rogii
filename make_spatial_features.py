# Build leakage-safe fold-specific spatial ANCC surface features for train_gru2_feature.py.
# Per fold f: IDW (k=20, 1/d weights) surface over (X,Y)->ANCC sampled 60 pts/well from
# the fold's TRAIN wells only; a train well's own samples are excluded when querying
# itself (self-exclusion), and validation wells are absent from the tree entirely.
# Output per fold: gru_spatial_fold{f}.parquet (well, md, spatial_u, dense_std, dense_dist)
#   spatial_u   = IDW ANCC at the well's (X,Y) minus its Z  -> TVT-like surface (shape only;
#                 the trainer re-anchors it on each sample's known prefix)
#   dense_std   = IDW-weighted std of neighbour ANCC (ft)
#   dense_dist  = nearest valid neighbour distance in std-scaled XY units
# RUN from ~/rogii:  python newgru/make_spatial_features.py [--fold k]     (~10-30min CPU)
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

ONLY_FOLD = int(sys.argv[sys.argv.index("--fold") + 1]) if "--fold" in sys.argv else None
SPW = 60
K = 20
ROW_STEP = 4


def horizontal_path(well):
    return os.path.join("train", f"{well}__horizontal_well.csv")


def query_surface(tree, tree_ancc, tree_wells, scale, xy, self_well=None):
    fetch = min(K + SPW, len(tree_ancc)) if self_well is not None else min(K, len(tree_ancc))
    dist, idx = tree.query(xy / scale, k=fetch, workers=-1)
    if fetch == 1:
        dist, idx = dist[:, None], idx[:, None]
    if self_well is not None:
        dist = np.where(tree_wells[idx] == self_well, np.inf, dist)
    take = np.argpartition(dist, K - 1, axis=1)[:, :K]
    dk = np.take_along_axis(dist, take, axis=1)
    ik = np.take_along_axis(idx, take, axis=1)
    valid = np.isfinite(dk)
    weight = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
    sw = weight.sum(1)
    if np.any(sw <= 0):
        raise RuntimeError(f"no valid spatial neighbours for {self_well}")
    neighbours = tree_ancc[ik]
    dense = (neighbours * weight).sum(1) / sw
    var = ((neighbours - dense[:, None]) ** 2 * weight).sum(1) / sw
    nearest = np.where(valid, dk, np.inf).min(1)
    return dense.astype(np.float32), np.sqrt(np.maximum(var, 0)).astype(np.float32), nearest.astype(np.float32)


t0 = time.time()
rowmap = pd.read_parquet("gru_rowmap.parquet", columns=["well"])
wells = sorted(rowmap["well"].unique())
fold_of = {}
for fold, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = fold

samples = {}
for i, well in enumerate(wells):
    d = pd.read_csv(horizontal_path(well), usecols=["X", "Y", "ANCC"]).dropna()
    take = np.linspace(0, len(d) - 1, min(SPW, len(d)), dtype=int)
    samples[well] = (d.iloc[take][["X", "Y"]].to_numpy(np.float64),
                     d.iloc[take]["ANCC"].to_numpy(np.float64))
    if (i + 1) % 100 == 0:
        print(f"[{time.time()-t0:.0f}s] sampled {i+1}/{len(wells)} wells", flush=True)

folds = [ONLY_FOLD] if ONLY_FOLD is not None else list(range(5))
for fold in folds:
    if fold not in range(5):
        raise ValueError("--fold must be 0..4")
    train_wells = [w for w in wells if fold_of[w] != fold]
    tree_xy = np.concatenate([samples[w][0] for w in train_wells])
    tree_ancc = np.concatenate([samples[w][1] for w in train_wells])
    tree_wells = np.concatenate([np.repeat(w, len(samples[w][1])) for w in train_wells])
    scale = np.where(tree_xy.std(0) < 1e-3, 1.0, tree_xy.std(0))
    tree = cKDTree(tree_xy / scale)
    records = []
    for i, well in enumerate(wells):
        hw = pd.read_csv(horizontal_path(well), usecols=["MD", "X", "Y", "Z"])
        pos = np.arange(0, len(hw), ROW_STEP, dtype=int)
        if pos[-1] != len(hw) - 1:
            pos = np.r_[pos, len(hw) - 1]
        q = hw.iloc[pos]
        self_well = well if fold_of[well] != fold else None
        dense, dense_std, dense_dist = query_surface(
            tree, tree_ancc, tree_wells, scale,
            q[["X", "Y"]].to_numpy(np.float64), self_well=self_well)
        # TVT-like surface BEFORE well-specific known-prefix calibration (trainer anchors it).
        spatial_u = -q["Z"].to_numpy(np.float64) + dense
        records.append(pd.DataFrame({
            "well": well,
            "md": q["MD"].to_numpy(np.float32),
            "spatial_u": spatial_u.astype(np.float32),
            "dense_std": dense_std,
            "dense_dist": dense_dist,
        }))
        if (i + 1) % 200 == 0:
            print(f"[{time.time()-t0:.0f}s] fold{fold}: {i+1}/{len(wells)} wells", flush=True)
    out = pd.concat(records, ignore_index=True)
    path = f"gru_spatial_fold{fold}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"[{time.time()-t0:.0f}s] wrote {path}: {len(out)} rows", flush=True)

print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
