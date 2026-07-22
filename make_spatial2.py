# Multi-surface spatial builder (v2): fold-safe IDW banks for ALL SIX surfaces
# (ANCC + ASTNU/ASTNL/EGFDU/EGFDL/BUDA), same recipe as make_spatial_features.py
# (60 samples/well, K=20, 1/d weights, self-exclusion, per-fold train-wells-only).
# Output per fold: gru_spatial2_fold{f}.parquet with the SAME base columns as v1
# (spatial_u/dense_std/dense_dist = ANCC) plus u_/std_/dist_{astnu,astnl,egfdu,egfdl,buda}.
# A surface with no bank samples in a fold yields NaN columns (trainer skips it).
# RUN from ~/rogii: python make_spatial2.py [--fold k]      (~15min CPU)
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
SURFACES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def horizontal_path(well):
    return os.path.join("train", f"{well}__horizontal_well.csv")


def query_surface(tree, tree_vals, tree_wells, scale, xy, self_well=None):
    fetch = min(K + SPW, len(tree_vals)) if self_well is not None else min(K, len(tree_vals))
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
    neighbours = tree_vals[ik]
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

# per-surface (xy, val) samples per well
samples = {s: {} for s in SURFACES}
for i, well in enumerate(wells):
    d = pd.read_csv(horizontal_path(well), usecols=["X", "Y"] + SURFACES)
    for s in SURFACES:
        ds = d[["X", "Y", s]].dropna()
        if len(ds) == 0:
            continue
        take = np.linspace(0, len(ds) - 1, min(SPW, len(ds)), dtype=int)
        samples[s][well] = (ds.iloc[take][["X", "Y"]].to_numpy(np.float64),
                            ds.iloc[take][s].to_numpy(np.float64))
    if (i + 1) % 200 == 0:
        print(f"[{time.time()-t0:.0f}s] sampled {i+1}/{len(wells)} wells", flush=True)
for s in SURFACES:
    print(f"  surface {s}: {len(samples[s])}/{len(wells)} wells with data", flush=True)

folds = [ONLY_FOLD] if ONLY_FOLD is not None else list(range(5))
for fold in folds:
    banks = {}
    for s in SURFACES:
        tr = [w for w in wells if fold_of[w] != fold and w in samples[s]]
        if not tr:
            banks[s] = None
            continue
        xy = np.concatenate([samples[s][w][0] for w in tr])
        vals = np.concatenate([samples[s][w][1] for w in tr])
        wds = np.concatenate([np.repeat(w, len(samples[s][w][1])) for w in tr])
        scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        banks[s] = (cKDTree(xy / scale), vals, wds, scale)
    records = []
    for i, well in enumerate(wells):
        hw = pd.read_csv(horizontal_path(well), usecols=["MD", "X", "Y", "Z"])
        pos = np.arange(0, len(hw), ROW_STEP, dtype=int)
        if pos[-1] != len(hw) - 1:
            pos = np.r_[pos, len(hw) - 1]
        q = hw.iloc[pos]
        xy = q[["X", "Y"]].to_numpy(np.float64)
        z = q["Z"].to_numpy(np.float64)
        rec = {"well": well, "md": q["MD"].to_numpy(np.float32)}
        for s in SURFACES:
            base = s == "ANCC"
            ku, ks, kd = (("spatial_u", "dense_std", "dense_dist") if base
                          else (f"u_{s.lower()}", f"std_{s.lower()}", f"dist_{s.lower()}"))
            if banks[s] is None:
                rec[ku] = rec[ks] = rec[kd] = np.full(len(q), np.nan, np.float32)
                continue
            tree, vals, wds, scale = banks[s]
            self_well = well if (fold_of[well] != fold and well in samples[s]) else None
            dense, dstd, ddist = query_surface(tree, vals, wds, scale, xy, self_well=self_well)
            rec[ku] = (-z + dense).astype(np.float32)
            rec[ks] = dstd
            rec[kd] = ddist
        records.append(pd.DataFrame(rec))
        if (i + 1) % 200 == 0:
            print(f"[{time.time()-t0:.0f}s] fold{fold}: {i+1}/{len(wells)} wells", flush=True)
    out = pd.concat(records, ignore_index=True)
    path = f"gru_spatial2_fold{fold}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    print(f"[{time.time()-t0:.0f}s] wrote {path}: {len(out)} rows, {len(out.columns)} cols", flush=True)

print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
