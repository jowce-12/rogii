# Measure items 1-4 before deploying (2026-07-12):
#   1. PF quality features (pf_pt_std / ll_spread / best_ll / gr_sig) as GBM features
#   2. include likpf_scale_*_d columns (currently filtered out)
#   3. full-data refit tree-count boost (x1.25 / x1.4 vs x1.0)
#   4. ridge meta with likpf_mean_d as an extra input column
# Proxy: v5 w150 cache (grcal=off to match how that cache's likpf was built),
# midreg single-LGB for feature arms, 3-LGB ridge for the meta arm. Relative deltas only.
import os, sys, time
os.environ["ROGII_GRCAL"] = "off"          # v5 cache likpf was built pre-S1 (raw PF)
os.environ.setdefault("ROGII_NJOBS", "24")

import numpy as np
import pandas as pd

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# --- pull functions out of train_stack.py (top half only; the driver would train) ---
src = open("train_stack.py", encoding="utf-8").read()
cut = src.index("# ===== full-stack retrain")
open("_ts_top.py", "w", encoding="utf-8").write(src[:cut])   # real file: numba cache needs a locator
import _ts_top as ns_mod
ns = vars(ns_mod)
load_well, lik_pf = ns["load_well"], ns["lik_pf"]
add_alias_metafeats, lgb_configs, rmse = ns["add_alias_metafeats"], ns["lgb_configs"], ns["rmse"]
from joblib import Parallel, delayed
log("train_stack top loaded (kernels compiled)")

CACHE = "train_features_v5_f1_w150.parquet"
df = pd.read_parquet(CACHE)
wells = sorted(df.well.unique())
log(f"cache {CACHE}: {len(df)} rows, {len(wells)} wells, {len(df.columns)} cols")

# --- A5 metafeats (production baseline includes them) ---
df = add_alias_metafeats(df, "train")
log("A5 metafeats joined")

# --- quality features: one raw 128-seed lik-PF pass per well ---
QUALITY = ["likpf_ptstd", "likpf_llspread", "likpf_bestll", "likpf_grsig"]
QCACHE = "quality_w150.parquet"
if os.path.exists(QCACHE):
    qdf = pd.read_parquet(QCACHE)
    log(f"quality loaded from {QCACHE}")
else:
    def qrows(wid):
        hw, tw = load_well(wid, "train")
        out, idx, q = lik_pf(hw, tw, with_quality=True)
        if not len(out):
            return None
        return pd.DataFrame({
            "id": [f"{wid}_{i}" for i in idx],
            "likpf_ptstd": q["pf_pt_std"].astype(np.float32),
            "likpf_llspread": np.float32(q["pf_ll_spread"]),
            "likpf_bestll": np.float32(q["pf_best_ll"]),
            "likpf_grsig": np.float32(q["pf_gr_sig"]),
            "_chk_mean": out["pf_mean"].astype(np.float32),   # sanity vs cached likpf_mean
        })
    res = Parallel(n_jobs=int(os.environ["ROGII_NJOBS"]), prefer="threads")(
        delayed(qrows)(w) for w in wells)
    qdf = pd.concat([r for r in res if r is not None], ignore_index=True)
    qdf.to_parquet(QCACHE, index=False)
    log(f"quality built + saved -> {QCACHE}")
df = df.merge(qdf, on="id", how="left")
for c in QUALITY:
    df[c] = df[c].astype(np.float32)
# sanity: fresh PF should reproduce the cached likpf_mean (same seeds/params)
if "_chk_mean" in df.columns and "likpf_mean" in df.columns:
    d = (df["_chk_mean"] - df["likpf_mean"]).abs()
    log(f"sanity fresh-vs-cache likpf_mean: median|d|={d.median():.4f} p99={d.quantile(.99):.3f}")
    df = df.drop(columns=["_chk_mean"])

# --- feature sets ---
drop = {"well", "id", "target"}
base = [c for c in df.columns if c not in drop
        and not (c.startswith("likpf_scale_") or c == "likpf_mean")
        and c not in QUALITY]
scale_d = sorted(c for c in df.columns if c.startswith("likpf_scale_") and c.endswith("_d"))
log(f"base={len(base)} feats | scale_d={scale_d} | quality={QUALITY}")

y = df["target"].values.astype(np.float32)
g = df["well"].values
if os.environ.get("ROGII_WINZ", "1") == "1":
    y_fit = np.clip(y, -90.0, 90.0)
else:
    y_fit = y

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
FOLDS = 4
cv = GroupKFold(FOLDS)
MID = lgb_configs("cpu")[1]   # midreg: the sweep winner, standard proxy config

def lgb_oof(feats, params, tag):
    X = df[feats].values.astype(np.float32)
    oof = np.zeros(len(df)); iters = []
    for tr, va in cv.split(X, y, groups=g):
        m = LGBMRegressor(**params)
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        iters.append(int(m.best_iteration_ or 0))
    log(f"  {tag}: OOF={rmse(y, oof):.4f} (iters {iters})")
    return oof

log("== items 1+2: feature arms (midreg LGB, 4-fold) ==")
oofA = lgb_oof(base, MID, "A base            ")
oofB = lgb_oof(base + scale_d, MID, "B +scale_d        ")
oofC = lgb_oof(base + QUALITY, MID, "C +quality        ")
oofD = lgb_oof(base + scale_d + QUALITY, MID, "D +scale_d+quality")

log("== item 4: ridge meta +likpf_mean_d (3-LGB stack on base feats) ==")
oof_cols = [oofA]
for ci in (0, 2):
    oof_cols.append(lgb_oof(base, lgb_configs("cpu")[ci], f"lgb{ci} base"))
OOF = np.column_stack(oof_cols)
anchor_col = df["likpf_mean_d"].values.astype(np.float32).reshape(-1, 1)
for tag, M in [("ridge 3-LGB          ", OOF),
               ("ridge 3-LGB +anchor  ", np.hstack([OOF, np.nan_to_num(anchor_col)]))]:
    meta = np.zeros(len(df))
    for tr, va in cv.split(M, y_fit, groups=g):
        r = Ridge(alpha=1.66, positive=True, fit_intercept=True)
        r.fit(M[tr], y_fit[tr]); meta[va] = r.predict(M[va])
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(M, y_fit)
    log(f"  {tag}: OOF={rmse(y, meta):.4f} coefs={np.round(r.coef_, 4)} icpt={r.intercept_:.3f}")

log("== item 3: refit-iteration boost (group holdout, midreg) ==")
for seed in (0, 1):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_i, ho_i = next(gss.split(df[base].values, y, groups=g))
    Xtr, Xho = df[base].values[tr_i].astype(np.float32), df[base].values[ho_i].astype(np.float32)
    ytr, yho = y_fit[tr_i], y[ho_i]; gtr = g[tr_i]
    inner = GroupKFold(4); iters = []
    for tri, vai in inner.split(Xtr, ytr, groups=gtr):
        m = LGBMRegressor(**MID)
        m.fit(Xtr[tri], ytr[tri], eval_set=[(Xtr[vai], ytr[vai])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        iters.append(int(m.best_iteration_ or 0))
    b = int(np.mean(iters))
    log(f"  seed{seed}: inner best_iters={iters} mean={b}")
    for f in (1.0, 1.25, 1.4):
        p = dict(MID); p["n_estimators"] = max(50, int(b * f))
        m = LGBMRegressor(**p); m.fit(Xtr, ytr)
        log(f"  seed{seed} refit x{f}: holdout RMSE={rmse(yho, m.predict(Xho)):.4f} (n={p['n_estimators']})")

log("ALL MEASUREMENTS DONE")
