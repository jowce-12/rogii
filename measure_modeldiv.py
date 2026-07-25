# fleongg MODEL-TYPE / PARAM diversity screen.
# Two lessons force this design:
#  (1) the 150-well proxy's ridge stack is BROKEN (plain mean 6.8555 beats ridge-cv 7.0030)
#      -> use 400 wells, 5 folds, and judge with metrics that do not need a stable ridge;
#  (2) v3-swap showed single-model OOF does not decide stack questions -> judge each
#      candidate by what it ADDS to the reference ensemble, not by its own RMSE.
# For every candidate we report: own OOF, error-correlation with the reference ensemble,
# the best 2-way blend against it, and the resulting gain. A candidate earns a slot only
# if it lowers the blend, which is exactly the property the ridge meta can exploit.
# RUN from ~/rogii: python measure_modeldiv.py     (~90min CPU, background)
import json
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

t0 = time.time()
N_WELLS = 400
FOLD_FILE = "proxy400_folds.json"

wells_all = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
wells = sorted(np.random.default_rng(11).choice(wells_all, N_WELLS, replace=False))
df = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet", filters=[("well", "in", wells)])
df = df.merge(pd.read_parquet("stride_join.parquet"), on="id", how="left")
F = [c for c in df.columns if c not in ("id", "well", "target")
     and pd.api.types.is_numeric_dtype(df[c])]
X = df[F].values.astype(np.float32)
y = df["target"].values.astype(np.float64)
y_fit = np.clip(y, -90, 90)
g = df["well"].values
# pinned folds (sklearn-version-proof, same lesson as gru_folds.json)
if os.path.exists(FOLD_FILE):
    fmap = {k: int(v) for k, v in json.load(open(FOLD_FILE)).items()}
else:
    fmap = {}
    for k, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
        for i in va:
            fmap[wells[i]] = k
    json.dump(fmap, open(FOLD_FILE, "w"))
fold_idx = np.array([fmap[w] for w in g])
folds = [(np.where(fold_idx != k)[0], np.where(fold_idx == k)[0]) for k in range(5)]
print(f"[{time.time()-t0:.0f}s] {len(df)} rows / {N_WELLS} wells / {len(F)} feats", flush=True)

LGB_BASE = dict(objective="regression", learning_rate=0.02, feature_fraction=0.6,
                bagging_fraction=0.7, bagging_freq=1, n_estimators=3000,
                n_jobs=8, verbose=-1)


def run_lgb(tag, **kw):
    oof = np.zeros(len(df))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**{**LGB_BASE, **kw})
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
    return oof


def run_xgb(tag, **kw):
    import xgboost as xgb
    P = dict(objective="reg:squarederror", learning_rate=0.02, max_depth=8,
             subsample=0.7, colsample_bytree=0.6, reg_lambda=20.0, min_child_weight=20,
             n_estimators=3000, tree_method="hist", n_jobs=8, early_stopping_rounds=100)
    P.update(kw)
    oof = np.zeros(len(df))
    for tr, va in folds:
        m = xgb.XGBRegressor(**P)
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], verbose=False)
        oof[va] = m.predict(X[va])
    return oof


def run_cb(tag, **kw):
    from catboost import CatBoostRegressor
    P = dict(loss_function="RMSE", learning_rate=0.03, depth=8, l2_leaf_reg=6.0,
             iterations=3000, allow_writing_files=False, verbose=0, thread_count=8)
    P.update(kw)
    oof = np.zeros(len(df))
    for tr, va in folds:
        m = CatBoostRegressor(**P)
        m.fit(X[tr], y_fit[tr], eval_set=(X[va], y_fit[va]),
              use_best_model=True, early_stopping_rounds=100)
        oof[va] = m.predict(X[va])
    return oof


def rmse(p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


# ---- reference ensemble: today's three LGB shapes ----
REF = {}
REF["lgb_deep"] = run_lgb("deep", num_leaves=255, min_child_samples=20, lambda_l2=20.0,
                          lambda_l1=1.0, seed=123)
print(f"[{time.time()-t0:.0f}s] ref lgb_deep     {rmse(REF['lgb_deep']):.4f}", flush=True)
REF["lgb_shallow"] = run_lgb("shallow", num_leaves=64, min_child_samples=40, lambda_l2=30.0, seed=0)
print(f"[{time.time()-t0:.0f}s] ref lgb_shallow  {rmse(REF['lgb_shallow']):.4f}", flush=True)
REF["lgb_shallow2"] = run_lgb("shallow2", num_leaves=31, feature_fraction=0.4,
                              bagging_fraction=0.6, lambda_l2=60.0, min_child_samples=60, seed=7)
print(f"[{time.time()-t0:.0f}s] ref lgb_shallow2 {rmse(REF['lgb_shallow2']):.4f}", flush=True)
R = np.mean([REF[k] for k in REF], 0)
print(f"[{time.time()-t0:.0f}s] REFERENCE ensemble (mean of 3) = {rmse(R):.4f}\n", flush=True)

CANDS = [
    ("lgb_dart", lambda: run_lgb("dart", boosting_type="dart", num_leaves=64,
                                 min_child_samples=40, lambda_l2=30.0, n_estimators=800,
                                 drop_rate=0.1, seed=5)),
    ("lgb_goss", lambda: run_lgb("goss", boosting_type="goss", num_leaves=96,
                                 min_child_samples=30, lambda_l2=25.0, seed=11)),
    ("lgb_xdeep", lambda: run_lgb("xdeep", num_leaves=511, min_child_samples=10,
                                  lambda_l2=5.0, feature_fraction=0.8, seed=21)),
    ("lgb_xstump", lambda: run_lgb("xstump", num_leaves=15, min_child_samples=120,
                                   lambda_l2=120.0, feature_fraction=0.3, seed=31)),
    ("xgb_hist", lambda: run_xgb("xgb")),
    ("xgb_deep_lowreg", lambda: run_xgb("xgb2", max_depth=12, reg_lambda=3.0,
                                        min_child_weight=5, colsample_bytree=0.8)),
    ("cb_d8", lambda: run_cb("cb8")),
    ("cb_d5_lossguide", lambda: run_cb("cb5", depth=5, grow_policy="Lossguide",
                                       max_leaves=48, l2_leaf_reg=12.0)),
]
print(f"{'candidate':18s} {'own OOF':>8s} {'corr(R)':>8s} {'best w':>7s} {'blend':>8s} {'gain':>8s}", flush=True)
rows = []
for name, fn in CANDS:
    try:
        p = fn()
    except Exception as e:
        print(f"{name:18s}  FAILED: {str(e)[:60]}", flush=True)
        continue
    er, ec = R - y, p - y
    c = float(np.corrcoef(er, ec)[0, 1])
    ws = np.arange(0.0, 0.65, 0.05)
    bl = [rmse((1 - w) * R + w * p) for w in ws]
    i = int(np.argmin(bl))
    rows.append((name, rmse(p), c, ws[i], bl[i], bl[i] - rmse(R)))
    print(f"{name:18s} {rmse(p):8.4f} {c:8.4f} {ws[i]:7.2f} {bl[i]:8.4f} {bl[i]-rmse(R):+8.4f}",
          flush=True)
    np.save(f"_mdiv_{name}.npy", p)
pd.DataFrame(rows, columns=["cand", "own", "corr", "w", "blend", "gain"]).to_csv("modeldiv.csv", index=False)
print("\nDONE (negative gain = the candidate improves the ensemble)", flush=True)
