# fleongg BASE-MODEL screening on the established 150-well proxy (single LGB, GKF-4,
# identical folds/seed; relative comparison only — the full stack is retrained later only
# for winners). Motivated by the diagnosis that fleongg's gap vs the GRU pole GROWS with
# distance from the cut (eval-length quartile gap 1.33 -> 1.70 ft).
#   A base      : deployed-ish midreg params, plain target
#   B huber     : robust objective (heavy tail: top 10% wells = 52% of SSE)
#   C rate      : predict target / (1 + md_since/1000) then scale back (drift-rate space)
#   D tiered    : separate near/far models split at median md_since
#   E capacity  : more leaves / lower l2 (params were tuned on the older, smaller feature set)
#   F cap_reg   : fewer leaves / higher l2 (other direction)
# RUN from ~/rogii: python measure_base.py        (~35min CPU)
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

t0 = time.time()
wells_all = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
wells150 = sorted(np.random.default_rng(0).choice(wells_all, 150, replace=False))
df = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     filters=[("well", "in", wells150)])
df = df.merge(pd.read_parquet("stride_join.parquet"), on="id", how="left")
cols = [c for c in df.columns if c not in ("id", "well", "target")
        and pd.api.types.is_numeric_dtype(df[c])]
X = df[cols].values.astype(np.float32)
y = df["target"].values.astype(np.float64)
y_fit = np.clip(y, -90, 90)
groups = df["well"].values
md = np.nan_to_num(df["md_since"].values.astype(np.float64))
scale = 1.0 + md / 1000.0
folds = list(GroupKFold(4).split(X, y_fit, groups=groups))
print(f"[{time.time()-t0:.0f}s] {len(df)} rows x {len(cols)} feats", flush=True)

P = dict(objective="regression", num_leaves=64, learning_rate=0.02, feature_fraction=0.6,
         bagging_fraction=0.7, bagging_freq=1, min_child_samples=40, lambda_l2=30.0,
         n_estimators=3000, n_jobs=8, verbose=-1, seed=0)


def run(tag, params=None, target=None, rescale=None, tier=None):
    p = dict(P); p.update(params or {})
    tgt = y_fit if target is None else target
    oof = np.zeros(len(df))
    for f, (tr, va) in enumerate(folds):
        if tier is None:
            m = lgb.LGBMRegressor(**p)
            m.fit(X[tr], tgt[tr], eval_set=[(X[va], tgt[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = m.predict(X[va])
        else:
            for side in (0, 1):
                tr_s = tr[tier[tr] == side]; va_s = va[tier[va] == side]
                if len(va_s) == 0:
                    continue
                m = lgb.LGBMRegressor(**p)
                m.fit(X[tr_s], tgt[tr_s], eval_set=[(X[va_s], tgt[va_s])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                oof[va_s] = m.predict(X[va_s])
    pred = oof if rescale is None else oof * rescale
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f"[{time.time()-t0:.0f}s] {tag:12s} OOF = {rmse:.4f}", flush=True)
    return rmse


rA = run("A_base")
run("B_huber", params=dict(objective="huber", alpha=8.0))
run("C_rate", target=y_fit / scale, rescale=scale)
run("D_tiered", tier=(md >= np.median(md)).astype(int))
run("E_capacity", params=dict(num_leaves=160, lambda_l2=12.0, min_child_samples=25))
run("F_cap_reg", params=dict(num_leaves=31, lambda_l2=80.0, min_child_samples=80))
print(f"baseline A = {rA:.4f}", flush=True)
print("DONE", flush=True)
