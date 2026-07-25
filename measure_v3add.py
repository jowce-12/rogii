# STRIDE-v3 as an ADDITIONAL feature (the untested combination).
# What was measured before, on the unreliable 150-well single-LGB proxy:
#   v1 four cols                 6.8736
#   v1 + s3_d + s3_vs_v1         7.1232   <- the disagreement column poisoned it (xtrk pattern)
#   v3 alone, v1 dropped         6.8241   <- looked good, but the FULL retrain got WORSE
#                                            (stack 8.0576 -> 8.1769: diversity loss)
# Never tested: v1 four cols + s3_d, with NO disagreement column. That is this script.
# Metric = the 400-well / 5-pinned-fold three-LGB ensemble (reference already 9.3315),
# because single-model OOF has misled twice and the 150-well ridge stack is broken.
import json, time
import numpy as np, pandas as pd, lightgbm as lgb

t0 = time.time()
V1 = ["stride_d", "stride_best_d", "stride_stiff_d", "stride_loose_d"]
wells_all = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
wells = sorted(np.random.default_rng(11).choice(wells_all, 400, replace=False))
df = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet", filters=[("well", "in", wells)])
df = df.merge(pd.read_parquet("stride_join.parquet"), on="id", how="left")
df = df.merge(pd.read_parquet("s3_all.parquet"), on="id", how="left")
BASE = [c for c in df.columns if c not in ("id", "well", "target", "s3_d")
        and pd.api.types.is_numeric_dtype(df[c])]
y = df["target"].values.astype(np.float64); y_fit = np.clip(y, -90, 90)
fmap = {k: int(v) for k, v in json.load(open("proxy400_folds.json")).items()}
fi = np.array([fmap[w] for w in df["well"].values])
folds = [(np.where(fi != k)[0], np.where(fi == k)[0]) for k in range(5)]
CFGS = [dict(num_leaves=255, min_child_samples=20, lambda_l2=20.0, lambda_l1=1.0, seed=123),
        dict(num_leaves=64, min_child_samples=40, lambda_l2=30.0, seed=0),
        dict(num_leaves=31, feature_fraction=0.4, bagging_fraction=0.6, lambda_l2=60.0,
             min_child_samples=60, seed=7)]
LGB = dict(objective="regression", learning_rate=0.02, feature_fraction=0.6,
           bagging_fraction=0.7, bagging_freq=1, n_estimators=3000, n_jobs=6, verbose=-1)
print(f"[{time.time()-t0:.0f}s] {len(df)} rows | v3 coverage {df['s3_d'].notna().mean():.1%}", flush=True)

def ens(cols, tag):
    X = df[cols].values.astype(np.float32)
    preds, singles, imps = [], [], np.zeros(len(cols))
    for cfg in CFGS:
        oof = np.zeros(len(df))
        for tr, va in folds:
            m = lgb.LGBMRegressor(**{**LGB, **cfg})
            m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = m.predict(X[va]); imps += m.booster_.feature_importance("gain")
        preds.append(oof); singles.append(float(np.sqrt(np.mean((oof - y) ** 2))))
    e = float(np.sqrt(np.mean((np.mean(preds, 0) - y) ** 2)))
    rank = {c: i + 1 for i, (c, _) in enumerate(sorted(zip(cols, imps), key=lambda kv: -kv[1]))}
    extra = "  ".join(f"{c}#{rank[c]}" for c in ["s3_d"] + V1 if c in rank)
    print(f"[{time.time()-t0:.0f}s] {tag:22s} ensemble {e:.4f} | singles "
          f"{' '.join(f'{s:.4f}' for s in singles)} | ranks {extra}", flush=True)
    return e

a = ens(BASE, "A v1 only (reference)")
b = ens(BASE + ["s3_d"], "B v1 + s3_d  (ADD)")
c = ens([x for x in BASE if x not in V1] + ["s3_d"], "C v3 only (replace)")
print(f"\nADD  B-A = {b-a:+.4f}   REPLACE  C-A = {c-a:+.4f}   (negative = better)", flush=True)
print("DONE", flush=True)
