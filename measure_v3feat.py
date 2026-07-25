# Last fleongg lever: STRIDE-v3 as a FEATURE (not a new axis — an upgrade of the already
# validated stride family, which entered the feature set as v1 decodes). Deployment cost
# is zero (patch48 already decodes v3 per well); the price is one fleongg retrain.
#   A base      : cache + v1 stride cols                        [deployed]
#   B +v3       : A + s3_d + (s3_d - stride_d) disagreement
#   C v3 only   : cache + v3 cols, v1 stride cols dropped
# Same 150-well proxy / folds / params as every earlier feature screen.
# RUN from ~/rogii: python measure_v3feat.py      (~15min CPU)
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

t0 = time.time()
V1 = ["stride_d", "stride_best_d", "stride_stiff_d", "stride_loose_d"]
wells_all = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
wells150 = sorted(np.random.default_rng(0).choice(wells_all, 150, replace=False))
df = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     filters=[("well", "in", wells150)])
df = df.merge(pd.read_parquet("stride_join.parquet"), on="id", how="left")
s3 = pd.read_parquet("s3_all.parquet")
df = df.merge(s3, on="id", how="left")
cov = float(df["s3_d"].notna().mean())
df["s3_vs_v1"] = df["s3_d"] - df["stride_d"]
print(f"[{time.time()-t0:.0f}s] {len(df)} rows | v3 coverage {cov:.1%}", flush=True)
assert cov > 0.9, "v3 coverage too low"

base_cols = [c for c in df.columns if c not in ("id", "well", "target", "s3_d", "s3_vs_v1")
             and pd.api.types.is_numeric_dtype(df[c])]
y = df["target"].values.astype(np.float64)
y_fit = np.clip(y, -90, 90)
groups = df["well"].values
folds = list(GroupKFold(4).split(df[base_cols].values, y_fit, groups=groups))
P = dict(objective="regression", num_leaves=64, learning_rate=0.02, feature_fraction=0.6,
         bagging_fraction=0.7, bagging_freq=1, min_child_samples=40, lambda_l2=30.0,
         n_estimators=3000, n_jobs=8, verbose=-1, seed=0)


def run(tag, cols):
    X = df[cols].values.astype(np.float32)
    oof = np.zeros(len(df))
    imp = np.zeros(len(cols))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**P)
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
        imp += m.booster_.feature_importance("gain")
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"[{time.time()-t0:.0f}s] {tag:10s} OOF = {r:.4f} ({len(cols)} feats)", flush=True)
    rank = {c: i + 1 for i, (c, _) in enumerate(sorted(zip(cols, imp), key=lambda kv: -kv[1]))}
    for c in ("s3_d", "s3_vs_v1", "stride_d"):
        if c in rank:
            print(f"    {c}: importance rank {rank[c]}/{len(cols)}", flush=True)
    return r


rA = run("A_base", base_cols)
rB = run("B_plus_v3", base_cols + ["s3_d", "s3_vs_v1"])
rC = run("C_v3_only", [c for c in base_cols if c not in V1] + ["s3_d"])
print(f"\nDELTA B-A = {rB - rA:+.4f} | C-A = {rC - rA:+.4f}", flush=True)
print("DONE", flush=True)
