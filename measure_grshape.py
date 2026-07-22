# A/B proxy for the distribution-shape features (grshape_join.parquet). Same harness as
# measure_xtrk: deployed-equivalent base (cache + 4 STRIDE cols), 150w fixed subset,
# midreg LGB, GKF-4, same folds. Pre-registered rule: adopt path only if <= -0.03 AND
# top-quartile importance; otherwise the fleongg feature axis is CLOSED for the campaign.
# RUN from ~/rogii (~20min CPU).
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
gs = pd.read_parquet("grshape_join.parquet")
GS_COLS = [c for c in gs.columns if c != "id"]
df = df.merge(gs, on="id", how="left")
print(f"[{time.time()-t0:.0f}s] {len(df)} rows, grshape coverage "
      f"{df[GS_COLS[0]].notna().mean():.1%}", flush=True)

base_cols = [c for c in df.columns
             if c not in ("id", "well", "target") and c not in GS_COLS
             and pd.api.types.is_numeric_dtype(df[c])]
y = df["target"].values.astype(np.float64)
y_fit = np.clip(y, -90, 90)
groups = df["well"].values
PARAMS = dict(objective="regression", num_leaves=64, learning_rate=0.02,
              feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1,
              min_child_samples=40, lambda_l2=30.0, n_estimators=3000,
              n_jobs=8, verbose=-1, seed=0)

def run(cols, tag):
    X = df[cols].values.astype(np.float32)
    oof = np.zeros(len(df))
    imps = np.zeros(len(cols))
    for f, (tr, va) in enumerate(GroupKFold(4).split(X, y_fit, groups=groups)):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
        imps += m.booster_.feature_importance("gain")
        print(f"[{time.time()-t0:.0f}s] {tag} fold{f} done", flush=True)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    print(f"== {tag}: OOF RMSE = {rmse:.4f} ({len(cols)} feats)", flush=True)
    return rmse, dict(zip(cols, imps))

rA, _ = run(base_cols, "A base+stride")
rB, impB = run(base_cols + GS_COLS, "B +grshape")
print(f"\nDELTA (B-A) = {rB - rA:+.4f}  ({'grshape helps' if rB < rA else 'no gain'})", flush=True)
rank = sorted(impB.items(), key=lambda kv: -kv[1])
pos = {c: i + 1 for i, (c, _) in enumerate(rank)}
for c in GS_COLS:
    print(f"  {c}: importance rank {pos[c]}/{len(rank)}", flush=True)
print("DONE", flush=True)
