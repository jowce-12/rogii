# GBM param re-sweep + pruning proxy on the v7 feature regime (single-LGB OOF proxy).
# Rationale: params were tuned on v5 features; v7 shifted gain mass to the likpf family
# (~60%) and the strongest-regularized config became the best single -> probe stronger reg.
# 150-well subset of the local w773 v7 cache; GroupKFold(4); winsorized fit target,
# raw-y OOF reported (comparable within this sweep only).
import os, time
os.environ.setdefault("ROGII_GRCAL", "blend")
import numpy as np
import pandas as pd

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

# train_stack top-half for add_alias_metafeats (A5 join, matches production features)
src = open("train_stack.py", encoding="utf-8").read()
open("_ts_top.py", "w", encoding="utf-8").write(src[:src.index("# ===== full-stack retrain")])
import _ts_top as TS
log("helpers loaded")

WELLS = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet", columns=["well"])["well"].unique())[:150]
df = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     filters=[("well", "in", WELLS)])
log(f"cache subset: {len(df)} rows, {df['well'].nunique()} wells, {len(df.columns)} cols")
df = TS.add_alias_metafeats(df, "train")
log("A5 joined")

feats = [c for c in df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") and not c.endswith("_d")) and c != "likpf_mean"]
log(f"features: {len(feats)}")
imp = pd.read_csv("feature_importance.csv").sort_values("gain", ascending=False)
top150 = [c for c in imp["feature"].head(150) if c in feats]
top120 = [c for c in imp["feature"].head(120) if c in feats]

y = df["target"].values.astype(np.float32)
y_fit = np.clip(y, -90.0, 90.0)
g = df["well"].values

from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
cv = GroupKFold(4)

def run(name, params, cols):
    X = df[cols].values.astype(np.float32)
    oof = np.zeros(len(df)); iters = []
    p = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1,
             max_bin=255, learning_rate=0.02, n_estimators=10000, random_state=0,
             subsample_freq=1, **params)
    for tr, va in cv.split(X, y, groups=g):
        m = LGBMRegressor(**p)
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        iters.append(int(m.best_iteration_ or 0))
    v = float(np.sqrt(np.mean((oof - y) ** 2)))
    log(f"{name:34s} OOF={v:.4f} (iters {iters})")
    return v

CFGS = [
    ("cur-lgb1 midreg l64/l2=30",  dict(num_leaves=64, min_child_samples=40, subsample=0.7, colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0), feats),
    ("cur-lgb2 l31/l2=60/ff.4",    dict(num_leaves=31, min_child_samples=60, subsample=0.6, colsample_bytree=0.4, reg_lambda=60.0, reg_alpha=2.0), feats),
    ("l31/l2=100/ff.4",            dict(num_leaves=31, min_child_samples=60, subsample=0.6, colsample_bytree=0.4, reg_lambda=100.0, reg_alpha=2.0), feats),
    ("l16/l2=60/ff.4",             dict(num_leaves=16, min_child_samples=60, subsample=0.6, colsample_bytree=0.4, reg_lambda=60.0, reg_alpha=2.0), feats),
    ("l16/l2=120/ff.3",            dict(num_leaves=16, min_child_samples=80, subsample=0.6, colsample_bytree=0.3, reg_lambda=120.0, reg_alpha=2.0), feats),
    ("l64/l2=100/ff.4",            dict(num_leaves=64, min_child_samples=40, subsample=0.7, colsample_bytree=0.4, reg_lambda=100.0, reg_alpha=1.0), feats),
    ("l31/l2=60/ff.25",            dict(num_leaves=31, min_child_samples=60, subsample=0.6, colsample_bytree=0.25, reg_lambda=60.0, reg_alpha=2.0), feats),
    ("PRUNE top150 (cur-lgb2)",    dict(num_leaves=31, min_child_samples=60, subsample=0.6, colsample_bytree=0.4, reg_lambda=60.0, reg_alpha=2.0), top150),
    ("PRUNE top120 (cur-lgb2)",    dict(num_leaves=31, min_child_samples=60, subsample=0.6, colsample_bytree=0.4, reg_lambda=60.0, reg_alpha=2.0), top120),
]
res = {}
for name, p, cols in CFGS:
    res[name] = run(name, p, cols)
log("=== ranked ===")
for k, v in sorted(res.items(), key=lambda x: x[1]):
    print(f"  {v:.4f}  {k}", flush=True)
