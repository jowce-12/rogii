# sub_1 (ravaghi/koolbox stack) param sweep on THEIR feature table (198 cols, no likpf —
# keeping the decorrelation vs our track B). Diagnosis from their notebook: lightgbm-2/3
# are SEED CLONES (leaves64/l2=95.75/lr0.0093) and lightgbm-1 is under-regularized
# (l2=3) — the same diseases our patch13 sweep fixed on the fleongg stack.
# Protocol: single-LGB 4-fold OOF proxy, tune on wells[:150], confirm on wells[150:300].
import os, time
import numpy as np
import pandas as pd

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

PQ = "ravaghi_train.parquet"
if not os.path.exists(PQ):
    log("converting ravaghi/data/train.csv -> parquet (one-time)...")
    df = pd.read_csv("ravaghi/data/train.csv", low_memory=False)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df.to_parquet(PQ, index=False)
    del df
    log("saved", PQ)

ALL = sorted(pd.read_parquet(PQ, columns=["well"])["well"].unique())
log(f"{len(ALL)} wells")

from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

CFGS = [
    ("their-1 deep255/l2=3/lr.03",  dict(num_leaves=255, min_child_samples=15, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03, n_estimators=5000, seed=123)),
    ("their-2 l64/l2=95.75/lr.009", dict(num_leaves=64, min_child_samples=40, subsample=0.4744, colsample_bytree=0.3928, reg_lambda=95.754, reg_alpha=10.788, min_child_weight=0.2408, learning_rate=0.009345, n_estimators=10000, random_state=0)),
    ("midreg l64/l2=30/ff.6",       dict(num_leaves=64, min_child_samples=40, subsample=0.7, colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0, learning_rate=0.02, n_estimators=10000, random_state=0)),
    ("l31/l2=60/ff.4",              dict(num_leaves=31, min_child_samples=60, subsample=0.6, colsample_bytree=0.4, reg_lambda=60.0, reg_alpha=2.0, learning_rate=0.02, n_estimators=10000, random_state=29)),
    ("l16/l2=120/ff.3",             dict(num_leaves=16, min_child_samples=80, subsample=0.6, colsample_bytree=0.3, reg_lambda=120.0, reg_alpha=2.0, learning_rate=0.02, n_estimators=10000, random_state=0)),
]

def run_sample(wells, tag, cfg_list):
    df = pd.read_parquet(PQ, filters=[("well", "in", wells)])
    feats = [c for c in df.columns if c not in {"well", "id", "target"}]
    X = df[feats].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    g = df["well"].values
    log(f"== {tag}: {df['well'].nunique()} wells, {len(df)} rows, {len(feats)} feats")
    out = {}
    for name, p in cfg_list:
        pp = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1,
                  max_bin=255, subsample_freq=1, **p)
        oof = np.zeros(len(df))
        for tr, va in GroupKFold(4).split(X, y, groups=g):
            m = LGBMRegressor(**pp)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
                  callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        v = float(np.sqrt(np.mean((oof - y) ** 2)))
        out[name] = v
        log(f"  {name:30s} OOF={v:.4f}")
    return out

A = run_sample(ALL[:150], "tune (wells 0-150)", CFGS)
top = sorted(A, key=A.get)[:3]
log(f"top-3 on tune: {top}")
B = run_sample(ALL[150:300], "confirm (wells 150-300)", [c for c in CFGS if c[0] in top or c[0].startswith("their")])
log("=== summary (tune / confirm) ===")
for name, _ in CFGS:
    a = A.get(name); b = B.get(name)
    print(f"  {name:30s} {a:.4f} / {b if b is None else f'{b:.4f}'}", flush=True)
