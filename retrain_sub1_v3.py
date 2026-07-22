# sub_1 Phase-2: comprehensive candidate pool, ALL judged later by the BLEND objective
# (0.3*sub1+0.7*selector on harness; Phase-1 showed sub_1-OOF ranking is misleading —
# all10 beats the OOF-best mix on both seeds). This script only TRAINS + saves OOFs/pkls;
# selection happens in blend_eval on the Windows side.
#
# Candidate axes (LGB/CB only — no new model families, per user):
#   B target handling  : winsorized +-90 fit (validated on the fleongg stack)
#   C likpf dosing     : their features + likpf_mean_d (dose1) / +4 scale_d (dose2)
#                        — attacks the actual 1.7-gap source; blend metric arbitrates
#                        the correlation-vs-information tradeoff empirically
#
# RUN (isic env):  python retrain_sub1_v3.py     (~1h GPU)
import glob, os, time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from koolbox import Trainer

t0 = time.time()
CV = GroupKFold(5)
PQ = "ravaghi_train.parquet"
train_df = pd.read_parquet(PQ) if os.path.exists(PQ) else pd.read_csv("ravaghi/data/train.csv", low_memory=False)
features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
y_raw = train_df["target"].astype(np.float64)
y_winz = y_raw.clip(-90.0, 90.0)
g = train_df["well"]
X_base = train_df[features]
print(f"[{time.time()-t0:.0f}s] {len(train_df)} rows | {len(features)} base feats", flush=True)

# likpf dosing frames (id-join from our v7 cache extract)
LK = pd.read_parquet("likpf_join.parquet")
lk = train_df[["id"]].merge(LK, on="id", how="left")
assert len(lk) == len(train_df)
DOSE1 = ["likpf_mean_d"]
DOSE2 = DOSE1 + ["likpf_scale_3_d", "likpf_scale_5_d", "likpf_scale_8_d", "likpf_scale_12_d"]
X_dose1 = pd.concat([X_base.reset_index(drop=True), lk[DOSE1].reset_index(drop=True)], axis=1)
X_dose2 = pd.concat([X_base.reset_index(drop=True), lk[DOSE2].reset_index(drop=True)], axis=1)
print(f"dose1 {X_dose1.shape[1]} feats | dose2 {X_dose2.shape[1]} feats | likpf NaN {lk['likpf_mean_d'].isna().mean():.4f}", flush=True)

MIDREG = dict(boosting_type="gbdt", num_leaves=64, min_child_samples=40, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0,
              objective="regression", verbose=-1, n_jobs=-1, max_bin=255,
              learning_rate=0.02, n_estimators=10000, random_state=0)
CB_BERN = dict(iterations=8000, depth=9, l2_leaf_reg=5.0, min_data_in_leaf=30, border_count=254,
               bootstrap_type="Bernoulli", subsample=0.7, loss_function="RMSE", task_type="GPU",
               devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=123)

def lgb_fit_args():
    from lightgbm import log_evaluation, early_stopping
    return {"eval_metric": "rmse", "callbacks": [log_evaluation(period=500), early_stopping(stopping_rounds=250)]}

CANDS = [
    # (name, kind, params, X, y) — model-diversity group removed at user request
    ("lgb-midreg-winz", "lgb", MIDREG, "base", "winz"),
    ("cb-bern-winz",    "cb",  CB_BERN, "base", "winz"),
    ("lgb-midreg-dose1", "lgb", dict(MIDREG, random_state=11), "dose1", "raw"),
    ("lgb-midreg-dose2", "lgb", dict(MIDREG, random_state=12), "dose2", "raw"),
    ("cb-bern-dose2",    "cb",  dict(CB_BERN, random_seed=13), "dose2", "raw"),
]

XMAP = {"base": X_base, "dose1": X_dose1, "dose2": X_dose2}
OUT = Path("sub1_oof_v3.parquet")
oof_df = pd.DataFrame({"id": train_df["id"]})
if OUT.exists():
    oof_df = pd.read_parquet(OUT)

for name, kind, params, xkey, ykey in CANDS:
    if name in oof_df.columns:
        print(f"[skip] {name} already done", flush=True)
        continue
    Xc = XMAP[xkey]; yc = y_winz if ykey == "winz" else y_raw
    print(f"\n=== {name} (X={xkey}, y={ykey}) ===", flush=True)
    try:
        if kind == "cb":
            tr = Trainer(estimator=CatBoostRegressor(**params), task="regression",
                         metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                         use_early_stopping=True, verbose=True, save=True,
                         save_path=f"ravaghi_new/models/cand-{name}")
            tr.fit(Xc, yc, fit_args={"verbose": 500, "early_stopping_rounds": 250, "use_best_model": True})
        elif kind == "lgb":
            tr = Trainer(estimator=LGBMRegressor(**params), task="regression",
                         metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                         use_early_stopping=True, verbose=True, save=True,
                         save_path=f"ravaghi_new/models/cand-{name}")
            tr.fit(Xc, yc, fit_args=lgb_fit_args())
        else:
            raise ValueError(f"unknown kind {kind}")
        o = np.asarray(tr.oof_preds, np.float32)
        single = float(np.sqrt(np.mean((o - y_raw.values) ** 2)))   # vs RAW target always
        oof_df[name] = o
        oof_df.to_parquet(OUT, index=False)                          # incremental checkpoint
        print(f"[{time.time()-t0:.0f}s] {name}: single(raw)={single:.4f}  -> saved", flush=True)
    except Exception as e:
        print(f"  {name} FAILED: {str(e)[:120]}", flush=True)

print(f"\nDONE [{time.time()-t0:.0f}s] -> {OUT} (+ pkls in ravaghi_new/models/cand-*)", flush=True)
print("Next: run blend_eval extensions on the Windows side to greedy-select winners on the", flush=True)
print("blend objective, then package + notebook loop-extension patch.", flush=True)
