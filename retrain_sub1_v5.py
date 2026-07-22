# sub_1 v5: the two axes the dose/v4/all10 failures left untried.
#   A "lgb-geo"       : THEIR original 195 features + 9 PURE-GEOMETRY/context cols from
#                       our v7 cache (head_cos/sin, incl, unc_x_*, extrap_*, gr_ev_valid_frac,
#                       twloc_std). No GR-matching signal -> decorrelation preserved by design
#                       (v4 died importing PF-track-derived diagnostics; this is the residue).
#   B "lgb-huber"     : THEIR 195 unchanged, objective=huber (delta 10ft). Error-profile
#                       decorrelation via the LOSS, not the features — immune to the
#                       correlation-import failure mode entirely.
#   C "lgb-geo-huber" : both combined.
# Judgment (my side, after OOF export): blend objective at w_sub1=0.18 AND 0.30 on BOTH
# harness seeds + fleongg err-corr guard (oof_stack). all10 lesson: harness pass alone is
# not an LB ticket — only clear margins (>=0.05 everywhere) get a slot proposal.
# RUN (isic env): python retrain_sub1_v5.py     (~1h, 3 candidates, checkpointed)
import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from koolbox import Trainer

t0 = time.time()
CV = GroupKFold(5)
train_df = pd.read_parquet("ravaghi_train.parquet")
features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
y = train_df["target"].astype(np.float64)
g = train_df["well"]
X_base = train_df[features]
print(f"[{time.time()-t0:.0f}s] {len(train_df)} rows | {len(features)} base feats", flush=True)

GEO = ["head_cos", "head_sin", "incl", "unc_x_frac", "unc_x_dist",
       "extrap_ratio", "extrap_vs_twrange", "gr_ev_valid_frac", "twloc_std"]
geo = train_df[["id"]].merge(
    pd.read_parquet("train_features_v7_f1_cblend_w773.parquet", columns=["id"] + GEO),
    on="id", how="left")
_nan = float(geo[GEO].isna().mean().mean())
assert _nan < 0.05, f"geo join NaN rate {_nan:.3f} — id alignment broken"
X_geo = pd.concat([X_base.reset_index(drop=True), geo[GEO].reset_index(drop=True)], axis=1)
print(f"geo join OK ({X_geo.shape[1]} feats, NaN {_nan:.4f})", flush=True)

MIDREG = dict(boosting_type="gbdt", num_leaves=64, min_child_samples=40, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0,
              verbose=-1, n_jobs=-1, max_bin=255, learning_rate=0.02, n_estimators=10000)
CANDS = [
    ("lgb-geo",       X_geo,  dict(MIDREG, objective="regression", random_state=31)),
    ("lgb-huber",     X_base, dict(MIDREG, objective="huber", alpha=10.0, random_state=32)),
    ("lgb-geo-huber", X_geo,  dict(MIDREG, objective="huber", alpha=10.0, random_state=33)),
]

OUT = Path("sub1_oof_v5.parquet")
oof_df = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame({"id": train_df["id"]})
for name, Xc, params in CANDS:
    if name in oof_df.columns:
        print(f"[skip] {name} already done", flush=True)
        continue
    print(f"\n=== {name} ({Xc.shape[1]} feats, obj={params['objective']}) ===", flush=True)
    tr = Trainer(estimator=LGBMRegressor(**params), task="regression",
                 metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                 use_early_stopping=True, verbose=True, save=True,
                 save_path=f"ravaghi_new/models/cand-{name}")
    tr.fit(Xc, y, fit_args={"eval_metric": "rmse",
                            "callbacks": [log_evaluation(period=500),
                                          early_stopping(stopping_rounds=250)]})
    o = np.asarray(tr.oof_preds, np.float32)
    oof_df[name] = o
    oof_df.to_parquet(OUT, index=False)
    print(f"[{time.time()-t0:.0f}s] {name}: single={float(np.sqrt(np.mean((o - y.values)**2))):.4f} -> saved", flush=True)

print(f"\nDONE [{time.time()-t0:.0f}s] -> {OUT} (+ pkls in ravaghi_new/models/cand-*)", flush=True)
print("Next: blend-objective judgment on the Windows side (w_sub1 0.18 & 0.30, both seeds,", flush=True)
print("fleongg err-corr guard). Only clear all-gate passes become packaging candidates.", flush=True)
