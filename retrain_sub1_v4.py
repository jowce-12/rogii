# sub_1 v4: the likpf-FREE feature upgrade — the axis the dose experiment should have
# been. X = our v7 feature table (their 195 improved + our derived/quality additions)
# MINUS every likpf-derived column, PLUS the raw-GR A5 alias diagnostics.
# Keeps sub_1's decorrelation (no selector signal) while giving it our feature
# engineering. Judged, as always, by the blend objective afterwards.
# RUN (isic env): python retrain_sub1_v4.py     (~40min GPU, 2 candidates)
import os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from koolbox import Trainer

t0 = time.time()
CV = GroupKFold(5)
V7 = "train_features_v7_f1_cblend_w773.parquet"

LIKPF = {"likpf_mean", "likpf_mean_d", "likpf_scale_3", "likpf_scale_3_d",
         "likpf_scale_5", "likpf_scale_5_d", "likpf_scale_8", "likpf_scale_8_d",
         "likpf_scale_12", "likpf_scale_12_d", "likpf_tspread",
         "likpf_ptstd", "likpf_llspread", "likpf_bestll", "likpf_grsig"}
schema = pq.ParquetFile(V7).schema.names
feats = [c for c in schema if c not in LIKPF and c not in {"well", "id", "target"}]
cols = ["well", "id", "target"] + feats
df = pd.read_parquet(V7, columns=cols)
a5 = pd.read_parquet("a5_join.parquet")
df = df.merge(a5, on="id", how="left")
A5 = ["gr_corr", "gr_corr_hf", "tw_hf_std", "alias_gap"]
X = df[feats + A5]
y = df["target"].astype(np.float64)
g = df["well"]
print(f"[{time.time()-t0:.0f}s] {len(df)} rows | {X.shape[1]} likpf-free feats "
      f"(v7-minus-likpf {len(feats)} + A5 {len(A5)})", flush=True)

# sanity: same target definition as ravaghi's table (id-aligned delta target)
rv = pd.read_parquet("ravaghi_train.parquet", columns=["id", "target"]).set_index("id")
chk = df[["id", "target"]].set_index("id").join(rv, rsuffix="_rv").dropna()
d = float((chk["target"] - chk["target_rv"]).abs().max())
print(f"target parity vs ravaghi table: max|diff|={d:.6f}", flush=True)
assert d < 1e-3, "target definition mismatch"

CANDS = [
    ("lgb-midreg-nolik", "lgb", dict(boosting_type="gbdt", num_leaves=64, min_child_samples=40,
                                     subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                                     reg_lambda=30.0, reg_alpha=1.0, objective="regression",
                                     verbose=-1, n_jobs=-1, max_bin=255,
                                     learning_rate=0.02, n_estimators=10000, random_state=21)),
    ("cb-bern-nolik",   "cb",  dict(iterations=8000, depth=9, l2_leaf_reg=5.0, min_data_in_leaf=30,
                                    border_count=254, bootstrap_type="Bernoulli", subsample=0.7,
                                    loss_function="RMSE", task_type="GPU", devices="0",
                                    od_type="Iter", od_wait=300, verbose=0,
                                    learning_rate=0.02, random_seed=22)),
]

OUT = "sub1_oof_v4.parquet"
oof_df = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame({"id": df["id"]})
for name, kind, params in CANDS:
    if name in oof_df.columns:
        print(f"[skip] {name}", flush=True)
        continue
    print(f"\n=== {name} ===", flush=True)
    if kind == "cb":
        tr = Trainer(estimator=CatBoostRegressor(**params), task="regression",
                     metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                     use_early_stopping=True, verbose=True, save=True,
                     save_path=f"ravaghi_new/models/cand-{name}")
        tr.fit(X, y, fit_args={"verbose": 500, "early_stopping_rounds": 250, "use_best_model": True})
    else:
        tr = Trainer(estimator=LGBMRegressor(**params), task="regression",
                     metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                     use_early_stopping=True, verbose=True, save=True,
                     save_path=f"ravaghi_new/models/cand-{name}")
        tr.fit(X, y, fit_args={"eval_metric": "rmse",
                               "callbacks": [log_evaluation(period=500),
                                             early_stopping(stopping_rounds=250)]})
    o = np.asarray(tr.oof_preds, np.float32)
    oof_df[name] = o
    oof_df.to_parquet(OUT, index=False)
    print(f"[{time.time()-t0:.0f}s] {name}: single={tr.overall_score:.4f} -> saved", flush=True)

print(f"\nDONE [{time.time()-t0:.0f}s] -> {OUT}; judgment = blend objective (my side)", flush=True)
