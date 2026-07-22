# Retrain the sub_1 (ravaghi/koolbox) stack on THEIR feature table, fixing the diagnosed
# diseases (lightgbm-2/3 seed clones, lightgbm-1 under-regularized) with sweep-validated
# configs. Output: ravaghi_new/ in the EXACT layout the notebooks load
# (models/lightgbm-N/*.pkl, models/catboost-N/*.pkl, data/train.csv) -> upload as a
# Kaggle dataset and point artifacts_path at it (notebook patch handled separately).
#
# REQUIRES:  pip install koolbox     (the Trainer class the pkl files must contain)
# RUN:       python retrain_sub1.py  (GPU: LGB cuda w/ cpu fallback, CatBoost GPU)
import os, shutil, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from koolbox import Trainer

t0 = time.time()
OUT = Path("ravaghi_new")
CV = GroupKFold(n_splits=5)

# ---------------- data (their features, verbatim) ----------------
PQ = "ravaghi_train.parquet"
train_df = pd.read_parquet(PQ) if os.path.exists(PQ) else pd.read_csv("ravaghi/data/train.csv", low_memory=False)
features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
X = train_df[features]
y = train_df["target"]
g = train_df["well"]
print(f"[{time.time()-t0:.0f}s] {len(train_df)} rows, {len(features)} feats, {g.nunique()} wells", flush=True)

# ---------------- model lineup ----------------
# FINALIZE FROM sweep_sub1.log (tune wells 0-150 / confirm 150-300) before running.
# Principles: break the lgb-2/3 clone with two DIFFERENT validated configs; keep one
# deep member only if the sweep supports it; diversify catboost-2 (patch16 recipe).
_dev = os.environ.get("SUB1_LGB_DEV", "cuda")   # cuda -> cpu fallback below
LGB_LINEUP = [
    # lightgbm-1: deep slot (their config, retuned reg) — REVIEW vs sweep
    dict(boosting_type="gbdt", num_leaves=255, min_child_samples=40, subsample=0.7,
         subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0, reg_alpha=1.0,
         objective="regression", verbose=-1, n_jobs=-1, max_bin=255,
         learning_rate=0.02, n_estimators=5000, seed=123),
    # lightgbm-2: midreg (patch13 winner class) — REVIEW vs sweep
    dict(boosting_type="gbdt", num_leaves=64, min_child_samples=40, subsample=0.7,
         subsample_freq=1, colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0,
         objective="regression", verbose=-1, n_jobs=-1, max_bin=255,
         learning_rate=0.02, n_estimators=10000, random_state=0),
    # lightgbm-3: their-2 config kept as the second diverse member (sweep runner-up
    # 10.911/11.316; differs from midreg in reg strength AND lr -> clone broken).
    # Our shallow strong-reg configs LOST on their likpf-free features (regime-dependent).
    dict(boosting_type="gbdt", num_leaves=64, min_child_samples=40, subsample=0.47437582748953966,
         subsample_freq=1, colsample_bytree=0.39283351290380497, reg_lambda=95.75401894533888,
         reg_alpha=10.788188919840913, min_child_weight=0.24081152127177283,
         objective="regression", verbose=-1, n_jobs=-1, max_bin=255,
         learning_rate=0.00934485794382918, n_estimators=10000, random_state=29),
]
CB_LINEUP = [
    # catboost-1: their config (kept)
    dict(iterations=8000, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
         loss_function="RMSE", task_type="GPU", devices="0", od_type="Iter", od_wait=300,
         verbose=0, learning_rate=0.020, random_seed=7),
    # catboost-2: diversified (patch16 recipe: deeper + Bernoulli — was an lr-only clone)
    dict(iterations=8000, depth=9, l2_leaf_reg=5.0, min_data_in_leaf=30, border_count=254,
         bootstrap_type="Bernoulli", subsample=0.7, loss_function="RMSE", task_type="GPU",
         devices="0", od_type="Iter", od_wait=300, verbose=0, learning_rate=0.020,
         random_seed=123),
]

def fit_lgb(i, params):
    for dev in ([_dev, "cpu"] if _dev != "cpu" else ["cpu"]):
        p = dict(params)
        if dev != "cpu":
            p["device_type"] = dev
        try:
            tr = Trainer(estimator=LGBMRegressor(**p), task="regression",
                         metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                         use_early_stopping=True, verbose=True, save=True,
                         save_path=str(OUT / f"models/lightgbm-{i+1}"))
            tr.fit(X, y, fit_args={"eval_metric": "rmse",
                                   "callbacks": [log_evaluation(period=250),
                                                 early_stopping(stopping_rounds=250)]})
            return tr
        except Exception as e:
            print(f"[lightgbm-{i+1}] {dev} failed ({str(e)[:70]}) -> retry cpu", flush=True)
    raise RuntimeError(f"lightgbm-{i+1} failed on all devices")

oof, scores = {}, {}
for i, params in enumerate(LGB_LINEUP):
    print(f"\n=== lightgbm-{i+1} ===", flush=True)
    tr = fit_lgb(i, params)
    oof[f"lightgbm-{i+1}"] = tr.oof_preds
    scores[f"lightgbm-{i+1}"] = tr.overall_score
    print(f"[{time.time()-t0:.0f}s] lightgbm-{i+1} OOF={tr.overall_score:.4f}", flush=True)

for i, params in enumerate(CB_LINEUP):
    print(f"\n=== catboost-{i+1} ===", flush=True)
    tr = Trainer(estimator=CatBoostRegressor(**params), task="regression",
                 metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                 use_early_stopping=True, verbose=True, save=True,
                 save_path=str(OUT / f"models/catboost-{i+1}"))
    tr.fit(X, y, fit_args={"verbose": 250, "early_stopping_rounds": 250,
                           "use_best_model": True})
    oof[f"catboost-{i+1}"] = tr.oof_preds
    scores[f"catboost-{i+1}"] = tr.overall_score
    print(f"[{time.time()-t0:.0f}s] catboost-{i+1} OOF={tr.overall_score:.4f}", flush=True)

# ---------------- ridge check (their exact meta) vs old stack (10.42) ----------------
oof_df = pd.DataFrame(oof)
ridge = Ridge(random_state=42, alpha=1.6602834637650032, tol=0.0005030247295617308,
              positive=True, fit_intercept=True)
meta = np.zeros(len(train_df))
for tr_i, va_i in CV.split(oof_df.values, y, groups=g):
    ridge.fit(oof_df.values[tr_i], y.values[tr_i])
    meta[va_i] = ridge.predict(oof_df.values[va_i])
rr = float(np.sqrt(np.mean((meta - y.values) ** 2)))
print("\nper-model OOF:", {k: round(v, 4) for k, v in scores.items()}, flush=True)
print(f"*** NEW sub_1 ridge OOF = {rr:.4f}  (old ravaghi stack: 10.4197) ***", flush=True)

# ---------------- package (exact artifacts layout) ----------------
(OUT / "data").mkdir(parents=True, exist_ok=True)
if not (OUT / "data" / "train.csv").exists():
    shutil.copy("ravaghi/data/train.csv", OUT / "data" / "train.csv")
print(f"\nDONE [{time.time()-t0:.0f}s] -> {OUT}/ (models/* + data/train.csv)", flush=True)
print("Upload ravaghi_new/ as a Kaggle dataset; the notebook's artifacts_path will be "
      "repointed in a separate patch once the dataset name is known.", flush=True)
