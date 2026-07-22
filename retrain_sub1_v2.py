# sub_1 v2: grow the ensemble with NEW DIVERSITY members (CatBoost-leaning — CB beats LGB
# on their likpf-free features; new cb2 was the best single at 10.471). Each candidate is
# trained once (koolbox Trainer, saved pkl) and judged by its MARGINAL effect on the
# best5 ridge OOF (best5 = old LGB x3 + new CB x2 = 10.3728). Only winners get packaged.
# RUN: python retrain_sub1_v2.py   (GPU env with koolbox; ~1.5-2h for 4 candidates)
import glob, os, time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, log_evaluation, early_stopping
from catboost import CatBoostRegressor
from koolbox import Trainer

t0 = time.time()
CV = GroupKFold(5)
PQ = "ravaghi_train.parquet"
train_df = pd.read_parquet(PQ) if os.path.exists(PQ) else pd.read_csv("ravaghi/data/train.csv", low_memory=False)
features = [c for c in train_df.columns if c not in {"well", "id", "target"}]
X = train_df[features]; y = train_df["target"]; g = train_df["well"]
print(f"[{time.time()-t0:.0f}s] {len(train_df)} rows, {g.nunique()} wells", flush=True)

SLOTS = ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]
def load_oof(root):
    return {s: np.asarray(joblib.load(sorted(glob.glob(f"{root}/models/{s}/*.pkl"))[0]).oof_preds, np.float64) for s in SLOTS}
OLD = load_oof("ravaghi"); NEW = load_oof("ravaghi_new")
BEST5 = [OLD["lightgbm-1"], OLD["lightgbm-2"], OLD["lightgbm-3"], NEW["catboost-1"], NEW["catboost-2"]]

def ridge_oof(cols):
    Xr = np.column_stack(cols); oof = np.zeros(len(y))
    for tr_i, va_i in CV.split(Xr, y, groups=g):
        r = Ridge(random_state=42, alpha=1.6602834637650032, tol=0.0005030247295617308,
                  positive=True, fit_intercept=True)
        r.fit(Xr[tr_i], y.values[tr_i]); oof[va_i] = r.predict(Xr[va_i])
    return float(np.sqrt(np.mean((oof - y.values) ** 2)))

base = ridge_oof(BEST5)
print(f"best5 baseline ridge OOF = {base:.4f}", flush=True)

# ---- diversity candidates (mechanism-diverse, not param-clones) ----
CANDS = [
    ("cb-mvs-d10",  "cb",  dict(iterations=8000, depth=10, l2_leaf_reg=3.0, min_data_in_leaf=50,
                                border_count=254, bootstrap_type="MVS", loss_function="RMSE",
                                task_type="GPU", devices="0", od_type="Iter", od_wait=300,
                                verbose=0, learning_rate=0.02, random_seed=31)),
    ("cb-rsm-d6",   "cb",  dict(iterations=8000, depth=6, l2_leaf_reg=10.0, min_data_in_leaf=15,
                                border_count=254, rsm=0.5, loss_function="RMSE",
                                task_type="GPU", devices="0", od_type="Iter", od_wait=300,
                                verbose=0, learning_rate=0.03, random_seed=57)),
    ("cb-depth5-lr05", "cb", dict(iterations=8000, depth=5, l2_leaf_reg=6.0, min_data_in_leaf=30,
                                border_count=254, loss_function="RMSE",
                                task_type="GPU", devices="0", od_type="Iter", od_wait=300,
                                verbose=0, learning_rate=0.05, random_seed=91)),
    ("lgb-dart",    "lgb", dict(boosting_type="dart", num_leaves=64, min_child_samples=40,
                                subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                                reg_lambda=30.0, reg_alpha=1.0, objective="regression",
                                verbose=-1, n_jobs=-1, max_bin=255, drop_rate=0.1,
                                learning_rate=0.03, n_estimators=1500, random_state=77)),
]

results = {}
kept = list(BEST5)
kept_names = []
for name, kind, params in CANDS:
    print(f"\n=== candidate {name} ===", flush=True)
    save_path = f"ravaghi_new/models/cand-{name}"
    try:
        if kind == "cb":
            tr = Trainer(estimator=CatBoostRegressor(**params), task="regression",
                         metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                         use_early_stopping=True, verbose=True, save=True, save_path=save_path)
            tr.fit(X, y, fit_args={"verbose": 500, "early_stopping_rounds": 250, "use_best_model": True})
        else:
            # dart: no early stopping (dart's best_iteration is unreliable) -> fixed trees
            tr = Trainer(estimator=LGBMRegressor(**params), task="regression",
                         metric=root_mean_squared_error, cv=CV, cv_args={"groups": g},
                         use_early_stopping=False, verbose=True, save=True, save_path=save_path)
            tr.fit(X, y)
        o = np.asarray(tr.oof_preds, np.float64)
        single = tr.overall_score
        marg = ridge_oof(BEST5 + [o])
        cum = ridge_oof(kept + [o])
        print(f"[{time.time()-t0:.0f}s] {name}: single={single:.4f}  best5+cand={marg:.4f} "
              f"(base {base:.4f})  cumulative={cum:.4f}", flush=True)
        results[name] = dict(single=single, marginal=marg, cumulative=cum)
        if cum < ridge_oof(kept) - 0.005:      # keep only real marginal gains
            kept.append(o); kept_names.append(name)
            print(f"  -> KEPT ({'+'.join(kept_names)})", flush=True)
    except Exception as e:
        print(f"  {name} FAILED: {str(e)[:100]}", flush=True)

print("\n=== summary ===", flush=True)
for k, v in results.items():
    print(f"  {k:15s} single={v['single']:.4f} best5+={v['marginal']:.4f} cum={v['cumulative']:.4f}", flush=True)
print(f"final kept set: best5 + {kept_names} -> ridge OOF {ridge_oof(kept):.4f} (baseline {base:.4f})", flush=True)
print("If winners exist: package = old lgb1-3 + new cb1-2 + kept candidates; the notebook's", flush=True)
print("cb_params/lgb_params lists then get extended to load the extra slots (my patch).", flush=True)
