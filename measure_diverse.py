# fleongg BASE DIVERSITY experiment (stack-level proxy, 150 wells, GKF-4, fixed folds).
# Motivation: today's v3-swap failure was a DIVERSITY failure — one column took 21.8% of
# gain, all five bases leaned on it and the ridge collapsed onto lgb0 (0.48). The five
# deployed bases already share the same two evidence axes (dense/geology ~19% + likpf
# ~40% of gain), so they correlate. Deliberately BLINDING extra bases to one axis should
# decorrelate them and give the positive-ridge meta something to lean on where that axis
# fails (exactly the monster wells, where sp45's optimal share measured 0.01).
# Unlike the single-LGB screens, this evaluates the STACK (bases + positive ridge), which
# is the structure the decision is actually about.
# RUN from ~/rogii: python measure_diverse.py      (~25min CPU)
import re
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

t0 = time.time()
wells_all = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
wells150 = sorted(np.random.default_rng(0).choice(wells_all, 150, replace=False))
df = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     filters=[("well", "in", wells150)])
df = df.merge(pd.read_parquet("stride_join.parquet"), on="id", how="left")
FEATS = [c for c in df.columns if c not in ("id", "well", "target")
         and pd.api.types.is_numeric_dtype(df[c])]
y = df["target"].values.astype(np.float64)
y_fit = np.clip(y, -90, 90)
groups = df["well"].values
folds = list(GroupKFold(4).split(df[FEATS].values, y_fit, groups=groups))

GEO = re.compile(r"^(tvt_dense|dense_|form_|frm_rmse_|spatial_|bw|bww|tvtF|tvtFw|pf_ancc)")
LIKPF = re.compile(r"^(likpf_|trk_)")
F_ALL = FEATS
F_NOGEO = [c for c in FEATS if not GEO.match(c)]
F_NOLIKPF = [c for c in FEATS if not LIKPF.match(c)]
print(f"[{time.time()-t0:.0f}s] {len(df)} rows | all {len(F_ALL)} | no-geo {len(F_NOGEO)} | "
      f"no-likpf {len(F_NOLIKPF)}", flush=True)

DEEP = dict(objective="regression", num_leaves=255, learning_rate=0.02, feature_fraction=0.6,
            bagging_fraction=0.7, bagging_freq=1, min_child_samples=20, lambda_l2=20.0,
            lambda_l1=1.0, n_estimators=3000, n_jobs=8, verbose=-1, seed=123)
SHAL = dict(objective="regression", num_leaves=64, learning_rate=0.02, feature_fraction=0.6,
            bagging_fraction=0.7, bagging_freq=1, min_child_samples=40, lambda_l2=30.0,
            n_estimators=3000, n_jobs=8, verbose=-1, seed=0)
SHAL2 = dict(SHAL, num_leaves=31, feature_fraction=0.4, bagging_fraction=0.6,
             lambda_l2=60.0, min_child_samples=60, seed=7)


def base_oof(tag, params, cols, target=None, rescale=None):
    X = df[cols].values.astype(np.float32)
    tgt = y_fit if target is None else target
    oof = np.zeros(len(df))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], tgt[tr], eval_set=[(X[va], tgt[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va])
    if rescale is not None:
        oof = oof * rescale
    print(f"[{time.time()-t0:.0f}s]   base {tag:12s} OOF {float(np.sqrt(np.mean((oof - y) ** 2))):.4f} "
          f"({len(cols)} feats)", flush=True)
    return oof


md = np.nan_to_num(df["md_since"].values.astype(np.float64))
scale = 1.0 + md / 1000.0
B = {}
B["deep"] = base_oof("deep", DEEP, F_ALL)
B["shallow"] = base_oof("shallow", SHAL, F_ALL)
B["shallow2"] = base_oof("shallow2", SHAL2, F_ALL)
B["nogeo"] = base_oof("nogeo", SHAL, F_NOGEO)
B["nolikpf"] = base_oof("nolikpf", SHAL, F_NOLIKPF)
B["rate"] = base_oof("rate", SHAL, F_ALL, target=y_fit / scale, rescale=scale)

anchor = np.nan_to_num(df["likpf_mean_d"].values.astype(np.float64))


def stack(names):
    M = np.column_stack([B[n] for n in names] + [anchor])
    oof = np.zeros(len(df))
    coefs = np.zeros(len(names) + 1)
    for tr, va in folds:
        r = Ridge(alpha=1.66, positive=True, fit_intercept=True)
        r.fit(M[tr], y_fit[tr]); oof[va] = r.predict(M[va])
        coefs += r.coef_ / len(folds)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    cs = "  ".join(f"{n}={c:.3f}" for n, c in zip(names + ["anchor"], coefs))
    print(f"[{time.time()-t0:.0f}s] stack {'+'.join(names):38s} = {rmse:.4f} | {cs}", flush=True)
    return rmse


print("\n== error correlation between bases ==", flush=True)
names = list(B)
E = np.column_stack([B[n] - y for n in names])
C = np.corrcoef(E.T)
print("        " + "".join(f"{n[:8]:>10s}" for n in names), flush=True)
for i, n in enumerate(names):
    print(f"{n[:8]:>8s}" + "".join(f"{C[i, j]:10.3f}" for j in range(len(names))), flush=True)

print("\n== stacks ==", flush=True)
s0 = stack(["deep", "shallow", "shallow2"])
stack(["deep", "shallow", "shallow2", "nogeo"])
stack(["deep", "shallow", "shallow2", "nolikpf"])
s1 = stack(["deep", "shallow", "shallow2", "nogeo", "nolikpf"])
stack(["deep", "shallow", "shallow2", "nogeo", "nolikpf", "rate"])
print(f"\nbaseline 3-base stack {s0:.4f} -> +blinded bases {s1:.4f} ({s1 - s0:+.4f})", flush=True)
print("DONE", flush=True)
