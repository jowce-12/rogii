# fleongg META-LAYER lab: the base models (lgb0-2, cb0, cb1) stay frozen; only the meta
# combiner changes. Deployed meta = Ridge(alpha1.66, positive) on [5 bases + likpf_mean_d].
# Every variant is scored (a) as fleongg-alone pooled RMSE via honest GroupKFold(5)-by-well
# meta-OOF, and (b) at the DEPLOYED blend on the 2x150 both-seed harness
# (normal tier .15/.45/.40, monster .20/.40/.40, ws3 .10, gamma 1.09).
# RUN from ~/rogii: python meta_lab.py            (~15min CPU)
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
import blend_eval as BE
from offline_tests import pooled

t0 = time.time()
BASES = ["lgb0", "lgb1", "lgb2", "cb0", "cb1"]
CTX = ["likpf_mean_d", "likpf_ptstd", "likpf_llspread", "likpf_grsig", "dense_dist",
       "dense_std", "extrap_ratio", "frac", "md_since", "known_len", "eval_len",
       "pfx_rmse", "tw_range", "trk_std"]

oof = pd.read_parquet("oof_stack.parquet")
feat = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet", columns=["id"] + CTX)
df = oof.merge(feat, on="id")
y = df["target"].values.astype(np.float64)
y_fit = np.clip(y, -90, 90)
g = df["well"].values
B = np.column_stack([df[c].values.astype(np.float64) for c in BASES])
C = {c: np.nan_to_num(df[c].values.astype(np.float64)) for c in CTX}
risk_well = df.groupby("well")["likpf_ptstd"].transform("mean").values
MON = np.nan_to_num(risk_well) >= 3.39
folds = list(GroupKFold(5).split(B, y_fit, groups=g))
print(f"[{time.time()-t0:.0f}s] {len(df)} rows | monster rows {MON.mean():.1%}", flush=True)


def ridge_oof(X):
    out = np.zeros(len(df))
    for tr, va in folds:
        r = Ridge(alpha=1.66, positive=True, fit_intercept=True)
        r.fit(X[tr], y_fit[tr]); out[va] = r.predict(X[va])
    return out


def tiered_ridge_oof(X):
    out = np.zeros(len(df))
    for tr, va in folds:
        for mask_val in (False, True):
            tr_m = tr[MON[tr] == mask_val]
            va_m = va[MON[va] == mask_val]
            if len(tr_m) < 1000 or len(va_m) == 0:
                if len(va_m):
                    r = Ridge(alpha=1.66, positive=True, fit_intercept=True)
                    r.fit(X[tr], y_fit[tr]); out[va_m] = r.predict(X[va_m])
                continue
            r = Ridge(alpha=1.66, positive=True, fit_intercept=True)
            r.fit(X[tr_m], y_fit[tr_m]); out[va_m] = r.predict(X[va_m])
    return out


def lgbm_meta_oof(X, leaves=15, l2=50.0, n=400):
    out = np.zeros(len(df))
    for tr, va in folds:
        m = lgb.LGBMRegressor(objective="regression", num_leaves=leaves, learning_rate=0.03,
                              n_estimators=n, min_child_samples=200, lambda_l2=l2,
                              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                              n_jobs=8, verbose=-1, seed=0)
        m.fit(X[tr], y_fit[tr])
        out[va] = m.predict(X[va])
    return out


X_base = np.column_stack([B, C["likpf_mean_d"]])
VARIANTS = {}
VARIANTS["M0_deployed"] = ridge_oof(X_base)
VARIANTS["M1_tiered_ridge"] = tiered_ridge_oof(X_base)
X_q = np.column_stack([B, C["likpf_mean_d"], C["likpf_ptstd"], C["likpf_llspread"], C["dense_dist"]])
VARIANTS["M2_ridge_moreq"] = ridge_oof(X_q)
X_ctx = np.column_stack([B, C["likpf_mean_d"]] + [C[c] for c in CTX if c != "likpf_mean_d"])
VARIANTS["M3_lgbm_ctx"] = lgbm_meta_oof(X_ctx)
VARIANTS["M4_lgbm_bases"] = lgbm_meta_oof(X_base)
VARIANTS["M5_half_lgbm"] = 0.5 * VARIANTS["M0_deployed"] + 0.5 * VARIANTS["M3_lgbm_ctx"]
for name, p in VARIANTS.items():
    print(f"[{time.time()-t0:.0f}s] fleongg-alone {name:16s} = "
          f"{float(np.sqrt(np.mean((p - y) ** 2))):.4f}", flush=True)

# ---- blend-level gate (deployed config) ----
risk_s = pd.Series(np.nan_to_num(df["likpf_ptstd"].values.astype(np.float64)), index=df["id"].values)
OLD5 = [f"old_{s}" for s in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]]
sub1_fn = BE.make_ridge_fn(OLD5)
gr_s = pd.Series(pd.read_parquet("gru_oof_dipfused5.parquet").set_index("id")["gru_d"])
s3_s = pd.Series(pd.read_parquet("s3_preds_tuned.parquet").set_index("id")["s3_tvt"])
META = {k: pd.Series(v, index=df["id"].values) for k, v in VARIANTS.items()}

for SEED in (7, 11):
    res, SEL = BE.selector_preds(SEED)
    sub = BE.OOF[BE.OOF["well"].isin([r_["wid"] for r_ in res])].copy()
    sub["sub1_tvt"] = sub1_fn(sub)
    sub["gr_tvt"] = sub["last_known_tvt"].values + gr_s.reindex(sub["id"].values).values
    sub["s3_tvt"] = s3_s.reindex(sub["id"].values).values
    sub["risk"] = risk_s.reindex(sub["id"].values).values
    parts = []
    for name, m in META.items():
        sub["fl_tvt"] = sub["last_known_tvt"].values + m.reindex(sub["id"].values).values
        by = {c: {w: gg[c].values for w, gg in sub.groupby("well")} for c in
              ("sub1_tvt", "fl_tvt", "gr_tvt", "s3_tvt", "last_known_tvt", "risk")}
        finals = []
        for r_, sel in zip(res, SEL):
            w = r_["wid"]
            s1, fl, gr, sv = (by["sub1_tvt"].get(w), by["fl_tvt"].get(w),
                              by["gr_tvt"].get(w), by["s3_tvt"].get(w))
            if any(v is None or len(v) != len(sel) or np.isnan(np.asarray(v, float)).any()
                   for v in (s1, fl, gr)):
                finals.append(np.asarray(sel, float)); continue
            risk = float(np.mean(by["risk"][w])); last = float(by["last_known_tvt"][w][0])
            monster = np.isfinite(risk) and risk >= 3.39
            wsp, wfl, wgr = (0.20, 0.40, 0.40) if monster else (0.15, 0.45, 0.40)
            b = wsp * (0.3 * s1 + 0.7 * sel) + wfl * fl + wgr * gr
            if sv is not None and len(sv) == len(sel) and np.isfinite(sv).all():
                b = 0.90 * b + 0.10 * sv
            if monster:
                b = last + 1.09 * (b - last)
            finals.append(b)
        parts.append(f"{name}={pooled(res, finals):.4f}")
    print(f"[{time.time()-t0:.0f}s] seed{SEED} BLEND: " + "  ".join(parts), flush=True)
print("DONE", flush=True)
