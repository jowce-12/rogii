# Blend judge for the spatial GRU legs at the DEPLOYED config (0.20/0.50/0.30 + gamma1.09).
# Compares gru_oof_dipfused (adopted, LB 6.663) vs spatialclean / spatialfused on the
# 2x150-well both-seed harness, plus err-corr vs the fleongg meta OOF (the 6-leg trap metric).
# RUN from ~/rogii: python judge_spatial.py          (~4min CPU)
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import blend_eval as BE
from offline_tests import pooled

t0 = time.time()
CANDS = {"dipfused": "gru_oof_dipfused.parquet",
         "spatialclean": "gru_oof_spatialclean.parquet",
         "spatialfused": "gru_oof_spatialfused.parquet"}

fs = pd.read_parquet("oof_stack.parquet").merge(
    pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                    columns=["id", "likpf_mean_d", "likpf_ptstd"]), on="id")
fy = fs["target"].values.astype(np.float64); fy_fit = np.clip(fy, -90, 90); fg = fs["well"].values
FX = np.column_stack([fs[c].values for c in ["lgb0", "lgb1", "lgb2", "cb0", "cb1"]]
                     + [np.nan_to_num(fs["likpf_mean_d"].values.astype(np.float64))])
meta = np.zeros(len(fs))
for tr, va in GroupKFold(5).split(FX, fy_fit, groups=fg):
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(FX[tr], fy_fit[tr]); meta[va] = r.predict(FX[va])
fl_s = pd.Series(meta, index=fs["id"].values)
fl_err = pd.Series(meta - fy, index=fs["id"].values)
risk_s = pd.Series(np.nan_to_num(fs["likpf_ptstd"].values.astype(np.float64)), index=fs["id"].values)
y_s = pd.Series(fy, index=fs["id"].values)
OLD5 = [f"old_{s}" for s in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]]
sub1_fn = BE.make_ridge_fn(OLD5)

# err-corr vs fleongg meta on the full OOF (deltas vs target)
print("=== err-corr vs fleongg meta (dip ref was ~0.794) ===", flush=True)
for name, path in CANDS.items():
    gr = pd.read_parquet(path).set_index("id")["gru_d"]
    common = gr.index.intersection(fl_err.index)
    ge = gr.reindex(common).values - y_s.reindex(common).values
    fe = fl_err.reindex(common).values
    print(f"  {name:13s}: corr={float(np.corrcoef(ge, fe)[0, 1]):.4f}  (n={len(common)})", flush=True)

for SEED in (7, 11):
    res, SEL = BE.selector_preds(SEED)
    wells = [r_["wid"] for r_ in res]
    sub = BE.OOF[BE.OOF["well"].isin(wells)].copy()
    sub["sub1_tvt"] = sub1_fn(sub)
    sub["fl_tvt"] = sub["last_known_tvt"].values + fl_s.reindex(sub["id"].values).values
    sub["risk"] = risk_s.reindex(sub["id"].values).values
    print(f"[{time.time()-t0:.0f}s] === seed{SEED} (150 wells) deployed 0.20/0.50/0.30 + gamma ===", flush=True)
    for name, path in CANDS.items():
        gr_s = pd.Series(pd.read_parquet(path).set_index("id")["gru_d"])
        sub["gr_tvt"] = sub["last_known_tvt"].values + gr_s.reindex(sub["id"].values).values
        by = {c: {w: g[c].values for w, g in sub.groupby("well")} for c in
              ("sub1_tvt", "fl_tvt", "gr_tvt", "last_known_tvt", "risk")}
        finals = []
        for r_, sel in zip(res, SEL):
            w = r_["wid"]
            s1, fl, gr = by["sub1_tvt"].get(w), by["fl_tvt"].get(w), by["gr_tvt"].get(w)
            if any(v is None or len(v) != len(sel) or np.isnan(v).any() for v in (s1, fl, gr)):
                finals.append(np.asarray(sel, float))
                continue
            b = 0.20 * (0.3 * s1 + 0.7 * sel) + 0.50 * fl + 0.30 * gr
            risk = float(np.mean(by["risk"][w])); last = float(by["last_known_tvt"][w][0])
            if np.isfinite(risk) and risk >= 3.39:
                b = last + 1.09 * (b - last)
            finals.append(b)
        print(f"  {name:13s}: pooled={pooled(res, finals):.4f}", flush=True)
print("DONE", flush=True)
