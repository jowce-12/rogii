# Phase-1 evaluator: the TRUE sub_1 objective = pooled RMSE of 0.3*sub_1 + 0.7*selector
# on the 2x150 harness wells (selector = deployed chain+STRIDE, TVT space).
# Judges sub_1 variants by BLEND contribution, not by sub_1 OOF alone.
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from offline_tests import load, pooled, b0, ba
import stride

S2_THR = 0.0083
OOF = pd.read_parquet("sub1_oof.parquet")

def chain_stride(rec):
    p = 0.65 * b0(rec) + 0.35 * ba(rec)
    if float(np.median(rec["dense_dist"])) <= S2_THR:
        p = 0.7 * p + 0.3 * rec["tvt_dense"]
    hw, tw = stride.load_well(rec["wid"])
    st, _ = stride.stride_track(hw, tw)
    if st is not None and len(st) == len(p) and np.all(np.isfinite(st)):
        p = 0.8 * p + 0.2 * st
    return p

_SEL = {}
def selector_preds(seed):
    if seed not in _SEL:
        res = load(seed)
        _SEL[seed] = (res, Parallel(n_jobs=24, prefer="threads")(delayed(chain_stride)(r) for r in res))
    return _SEL[seed]

def eval_blend(ridge_cols_fn, label, w_sub1=0.3):
    """ridge_cols_fn(sub) -> per-row sub_1 TVT prediction array for a harness subset."""
    out = []
    for seed in (7, 11):
        res, SEL = selector_preds(seed)
        wells = [r["wid"] for r in res]
        sub = OOF[OOF["well"].isin(wells)].copy()
        sub["sub1_tvt"] = ridge_cols_fn(sub)
        by_well = {w: gdf["sub1_tvt"].values for w, gdf in sub.groupby("well")}
        B = []
        for r, sel in zip(res, SEL):
            s1 = by_well.get(r["wid"])
            if s1 is None or len(s1) != len(sel):
                B.append(sel)
                continue
            B.append(w_sub1 * s1 + (1 - w_sub1) * sel)
        v = pooled(res, B)
        out.append(v)
    print(f"  {label:36s} blend: seed7={out[0]:.4f} seed11={out[1]:.4f}", flush=True)
    return out

def make_ridge_fn(cols, alpha=1.6602834637650032):
    """Fit positive ridge on FULL-data OOF cols (grouped CV OOF-of-OOF is overkill for
    weighting; the members are already OOF), return per-row TVT predictor."""
    from sklearn.linear_model import Ridge
    X = OOF[cols].values.astype(np.float64)
    y = OOF["target"].values.astype(np.float64)
    r = Ridge(random_state=42, alpha=alpha, tol=0.0005030247295617308, positive=True, fit_intercept=True)
    r.fit(X, y)
    def fn(sub):
        return sub["last_known_tvt"].values + r.predict(sub[cols].values.astype(np.float64))
    return fn

if __name__ == "__main__":
    OLD5 = [f"old_{s}" for s in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]]
    NEW5 = [f"new_{s}" for s in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]]
    BEST5 = ["old_lightgbm-1", "old_lightgbm-2", "old_lightgbm-3", "new_catboost-1", "new_catboost-2"]
    print("== blend objective (0.3*sub1 + 0.7*selector), harness 2x150 ==", flush=True)
    eval_blend(lambda sub: sub["last_known_tvt"].values * np.nan, "no-sub1 (selector only)", w_sub1=0.0)
    eval_blend(make_ridge_fn(OLD5), "old5 (deployed)")
    eval_blend(make_ridge_fn(NEW5), "new5 (retrained)")
    eval_blend(make_ridge_fn(BEST5), "best5 mix")
    eval_blend(make_ridge_fn(OLD5 + NEW5), "all10")
