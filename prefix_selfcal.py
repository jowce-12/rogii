# TRANSDUCTIVE self-calibration probe: at test time a well's VISIBLE PREFIX is labeled,
# so we can cut it artificially, predict the masked tail with a cheap tracker (STRIDE-v3,
# already ~4s/well in the pipeline), and measure that tracker's SIGNED bias against known
# TVT. Question: does that backtest bias predict the DEPLOYED BLEND's bias in the real
# eval zone? If yes, a per-well correction is implementable in-notebook for ~+13min.
# Uses train wells (truth everywhere) but measures the bias EXACTLY as test time would.
# RUN from ~/rogii: python prefix_selfcal.py       (~25min CPU, 10 procs)
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

CUT_FRAC = 0.75          # keep this fraction of the visible prefix, mask the rest
t0 = time.time()


def backtest_bias(wid):
    """Mask the tail of the visible prefix, decode with v3, return signed bias + rmse."""
    import sys as _s
    _a = _s.argv
    _s.argv = ["x", "--wlen", "0.5"]
    import stride3 as S3
    _s.argv = _a
    from stride import load_well
    try:
        hw, tw = load_well(wid, "train")
        kn_idx = hw.index[hw["TVT_input"].notna()].values
        if len(kn_idx) < 200:
            return wid, None
        keep = int(len(kn_idx) * CUT_FRAC)
        mask_idx = kn_idx[keep:]
        if len(mask_idx) < 50:
            return wid, None
        truth_masked = hw.loc[mask_idx, "TVT_input"].values.astype(float)
        hw_cut = hw.copy()
        hw_cut.loc[mask_idx, "TVT_input"] = np.nan
        pred = S3.decode(hw_cut, tw)
        if pred is None:
            return wid, None
        ev_idx = hw_cut.index[hw_cut["TVT_input"].isna()].values
        pos = {v: i for i, v in enumerate(ev_idx)}
        sel = [pos[m] for m in mask_idx if m in pos]
        if len(sel) < 30:
            return wid, None
        p = np.asarray(pred, float)[sel]
        fin = np.isfinite(p) & np.isfinite(truth_masked[:len(sel)])
        if fin.sum() < 30:
            return wid, None
        e = p[fin] - truth_masked[:len(sel)][fin]
        return wid, (float(np.mean(e)), float(np.sqrt(np.mean(e ** 2))), int(fin.sum()))
    except Exception as ex:
        return wid, ("err", str(ex)[:60])


if __name__ == "__main__":
    import blend_eval as BE
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from offline_tests import pooled

    res7, SEL7 = BE.selector_preds(7)
    res11, SEL11 = BE.selector_preds(11)
    wells = sorted({r["wid"] for r in res7} | {r["wid"] for r in res11})
    outs = Parallel(n_jobs=10, backend="loky")(delayed(backtest_bias)(w) for w in wells)
    bias = {w: o for w, o in outs if o is not None and not isinstance(o[0], str)}
    print(f"[{time.time()-t0:.0f}s] backtest bias for {len(bias)}/{len(wells)} wells", flush=True)
    pd.DataFrame([dict(well=w, bias=v[0], rmse=v[1], n=v[2]) for w, v in bias.items()]
                 ).to_parquet("prefix_backtest_bias.parquet", index=False)

    # deployed blend per well (dip5 + v3 pole + tiered weights + gamma), and ITS true bias
    fs = pd.read_parquet("oof_stack.parquet").merge(
        pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                        columns=["id", "likpf_mean_d", "likpf_ptstd"]), on="id")
    fy = fs["target"].values.astype(np.float64); fy_fit = np.clip(fy, -90, 90)
    FX = np.column_stack([fs[c].values for c in ["lgb0", "lgb1", "lgb2", "cb0", "cb1"]]
                         + [np.nan_to_num(fs["likpf_mean_d"].values.astype(np.float64))])
    meta = np.zeros(len(fs))
    for tr, va in GroupKFold(5).split(FX, fy_fit, groups=fs["well"].values):
        r = Ridge(alpha=1.66, positive=True, fit_intercept=True)
        r.fit(FX[tr], fy_fit[tr]); meta[va] = r.predict(FX[va])
    fl_s = pd.Series(meta, index=fs["id"].values)
    risk_s = pd.Series(np.nan_to_num(fs["likpf_ptstd"].values.astype(np.float64)), index=fs["id"].values)
    OLD5 = [f"old_{s}" for s in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]]
    sub1_fn = BE.make_ridge_fn(OLD5)
    gr_s = pd.Series(pd.read_parquet("gru_oof_dipfused5.parquet").set_index("id")["gru_d"])
    s3_s = pd.Series(pd.read_parquet("s3_preds_tuned.parquet").set_index("id")["s3_tvt"])

    for SEED, res, SEL in ((7, res7, SEL7), (11, res11, SEL11)):
        sub = BE.OOF[BE.OOF["well"].isin([r_["wid"] for r_ in res])].copy()
        sub["sub1_tvt"] = sub1_fn(sub)
        sub["fl_tvt"] = sub["last_known_tvt"].values + fl_s.reindex(sub["id"].values).values
        sub["gr_tvt"] = sub["last_known_tvt"].values + gr_s.reindex(sub["id"].values).values
        sub["s3_tvt"] = s3_s.reindex(sub["id"].values).values
        sub["risk"] = risk_s.reindex(sub["id"].values).values
        by = {c: {w: g[c].values for w, g in sub.groupby("well")} for c in
              ("sub1_tvt", "fl_tvt", "gr_tvt", "s3_tvt", "last_known_tvt", "risk")}
        base, truth_bias = {}, {}
        for r_, sel in zip(res, SEL):
            w = r_["wid"]
            s1, fl, gr, sv = (by["sub1_tvt"].get(w), by["fl_tvt"].get(w),
                              by["gr_tvt"].get(w), by["s3_tvt"].get(w))
            if any(v is None or len(v) != len(sel) or np.isnan(np.asarray(v, float)).any()
                   for v in (s1, fl, gr)):
                base[w] = np.asarray(sel, float); continue
            risk = float(np.mean(by["risk"][w])); last = float(by["last_known_tvt"][w][0])
            monster = np.isfinite(risk) and risk >= 3.39
            wsp, wfl, wgr = (0.20, 0.40, 0.40) if monster else (0.15, 0.45, 0.40)
            b = wsp * (0.3 * s1 + 0.7 * sel) + wfl * fl + wgr * gr
            if sv is not None and len(sv) == len(sel) and np.isfinite(sv).all():
                b = 0.90 * b + 0.10 * sv
            if monster:
                b = last + 1.09 * (b - last)
            base[w] = b
            truth_bias[w] = float(np.mean(b - np.asarray(r_["y"], float)))
        common = [w for w in truth_bias if w in bias]
        bt = np.array([bias[w][0] for w in common])
        tb = np.array([truth_bias[w] for w in common])
        print(f"[{time.time()-t0:.0f}s] seed{SEED}: corr(v3-backtest bias, blend eval bias) = "
              f"{float(np.corrcoef(bt, tb)[0, 1]):.3f} on {len(common)} wells", flush=True)
        parts = []
        for k in (0.0, 0.15, 0.25, 0.4):
            finals = []
            for r_ in res:
                w = r_["wid"]
                b = base[w].copy()
                if k > 0 and w in bias:
                    b = b - k * bias[w][0]
                finals.append(b)
            parts.append(f"k={k:.2f}: {pooled(res, finals):.4f}")
        print(f"   correction grid: " + "  ".join(parts), flush=True)
    print("DONE", flush=True)
