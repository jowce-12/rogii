# PF seed-branch midpoint hedge (v599 fork's mechanism) measured on OUR harness at the
# deployed config (spatialfused GRU, 0.15/0.45/0.40 + gamma). Seed levels from the same
# 32-seed/500-particle PF the deployment caches for the GRU -> measurement == deployment.
# Stage 1: per-well branch stats (weighted 2-cluster split of seed eval-medians).
# Stage 2: apply THEIR gates (strength .60, minor mass >= .25, sep 4-40ft, cap 2ft) plus a
# small pre-registered grid (strength .4/.6, cap 1/2) -> pooled before/after, both seeds.
# RUN from ~/rogii: python branch_hedge_eval.py       (~25min CPU, 10 procs)
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import blend_eval as BE
from offline_tests import pooled

t0 = time.time()


def branch_stats(wid):
    import numpy as np
    import _t1_pf as PF
    import stride as ST
    try:
        hw, tw = ST.load_well(wid, "train")
        ev_mask = hw["TVT_input"].isna().values
        if int(ev_mask.sum()) < 10:
            return wid, None
        pairs = PF._pf_seed_batch(hw, tw, 500, range(32))
        pred_arr = np.stack([np.asarray(p, float) for p, _l in pairs], 0)
        liks = np.array([l for _p, l in pairs], float)
        liks_n = liks - liks.max()
        w = np.exp(liks_n / 5.0)
        w = w / max(float(w.sum()), 1e-12)
        level = np.nanmedian(pred_arr[:, ev_mask], axis=1)
        valid = np.isfinite(level) & np.isfinite(w) & (w > 0)
        level, w = level[valid], w[valid]
        w = w / max(float(w.sum()), 1e-12)
        if len(level) < 4:
            return wid, None
        order = np.argsort(level)
        x, ww = level[order], w[order]
        cw, cx, cx2 = np.cumsum(ww), np.cumsum(ww * x), np.cumsum(ww * x * x)
        tw_, tx_, tx2_ = float(cw[-1]), float(cx[-1]), float(cx2[-1])
        best = None
        for cut in range(1, len(x)):
            wl = float(cw[cut - 1])
            wr = tw_ - wl
            if wl < 0.05 or wr < 0.05:
                continue
            xl = float(cx[cut - 1]); xr = tx_ - xl
            ssel = float(cx2[cut - 1] - xl * xl / wl)
            sser = float((tx2_ - cx2[cut - 1]) - xr * xr / wr)
            sse = ssel + sser
            if best is None or sse < best[0]:
                best = (sse, cut, xl / wl, xr / wr, wl, wr)
        if best is None:
            return wid, None
        _, cut, c_lo, c_hi, m_lo, m_hi = best
        return wid, dict(center_low=c_lo, center_high=c_hi, mass_low=m_lo, mass_high=m_hi,
                         weighted_center=float((w * level).sum()))
    except Exception as e:
        return wid, {"err": str(e)[:60]}


if __name__ == "__main__":
    res7, SEL7 = BE.selector_preds(7)
    res11, SEL11 = BE.selector_preds(11)
    wells_all = sorted({r["wid"] for r in res7} | {r["wid"] for r in res11})
    print(f"[{time.time()-t0:.0f}s] {len(wells_all)} unique wells; computing 32-seed branch stats...", flush=True)
    outs = Parallel(n_jobs=10, backend="loky")(delayed(branch_stats)(w) for w in wells_all)
    stats = {w: s for w, s in outs if s is not None and "err" not in s}
    n_err = sum(1 for _w, s in outs if s is not None and "err" in s)
    print(f"[{time.time()-t0:.0f}s] stats ok={len(stats)} none={sum(1 for _w,s in outs if s is None)} err={n_err}", flush=True)
    pd.DataFrame([dict(well=w, **s) for w, s in stats.items()]).to_parquet("branch_stats.parquet", index=False)

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
    risk_s = pd.Series(np.nan_to_num(fs["likpf_ptstd"].values.astype(np.float64)), index=fs["id"].values)
    OLD5 = [f"old_{s}" for s in ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2"]]
    sub1_fn = BE.make_ridge_fn(OLD5)
    gr_s = pd.Series(pd.read_parquet("gru_oof_spatialfused.parquet").set_index("id")["gru_d"])

    GATES = dict(minmass=0.25, seplo=4.0, sephi=40.0)
    GRID = [(0.60, 2.0), (0.40, 2.0), (0.60, 1.0), (0.40, 1.0)]
    for SEED, res, SEL in ((7, res7, SEL7), (11, res11, SEL11)):
        sub = BE.OOF[BE.OOF["well"].isin([r_["wid"] for r_ in res])].copy()
        sub["sub1_tvt"] = sub1_fn(sub)
        sub["fl_tvt"] = sub["last_known_tvt"].values + fl_s.reindex(sub["id"].values).values
        sub["gr_tvt"] = sub["last_known_tvt"].values + gr_s.reindex(sub["id"].values).values
        sub["risk"] = risk_s.reindex(sub["id"].values).values
        by = {c: {w: g[c].values for w, g in sub.groupby("well")} for c in
              ("sub1_tvt", "fl_tvt", "gr_tvt", "last_known_tvt", "risk")}
        base = {}
        for r_, sel in zip(res, SEL):
            w = r_["wid"]
            s1, fl, gr = by["sub1_tvt"].get(w), by["fl_tvt"].get(w), by["gr_tvt"].get(w)
            if any(v is None or len(v) != len(sel) or np.isnan(v).any() for v in (s1, fl, gr)):
                base[w] = np.asarray(sel, float)
                continue
            b = 0.15 * (0.3 * s1 + 0.7 * sel) + 0.45 * fl + 0.40 * gr
            risk = float(np.mean(by["risk"][w])); last = float(by["last_known_tvt"][w][0])
            if np.isfinite(risk) and risk >= 3.39:
                b = last + 1.09 * (b - last)
            base[w] = b
        print(f"[{time.time()-t0:.0f}s] seed{SEED} base pooled = {pooled(res, [base[r_['wid']] for r_ in res]):.4f}", flush=True)
        for strength, cap in GRID:
            finals, n_app = [], 0
            for r_ in res:
                w = r_["wid"]
                b = base[w].copy()
                st = stats.get(w)
                if st:
                    sep = abs(st["center_high"] - st["center_low"])
                    minor = min(st["mass_low"], st["mass_high"])
                    if minor >= GATES["minmass"] and GATES["seplo"] <= sep <= GATES["sephi"]:
                        target = 0.5 * (st["center_low"] + st["center_high"])
                        shift = float(np.clip(strength * (target - st["weighted_center"]), -cap, cap))
                        if abs(shift) >= 0.01:
                            b = b + shift
                            n_app += 1
                finals.append(b)
            print(f"   strength={strength:.2f} cap={cap:.0f}: pooled={pooled(res, finals):.4f} (applied {n_app} wells)", flush=True)
    print("DONE", flush=True)
