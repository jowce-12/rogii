# STRIDE v2 sweep: label-fitted priors + changepoint cost + widened rate grid.
# Tune on seed7 only; the chosen config is then confirmed frozen on seed11.
import sys, time
import numpy as np
from joblib import Parallel, delayed
import stride
from offline_tests import load, pooled

def decode(rec, seg_len, K, sig_pers, jump_pen, lik_w, c0, rate_max, rate_step):
    hw, tw = stride.load_well(rec["wid"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    tw_gr = stride.grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
    gmin = tw_tvt[0]
    gg = np.interp(np.arange(gmin, tw_tvt[-1] + 0.5, 0.5), tw_tvt, tw_gr)
    last = kn.iloc[-1]
    u0 = float(last["TVT_input"]) + float(last["Z"])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    s0 = float(np.clip(np.median((dt + dz)[m] / dm[m]) if m.sum() >= 3 else 0.0, -rate_max, rate_max))
    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gam = float(np.clip(np.nanmedian(np.abs(kn["GR"].values.astype(float) - tw_at_k)), 5.0, 40.0))
    md = ev["MD"].values.astype(float); z = ev["Z"].values.astype(float)
    gr = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    msk = np.ones(len(md), np.int8)
    bnds = [0]; cur = md[0]
    for i in range(len(md)):
        if md[i] >= cur + seg_len:
            bnds.append(i); cur = md[i]
    if bnds[-1] != len(md):
        bnds.append(len(md))
    bnds = np.array(bnds, dtype=np.int64)
    rates = np.arange(-rate_max, rate_max + rate_step / 2, rate_step)
    br, sc = stride._decode(md, z, gr, msk, gg, gmin, 0.5, u0, s0, gam, rates, bnds,
                            K, sig_pers, jump_pen, 0.0, lik_w, c0)
    paths = stride._paths_from_rates(md, bnds, u0, rates, br)
    order = np.argsort(sc)[::-1][:32]
    s = (sc[order] - sc[order][0]) / max(len(z), 1)
    w = np.exp(s / 0.02); w /= w.sum()
    return (w[:, None] * paths[order]).sum(0) - z

def run(res, **cfg):
    return Parallel(n_jobs=24, prefer="threads")(delayed(decode)(r, **cfg) for r in res)

if __name__ == "__main__":
    res = load(7)
    t0 = time.time()
    V1 = dict(seg_len=200.0, K=96, sig_pers=0.012, jump_pen=25.0, lik_w=0.1,
              c0=0.0, rate_max=0.06, rate_step=0.002)
    print(f"v1 baseline: {pooled(res, run(res, **V1)):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    # A: widen the rate grid (labels: |s| p99=0.082 > old 0.06 cap)
    for rm, rs in [(0.10, 0.002), (0.10, 0.0025)]:
        cfg = dict(V1); cfg.update(rate_max=rm, rate_step=rs)
        print(f"A rate_max={rm} step={rs}: {pooled(res, run(res, **cfg)):.4f}", flush=True)
    # B: fine grid + changepoint cost (labels: 38.6% segments unchanged, robust sig 0.004)
    for seg in (100.0, 150.0):
        for c0 in (0.0, 0.5, 1.0, 2.0, 4.0):
            cfg = dict(V1); cfg.update(seg_len=seg, sig_pers=0.006, c0=c0, rate_max=0.10, rate_step=0.0025)
            print(f"B seg={seg:.0f} c0={c0}: {pooled(res, run(res, **cfg)):.4f}", flush=True)
    print(f"done [{time.time()-t0:.0f}s]", flush=True)
