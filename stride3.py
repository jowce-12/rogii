# STRIDE v3 — variable-length (h_len, dTVT) lattice beam decode (shreygandhi backbone
# design): log-normal segment-length prior (fit from labels: seg_prior.json, median 349ft
# — v1's fixed 200ft over-segments), rate-persistence prior, Cauchy emission with the
# v1 lik_w=0.1 autocorrelation discount (THE key evidence calibration), band-affine
# typewell recal. Deterministic, CPU. v2 lesson honored: rate freedom (grid to ±0.10)
# is only opened TOGETHER with the length prior that regularizes it.
# Standalone eval: python stride3.py [--wlen 1.0] [--sigp 0.012] [--seed7only]
import json
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


W_LEN = _arg("--wlen", 1.0, float)     # length-prior vote weight
SIG_P = _arg("--sigp", 0.012, float)   # rate-persistence scale (v1 value; grid later)
LIK_W = 0.1
K_BEAM = 96
GSTEP = 10.0
LEN_GRID = np.array([100.0, 160.0, 240.0, 360.0, 520.0, 760.0])
RATE_GRID = np.arange(-0.10, 0.1001, 0.005)
PRIOR = json.load(open("seg_prior.json"))
LMU, LSG = PRIOR["len_lognorm_mu"], PRIOR["len_lognorm_sigma"]
LEN_LP = -0.5 * ((np.log(LEN_GRID) - LMU) / LSG) ** 2 - np.log(LEN_GRID)  # lognormal logpdf (unnorm)
TOP_AGG = 32
TEMP = 0.02


def decode(hw, tw):
    """Returns per-eval-row predicted TVT (absolute) or None."""
    from stride import grcal_tw
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 5 or len(kn) < 30 or len(tw_tvt) < 20:
        return None
    mu = float(np.nanmean(kn["GR"].values)); sd = float(np.nanstd(kn["GR"].values)) + 1e-3
    tw_cal = grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
    twz_tab = (tw_cal - mu) / sd
    gr_fill = hw["GR"].interpolate(limit_direction="both")
    gr_fill = gr_fill.fillna(float(np.nanmean(hw["GR"].values))).values
    # known-zone Cauchy scale (v1 recipe)
    kn_u = kn["TVT_input"].values + kn["Z"].values
    kn_z = (gr_fill[kn.index.values] - mu) / sd
    ref_kn = np.interp(np.clip(kn["TVT_input"].values, tw_tvt[0], tw_tvt[-1]), tw_tvt, twz_tab)
    gam = float(np.clip(np.median(np.abs(kn_z - ref_kn)) * sd, 5.0, 40.0)) / sd
    # uniform decode grid over the eval zone (u-space state)
    md_ev = ev["MD"].values.astype(float)
    z_ev = ev["Z"].values.astype(float)
    md0, md1 = float(kn["MD"].iloc[-1]), float(md_ev[-1])
    grid_md = np.arange(md0, md1 + GSTEP / 2, GSTEP)
    n = len(grid_md)
    if n < 4:
        return None
    grz_g = np.interp(grid_md, hw["MD"].values.astype(float), (gr_fill - mu) / sd)
    z_g = np.interp(grid_md, hw["MD"].values.astype(float), hw["Z"].values.astype(float))
    u0 = float(kn["TVT_input"].iloc[-1]) + float(kn["Z"].iloc[-1])
    # drilled-trend init (v1's s0 / writeup's trend centering): first segment's
    # persistence anchor is the known-zone exit slope, not 0 (flat-lock prevention)
    tail30 = kn.tail(30)
    dt_ = np.diff(tail30["TVT_input"].values); dz_ = np.diff(tail30["Z"].values)
    dm_ = np.diff(tail30["MD"].values)
    mm_ok = dm_ > 0
    s0 = float(np.clip(np.median((dt_ + dz_)[mm_ok] / dm_[mm_ok]), -0.08, 0.08)) \
        if mm_ok.sum() >= 5 else 0.0

    def emis(u_seg, rows):
        """Cauchy log-lik of candidate u path over grid rows. u_seg: (C, m)."""
        tvt_c = np.clip(u_seg - z_g[rows], tw_tvt[0], tw_tvt[-1])
        ref = np.interp(tvt_c, tw_tvt, twz_tab)
        r = (grz_g[rows][None, :] - ref) / gam
        return -np.log1p(r * r).sum(axis=1) * LIK_W

    # position-bucketed lattice DP: states arriving at the same grid index compete
    # (fair pruning — cumulative scores are only comparable at equal rows consumed).
    LSTEPS = np.maximum((LEN_GRID / GSTEP).astype(int), 2)
    st_u = [u0]; st_rate = [s0]; st_score = [0.0]; st_parent = [-1]
    st_from = [0]; st_pos = [0]
    buckets = {0: [0]}
    for p in range(n - 1):
        ids = buckets.pop(p, None)
        if not ids:
            continue
        sc = np.array([st_score[i] for i in ids])
        keep = np.argsort(-sc)[:K_BEAM]
        ids = [ids[i] for i in keep]
        us = np.array([st_u[i] for i in ids])
        rates = np.array([st_rate[i] for i in ids])
        scores = np.array([st_score[i] for i in ids])
        for li, ls in enumerate(LSTEPS):
            p2 = min(p + int(ls), n - 1)
            if p2 <= p:
                continue
            m = p2 - p
            t = np.arange(1, m + 1) * GSTEP
            # candidates: (K, R) grid of end states; emission vectorized over K*R
            u_seg = us[:, None, None] + RATE_GRID[None, :, None] * t[None, None, :]
            C = u_seg.reshape(-1, m)
            e = emis(C, slice(p + 1, p2 + 1)).reshape(len(ids), len(RATE_GRID))
            pen = -0.5 * ((RATE_GRID[None, :] - rates[:, None]) / SIG_P) ** 2
            pen = np.maximum(pen, -25.0)
            tot = scores[:, None] + e + W_LEN * LEN_LP[li] + pen
            # top candidates per (L) expansion to bound growth
            flat = tot.ravel()
            top = np.argsort(-flat)[:K_BEAM]
            arr = buckets.setdefault(p2, [])
            for f in top:
                ki, ri = divmod(int(f), len(RATE_GRID))
                st_u.append(float(u_seg[ki, ri, -1]))
                st_rate.append(float(RATE_GRID[ri]))
                st_score.append(float(flat[f]))
                st_parent.append(ids[ki])
                st_from.append(p)
                st_pos.append(p2)
                arr.append(len(st_u) - 1)
    finals = buckets.get(n - 1, [])
    if not finals:
        return None
    fsc = np.array([st_score[i] for i in finals])
    order = np.argsort(-fsc)[:TOP_AGG]
    w = np.exp(((fsc[order] - fsc[order].max()) / n) / TEMP)
    w /= w.sum()
    P = np.full((len(order), n), np.nan)
    for k, oi in enumerate(order):
        i = finals[oi]
        while i >= 0 and st_parent[i] >= 0:
            p0, p1 = st_from[i], st_pos[i]
            r = st_rate[i]
            u_end = st_u[i]
            mm_ = p1 - p0
            P[k, p0 + 1:p1 + 1] = u_end - r * GSTEP * np.arange(mm_ - 1, -1, -1)
            i = st_parent[i]
        P[k, 0] = u0
    u_hat = (w[:, None] * P).sum(0)
    u_hat[0] = u0
    fin = np.isfinite(u_hat)
    u_row = np.interp(md_ev, grid_md[fin], u_hat[fin])
    return u_row - z_ev


def one(wid):
    from stride import load_well
    try:
        hw, tw = load_well(wid, "train")
        pred = decode(hw, tw)
        if pred is None:
            return wid, None
        ev = hw[hw["TVT_input"].isna()]
        truth = ev["TVT"].values.astype(float)
        fin = np.isfinite(truth) & np.isfinite(pred)
        if fin.sum() < 5:
            return wid, None
        return wid, (float(((pred[fin] - truth[fin]) ** 2).sum()), int(fin.sum()))
    except Exception as e:
        return wid, ("err", str(e)[:60])


if __name__ == "__main__":
    import blend_eval as BE
    t0 = time.time()
    seeds = (7,) if "--seed7only" in sys.argv else (7, 11)
    for SEED in seeds:
        res, _ = BE.selector_preds(SEED)
        wells = [r_["wid"] for r_ in res]
        outs = Parallel(n_jobs=10, backend="loky")(delayed(one)(w) for w in wells)
        se = n = 0.0
        errs = 0
        for _w, o in outs:
            if o is None:
                continue
            if isinstance(o[0], str):
                errs += 1
                continue
            se += o[0]; n += o[1]
        print(f"[{time.time()-t0:.0f}s] seed{SEED}: v3 standalone pooled = {(se/max(n,1))**0.5:.4f} "
              f"(wlen={W_LEN}, sigp={SIG_P}; {errs} errs; v1 ref 13.25/12.18)", flush=True)
    print("DONE", flush=True)
