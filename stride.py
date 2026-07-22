"""stride.py — STRIDE-style segment-lattice sequence decoder (v1).

Decodes the smooth structural surface U = TVT + Z (the "trend"; the wiggle -Z is
free) as a chain of piecewise-linear segments, beam-searched JOINTLY over the whole
eval zone:
    score(path) = sum_rows log Cauchy(GR_obs - GR_typewell(U - Z))     # anti-alias
                - sum_segs min(((s_i - s_{i-1})/sig_pers)^2, jump_pen) # persistence
                                                                       #  w/ fault jumps
Key differences from the deployed lik-PF: sequence-level optimization (later evidence
revises earlier segments), heavy-tailed likelihood (a repeated bed cannot out-argue a
long stretch of honest agreement), and explicit piecewise-linear structure (matching
the human-drawn labels: 64.5% of eval points have ~zero curvature).

Standalone selector channel: no GBM / features.json coupling. Validate with
measure_stride.py on the 2x150 disjoint eval samples before wiring into notebooks.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

DATA = Path(os.environ.get("ROGII_DATA", "."))


# ---------- typewell band-affine GR recalibration (same guards as deployed S1) ----------
def grcal_tw(tw_tvt, tw_gr, ktvt, kgr):
    ktvt = np.asarray(ktvt, float); kgr = np.asarray(kgr, float)
    v = np.isfinite(kgr)
    if v.sum() < 30:
        return tw_gr
    band = tw_gr[(tw_tvt >= np.nanmin(ktvt) - 10.0) & (tw_tvt <= np.nanmax(ktvt) + 10.0)]
    if len(band) < 20:
        band = tw_gr
    bm = float(np.mean(band)); km = float(np.mean(kgr[v]))
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    vv = v & np.isfinite(tw_at_k)
    if vv.sum() < 30 or np.std(tw_at_k[vv]) < 1e-6:
        a, b = 1.0, km - bm
    else:
        a, b = np.polyfit(tw_at_k[vv], kgr[vv], 1)
        if (not np.isfinite(a)) or a < 0.2 or a > 5.0:
            a, b = 1.0, km - bm
    return a * tw_gr + b


@njit(cache=True, nogil=True)
def _decode(md, z, gr, msk, gg, gmin, gstep, u0, s0, gam, rates, bnds,
            K, sig_pers, jump_pen, lam_s, lik_w, c0=0.0):
    """Beam decode over surface slope per segment. Returns (paths_rates, scores):
    paths_rates[k, seg] = chosen rate index of beam k, scores[k] = total log-score."""
    n_seg = len(bnds) - 1
    R = len(rates)
    ng = len(gg)
    # active beams
    beam_u = np.empty(K); beam_s = np.empty(K); beam_sc = np.empty(K)
    beam_rates = np.full((K, n_seg), -1, dtype=np.int16)
    beam_u[0] = u0; beam_s[0] = s0; beam_sc[0] = 0.0
    kb = 1
    cand_sc = np.empty(K * R); cand_u = np.empty(K * R)
    cand_par = np.empty(K * R, dtype=np.int32); cand_r = np.empty(K * R, dtype=np.int16)
    new_u = np.empty(K); new_s = np.empty(K); new_sc = np.empty(K)
    new_rates = np.full((K, n_seg), -1, dtype=np.int16)
    keys = np.empty(K, dtype=np.int64)
    for seg in range(n_seg):
        i0, i1 = bnds[seg], bnds[seg + 1]
        md0 = md[i0 - 1] if i0 > 0 else md[0] - (md[1] - md[0])
        u_prev = 0.0
        nc = 0
        for b in range(kb):
            for ri in range(R):
                r = rates[ri]
                # segment log-lik: Cauchy on GR mismatch along the candidate line
                ll = 0.0
                ub = beam_u[b]
                for i in range(i0, i1):
                    if msk[i] == 0:
                        continue   # interpolated GR row: no real evidence
                    u_i = ub + r * (md[i] - md0)
                    tvt = u_i - z[i]
                    gi = int((tvt - gmin) / gstep)
                    if gi < 0:
                        gi = 0
                    elif gi >= ng:
                        gi = ng - 1
                    d = (gr[i] - gg[gi]) / gam
                    ll += -np.log(1.0 + d * d)
                # persistence prior with fault-jump cap + weak anchor to initial trend.
                # v2: changepoint cost c0 — keeping the SAME rate is free, changing it
                # costs c0 + capped quadratic => encourages long piecewise-linear runs
                # (the labels' actual structure); c0=0 reproduces v1 exactly.
                if r == beam_s[b]:
                    pen = 0.0
                else:
                    ds = (r - beam_s[b]) / sig_pers
                    pen = ds * ds
                    if pen > jump_pen:
                        pen = jump_pen
                    pen += c0
                pen += lam_s * (r - s0) * (r - s0)
                # lik_w corrects for GR residual autocorrelation (~50-sample runs):
                # raw per-row log-lik sums overcount evidence and let GR out-vote
                # the continuity prior — the known 30-ft "GR-trusting" failure mode.
                cand_sc[nc] = beam_sc[b] + lik_w * ll - pen
                cand_u[nc] = beam_u[b] + r * (md[i1 - 1] - md0)
                cand_par[nc] = b
                cand_r[nc] = ri
                nc += 1
        # top-K with dedup on (u-bin, rate)
        order = np.argsort(cand_sc[:nc])[::-1]
        nk = 0
        for oi in range(nc):
            c = order[oi]
            key = np.int64(np.round(cand_u[c] * 2.0)) * 1000 + cand_r[c]
            dup = False
            for j in range(nk):
                if keys[j] == key:
                    dup = True
                    break
            if dup:
                continue
            keys[nk] = key
            new_u[nk] = cand_u[c]
            new_s[nk] = rates[cand_r[c]]
            new_sc[nk] = cand_sc[c]
            p = cand_par[c]
            for s2 in range(seg):
                new_rates[nk, s2] = beam_rates[p, s2]
            new_rates[nk, seg] = cand_r[c]
            nk += 1
            if nk >= K:
                break
        kb = nk
        beam_u[:kb] = new_u[:kb]; beam_s[:kb] = new_s[:kb]; beam_sc[:kb] = new_sc[:kb]
        beam_rates[:kb] = new_rates[:kb]
    return beam_rates[:kb], beam_sc[:kb]


@njit(cache=True, nogil=True)
def _paths_from_rates(md, bnds, u0, rates, beam_rates):
    kb, n_seg = beam_rates.shape
    n = len(md)
    paths = np.empty((kb, n))
    for k in range(kb):
        u = u0
        for seg in range(n_seg):
            i0, i1 = bnds[seg], bnds[seg + 1]
            md0 = md[i0 - 1] if i0 > 0 else md[0] - (md[1] - md[0])
            r = rates[beam_rates[k, seg]]
            for i in range(i0, i1):
                paths[k, i] = u + r * (md[i] - md0)
            u = u + r * (md[i1 - 1] - md0)
    return paths


def stride_track(hw, tw, seg_len=200.0, K=96, rate_max=0.06, rate_step=0.002,
                 sig_pers=0.012, jump_pen=25.0, lam_s=0.0, temp=8.0, top_m=32,
                 grcal=True, grid_step=0.5, lik_w=0.1):
    # defaults = the deployed patch26 config (tuned seed7, confirmed seed11)
    """Decode one well's eval zone. Returns (tvt_pred_eval, info) or (None, reason)."""
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 5 or len(kn) < 10:
        return None, "too short"
    if grcal:
        tw_gr = grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
    # uniform typewell grid (cheap likelihood lookup)
    gmin, gmax = tw_tvt[0], tw_tvt[-1]
    gg_x = np.arange(gmin, gmax + grid_step, grid_step)
    gg = np.interp(gg_x, tw_tvt, tw_gr)
    # anchors: PS point surface + known-tail surface slope (same recipe as the PF)
    last = kn.iloc[-1]
    u0 = float(last["TVT_input"]) + float(last["Z"])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    s0 = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    s0 = float(np.clip(s0, -rate_max, rate_max))
    # Cauchy scale from the known-zone GR mismatch
    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    resid = kn["GR"].values.astype(float) - tw_at_k
    gam = float(np.clip(np.nanmedian(np.abs(resid)), 5.0, 40.0))
    # eval arrays
    md = ev["MD"].values.astype(float)
    z = ev["Z"].values.astype(float)
    gr = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    # segment boundaries
    bnds = [0]
    cur = md[0]
    for i in range(len(md)):
        if md[i] >= cur + seg_len:
            bnds.append(i); cur = md[i]
    if bnds[-1] != len(md):
        bnds.append(len(md))
    bnds = np.array(bnds, dtype=np.int64)
    rates = np.arange(-rate_max, rate_max + rate_step / 2, rate_step)
    msk = np.ones(len(md), np.int8)
    beam_rates, scores = _decode(md, z, gr, msk, gg, gmin, grid_step, u0, s0, gam,
                                 rates, bnds, K, sig_pers, jump_pen, lam_s, lik_w)
    paths = _paths_from_rates(md, bnds, u0, rates, beam_rates)  # surface U per row
    # posterior-mean over the top-M beams (softmax on total score)
    mtop = min(top_m, len(scores))
    order = np.argsort(scores)[::-1][:mtop]
    sc = scores[order]
    w = np.exp((sc - sc.max()) / max(temp, 1e-6)); w /= w.sum()
    u_mean = (w[:, None] * paths[order]).sum(0)
    tvt = u_mean - z
    info = dict(n_seg=len(bnds) - 1, gam=gam, s0=s0, n_beam=len(scores),
                best_score=float(sc[0]), tvt_best=paths[order[0]] - z)
    return tvt, info


def load_well(wid, split="train"):
    hw = pd.read_csv(DATA / split / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA / split / f"{wid}__typewell.csv")
    return hw, tw


if __name__ == "__main__":
    # smoke test on 3 wells with ground truth
    import time
    wids = sorted(p.stem.replace("__horizontal_well", "")
                  for p in (DATA / "train").glob("*__horizontal_well.csv"))[:3]
    for wid in wids:
        hw, tw = load_well(wid)
        ev = hw[hw.TVT_input.isna()]
        t0 = time.time()
        pred, info = stride_track(hw, tw)
        dt = time.time() - t0
        if pred is None:
            print(wid, "skipped:", info); continue
        y = ev.TVT.values.astype(float)
        last = hw[hw.TVT_input.notna()].iloc[-1]["TVT_input"]
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        flat = float(np.sqrt(np.mean((last - y) ** 2)))
        rb = float(np.sqrt(np.mean((info["tvt_best"] - y) ** 2)))
        print(f"{wid}: stride={rmse:.2f} (best-beam {rb:.2f}) vs flat={flat:.2f} "
              f"| segs={info['n_seg']} gam={info['gam']:.1f} {dt:.2f}s")
