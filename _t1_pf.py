# _t1_pf.py — EXTRACTED VERBATIM from the submission notebook cell 7 (selector PF).
# Only _pf_all_seeds is overridden below to a serial loop (seed order preserved ->
# results identical to the notebook's chunked-parallel version, per its own comment).
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 4.5 * rng.standard_normal(N)  # sp45 patch (sel15 vb best)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


_PF_PROCS = max(1, min(4, os.cpu_count() or 1))

def _pf_seed_batch(hw, tw, n_particles, seeds):
    return [run_particle_filter(hw, tw, n_particles=n_particles, seed=int(s)) for s in seeds]

def _pf_all_seeds(hw, tw, n_particles, n_seeds):
    # process-parallel over seed chunks (the pure-python PF holds the GIL, so threads
    # cannot help). Chunks keep seed order -> identical results to the serial loop.
    if _PF_PROCS == 1 or n_seeds < 2 * _PF_PROCS:
        return _pf_seed_batch(hw, tw, n_particles, range(n_seeds))
    chunks = [c for c in np.array_split(np.arange(n_seeds), _PF_PROCS) if len(c)]
    try:
        res = Parallel(n_jobs=len(chunks), backend="loky")(
            delayed(_pf_seed_batch)(hw, tw, n_particles, c) for c in chunks)
        return [pair for batch in res for pair in batch]
    except Exception as _e:
        print(f"[pf] process pool failed ({str(_e)[:60]}); serial fallback", flush=True)
        return _pf_seed_batch(hw, tw, n_particles, range(n_seeds))


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):
    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)
    preds = [p for p, _ll in pairs]
    liks  = [ll for _p, ll in pairs]

    liks   = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()

    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):
    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)
    preds = [p for p, _ll in pairs]
    liks = [ll for _p, ll in pairs]
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    return out



# serial override (identical results; avoids nested pools in the well-parallel builder)
def _pf_all_seeds(hw, tw, n_particles, n_seeds):
    return _pf_seed_batch(hw, tw, n_particles, range(n_seeds))
