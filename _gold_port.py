# _gold_port.py — VERBATIM extraction: notebook cell 7 (selector PF/beam machinery)
# + gold FUNCTIONS/CONFIG only (run section excluded). Harness-side gold reproduction.
import os, sys, glob, time, warnings, multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
class CFG:
    dataset_path = Path('.')

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)

SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}

SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

FORMATION_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset


def load_well(wid, split='train'):
    base = CFG.dataset_path / split
    hw = pd.read_csv(base / f'{wid}__horizontal_well.csv')
    tw = pd.read_csv(base / f'{wid}__typewell.csv')
    return hw, tw


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


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128, pairs_out=None):
    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)
    if pairs_out is not None:
        pairs_out.extend(pairs)   # patch38: expose per-seed (pred, loglik) for the GRU pole
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


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten()
        tot_f = tot.flatten()
        vf    = valid.flatten()
        ni_f  = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]

        bidx[:kept]  = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s  = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if 'beam' in parts:
        beam_weight = float(parts[parts.index('beam') + 1])
    if 'hold' in parts:
        hold_weight = float(parts[parts.index('hold') + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred

import os as _gold_os

import glob as _gold_glob

import json as _gold_json

import time as _gold_time

import hashlib as _gold_hashlib

from pathlib import Path as _GoldPath

import numpy as _gold_np

import pandas as _gold_pd

_GOLD_ENABLE = _gold_os.environ.get('ROGII_GOLD_PREFIX_CAL', '1') == '1'

_GOLD_PROFILE = _gold_os.environ.get('ROGII_GOLD_PROFILE', 'balanced').strip().lower()

_GOLD_INCLUDE_PF = _gold_os.environ.get('ROGII_GOLD_INCLUDE_PF', '1') == '1'

_GOLD_CAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_CAL_SEEDS', '24'))

_GOLD_FINAL_SEEDS = int(_gold_os.environ.get('ROGII_GOLD_FINAL_SEEDS', '48'))

_GOLD_PARTICLES = int(_gold_os.environ.get('ROGII_GOLD_PARTICLES', '350'))

_GOLD_CUT_FRACS = (0.50, 0.65, 0.75)

_GOLD_MAX_WELLS = int(_gold_os.environ.get('ROGII_GOLD_MAX_WELLS', '1000000'))

_GOLD_PROFILES = {
    'conservative': dict(min_gain=1.00, max_best=12.0, min_consistency=0.67, base=0.06, gain_scale=0.12, margin_scale=0.04, quality_bonus=0.02, cap=0.22, clip_base=8.0, clip_gain=3.0, clip_max=18.0, delta_soft=22.0, p95_hard=55.0),
    'balanced':     dict(min_gain=0.55, max_best=12.0, min_consistency=0.50, base=0.08, gain_scale=0.20, margin_scale=0.06, quality_bonus=0.04, cap=0.36, clip_base=10.0, clip_gain=4.5, clip_max=28.0, delta_soft=30.0, p95_hard=75.0),
    'aggressive':   dict(min_gain=0.25, max_best=15.0, min_consistency=0.34, base=0.12, gain_scale=0.32, margin_scale=0.10, quality_bonus=0.06, cap=0.56, clip_base=14.0, clip_gain=7.0, clip_max=45.0, delta_soft=42.0, p95_hard=110.0),
}

if _GOLD_PROFILE not in _GOLD_PROFILES:
    print(f'Unknown ROGII_GOLD_PROFILE={_GOLD_PROFILE!r}; using balanced')
    _GOLD_PROFILE = 'balanced'

def _gold_work_dir():
    return _GoldPath('/kaggle/working') if _GoldPath('/kaggle/working').exists() else _GoldPath('.')

def _gold_find_data():
    candidates = []
    obj = globals().get('CFG')
    if obj is not None:
        for attr in ('dataset_path', 'DATA'):
            if hasattr(obj, attr):
                candidates.append(_GoldPath(getattr(obj, attr)))
    candidates.extend([
        _GoldPath('/kaggle/input/competitions/rogii-wellbore-geology-prediction'),
        _GoldPath('/kaggle/input/rogii-wellbore-geology-prediction'),
        _GoldPath('.'),
    ])
    for c in candidates:
        try:
            if (c / 'train').exists() and (c / 'test').exists() and (c / 'sample_submission.csv').exists():
                return c
        except Exception:
            pass
    for p in _gold_glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True):
        c = _GoldPath(p).parent
        if (c / 'train').exists() and (c / 'test').exists():
            return c
    raise RuntimeError('Could not locate ROGII data directory')

def _gold_split_ids(df):
    out = df.copy()
    parts = out['id'].astype(str).str.rsplit('_', n=1, expand=True)
    if parts.shape[1] != 2:
        raise RuntimeError('Unexpected id format; expected well_rowindex')
    out['well'] = parts[0]
    out['row_idx'] = parts[1].astype(int)
    return out

def _gold_rmse(a, b):
    a = _gold_np.asarray(a, dtype=float)
    b = _gold_np.asarray(b, dtype=float)
    m = _gold_np.isfinite(a) & _gold_np.isfinite(b)
    if int(m.sum()) == 0:
        return float('inf')
    d = a[m] - b[m]
    return float(_gold_np.sqrt(_gold_np.mean(d * d)))

def _gold_sha256(path):
    h = _gold_hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def _gold_robust_poly_predict(x_known, y_known, x_all, deg):
    x_known = _gold_np.asarray(x_known, dtype=float)
    y_known = _gold_np.asarray(y_known, dtype=float)
    x_all = _gold_np.asarray(x_all, dtype=float)
    m = _gold_np.isfinite(x_known) & _gold_np.isfinite(y_known)
    x_known = x_known[m]
    y_known = y_known[m]
    if len(x_known) < 3:
        fill = float(_gold_np.nanmedian(y_known)) if len(y_known) else 0.0
        return _gold_np.full_like(x_all, fill, dtype=float)
    deg = int(min(max(1, deg), len(x_known) - 1))
    x0 = float(x_known[0])
    xs = float(_gold_np.nanmax(x_known) - _gold_np.nanmin(x_known))
    if (not _gold_np.isfinite(xs)) or xs < 1e-6:
        xs = 1.0
    xk = (x_known - x0) / xs
    xa = (x_all - x0) / xs
    try:
        coef = _gold_np.polyfit(xk, y_known, deg)
        for _ in range(5):
            fit = _gold_np.polyval(coef, xk)
            res = y_known - fit
            sc = 1.4826 * float(_gold_np.nanmedian(_gold_np.abs(res - _gold_np.nanmedian(res)))) + 1e-6
            weights = 1.0 / (1.0 + (res / (2.5 * sc)) ** 2)
            coef = _gold_np.polyfit(xk, y_known, deg, w=weights)
        return _gold_np.polyval(coef, xa).astype(float)
    except Exception:
        return _gold_np.full_like(x_all, float(_gold_np.nanmedian(y_known)), dtype=float)

def _gold_variant_grid():
    variants = set()
    try:
        variants.update(SELECTOR_BIN_VARIANTS.values())
        variants.add(SELECTOR_GLOBAL_VARIANT)
    except Exception:
        pass
    for scale in (3, 5, 8, 12):
        for hold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            variants.add(f'pf_scale_{scale:g}_hold_{hold:g}')
        for beam in (0.05, 0.10, 0.20, 0.30):
            for hold in (0.0, 0.05, 0.10, 0.15, 0.20):
                variants.add(f'pf_scale_{scale:g}_beam_{beam:g}_hold_{hold:g}')
    return sorted(variants)

def _gold_tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology']) if 'Geology' in tw_tr.columns else tw_tr.copy()
    if 'Geology' in tw_g.columns and ref_col in hw_tr.columns:
        ref_tvt = tw_g.loc[tw_g['Geology'] == ref_col, 'TVT'].min()
        if _gold_np.isnan(ref_tvt):
            ref_col = tw_g['Geology'].iloc[0]
            ref_tvt = tw_g.loc[tw_g['Geology'] == ref_col, 'TVT'].min()
        offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
        return (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset).to_numpy(dtype=float)
    return hw_tr['TVT'].to_numpy(dtype=float)

def _gold_contact_candidate(wid, hw, data_dir):
    out = {}
    try:
        hw_tr_path = data_dir / 'train' / f'{wid}__horizontal_well.csv'
        tw_tr_path = data_dir / 'train' / f'{wid}__typewell.csv'
        if not hw_tr_path.exists() or not tw_tr_path.exists():
            return out
        hw_tr = _gold_pd.read_csv(hw_tr_path)
        tw_tr = _gold_pd.read_csv(tw_tr_path)
        phys = _gold_tvt_from_contacts(hw_tr, tw_tr)
        md = hw_tr['MD'].to_numpy(dtype=float)
        m = _gold_np.isfinite(md) & _gold_np.isfinite(phys)
        if int(m.sum()) < 100:
            return out
        order = _gold_np.argsort(md[m])
        md_s = md[m][order]
        ph_s = phys[m][order]
        pred = _gold_np.interp(hw['MD'].to_numpy(dtype=float), md_s, ph_s, left=_gold_np.nan, right=_gold_np.nan)
        out['contact_md_lookup'] = pred.astype(float)
    except Exception:
        return out
    return out

def _gold_poly_candidates(hw_masked):
    out = {}
    tvt = hw_masked['TVT_input'].to_numpy(dtype=float)
    md = hw_masked['MD'].to_numpy(dtype=float)
    z = hw_masked['Z'].to_numpy(dtype=float)
    kn = _gold_np.flatnonzero(_gold_np.isfinite(tvt) & _gold_np.isfinite(md) & _gold_np.isfinite(z))
    if len(kn) < 30:
        return out
    u = tvt + z
    for tail in (80, 160, 320, 640, 1000000):
        sel = kn[-min(int(tail), len(kn)):]
        if len(sel) < 30:
            continue
        tag = 'all' if tail >= 1000000 else f'tail{tail}'
        for deg in (1, 2, 3):
            if len(sel) < deg + 12:
                continue
            uhat = _gold_robust_poly_predict(md[sel], u[sel], md, deg)
            out[f'poly_u_deg{deg}_{tag}'] = (uhat - z).astype(float)
    return out

def _gold_surface_candidates(hw_masked, wid, data_dir):
    out = {}
    tvt = hw_masked['TVT_input'].to_numpy(dtype=float)
    z = hw_masked['Z'].to_numpy(dtype=float)
    xy = hw_masked[['X', 'Y']].to_numpy(dtype=float)
    kn = _gold_np.isfinite(tvt) & _gold_np.isfinite(z) & _gold_np.isfinite(xy).all(axis=1)
    if int(kn.sum()) < 30:
        return out
    formations = list(globals().get('FORMATIONS', ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']))
    fi = globals().get('_FI', globals().get('FI', None))
    di = globals().get('_DI', globals().get('DI', None))
    surf_names = []
    try:
        if fi is not None:
            form_all, _ = fi.impute(xy, self_wid=None)
            form_all = _gold_np.asarray(form_all, dtype=float)
            for i, fn in enumerate(formations[:form_all.shape[1]]):
                f = form_all[:, i]
                good = kn & _gold_np.isfinite(f)
                if int(good.sum()) < 30:
                    continue
                b_med = float(_gold_np.nanmedian(tvt[good] + z[good] - f[good]))
                out[f'surface_{fn}_median'] = (-z + f + b_med).astype(float)
                surf_names.append(f'surface_{fn}_median')
                if callable(globals().get('seg_b_well')):
                    try:
                        b_full, _, _, b_late, b_wls = seg_b_well(
                            tvt[good].astype(_gold_np.float32),
                            z[good].astype(_gold_np.float32),
                            f[good].astype(_gold_np.float32),
                        )
                        out[f'surface_{fn}_full'] = (-z + f + float(b_full)).astype(float)
                        out[f'surface_{fn}_late'] = (-z + f + float(b_late)).astype(float)
                        out[f'surface_{fn}_wls'] = (-z + f + float(b_wls)).astype(float)
                        surf_names.extend([f'surface_{fn}_full', f'surface_{fn}_late', f'surface_{fn}_wls'])
                    except Exception:
                        pass
    except Exception as e:
        print('surface imputer skipped', wid, e)
    try:
        if di is not None:
            dense, _, _ = di.impute(xy, self_wid=None)
            dense = _gold_np.asarray(dense, dtype=float)
            good = kn & _gold_np.isfinite(dense)
            if int(good.sum()) >= 30:
                b_med = float(_gold_np.nanmedian(tvt[good] + z[good] - dense[good]))
                out['dense_ancc_median'] = (-z + dense + b_med).astype(float)
                surf_names.append('dense_ancc_median')
                if callable(globals().get('seg_b_well')):
                    try:
                        b_full, _, _, b_late, b_wls = seg_b_well(
                            tvt[good].astype(_gold_np.float32),
                            z[good].astype(_gold_np.float32),
                            dense[good].astype(_gold_np.float32),
                        )
                        out['dense_ancc_full'] = (-z + dense + float(b_full)).astype(float)
                        out['dense_ancc_late'] = (-z + dense + float(b_late)).astype(float)
                        out['dense_ancc_wls'] = (-z + dense + float(b_wls)).astype(float)
                        surf_names.extend(['dense_ancc_full', 'dense_ancc_late', 'dense_ancc_wls'])
                    except Exception:
                        pass
    except Exception as e:
        print('dense imputer skipped', wid, e)
    ens_names = [n for n in surf_names if n in out]
    if len(ens_names) >= 2:
        errs = _gold_np.array([_gold_rmse(out[n][kn], tvt[kn]) for n in ens_names], dtype=float)
        finite = _gold_np.isfinite(errs)
        if int(finite.sum()) >= 2:
            names = [n for n, ok in zip(ens_names, finite) if ok]
            errs = errs[finite]
            weights = 1.0 / _gold_np.maximum(errs, 0.25) ** 2
            weights = weights / weights.sum()
            mat = _gold_np.vstack([out[n] for n in names])
            out['surface_weighted_prefix'] = (weights[:, None] * mat).sum(axis=0).astype(float)
    out.update(_gold_contact_candidate(wid, hw_masked, data_dir))
    return out

def _gold_pf_candidates(hw_masked, tw, variants, n_seeds, n_particles):
    out = {}
    if not _GOLD_INCLUDE_PF:
        return out
    if not callable(globals().get('run_pf_lik_ensemble_scales')) or not callable(globals().get('apply_selector_variant')):
        return out
    kn = hw_masked[hw_masked['TVT_input'].notna()]
    ev = hw_masked[hw_masked['TVT_input'].isna()]
    if len(kn) < 30 or len(ev) == 0:
        return out
    try:
        pf_by_scale = run_pf_lik_ensemble_scales(
            hw_masked,
            tw,
            scales=tuple(globals().get('SELECTOR_SCALES', (3.0, 5.0, 8.0, 12.0))),
            n_particles=int(n_particles),
            n_seeds=int(n_seeds),
        )
        try:
            tvt_beam = run_beam_ensemble(hw_masked, tw)
        except Exception:
            tvt_beam = pf_by_scale.get('pf_mean')
            if tvt_beam is None:
                tvt_beam = next(iter(pf_by_scale.values()))
        last_known_tvt = float(kn['TVT_input'].iloc[-1])
        for variant in variants:
            try:
                pred = apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)
                if pred is not None and len(pred) == len(hw_masked):
                    out['pf|' + variant] = _gold_np.asarray(pred, dtype=float)
            except Exception:
                pass
    except Exception as e:
        print('PF calibration skipped:', e)
    return out

def _gold_candidate_pool(wid, hw_masked, tw, data_dir, variants, include_pf=True, n_seeds=24, n_particles=350):
    pool = {}
    pool.update(_gold_poly_candidates(hw_masked))
    pool.update(_gold_surface_candidates(hw_masked, wid, data_dir))
    if include_pf:
        pool.update(_gold_pf_candidates(hw_masked, tw, variants, n_seeds=n_seeds, n_particles=n_particles))
    clean = {}
    for name, pred in pool.items():
        arr = _gold_np.asarray(pred, dtype=float)
        if len(arr) == len(hw_masked) and _gold_np.isfinite(arr).sum() >= max(20, len(hw_masked) // 20):
            clean[name] = arr
    return clean

def _gold_default_pf_name(hw):
    try:
        return 'pf|' + selector_well_code(hw)[1]
    except Exception:
        try:
            return 'pf|' + SELECTOR_GLOBAL_VARIANT
        except Exception:
            return None

def _gold_calibrate_well(wid, hw, tw, data_dir, variants):
    tvt = hw['TVT_input'].to_numpy(dtype=float)
    is_known = _gold_np.isfinite(tvt)
    is_hidden = ~is_known
    if not bool(is_hidden.any()):
        return None
    first_hidden = int(_gold_np.flatnonzero(is_hidden)[0])
    known_prefix = _gold_np.flatnonzero(is_known & (_gold_np.arange(len(hw)) < first_hidden))
    if len(known_prefix) < 140:
        return dict(well=wid, status='skip_short_prefix', known_prefix=int(len(known_prefix)))
    cuts = []
    for frac in _GOLD_CUT_FRACS:
        cut_pos = int(round(len(known_prefix) * float(frac)))
        cut_pos = max(50, min(cut_pos, len(known_prefix) - 35))
        if cut_pos <= 0 or cut_pos >= len(known_prefix):
            continue
        cutoff_idx = int(known_prefix[cut_pos - 1])
        hold_idx = known_prefix[cut_pos:]
        if len(hold_idx) >= 35:
            cuts.append((float(frac), cutoff_idx, hold_idx))
    if not cuts:
        return dict(well=wid, status='skip_no_holdout', known_prefix=int(len(known_prefix)))
    scores = {}
    cut_rows = []
    default_name = _gold_default_pf_name(hw)
    for frac, cutoff_idx, hold_idx in cuts:
        hw_m = hw.copy(deep=True)
        hw_m.loc[hw_m.index > cutoff_idx, 'TVT_input'] = _gold_np.nan
        pool = _gold_candidate_pool(
            wid, hw_m, tw, data_dir, variants,
            include_pf=_GOLD_INCLUDE_PF,
            n_seeds=_GOLD_CAL_SEEDS,
            n_particles=_GOLD_PARTICLES,
        )
        y = tvt[hold_idx]
        row = {'well': wid, 'cut_frac': frac, 'holdout_rows': int(len(hold_idx)), 'candidates': int(len(pool))}
        local = []
        for name, pred in pool.items():
            err = _gold_rmse(pred[hold_idx], y)
            if _gold_np.isfinite(err):
                scores.setdefault(name, []).append(err)
                local.append((err, name))
        local.sort()
        if local:
            row['best_name'] = local[0][1]
            row['best_rmse'] = float(local[0][0])
            if default_name in pool:
                row['default_rmse'] = float(_gold_rmse(pool[default_name][hold_idx], y))
            else:
                row['default_rmse'] = float('nan')
        cut_rows.append(row)
    if not scores:
        return dict(well=wid, status='skip_no_scores', known_prefix=int(len(known_prefix)))
    agg = {}
    for name, vals in scores.items():
        vals = _gold_np.asarray(vals, dtype=float)
        agg[name] = float(_gold_np.nanmedian(vals) + 0.10 * _gold_np.nanstd(vals))
    ordered = sorted((v, k) for k, v in agg.items() if _gold_np.isfinite(v))
    if not ordered:
        return dict(well=wid, status='skip_nonfinite_scores', known_prefix=int(len(known_prefix)))
    best_score, best_name = ordered[0]
    second_score = ordered[1][0] if len(ordered) > 1 else best_score
    if default_name is not None and default_name in agg:
        default_score = float(agg[default_name])
    else:
        pf_scores = [v for k, v in agg.items() if k.startswith('pf|')]
        default_score = float(_gold_np.nanmedian(pf_scores)) if pf_scores else float(second_score)
    consistency = 0.0
    comparable = 0
    for row in cut_rows:
        if _gold_np.isfinite(row.get('default_rmse', _gold_np.nan)):
            comparable += 1
            if row.get('best_rmse', float('inf')) <= row['default_rmse'] - 0.25:
                consistency += 1.0
    if comparable:
        consistency /= comparable
    else:
        winners = [r.get('best_name') for r in cut_rows if r.get('best_name')]
        consistency = float(sum(w == best_name for w in winners)) / max(1, len(winners))
    return dict(
        well=wid,
        status='ok',
        known_prefix=int(len(known_prefix)),
        cuts=int(len(cut_rows)),
        candidate_count=int(len(agg)),
        best_name=best_name,
        best_score=float(best_score),
        second_score=float(second_score),
        default_name=default_name,
        default_score=float(default_score),
        gain=float(default_score - best_score),
        rank_margin=float(second_score - best_score),
        consistency=float(consistency),
        cut_rows=cut_rows,
    )

def _gold_alpha(report, delta_rmse, delta_p95, profile_name):
    p = _GOLD_PROFILES[profile_name]
    if report.get('status') != 'ok':
        return 0.0
    gain = float(report.get('gain', 0.0))
    best = float(report.get('best_score', float('inf')))
    margin = float(report.get('rank_margin', 0.0))
    consistency = float(report.get('consistency', 0.0))
    if (not _gold_np.isfinite(best)) or best > p['max_best'] or gain < p['min_gain'] or consistency < p['min_consistency']:
        return 0.0
    alpha = p['base']
    alpha += p['gain_scale'] * min(max(gain, 0.0), 5.0) / 5.0
    alpha += p['margin_scale'] * min(max(margin, 0.0), 3.0) / 3.0
    if best <= 5.0:
        alpha += p['quality_bonus']
    best_name = str(report.get('best_name', ''))
    if (best_name.startswith('surface_') or best_name.startswith('dense_') or best_name.startswith('poly_') or best_name.startswith('contact_')) and consistency >= 0.67:
        alpha += 0.03 if profile_name != 'aggressive' else 0.06
    if _gold_np.isfinite(delta_rmse) and delta_rmse > p['delta_soft']:
        alpha *= max(0.20, p['delta_soft'] / max(delta_rmse, 1e-6))
    if _gold_np.isfinite(delta_p95) and delta_p95 > p['p95_hard']:
        return 0.0
    return float(min(p['cap'], max(0.0, alpha)))

def _gold_profile_output(base_sub, candidate_by_id, reports_by_well, profile_name):
    prof = _GOLD_PROFILES[profile_name]
    out = base_sub.copy()
    move_rows = []
    for wid, rep in reports_by_well.items():
        ids = out.loc[out['well'] == wid, 'id'].astype(str).tolist()
        if not ids:
            continue
        cand = _gold_np.array([candidate_by_id.get(i, _gold_np.nan) for i in ids], dtype=float)
        idx = out.index[out['well'] == wid].to_numpy()
        base = out.loc[idx, 'tvt'].to_numpy(dtype=float)
        ok = _gold_np.isfinite(cand) & _gold_np.isfinite(base)
        if int(ok.sum()) != len(base):
            rep = dict(rep)
            rep['apply_status'] = 'skip_nonfinite_candidate'
            move_rows.append(rep)
            continue
        diff = cand - base
        delta_rmse = float(_gold_np.sqrt(_gold_np.mean(diff * diff))) if len(diff) else float('nan')
        delta_p95 = float(_gold_np.quantile(_gold_np.abs(diff), 0.95)) if len(diff) else float('nan')
        alpha = _gold_alpha(rep, delta_rmse, delta_p95, profile_name)
        gain = max(0.0, float(rep.get('gain', 0.0)))
        max_move = min(prof['clip_max'], prof['clip_base'] + prof['clip_gain'] * _gold_np.sqrt(gain + 1e-9))
        ramp = 1.0 - _gold_np.exp(-_gold_np.arange(len(diff), dtype=float) / max(80.0, 0.12 * max(1, len(diff))))
        move = _gold_np.clip(alpha * ramp * diff, -max_move, max_move)
        out.loc[idx, 'tvt'] = base + move
        row = dict(rep)
        row.update(dict(
            profile=profile_name,
            alpha=float(alpha),
            delta_rmse_vs_public=float(delta_rmse),
            delta_p95_vs_public=float(delta_p95),
            max_move_clip=float(max_move),
            applied_rows=int(len(idx)),
            mean_abs_move=float(_gold_np.mean(_gold_np.abs(move))) if len(move) else 0.0,
            max_abs_move=float(_gold_np.max(_gold_np.abs(move))) if len(move) else 0.0,
            apply_status='applied' if alpha > 0 else 'kept_public',
        ))
        move_rows.append(row)
    return out, move_rows

def _gold_reapply_guarded_contact_override(sub_df, data_dir):
    sub = _gold_split_ids(sub_df[['id', 'tvt']])
    pred = dict(zip(sub['id'].astype(str), sub['tvt'].astype(float)))
    train_wells = set(p.stem.replace('__horizontal_well', '') for p in (data_dir / 'train').glob('*__horizontal_well.csv'))
    n_ok = 0
    n_skip = 0
    for wid, g in sub.groupby('well'):
        if wid not in train_wells:
            continue
        try:
            hw_te = _gold_pd.read_csv(data_dir / 'test' / f'{wid}__horizontal_well.csv')
            hw_tr = _gold_pd.read_csv(data_dir / 'train' / f'{wid}__horizontal_well.csv')
            tw_tr = _gold_pd.read_csv(data_dir / 'train' / f'{wid}__typewell.csv')
            phys = _gold_tvt_from_contacts(hw_tr, tw_tr)
            md_raw = hw_tr['MD'].to_numpy(dtype=float)
            m = _gold_np.isfinite(phys) & _gold_np.isfinite(md_raw)
            if int(m.sum()) < 100:
                n_skip += 1
                continue
            order = _gold_np.argsort(md_raw[m])
            md_tr = md_raw[m][order]
            ph_tr = phys[m][order]
            kn = hw_te[hw_te['TVT_input'].notna()]
            kn = kn[(kn['MD'] >= md_tr[0]) & (kn['MD'] <= md_tr[-1])]
            if len(kn) < 50:
                n_skip += 1
                continue
            rk = _gold_rmse(_gold_np.interp(kn['MD'].to_numpy(dtype=float), md_tr, ph_tr), kn['TVT_input'].to_numpy(dtype=float))
            if (not _gold_np.isfinite(rk)) or rk > 1.0:
                n_skip += 1
                continue
            md_te = hw_te['MD'].to_numpy(dtype=float)
            n_row = 0
            for rid, ri in zip(g['id'].astype(str).values, g['row_idx'].values):
                ri = int(ri)
                if 0 <= ri < len(md_te):
                    mte = float(md_te[ri])
                    if md_tr[0] <= mte <= md_tr[-1]:
                        pred[rid] = float(_gold_np.interp(mte, md_tr, ph_tr))
                        n_row += 1
            print('gold contact override OK %s rmse=%.4f rows=%d/%d' % (wid, rk, n_row, len(g)))
            n_ok += 1
        except Exception as e:
            print('gold contact override fallback', wid, e)
            n_skip += 1
    sub['tvt'] = sub['id'].astype(str).map(pred).astype(float)
    print('gold contact override done: overridden=%d skipped=%d' % (n_ok, n_skip))
    return sub[['id', 'tvt']]

def _gold_validate_and_write(sub, sample, path):
    out = sub[['id', 'tvt']].copy()
    out['id'] = out['id'].astype(str)
    out['tvt'] = out['tvt'].astype(float)
    if list(out.columns) != ['id', 'tvt']:
        raise RuntimeError('bad output columns')
    if len(out) != len(sample):
        raise RuntimeError('bad output length')
    if not out['id'].equals(sample['id'].astype(str)):
        raise RuntimeError('id order mismatch')
    if not _gold_np.isfinite(out['tvt'].to_numpy(dtype=float)).all():
        raise RuntimeError('non-finite tvt in output')
    out.to_csv(path, index=False)
    return out
