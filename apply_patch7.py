"""Patch 7: extend add_derived_features with a 2nd batch (v2) of derived features
and bump the feature-cache version. Edits build_well (CELL 30, id b515d5c6) in the
main notebook; the standalone trainers pull this cell so they inherit it too.
Idempotent (keys off the v2 marker).
"""
import json, io, sys

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'b515d5c6')
src = ''.join(c['source'])

if 'derived v2 batch' in src:
    print('Patch 7 already applied; aborting.')
    sys.exit(0)

NEW_FUNC = '''def add_derived_features(feats, hgr, tw_tvt, tw_gr, pf_use, pf_z, has_z, std_use,
                         beam_mean, sc_ens, tvt_dense, tvtF_ANCC, z_ev, dzdmd, dxdmd, dydmd,
                         last_tvt, a_cal, b_cal, md_since, known_len, nh):
    """Physics/PPT-grounded derived features. All computable in the eval zone."""
    import os as _os
    if _os.environ.get("ROGII_FEATS", "1") != "1":
        return
    f32 = np.float32
    def _f(a):
        return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0., posinf=0., neginf=0.).astype(f32)
    tw_tvt = np.asarray(tw_tvt, np.float64); tw_gr = np.asarray(tw_gr, np.float64)
    hgr64 = np.asarray(hgr, np.float64); pf64 = np.asarray(pf_use, np.float64)
    # (1) local GR-match-quality profile -- rising = the track is drifting off the log
    tw_at_pf = np.interp(pf64, tw_tvt, tw_gr)
    mq = np.abs(hgr64 - tw_at_pf)
    s = pd.Series(mq)
    feats["mq_res"] = _f(mq)
    for w in (21, 51, 101):
        feats[f"mq_roll{w}"] = _f(s.rolling(w, center=True, min_periods=1).mean().values)
    feats["mq_roll51_std"] = _f(s.rolling(51, center=True, min_periods=1).std().fillna(0).values)
    feats["mq_cum"] = _f(s.expanding().mean().values)
    # (2) calibrated match offsets (slide-9: typewell GR -> horizontal GR units)
    for o in (-8, -4, 0, 4, 8):
        feats[f"tdpf_cal{int(o)}"] = _f(hgr64 - (a_cal * np.interp(pf64 + o, tw_tvt, tw_gr) + b_cal))
    for o in (-20, -5, 0, 5, 20):
        feats[f"tda_cal{int(o)}"] = _f(hgr64 - (a_cal * np.interp(last_tvt + o, tw_tvt, tw_gr) + b_cal))
    # (3) cross-tracker disagreement / rank of the PF within the panel
    cols = [pf64, (np.asarray(pf_z, np.float64) if has_z else pf64),
            np.asarray(beam_mean, np.float64), np.asarray(sc_ens, np.float64),
            np.asarray(tvt_dense, np.float64), np.asarray(tvtF_ANCC, np.float64)]
    trk = np.stack(cols, 1)
    med = np.median(trk, 1)
    feats["trk_std"] = _f(trk.std(1))
    feats["trk_iqr"] = _f(np.percentile(trk, 75, 1) - np.percentile(trk, 25, 1))
    feats["trk_med_d"] = _f(med - last_tvt)
    feats["pf_minus_med"] = _f(pf64 - med)
    # (4) GR-direction alignment (slides 6-7): does GR change match the expected log slope?
    d_gr = np.gradient(tw_gr); d_tvt = np.gradient(tw_tvt)
    tw_grad = d_gr / np.where(np.abs(d_tvt) < 1e-6, 1e-6, d_tvt)
    tw_grad_pf = np.interp(pf64, tw_tvt, tw_grad)
    grd = np.gradient(hgr64)
    dzc = np.nan_to_num(np.asarray(dzdmd, np.float64), nan=0.)
    feats["tw_grad_pf"] = _f(tw_grad_pf)
    feats["grdir_align"] = _f(grd * tw_grad_pf * dzc)
    # (6) normalized extrapolation distance + uncertainty interactions
    kl = max(float(known_len), 1.0)
    std64 = np.asarray(std_use, np.float64); ms = np.asarray(md_since, np.float64)
    feats["extrap_ratio"] = _f(ms / kl)
    feats["unc_x_dist"] = _f(std64 * ms)
    feats["unc_x_frac"] = _f(std64 * (np.arange(nh) / max(nh - 1, 1)))
    # ===== derived v2 batch =====
    # signed match residual (drift DIRECTION/bias; distinct from |mq|)
    ss = pd.Series(hgr64 - tw_at_pf)
    feats["sres_roll21"] = _f(ss.rolling(21, center=True, min_periods=1).mean().values)
    feats["sres_roll51"] = _f(ss.rolling(51, center=True, min_periods=1).mean().values)
    # best local shift over a small offset grid (is pf biased up/down here?)
    offs = np.array([-12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12], np.float64)
    rstack = np.stack([np.abs(hgr64 - np.interp(pf64 + o, tw_tvt, tw_gr)) for o in offs], 0)
    bi = rstack.argmin(0)
    feats["best_shift"] = _f(offs[bi])
    feats["best_absres"] = _f(rstack.min(0))
    # typewell local discriminability around pf (flat log => ambiguous => uncertain)
    tw_rstd = pd.Series(tw_gr).rolling(41, center=True, min_periods=1).std().fillna(0).values
    feats["twloc_std"] = _f(np.interp(pf64, tw_tvt, tw_rstd))
    # GR texture (hi/lo-freq energy ratio) + local slope
    hs = pd.Series(hgr64)
    g21 = hs.rolling(21, center=True, min_periods=1).std().fillna(0).values
    g101 = hs.rolling(101, center=True, min_periods=1).std().fillna(0).values
    feats["gr_hf_ratio"] = _f(g21 / (g101 + 1e-6))
    feats["gr_slope21"] = _f(np.gradient(hs.rolling(21, center=True, min_periods=1).mean().values))
    # PF trajectory velocity / acceleration (predicted TVT change rate)
    pv = np.gradient(pf64)
    feats["pf_vel"] = _f(pv)
    feats["pf_accel"] = _f(np.gradient(pv))
    # trajectory heading + inclination (slide 12: azimuth affects the experienced dip)
    dxm = np.nan_to_num(np.asarray(dxdmd, np.float64)); dym = np.nan_to_num(np.asarray(dydmd, np.float64))
    hyp = np.hypot(dxm, dym) + 1e-6
    feats["head_sin"] = _f(dym / hyp); feats["head_cos"] = _f(dxm / hyp)
    feats["incl"] = _f(dzc / hyp)
    # normalized extrapolation vs typewell range
    tw_rng = float(np.ptp(tw_tvt)) if np.ptp(tw_tvt) > 1e-6 else 1.0
    feats["extrap_vs_twrange"] = _f(ms / tw_rng)


'''

# replace the whole add_derived_features function (helper sits before build_well)
i = src.index('def add_derived_features')
j = src.index('def build_well')
src = src[:i] + NEW_FUNC + src[j:]

# update the call to pass dxdmd, dydmd
old_call = 'sc_ens, tvt_dense, tvt_fs["tvtF_ANCC"], z_ev, dzdmd,\n'
new_call = 'sc_ens, tvt_dense, tvt_fs["tvtF_ANCC"], z_ev, dzdmd, dxdmd, dydmd,\n'
assert src.count(old_call) == 1, ('call anchor', src.count(old_call))
src = src.replace(old_call, new_call)

import ast
ast.parse(src)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('extended add_derived_features with v2 batch (+ dxdmd/dydmd)')
