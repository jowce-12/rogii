"""Patch 6: add physics/PPT-grounded derived features to the fleongg build_well
(CELL 30, id b515d5c6). Groups: (1) local GR-match-quality profile, (2) calibrated
match offsets (slide-9), (3) tracker disagreement/rank, (4) GR-direction alignment
(slides 6-7), (6) normalized extrapolation + uncertainty interactions.
Gated by ROGII_FEATS (default '1'). train_stack/train_lgb/train_gru all pull this
cell, so the batch propagates. Idempotent.
"""
import json, io, sys

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'b515d5c6')
src = ''.join(c['source'])

if 'add_derived_features' in src:
    print('Patch 6 already applied; aborting.')
    sys.exit(0)

HELPER = '''def add_derived_features(feats, hgr, tw_tvt, tw_gr, pf_use, pf_z, has_z, std_use,
                         beam_mean, sc_ens, tvt_dense, tvtF_ANCC, z_ev, dzdmd,
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
    dz = np.nan_to_num(np.asarray(dzdmd, np.float64), nan=0.)
    feats["tw_grad_pf"] = _f(tw_grad_pf)
    feats["grdir_align"] = _f(grd * tw_grad_pf * dz)
    # (6) normalized extrapolation distance + uncertainty interactions
    kl = max(float(known_len), 1.0)
    std64 = np.asarray(std_use, np.float64); ms = np.asarray(md_since, np.float64)
    feats["extrap_ratio"] = _f(ms / kl)
    feats["unc_x_dist"] = _f(std64 * ms)
    feats["unc_x_frac"] = _f(std64 * (np.arange(nh) / max(nh - 1, 1)))


'''

# 1) prepend helper to the cell
src = HELPER + src

# 2) insert the call right before the DataFrame is built
anchor = '    for k,v in rolls.items(): feats[k]=v'
call = (anchor + '\n'
        '    add_derived_features(feats, hgr, tw_tvt, tw_gr, pf_use, pf_z, has_z, std_use,\n'
        '                         np.stack(list(bpaths.values()), 1).mean(1).astype(np.float32),\n'
        '                         sc_ens, tvt_dense, tvt_fs["tvtF_ANCC"], z_ev, dzdmd,\n'
        '                         last_tvt, a_cal, b_cal, md_since, len(kn), nh)')
assert src.count(anchor) == 1
src = src.replace(anchor, call)

c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('added derived-feature batch to build_well (CELL 30)')
