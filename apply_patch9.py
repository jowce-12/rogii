"""Patch 9 (from rogii_best.py): 'gr fill' — feed interpolated GR to the single
particle filters run_pf_ancc / run_pf_z (fleongg cell c3ab97ce), instead of raw
ev.GR (which the PF kernel skips at NaN steps -> free drift in GR gaps). This
improves pf_ancc / pf_z (both important features). Only the fleongg PF cell is
patched (the 1st-stack CELL 5 keeps raw GR to stay consistent with its
precomputed models). Bumps feature cache to v4. Idempotent.
"""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'c3ab97ce')
src = ''.join(c['source'])

if 'gr fill: feed filled GR' in src:
    print('Patch 9 already applied; aborting.')
    sys.exit(0)

# --- run_pf_ancc: raw ev.GR -> gr_fill ---
anc_old = (
    "    gg, gmin, gst = _grid(tw_tvt, tw_gr)\n"
    "    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),\n"
    "                        gg, gmin, gst, gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)"
)
anc_new = (
    "    gg, gmin, gst = _grid(tw_tvt, tw_gr)\n"
    "    gr_fill = hw.GR.astype(float).interpolate(limit_direction=\"both\").fillna(float(np.nanmean(tw_gr)))  # gr fill: feed filled GR to the PF (fewer skipped NaN steps)\n"
    "    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), gr_fill.loc[ev.index].values.astype(np.float64),\n"
    "                        gg, gmin, gst, gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)"
)
assert src.count(anc_old) == 1, ('ancc anchor', src.count(anc_old))
src = src.replace(anc_old, anc_new)

# --- run_pf_z: raw ev.GR + gr_sm on raw -> gr_fill for both ---
z_old = (
    "    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)\n"
    "    gr_sm = hw.GR.rolling(PF_GR_WIN, center=True, min_periods=1).mean()\n"
    "    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), ev.GR.values.astype(np.float64),\n"
    "                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,"
)
z_new = (
    "    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)\n"
    "    gr_fill = hw.GR.astype(float).interpolate(limit_direction=\"both\").fillna(float(np.nanmean(tw_gr)))  # gr fill\n"
    "    gr_sm = gr_fill.rolling(PF_GR_WIN, center=True, min_periods=1).mean()\n"
    "    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), gr_fill.loc[ev.index].values.astype(np.float64),\n"
    "                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,"
)
assert src.count(z_old) == 1, ('z anchor', src.count(z_old))
src = src.replace(z_old, z_new)

ast.parse(src)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('applied gr fill to fleongg run_pf_ancc / run_pf_z')
