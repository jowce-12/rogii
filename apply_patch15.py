"""Patch 15: gate the gr-fill behavior behind ROGII_GRFILL (default 1 = current).
ROGII_GRFILL=0 restores the pre-patch9 behavior (raw ev.GR -> PF kernel skips NaN
steps). Needed to A/B-diagnose the full-stack OOF regression (9.51 -> 9.61).
Also: trainer cache names get a _g0 suffix when ROGII_GRFILL=0 (g=1 keeps the
old name so existing caches stay valid). Idempotent."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'c3ab97ce')
s = ''.join(c['source'])

if 'ROGII_GRFILL' in s:
    print('Patch 15 already applied to notebook.')
else:
    # --- run_pf_ancc ---
    old_a = ('    gr_fill = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))  # gr fill: feed filled GR to the PF (fewer skipped NaN steps)\n'
             '    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), gr_fill.loc[ev.index].values.astype(np.float64),')
    new_a = ('    if os.environ.get("ROGII_GRFILL", "1") == "1":\n'
             '        _gr_in = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).loc[ev.index].values.astype(np.float64)  # gr fill\n'
             '    else:\n'
             '        _gr_in = ev.GR.values.astype(np.float64)   # raw: kernel skips NaN steps (pre-gr-fill behavior)\n'
             '    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), _gr_in,')
    assert s.count(old_a) == 1, ('ancc anchor', s.count(old_a))
    s = s.replace(old_a, new_a)

    # --- run_pf_z ---
    old_z = ('    gr_fill = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))  # gr fill\n'
             '    gr_sm = gr_fill.rolling(PF_GR_WIN, center=True, min_periods=1).mean()\n'
             '    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), gr_fill.loc[ev.index].values.astype(np.float64),\n'
             '                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,')
    new_z = ('    if os.environ.get("ROGII_GRFILL", "1") == "1":\n'
             '        _gr_src = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))  # gr fill\n'
             '    else:\n'
             '        _gr_src = hw.GR   # raw (pre-gr-fill behavior)\n'
             '    gr_sm = _gr_src.rolling(PF_GR_WIN, center=True, min_periods=1).mean()\n'
             '    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), _gr_src.loc[ev.index].values.astype(np.float64),\n'
             '                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,')
    assert s.count(old_z) == 1, ('z anchor', s.count(old_z))
    s = s.replace(old_z, new_z)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('gated gr fill behind ROGII_GRFILL (default 1)')

# --- trainer cache names: _g0 suffix when GRFILL=0 (backward compatible for g=1) ---
for f in ['build_stack_notebook.py', 'build_lgb_notebook.py', 'build_train_notebook.py', 'build_quick_notebook.py']:
    t = io.open(f, encoding='utf-8').read()
    if '_g0' in t:
        print(f, 'cache name already gated'); continue
    old = 'return f"train_features_v5_f{os.environ.get(\'ROGII_FEATS\',\'1\')}_w{_n}.parquet"'
    new = ('_g = "" if os.environ.get("ROGII_GRFILL", "1") == "1" else "_g0"\n'
           '    return f"train_features_v5_f{os.environ.get(\'ROGII_FEATS\',\'1\')}{_g}_w{_n}.parquet"')
    assert t.count(old) == 1, f
    io.open(f, 'w', encoding='utf-8').write(t.replace(old, new))
    print(f, 'cache name gated')
