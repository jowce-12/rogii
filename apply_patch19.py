"""Patch 19 (S1): band-restricted GR recalibration of the typewell INTO the PF
likelihood input. Modes via ROGII_GRCAL: off(default) | affine | var | offset.
- affine: polyfit(tw_at_known, kn.GR) with slope guards (aug-PF-style)
- var:    moment matching on the known TVT band (+-10ft): a=std(hwGR)/std(tw_band)
- offset: level shift only
The typewell curve is transformed tw' = a*tw + b BEFORE the grid/gs are built, so
gs keeps its exact fillna(0) form (bug-as-regularizer preserved) but is measured
against the calibrated curve. Applied to BOTH PF branches:
  - cell ec5d18d6  run_particle_filter (sub_2 main branch, weight 0.7)
  - cell 83876170  lik_pf (fleongg likpf features / anchor)
Default OFF -> production unchanged until the harness 4-way picks a winner.
Idempotent."""
import json, io, sys, ast

HELPER = '''def _grcal_tw(tw_tvt, tw_gr, ktvt, kgr, mode):
    """S1: band-restricted recalibration of the typewell GR into hw-GR units.
    Fits on the known zone only (leak-free); returns (tw_gr', a, b)."""
    ktvt = np.asarray(ktvt, float); kgr = np.asarray(kgr, float)
    v = np.isfinite(kgr)
    if v.sum() < 30:
        return tw_gr, 1.0, 0.0
    lo = float(np.nanmin(ktvt)) - 10.0; hi = float(np.nanmax(ktvt)) + 10.0
    band = tw_gr[(tw_tvt >= lo) & (tw_tvt <= hi)]
    if len(band) < 20:
        band = tw_gr
    bm = float(np.mean(band)); km = float(np.mean(kgr[v]))
    if mode == "offset":
        a, b = 1.0, km - bm
    elif mode == "var":
        sb = float(np.std(band)); sk = float(np.std(kgr[v]))
        a = float(np.clip(sk / sb, 0.2, 5.0)) if sb > 1e-6 else 1.0
        b = km - a * bm
    else:  # affine
        tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
        vv = v & np.isfinite(tw_at_k)
        if vv.sum() < 30 or np.std(tw_at_k[vv]) < 1e-6:
            a, b = 1.0, km - bm
        else:
            a, b = np.polyfit(tw_at_k[vv], kgr[vv], 1)
            if (not np.isfinite(a)) or a < 0.2 or a > 5.0:
                a, b = 1.0, km - bm
    return a * tw_gr + b, float(a), float(b)


'''

GATE_FLEONGG = '''    _gc = os.environ.get("ROGII_GRCAL", "off").lower()
    if _gc in ("affine", "var", "offset"):
        tw_gr, _, _ = _grcal_tw(tw_tvt, tw_gr, kn.TVT_input.values, kn.GR.values, _gc)   # S1
'''

GATE_CELL4 = '''    _gc = os.environ.get("ROGII_GRCAL", "off").lower()
    if _gc in ("affine", "var", "offset"):
        tw_gr, _, _ = _grcal_tw(tw_tvt, tw_gr, kn['TVT_input'].values, kn['GR'].values, _gc)   # S1
'''

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))

# ---- fleongg lik_pf (83876170) ----
c = next(x for x in nb['cells'] if x.get('id') == '83876170')
s = ''.join(c['source'])
if 'ROGII_GRCAL' in s:
    print('83876170 already patched')
else:
    s = s.replace('def lik_pf(', HELPER + 'def lik_pf(', 1)
    o = '''    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([]), {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)'''
    n = '''    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([]), {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
''' + GATE_FLEONGG + '''    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)'''
    assert s.count(o) == 1, ('fleongg anchor', s.count(o))
    s = s.replace(o, n)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print('patched 83876170 (fleongg lik_pf)')

# ---- CELL 4 run_particle_filter (ec5d18d6) ----
c = next(x for x in nb['cells'] if x.get('id') == 'ec5d18d6')
s = ''.join(c['source'])
if 'ROGII_GRCAL' in s:
    print('ec5d18d6 already patched')
else:
    s = s.replace('def run_particle_filter(', HELPER + 'def run_particle_filter(', 1)
    o = '''    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)'''
    n = '''    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

''' + GATE_CELL4 + '''    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)'''
    assert s.count(o) == 1, ('cell4 anchor', s.count(o))
    s = s.replace(o, n)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print('patched ec5d18d6 (sub_2 run_particle_filter)')

json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook saved')
