"""Patch 21: deploy the VALIDATED sub_2 chain + projection upgrade to BOTH
submission notebooks. Validation: tuned on 150 train wells (seed7), confirmed on
a disjoint 150 (seed11) — selector-level pooled RMSE 11.97->9.87 / 10.08->8.46.

  T1  S1-affine 2nd PF channel, blend w=0.35 (64 seeds for the affine pass)
  T2  neighbor-surface anchor blend wd=0.3, gate: median scaled dense-dist<=0.0083
  T3  projection: PS-anchored (no-intercept) robust deg-4 fit (dU(0)=0 physical)
  T4  alias hold-gate: gr_corr>0.85 & tw_hf_std<14.0112 & excursion<15ft -> h=0.35

Targets:
  rogii-geology-aware-ensembling-lb-7-129.ipynb  cells [24] sub_2 loop, [28] projection
  public-score-rogii-lb-7-159.ipynb              cells f9218767, d48e402e
Current notebook: replaces bin-variant selector + aug-PF block with the same
validated chain (aug's affine content is superseded by the cleaner S1 channel).
Idempotent."""
import json, io, sys, ast

PRE = '''
    # === validated sub_2 upgrade: S1-affine blend + neighbor anchor + alias gate ===
    # Tuned on 150 train wells (seed7), CONFIRMED on disjoint 150 (seed11):
    # selector pooled RMSE 11.97->9.87 / 10.08->8.46. Constants train-derived.
'''

HELPERS = '''# --- sub_2 upgrade helpers (validated chain; see IMPROVEMENTS.md) ---
from scipy.spatial import cKDTree as _s2KDT
_S2_THR = 0.0083     # dense-dist gate (median of train wells' median scaled NN dist)
_S2_HF = 14.0112     # tw_hf_std median (alias gate)
def _s2_grcal(tw_tvt, tw_gr, ktvt, kgr):
    ktvt = np.asarray(ktvt, float); kgr = np.asarray(kgr, float)
    v = np.isfinite(kgr)
    if v.sum() < 30: return tw_gr
    band = tw_gr[(tw_tvt >= float(np.nanmin(ktvt)) - 10.0) & (tw_tvt <= float(np.nanmax(ktvt)) + 10.0)]
    if len(band) < 20: band = tw_gr
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
class _s2Dense:
    def __init__(self, wids, ddir, spw=60):
        xs, ys, an = [], [], []
        for w in wids:
            try:
                df = pd.read_csv(ddir / f"{w}__horizontal_well.csv", usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int); s_ = df.iloc[ix]
            xs.append(s_.X.values); ys.append(s_.Y.values); an.append(s_.ANCC.values)
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.an = np.concatenate(an).astype(np.float64)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0))
        self.tree = _s2KDT(self.xy / self.scale)
    def impute(self, xy_q, k=20, nfetch=3000):
        q = np.atleast_2d(xy_q) / self.scale; nf = min(nfetch, len(self.an))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        o = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, o, 1); ik = np.take_along_axis(idx, o, 1)
        w = 1.0 / (dk + 1e-3)
        return (self.an[ik] * w).sum(1) / w.sum(1), dist.min(1)
_s2_imp = _s2Dense(train_wells, CFG.dataset_path / 'train')
print(f'sub_2 upgrade ready (dense imputer over {len(train_wells)} train wells)')

'''

CHAIN = '''    _sel_off = (pf_by_scale['pf_scale_3'] + pf_by_scale['pf_scale_5'] + pf_by_scale['pf_scale_8']) / 3.0
    tvt_selector = np.asarray(_sel_off, float).copy()
    _kn2 = hw_te[hw_te['TVT_input'].notna()]; _ev2 = hw_te[hw_te['TVT_input'].isna()]
    _ei2 = list(_ev2.index)
    _tws = tw_ref.sort_values('TVT')
    _ttvt2 = _tws['TVT'].values.astype(float)
    _tgr2 = _tws['GR'].fillna(_tws['GR'].mean()).values.astype(float)
    # (T1) S1-affine second PF channel, blend w=0.35
    try:
        _tw_aff = pd.DataFrame({'TVT': _ttvt2,
                                'GR': _s2_grcal(_ttvt2, _tgr2, _kn2['TVT_input'].values, _kn2['GR'].values),
                                'Geology': np.nan})
        _pf_aff = run_pf_lik_ensemble_scales(hw_te, _tw_aff, n_particles=500, n_seeds=64)
        _sel_aff = (_pf_aff['pf_scale_3'] + _pf_aff['pf_scale_5'] + _pf_aff['pf_scale_8']) / 3.0
        tvt_selector = 0.65 * tvt_selector + 0.35 * np.asarray(_sel_aff, float)
        print('  S1-affine blend OK (w=0.35)')
    except Exception as _e:
        print(f'  S1-affine skipped: {_e}')
    # (T2) neighbor-surface anchor blend (validated wd=0.3)
    try:
        _dv, _dd = _s2_imp.impute(_ev2[['X', 'Y']].to_numpy(float))
        _dk, _ = _s2_imp.impute(_kn2[['X', 'Y']].to_numpy(float))
        _b2 = float(np.median(_kn2['TVT_input'].values + _kn2['Z'].values - _dk))
        if float(np.median(_dd)) <= _S2_THR and len(_ei2):
            tvt_selector[_ei2] = 0.7 * tvt_selector[_ei2] + 0.3 * (-_ev2['Z'].values.astype(float) + _dv + _b2)
            print('  dense-anchor blend applied')
    except Exception as _e:
        print(f'  dense-anchor skipped: {_e}')
    # (T4) alias hold-gate (high raw gr_corr, low HF signal, low excursion)
    try:
        _kg2 = _kn2['GR'].values.astype(float); _v2 = np.isfinite(_kg2)
        _ta2 = np.interp(_kn2['TVT_input'].values.astype(float), _ttvt2, _tgr2)
        _gcorr = float(np.corrcoef(_kg2[_v2], _ta2[_v2])[0, 1]) if _v2.sum() > 100 else 0.0
        _band = _tgr2[(_ttvt2 >= _kn2['TVT_input'].min() - 10) & (_ttvt2 <= _kn2['TVT_input'].max() + 10)]
        _hf = float((pd.Series(_band) - pd.Series(_band).rolling(101, center=True, min_periods=1).mean()).std()) if len(_band) >= 40 else 99.0
        _exc = float(np.max(np.abs(tvt_selector[_ei2] - last_known_tvt))) if len(_ei2) else 0.0
        if _gcorr > 0.85 and _hf < _S2_HF and _exc < 15:
            tvt_selector[_ei2] = 0.65 * tvt_selector[_ei2] + 0.35 * last_known_tvt
            print('  alias-gate hold applied')
    except Exception:
        pass'''

ROBFIT_NEW = '''def _robfit(s, y, deg=4):
    # PS-anchored robust fit: no intercept => dU(0)=0 (known->eval boundary jump is
    # physically zero — probe median 0.000). Validated vs free-intercept deg3/4 on
    # two disjoint 150-well samples.
    if len(s) < deg + 2:
        return y.copy()
    A = _np.column_stack([s**k for k in range(1, deg + 1)])
    w = _np.ones(len(s))
    for _ in range(4):
        try:
            c = _np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        except Exception:
            return y.copy()
        r = y - A @ c
        sc = _np.median(_np.abs(r)) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (r / (2.0 * sc)) ** 2)
    return A @ c'''

ROBFIT_OLD_TPL = '''def _robfit(s, y, deg={D}):
    if len(s) < deg + 2:
        return y.copy()
    c = _np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - _np.polyval(c, s)
        sc = _np.median(_np.abs(r)) * 1.4826 + 1e-6
        c = _np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return _np.polyval(c, s)'''


def patch_projection(cellsrc, old_deg_def, old_deg_call):
    o = ROBFIT_OLD_TPL.replace('{D}', str(old_deg_def))
    assert cellsrc.count(o) == 1, ('robfit def', cellsrc.count(o))
    cellsrc = cellsrc.replace(o, ROBFIT_NEW)
    oc = f"_fit = _robfit(_s, (_tvt + _Z) - _anchor, {old_deg_call})"
    assert cellsrc.count(oc) == 1, ('robfit call', cellsrc.count(oc))
    return cellsrc.replace(oc, "_fit = _robfit(_s, (_tvt + _Z) - _anchor, 4)")


# ============ notebook A: 7.129 ============
PA = 'rogii-geology-aware-ensembling-lb-7-129.ipynb'
nbA = json.load(io.open(PA, encoding='utf-8'))
sA = ''.join(nbA['cells'][24]['source'])
if '_s2_imp' in sA:
    print('A[24] already patched')
else:
    o = "rows = []\nfor i, wid in enumerate(test_wells):"
    assert sA.count(o) == 1
    sA = sA.replace(o, HELPERS + o)
    o = """    # أخذ المتوسط الحسابي للمقاييس الثلاثة لقتل التذبذب
    tvt_selector = (pf_by_scale['pf_scale_3'] + pf_by_scale['pf_scale_5'] + pf_by_scale['pf_scale_8']) / 3.0
    print(f'  Selector: USING AVERAGE OF SCALES (3, 5, 8)')"""
    assert sA.count(o) == 1, ('A selector', sA.count(o))
    sA = sA.replace(o, PRE + CHAIN)
    ast.parse(sA)
    nbA['cells'][24]['source'] = sA.splitlines(keepends=True)
    print('A[24] sub_2 chain deployed')
s28 = ''.join(nbA['cells'][28]['source'])
if 'PS-anchored robust fit' in s28:
    print('A[28] already patched')
else:
    s28 = patch_projection(s28, 3, 3)
    ast.parse(s28)
    nbA['cells'][28]['source'] = s28.splitlines(keepends=True)
    print('A[28] projection -> PS-anchored deg4')
json.dump(nbA, io.open(PA, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ============ notebook B: current ============
PB = 'public-score-rogii-lb-7-159.ipynb'
nbB = json.load(io.open(PB, encoding='utf-8'))
cB = next(x for x in nbB['cells'] if x.get('id') == 'f9218767')
sB = ''.join(cB['source'])
if '_s2_imp' in sB:
    print('B sub_2 already patched')
else:
    o = "rows = []\nfor i, wid in enumerate(test_wells):"
    assert sB.count(o) == 1
    sB = sB.replace(o, HELPERS + o)
    # replace bin-variant selector + aug-PF block with the validated chain
    i0 = sB.find("    tvt_selector = apply_selector_variant(")
    i1 = sB.find("        print(f'  Self-augmented PF skipped: {e}')")
    assert i0 > 0 and i1 > i0
    i1 = sB.index('\n', i1) + 1
    sB = sB[:i0] + PRE + CHAIN + '\n' + sB[i1:]
    ast.parse(sB)
    cB['source'] = sB.splitlines(keepends=True)
    print('B sub_2 chain deployed (bin-variant + aug-PF superseded)')
cP = next(x for x in nbB['cells'] if x.get('id') == 'd48e402e')
sP = ''.join(cP['source'])
if 'PS-anchored robust fit' in sP:
    print('B projection already patched')
else:
    sP = patch_projection(sP, 5, 4)
    ast.parse(sP)
    cP['source'] = sP.splitlines(keepends=True)
    print('B projection -> PS-anchored deg4')
json.dump(nbB, io.open(PB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('done')
