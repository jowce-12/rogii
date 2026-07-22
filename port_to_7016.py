"""Port every improvement from rogii-geology-aware-ensembling-lb-7-129.ipynb into
target-free-tvt-geosteering-7-016.ipynb (new public best, LB 7.016) so the current
train_stack.py output dataset (lgb*.pkl + cb*.cbm + ridge.pkl + stack_meta.json +
features.json, cache v7) runs there.

Replaced wholesale (learned/fleongg branch; matched by position + signature):
  [33] imports/CFG            <- 44e50dd0 (NJOBS uncap, data guard)
  [36] PF kernels             <- c3ab97ce (A6 beam resample, GRFILL gate, _grcal_tw, nogil)
  [37] lik-PF                 <- 83876170 (S1 blend mode + quality)
  [39] spatial imputers       <- c9bba3da
  [40] build_well pipeline    <- b515d5c6 (A5 + 50 derived feats + quality likpf rows)
  [41] _device/configs        <- cc1a056a (gpu cascade, retuned configs)
  [43] _find_models/main      <- 209071f6 (FULL-STACK inference + meta_extra + grcal warn
                                 + A5 test join + missing-feature warn)
Edited in place (7.016 score-makers preserved):
  [ 6] SP45_PROJECTION_DEGREE 3 -> 4 (pairs with the PS-anchored basis in [30])
  [ 7] + ROGII_GRCAL=blend (must match training)
  [11] patch24: PF seed loop -> 4-process parallel (identical results, ~4x sub_2+gold)
  [14] patch23: 5 track-A numba kernels get nogil=True
  [26] patch21 chain LAYERED ON TOP of the contact-gated selector: T1 S1-affine blend
       (w=0.35) + T2 dense-anchor blend (wd=0.3, gated). T4 alias-hold SKIPPED — the
       notebook's own prefix-trust gate covers that failure mode.
  [30] _robfit -> PS-anchored no-intercept IRLS (their 0.75 blend weight kept)
  [42] PP kept as-is (their tuning) + A8 clip only
Preserved untouched: control-panel profiles (vp_balanced_final, sp45 0.60, cut fracs),
contact-gated selector variants + heel calibration + adaptive temp + trust gate +
bimodal machinery, guarded overlap override, VP calibration cells, model-package cell.
"""
import json, io, ast, shutil, sys

SRC = 'public-score-rogii-lb-7-159.ipynb'
TGT = 'target-free-tvt-geosteering-7-016.ipynb'
BAK = 'target-free-tvt-geosteering-7-016.BACKUP.ipynb'

src_nb = json.load(io.open(SRC, encoding='utf-8'))
tgt_nb = json.load(io.open(TGT, encoding='utf-8'))

def src_cell(cid):
    return ''.join(next(c for c in src_nb['cells'] if c.get('id') == cid)['source'])

if any('add_alias_metafeats' in ''.join(c['source']) for c in tgt_nb['cells']):
    print('already ported; aborting'); sys.exit(0)
shutil.copy(TGT, BAK)

def cell(idx, sig):
    c = tgt_nb['cells'][idx]
    cur = ''.join(c['source'])
    assert c['cell_type'] == 'code' and cur.strip().startswith(sig), (idx, sig, cur[:70])
    return c, cur

def setcell(c, s, tag):
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print(tag)

def rep(cur, old, new, tag):
    n = cur.count(old)
    assert n == 1, f"{tag}: matched {n}"
    return cur.replace(old, new)

# ---------------- wholesale replacements ----------------
PLAN = [
    (33, 'import os, sys, glob, time, warnings, multiprocessing', '44e50dd0'),
    (36, '# ---- single particle filters', 'c3ab97ce'),
    (37, '# ---- 128-seed likelihood-weighted particle filter', '83876170'),
    (39, 'PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20', 'c9bba3da'),
    (40, 'def build_well(hw_path, tw_path, is_train', 'b515d5c6'),
    (41, 'def _device():', 'cc1a056a'),
    (43, 'def _find_models():', '209071f6'),
]
for idx, sig, cid in PLAN:
    c, cur = cell(idx, sig)
    new = src_cell(cid)
    setcell(c, new, f'[{idx:2d}] <- source {cid} ({len(cur)} -> {len(new)} chars)')

# ---------------- [6] projection degree 3 -> 4 ----------------
c6, cur = cell(6, '# Profile choices:')
cur = rep(cur, 'SP45_PROJECTION_DEGREE = 3',
          'SP45_PROJECTION_DEGREE = 4   # deg-4 with the PS-anchored no-intercept basis (validated on 2x150 wells)',
          '[6] degree')
setcell(c6, cur, '[ 6] SP45_PROJECTION_DEGREE 3 -> 4')

# ---------------- [7] grcal env ----------------
c7, cur = cell(7, '# Runtime bridge')
assert 'ROGII_GRCAL' not in cur
cur = cur.rstrip() + "\nos.environ['ROGII_GRCAL'] = 'blend'   # S1 GR-recal blend; MUST match train_stack training\n"
setcell(c7, cur, '[ 7] ROGII_GRCAL=blend added')

# ---------------- [11] patch24: process-parallel PF seed loop ----------------
HELP24 = '''def _pf_seed_batch(hw, tw, n_particles, seeds):
    return [run_particle_filter(hw, tw, n_particles=n_particles, seed=int(s)) for s in seeds]

def _pf_all_seeds(hw, tw, n_particles, n_seeds):
    # process-parallel over seed chunks (the pure-python PF holds the GIL, so threads
    # cannot help). Chunks keep seed order -> identical results to the serial loop.
    import os as _os
    from joblib import Parallel as _Par, delayed as _Del
    _procs = max(1, min(4, _os.cpu_count() or 1))
    if _procs == 1 or n_seeds < 2 * _procs:
        return _pf_seed_batch(hw, tw, n_particles, range(n_seeds))
    chunks = [c for c in np.array_split(np.arange(n_seeds), _procs) if len(c)]
    try:
        res = _Par(n_jobs=len(chunks), backend="loky")(
            _Del(_pf_seed_batch)(hw, tw, n_particles, c) for c in chunks)
        return [pair for batch in res for pair in batch]
    except Exception as _e:
        print(f"[pf] process pool failed ({str(_e)[:60]}); serial fallback", flush=True)
        return _pf_seed_batch(hw, tw, n_particles, range(n_seeds))


def run_pf_lik_ensemble('''
L1_OLD = '''    preds = []
    liks  = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)'''
L1_NEW = '''    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)
    preds = [p for p, _ll in pairs]
    liks  = [ll for _p, ll in pairs]'''
L2_OLD = L1_OLD.replace('liks  = []', 'liks = []')
L2_NEW = '''    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)
    preds = [p for p, _ll in pairs]
    liks = [ll for _p, ll in pairs]'''
c11, cur = cell(11, 'SELECTOR_N_EVAL_THRESHOLD')
cur = rep(cur, 'def run_pf_lik_ensemble(', HELP24, '[11] helpers')
cur = rep(cur, L1_OLD, L1_NEW, '[11] loop1')
cur = rep(cur, L2_OLD, L2_NEW, '[11] loop2')
setcell(c11, cur, '[11] PF seed loop -> 4-process parallel')

# ---------------- [14] patch23: nogil on track-A kernels ----------------
c14, cur = cell(14, 'SEED=42')
n = cur.count('@njit(cache=True)\n')
assert n == 5, f'[14] expected 5 kernels, found {n}'
cur = cur.replace('@njit(cache=True)\n', '@njit(cache=True, nogil=True)\n')
setcell(c14, cur, '[14] 5 kernels -> nogil')

# ---------------- [26] patch21 chain (T1 + T2, layered on their selector) ----------------
sub2_src = ''.join(next(c for c in src_nb['cells'] if c.get('id') == 'f9218767')['source'])
h0 = sub2_src.index('# --- sub_2 upgrade helpers')
h1 = sub2_src.index('\n\nrows = []', h0)
HELPERS = sub2_src[h0:h1].replace('    _S2_HF = 14.0112', '_S2_HF = 14.0112')  # no-op safety
assert '_s2_imp = _s2Dense(train_wells' in HELPERS and 'def _s2_grcal' in HELPERS

CHAIN = '''    # === ported sub_2 upgrades (validated on 2x150 disjoint train wells; IMPROVEMENTS.md) ===
    # Layered ON TOP of the contact-gated selector output. T1: S1-affine second PF channel
    # blended at w=0.35 (decorrelated GR-calibrated tracker). T2: neighbor-surface anchor
    # blend wd=0.3, gated by scaled dense-dist. T4 alias-hold intentionally NOT ported:
    # this notebook's prefix-trust gate already covers that failure mode.
    tvt_selector = np.asarray(tvt_selector, float).copy()
    _kn2 = hw_te[hw_te['TVT_input'].notna()]; _ev2 = hw_te[hw_te['TVT_input'].isna()]
    _ei2 = list(_ev2.index)
    _tws = tw_ref.sort_values('TVT')
    _ttvt2 = _tws['TVT'].values.astype(float)
    _tgr2 = _tws['GR'].fillna(_tws['GR'].mean()).values.astype(float)
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
    try:
        _dv, _dd = _s2_imp.impute(_ev2[['X', 'Y']].to_numpy(float))
        _dk, _ = _s2_imp.impute(_kn2[['X', 'Y']].to_numpy(float))
        _b2 = float(np.median(_kn2['TVT_input'].values + _kn2['Z'].values - _dk))
        if float(np.median(_dd)) <= _S2_THR and len(_ei2):
            tvt_selector[_ei2] = 0.7 * tvt_selector[_ei2] + 0.3 * (-_ev2['Z'].values.astype(float) + _dv + _b2)
            print('  dense-anchor blend applied')
    except Exception as _e:
        print(f'  dense-anchor skipped: {_e}')

    ws = sample[sample['well'] == wid]'''
c26, cur = cell(26, 'sample = pd.read_csv')
cur = rep(cur, 'rows = []\nbimodal_report_rows = []', HELPERS + '\nrows = []\nbimodal_report_rows = []', '[26] helpers')
cur = rep(cur, "    ws = sample[sample['well'] == wid]", CHAIN, '[26] chain')
setcell(c26, cur, '[26] sub_2 chain T1+T2 layered onto contact-gated selector')

# ---------------- [30] PS-anchored projection basis ----------------
ROB_OLD = '''def _robfit(s, y, deg=5):
    if len(s) < deg + 2:
        return y.copy()
    c = _np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - _np.polyval(c, s)
        sc = _np.median(_np.abs(r)) * 1.4826 + 1e-6
        c = _np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return _np.polyval(c, s)'''
ROB_NEW = '''def _robfit(s, y, deg=4):
    # PS-anchored robust fit: no intercept => dU(0)=0 (the known->eval boundary jump is
    # physically zero). Validated vs the free-intercept polyfit on 2x150 disjoint wells.
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
c30, cur = cell(30, '# Robust low-order projection')
cur = rep(cur, ROB_OLD, ROB_NEW, '[30] robfit')
setcell(c30, cur, '[30] projection -> PS-anchored deg4 (their 0.75 blend kept)')

# ---------------- [42] PP: A8 clip only ----------------
c42, cur = cell(42, 'class PP:')
o = '''    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    pred = last + delta'''
cur = rep(cur, o, '''    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    delta = np.clip(delta, -110.0, 110.0)   # A8: train max|target|=98.9ft -> free runaway guard
    pred = last + delta''', '[42] clip')
setcell(c42, cur, '[42] PP kept (their tuning) + A8 clip')

json.dump(tgt_nb, io.open(TGT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved', TGT)

# ---------------- verification ----------------
tgt = json.load(io.open(TGT, encoding='utf-8'))
a = '\n'.join(''.join(c['source']) for c in tgt['cells'])
checks = [
    ('FULL-STACK 추론', 'FULL-STACK inference'), ('meta_extra 게이트', '_meta.get("meta_extra")'),
    ('missing-feat warn', 'zero-filled'), ('A5 join(test)', 'add_alias_metafeats(test_df'),
    ('quality PF', 'with_quality=True'), ('quality skip', 'LIKPF_QUALITY'),
    ('파생피처', 'def add_derived_features'), ('S1 blend likpf', '_pf_lik_allseeds'),
    ('A6 beam', 'A6: resample typewell'), ('grcal env', "os.environ['ROGII_GRCAL'] = 'blend'"),
    ('grcal warn', 'grcal mismatch'), ('A8 clip', 'A8: train max|target|'),
    ('T1 chain', 'S1-affine blend OK'), ('T2 chain', 'dense-anchor blend applied'),
    ('patch24', '_pf_all_seeds'), ('PS-anchored proj', 'PS-anchored robust fit'),
    ('[보존] 프로파일', "SUBMISSION_PROFILE = 'vp_balanced_final'"),
    ('[보존] selector bins', 'SELECTOR_BIN_VARIANTS'),
    ('[보존] n_eval 임계', 'SELECTOR_N_EVAL_THRESHOLD = 4840.0'),
    ('[보존] heel cal', 'RUN_HEEL_CALIBRATION = True'),
    ('[보존] trust gate', 'RUN_PREFIX_TRUST_GATE = True'),
    ('[보존] selector 호출', 'tvt_selector, bimodal_info = apply_selector_variant('),
    ('[보존] proj blend 0.75', 'SP45_PROJECTION_BLEND_WEIGHT = 0.75'),
    ('[보존] VP cal seeds', 'VISIBLE_PREFIX_CAL_SEEDS = 24'),
    ('[의도적 미포팅] T4', '!alias-gate hold'),
]
for name, m in checks:
    if m.startswith('!'):
        print(('OK  ' if m[1:] not in a else 'FAIL') + ' ' + name)
    else:
        print(('OK  ' if m in a else 'FAIL') + ' ' + name)
for c in tgt['cells']:
    if c['cell_type'] == 'code' and ''.join(c['source']).strip():
        ast.parse(''.join(c['source']))
print('all code cells parse OK')
