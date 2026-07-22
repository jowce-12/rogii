# Patch 26 — STRIDE channel into sub_2 (all three notebooks).
# Deterministic segment-lattice sequence decoder (see stride.py + writeup_shreygandhi.md):
# beam decode of the structural surface U=TVT+Z as piecewise-linear segments with a
# heavy-tailed Cauchy GR likelihood (autocorrelation-corrected weight lik_w=0.1) and a
# persistence prior. Decorrelated from the PF chain (err-corr 0.46-0.47).
# Validated (tuned seed7 / confirmed disjoint seed11, deployment-equivalent chain+projection):
#   w=0.20: seed7 9.8885 -> 9.4844 (-0.40) | seed11 8.4779 -> 8.2290 (-0.25)
# Runtime: ~0.3 s/well, deterministic (adds no seed variance).
import json, io, ast, shutil

STRIDE_DEFS = '''
# --- STRIDE channel (patch26): segment-lattice sequence decoder, see stride.py ---
from numba import njit as _s3njit

@_s3njit(cache=True, nogil=True)
def _stride_decode(md, z, gr, gg, gmin, gstep, u0, s0, gam, rates, bnds,
                   K, sig_pers, jump_pen, lik_w):
    n_seg = len(bnds) - 1; R = len(rates); ng = len(gg)
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
        nc = 0
        for b in range(kb):
            for ri in range(R):
                r = rates[ri]; ll = 0.0; ub = beam_u[b]
                for i in range(i0, i1):
                    tvt = ub + r * (md[i] - md0) - z[i]
                    gi = int((tvt - gmin) / gstep)
                    if gi < 0: gi = 0
                    elif gi >= ng: gi = ng - 1
                    d = (gr[i] - gg[gi]) / gam
                    ll += -np.log(1.0 + d * d)
                ds = (r - beam_s[b]) / sig_pers
                pen = ds * ds
                if pen > jump_pen: pen = jump_pen
                cand_sc[nc] = beam_sc[b] + lik_w * ll - pen
                cand_u[nc] = beam_u[b] + r * (md[i1 - 1] - md0)
                cand_par[nc] = b; cand_r[nc] = ri; nc += 1
        order = np.argsort(cand_sc[:nc])[::-1]
        nk = 0
        for oi in range(nc):
            c = order[oi]
            key = np.int64(np.round(cand_u[c] * 2.0)) * 1000 + cand_r[c]
            dup = False
            for j in range(nk):
                if keys[j] == key:
                    dup = True; break
            if dup: continue
            keys[nk] = key
            new_u[nk] = cand_u[c]; new_s[nk] = rates[cand_r[c]]; new_sc[nk] = cand_sc[c]
            p = cand_par[c]
            for s2 in range(seg):
                new_rates[nk, s2] = beam_rates[p, s2]
            new_rates[nk, seg] = cand_r[c]
            nk += 1
            if nk >= K: break
        kb = nk
        beam_u[:kb] = new_u[:kb]; beam_s[:kb] = new_s[:kb]; beam_sc[:kb] = new_sc[:kb]
        beam_rates[:kb] = new_rates[:kb]
    return beam_rates[:kb], beam_sc[:kb]

@_s3njit(cache=True, nogil=True)
def _stride_paths(md, bnds, u0, rates, beam_rates):
    kb, n_seg = beam_rates.shape
    paths = np.empty((kb, len(md)))
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

def _stride_track(hw, tw, seg_len=200.0, K=96, sig_pers=0.012, jump_pen=25.0, lik_w=0.1):
    """Eval-zone TVT from a joint beam decode of the surface (deterministic)."""
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) < 5 or len(kn) < 10 or len(tw_tvt) < 20:
        return None
    tw_gr = _s2_grcal(tw_tvt, tw_gr, kn['TVT_input'].values, kn['GR'].values)
    gmin = tw_tvt[0]
    gg = np.interp(np.arange(gmin, tw_tvt[-1] + 0.5, 0.5), tw_tvt, tw_gr)
    _l = kn.iloc[-1]
    u0 = float(_l['TVT_input']) + float(_l['Z'])
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    s0 = float(np.clip(np.median((dt + dz)[m] / dm[m]) if m.sum() >= 3 else 0.0, -0.06, 0.06))
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gam = float(np.clip(np.nanmedian(np.abs(kn['GR'].values.astype(float) - tw_at_k)), 5.0, 40.0))
    md_v = ev['MD'].values.astype(float); z_v = ev['Z'].values.astype(float)
    gr_v = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index]
    bnds = [0]; cur = md_v[0]
    for i in range(len(md_v)):
        if md_v[i] >= cur + seg_len:
            bnds.append(i); cur = md_v[i]
    if bnds[-1] != len(md_v):
        bnds.append(len(md_v))
    bnds = np.array(bnds, dtype=np.int64)
    rates = np.arange(-0.06, 0.06 + 0.001, 0.002)
    br, sc = _stride_decode(md_v, z_v, gr_v, gg, gmin, 0.5, u0, s0, gam, rates, bnds,
                            K, sig_pers, jump_pen, lik_w)
    paths = _stride_paths(md_v, bnds, u0, rates, br)
    order = np.argsort(sc)[::-1][:32]
    s = (sc[order] - sc[order][0]) / max(len(z_v), 1)
    w = np.exp(s / 0.02); w /= w.sum()
    return (w[:, None] * paths[order]).sum(0) - z_v

rows = []'''

STRIDE_BLEND = '''    # (T3s) STRIDE blend (patch26): decorrelated sequence decoder (err-corr ~0.46 vs the
    # PF chain). Validated on 2x150 disjoint wells at the deployed chain+projection level:
    # w=0.20 -> 9.8885->9.4844 (seed7) / 8.4779->8.2290 (seed11). Deterministic, ~0.3s/well.
    try:
        _st = _stride_track(hw_te, tw_ref)
        if _st is not None and len(_st) == len(_ei2) and np.all(np.isfinite(_st)):
            tvt_selector[_ei2] = 0.8 * tvt_selector[_ei2] + 0.2 * _st
            print('  STRIDE blend OK (w=0.20)')
    except Exception as _e:
        print(f'  STRIDE skipped: {_e}')
'''

def apply(text, old, new, tag, where):
    n = text.count(old)
    assert n == 1, f"{where}: '{tag}' matched {n} times"
    return text.replace(old, new)

TARGETS = [
    ("public-score-rogii-lb-7-159.ipynb", "public-score-rogii-lb-7-159.BACKUP26.ipynb",
     "    # T4 hold-gate removed:"),
    ("rogii-geology-aware-ensembling-lb-7-129.ipynb", "rogii-geology-aware-ensembling-lb-7-129.BACKUP7.ipynb",
     "    # T4 hold-gate removed:"),
    ("target-free-tvt-geosteering-7-016.ipynb", "target-free-tvt-geosteering-7-016.BACKUP3.ipynb",
     "    ws = sample[sample['well'] == wid]"),
]
for path, bak, blend_anchor in TARGETS:
    shutil.copy(path, bak)
    nb = json.load(io.open(path, encoding='utf-8'))
    done = 0
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        if '_s2_imp = _s2Dense(train_wells' in s and 'S1-affine blend OK' in s:
            s = apply(s, '\nrows = []', '\n' + STRIDE_DEFS, 'stride defs', path)
            s = apply(s, blend_anchor, STRIDE_BLEND + blend_anchor, 'stride blend', path)
            ast.parse(s)
            c['source'] = s.splitlines(keepends=True)
            done += 1
    assert done == 1, (path, done)
    json.dump(nb, io.open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"{path}: STRIDE wired (backup {bak})")

# regen 7-159 mirror + full verify
nb = json.load(io.open('public-score-rogii-lb-7-159.ipynb', encoding='utf-8'))
out = io.open('public-score-rogii-lb-7-159.py', 'w', encoding='utf-8'); out.write('#!/usr/bin/env python3\n')
for i, cc in enumerate(nb['cells']):
    t = ''.join(cc['source'])
    if cc['cell_type'] == 'markdown':
        [out.write('# ' + ln + '\n') for ln in t.splitlines()]; continue
    out.write(f"\n# ===== CELL {i} id={cc.get('id','')} =====\n" + t + ('' if t.endswith('\n') else '\n'))
out.close(); ast.parse(io.open('public-score-rogii-lb-7-159.py', encoding='utf-8').read())
for P, _, _ in TARGETS:
    nb = json.load(io.open(P, encoding='utf-8'))
    a = '\n'.join(''.join(c['source']) for c in nb['cells'])
    for c in nb['cells']:
        s = ''.join(c['source'])
        if c['cell_type'] == 'code' and s.strip():
            ast.parse(s)
    assert 'STRIDE blend OK' in a and 'def _stride_track' in a
    print(P, 'verify OK')
print('done')
