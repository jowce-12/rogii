"""Patch 8: v3 rolling batch (sequential info not yet featurized) into
add_derived_features (build_well, CELL 30). Reuses already-computed series.
Idempotent (keys off the v3 marker)."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'b515d5c6')
src = ''.join(c['source'])

if 'derived v3 batch' in src:
    print('Patch 8 already applied; aborting.')
    sys.exit(0)

anchor = '    feats["extrap_vs_twrange"] = _f(ms / tw_rng)'
assert src.count(anchor) == 1, ('anchor', src.count(anchor))

V3 = anchor + '''
    # ===== derived v3 batch: sequential signals not yet rolled (drift/steadiness) =====
    pfs = pd.Series(pf64)
    feats["pf_std21"] = _f(pfs.rolling(21, center=True, min_periods=1).std().fillna(0).values)   # PF jitter
    feats["pf_std51"] = _f(pfs.rolling(51, center=True, min_periods=1).std().fillna(0).values)
    dps = pd.Series(dzc)
    feats["dip_mean21"] = _f(dps.rolling(21, center=True, min_periods=1).mean().values)           # trajectory dip
    feats["dip_std51"] = _f(dps.rolling(51, center=True, min_periods=1).std().fillna(0).values)   # dip steadiness
    feats["grm201"] = _f(hs.rolling(201, center=True, min_periods=1).mean().values)               # slow GR trend
    feats["mq_roll201"] = _f(s.rolling(201, center=True, min_periods=1).mean().values)            # sustained mismatch
    feats["sres_cum"] = _f(ss.expanding().mean().values)                                          # systematic bias dir
    feats["gr_trend101"] = _f(np.gradient(hs.rolling(101, center=True, min_periods=1).mean().values))'''

src = src.replace(anchor, V3)
ast.parse(src)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('added v3 rolling batch (8 feats)')
