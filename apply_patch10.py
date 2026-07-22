"""Patch 10: GR coverage / reliability features. Horizontal GR is ~28% NaN
(0-73% per well); we now fill it for the trackers, so flag WHERE it was
interpolated so the GBM can down-weight GR-derived signals in sparse regions.
Adds gr_is_interp, gr_ev_valid_frac, gr_gap21, gr_gap101 to add_derived_features
(build_well, CELL 30) via a new gr_isna_ev arg. Idempotent."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'b515d5c6')
src = ''.join(c['source'])

if 'gr coverage / reliability' in src:
    print('Patch 10 already applied; aborting.')
    sys.exit(0)

# 1) extend signature (add optional gr_isna_ev)
sig_old = 'last_tvt, a_cal, b_cal, md_since, known_len, nh):'
sig_new = 'last_tvt, a_cal, b_cal, md_since, known_len, nh, gr_isna_ev=None):'
assert src.count(sig_old) == 1
src = src.replace(sig_old, sig_new)

# 2) append the coverage block after the v3 batch (last derived line)
v3_end = '    feats["gr_trend101"] = _f(np.gradient(hs.rolling(101, center=True, min_periods=1).mean().values))'
block = v3_end + '''
    # ===== gr coverage / reliability (horizontal GR is ~28% NaN; flag interpolated regions) =====
    if gr_isna_ev is not None:
        miss = np.asarray(gr_isna_ev, np.float64)
        if len(miss) == nh:
            mser = pd.Series(miss)
            feats["gr_is_interp"] = _f(miss)                                           # per-point: GR was interpolated
            feats["gr_ev_valid_frac"] = _f(np.full(nh, 1.0 - float(miss.mean())))      # well-level eval GR coverage
            feats["gr_gap21"] = _f(mser.rolling(21, center=True, min_periods=1).mean().values)    # local missing density
            feats["gr_gap101"] = _f(mser.rolling(101, center=True, min_periods=1).mean().values)'''
assert src.count(v3_end) == 1
src = src.replace(v3_end, block)

# 3) pass the raw eval-zone GR NaN mask in the call
call_old = 'last_tvt, a_cal, b_cal, md_since, len(kn), nh)'
call_new = 'last_tvt, a_cal, b_cal, md_since, len(kn), nh, ev["GR"].isna().to_numpy())'
assert src.count(call_old) == 1
src = src.replace(call_old, call_new)

ast.parse(src)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('added gr coverage features (gr_is_interp, gr_ev_valid_frac, gr_gap21, gr_gap101)')
