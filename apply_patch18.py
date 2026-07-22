"""Patch 18: A6 (beam 0.5ft typewell resample) + A8 (target winsorize 90ft +
final delta clip 110ft).
- A6 -> notebook cell c3ab97ce beam_search (fleongg path; propagates to
  train_stack/quick/lgb/gru). _beam_jit moves +-2 GRID indices per row, so dip
  limits scale with the native grid step (0.1-1.0ft across 120 wells); resampling
  to the dominant 0.5ft grid normalizes dynamics with zero information loss
  (oversampled typewells are master-curve duplicates — probe-verified).
- A8 winsorize -> build_stack driver: y_fit trains on clip(target,+-90) while all
  printed RMSEs keep the RAW y (comparability preserved).
- A8 clip -> notebook make_prediction: final fleongg delta clipped to +-110ft
  (train max|target| = 98.9ft -> 0 rows affected in-sample; pure insurance).
- Cache bump v5 -> v6 in all 4 builders (beam features change).
Idempotent."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))

# ---------- A6: beam resample (cell c3ab97ce) ----------
c = next(x for x in nb['cells'] if x.get('id') == 'c3ab97ce')
s = ''.join(c['source'])
if 'A6: resample typewell' in s:
    print('A6 already applied')
else:
    o = '''def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)'''
    n = '''def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    # A6: resample typewell to a uniform 0.5ft grid. _beam_jit moves +-2 grid
    # INDICES per row, so dip limits/move costs scale with the native grid step
    # (0.1-1.0ft across wells; 0.1ft wells were effectively frozen). Oversampled
    # typewells are master-curve duplicates, so 0.5ft resampling loses nothing.
    _t0 = float(np.min(tw_tvt)); _t1 = float(np.max(tw_tvt))
    if _t1 - _t0 > 1.0:
        _tg = np.arange(_t0, _t1 + 0.25, 0.5)
        tw_gr = np.interp(_tg, np.asarray(tw_tvt, np.float64), np.asarray(tw_gr, np.float64))
        tw_tvt = _tg
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, np.asarray(tw_gr, np.float64), si, bs, float(mc), float(es))].astype(np.float32)'''
    assert s.count(o) == 1, ('beam anchor', s.count(o))
    s = s.replace(o, n)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print('A6 applied to c3ab97ce')

# ---------- A8 clip: make_prediction (cell e9fe10a5) ----------
c2 = next(x for x in nb['cells'] if x.get('id') == 'e9fe10a5')
p = ''.join(c2['source'])
if 'A8: hard-stop' in p:
    print('A8 clip already applied')
else:
    o = '''    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    pred = last + delta'''
    n = '''    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    delta = np.clip(delta, -110.0, 110.0)   # A8: hard-stop — train max|target|=98.9ft over ~1M rows, so this is a free runaway guard
    pred = last + delta'''
    assert p.count(o) == 1, ('mp anchor', p.count(o))
    p = p.replace(o, n)
    ast.parse(p)
    c2['source'] = p.splitlines(keepends=True)
    print('A8 clip applied to make_prediction')

json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------- A8 winsorize: build_stack driver ----------
f = 'build_stack_notebook.py'
t = io.open(f, encoding='utf-8').read()
if 'winsorized at' in t:
    print('A8 winsorize already applied')
else:
    o = '''_resid = os.environ.get("ROGII_RESID", "1") == "1"
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y - anchor'''
    n = '''# A8: winsorize the TRAINING target at +-90ft (train p99.9=91.2, max 98.9 over ~1M
# rows -> ~0.1% of rows). All printed RMSEs still use the RAW y (comparable history).
_nw = int((np.abs(y) > 90).sum())
y_w = np.clip(y, -90.0, 90.0)
print(f"[winz] training target winsorized at +-90ft ({_nw} rows affected)", flush=True)
_resid = os.environ.get("ROGII_RESID", "1") == "1"
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y_w - anchor'''
    assert t.count(o) == 1, ('winz anchor', t.count(o))
    t = t.replace(o, n)
    io.open(f, 'w', encoding='utf-8').write(t)
    print('A8 winsorize applied to build_stack driver')

# ---------- cache bump v5 -> v6 (beam features changed) ----------
for bf in ['build_stack_notebook.py', 'build_lgb_notebook.py', 'build_train_notebook.py', 'build_quick_notebook.py']:
    bt = io.open(bf, encoding='utf-8').read()
    if 'train_features_v6_f' in bt:
        print(bf, 'cache already v6'); continue
    assert bt.count('train_features_v5_f') == 1, (bf, bt.count('train_features_v5_f'))
    io.open(bf, 'w', encoding='utf-8').write(bt.replace('train_features_v5_f', 'train_features_v6_f'))
    print(bf, '-> cache v6')
