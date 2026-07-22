"""Patch 5: make the fleongg INFERENCE branch full-stack aware.
If the attached model dir has stack_meta.json + ridge.pkl, apply LGB+CB base
models -> Ridge meta (the stronger stack). Otherwise fall back to the LGB mean.
The GRU-bundle blend that follows still applies to meta_test. Idempotent.
"""
import json, io, sys

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == '209071f6')
src = ''.join(c['source'])

if 'FULL-STACK inference' in src:
    print('Patch 5 already applied; aborting.')
    sys.exit(0)

anchor = (
    '        models = [joblib.load(p) for p in sorted(models_dir.glob("lgb*.pkl"))]\n'
    '        for c in feats:\n'
    '            if c not in test_df.columns: test_df[c] = 0.0\n'
    '        Xt = test_df[feats].values.astype(np.float32)\n'
    '        meta_test = np.mean([m.predict(Xt) for m in models], axis=0)'
)
assert src.count(anchor) == 1, ('anchor count', src.count(anchor))

replacement = (
    '        for c in feats:\n'
    '            if c not in test_df.columns: test_df[c] = 0.0\n'
    '        Xt = test_df[feats].values.astype(np.float32)\n'
    '        _stack_meta = models_dir / "stack_meta.json"; _ridge_pkl = models_dir / "ridge.pkl"\n'
    '        if _stack_meta.exists() and _ridge_pkl.exists():\n'
    '            # FULL-STACK inference: LGB + CatBoost base models -> Ridge meta\n'
    '            _meta = json.load(open(_stack_meta)); _bn = _meta["base_names"]\n'
    '            _pred = {}\n'
    '            for _p in sorted(models_dir.glob("lgb*.pkl")):\n'
    '                _pred[_p.stem] = joblib.load(_p).predict(Xt)\n'
    '            for _p in sorted(models_dir.glob("cb*.cbm")):\n'
    '                from catboost import CatBoostRegressor as _CB\n'
    '                _m = _CB(); _m.load_model(str(_p)); _pred[_p.stem] = _m.predict(Xt)\n'
    '            _cols = np.column_stack([_pred[_n] for _n in _bn if _n in _pred])\n'
    '            meta_test = joblib.load(_ridge_pkl).predict(_cols)\n'
    '            print(f"FULL-STACK inference: {[_n for _n in _bn if _n in _pred]} + ridge", flush=True)\n'
    '        else:\n'
    '            models = [joblib.load(p) for p in sorted(models_dir.glob("lgb*.pkl"))]\n'
    '            meta_test = np.mean([m.predict(Xt) for m in models], axis=0)\n'
    '            print("LGB-average inference", flush=True)'
)

src = src.replace(anchor, replacement)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched INFERENCE branch to be full-stack aware')
