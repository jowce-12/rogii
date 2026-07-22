"""Patch 3 — Option C: offline-trained GRU bundle, blended in the INFERENCE path.
  1. Replace the gru00001 cell with the full gru_offline module + helpers +
     a train_gru_oof shim (keeps the train-from-scratch meta-stack path working).
  2. Insert an offline-training cell (gated by ROGII_GRU_TRAIN=1) after main().
  3. Blend the loaded GRU bundle into meta_test in the INFERENCE branch.
Idempotent.
"""
import json, io, sys

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
joined = '\n'.join(''.join(c['source']) for c in nb['cells'])
if 'def predict_bundle' in joined:
    print('Option C already applied; aborting.')
    sys.exit(0)

# --- module body (strip the __main__ smoke test) ---
mod = io.open('gru_offline.py', encoding='utf-8').read()
body = mod.split('# ----------------------------------------------------------------------------- smoke test')[0].rstrip() + '\n'

SHIM = '''

# ----------------------------------------------------------------------------- notebook glue
def _torch_ok():
    try:
        import torch  # noqa
        return True
    except Exception:
        return False

def _find_gru_bundle():
    import glob as _g
    cands = _g.glob('/kaggle/input/**/gru_bundle.pt', recursive=True)
    try:
        wk = str(getattr(CFG, 'OUT', '.')) + '/gru_bundle.pt'
        if os.path.exists(wk):
            cands.append(wk)
    except Exception:
        pass
    cands = [c for c in cands if os.path.exists(c)]
    return cands[0] if cands else None

def train_gru_oof(train_df, test_df, features, cv, seed=42):
    """Shim for the train-from-scratch meta-stack path (train_stack, CELL 31).
    No-op unless torch+CUDA (or ROGII_GRU_FORCE_CPU=1); ROGII_GRU=off disables."""
    mode = os.environ.get('ROGII_GRU', 'add').strip().lower()
    if mode == 'off' or not _torch_ok():
        print(f'  GRU disabled (ROGII_GRU={mode}, torch_ok={_torch_ok()})')
        return None
    import torch
    if not torch.cuda.is_available() and os.environ.get('ROGII_GRU_FORCE_CPU', '0') != '1':
        print('  GRU skipped: no CUDA (set ROGII_GRU_FORCE_CPU=1 to run on CPU)')
        return None
    cfg = dict(DEFAULT_CFG)
    cfg['epochs'] = int(os.environ.get('ROGII_GRU_EPOCHS', cfg['epochs']))
    oof, states, scaler, rmse = kfold_oof(train_df, features, cv, cfg, seed=seed)
    bundle = dict(states=states, cfg=cfg, mu=scaler[0], sd=scaler[1], features=list(features))
    test = predict_bundle(bundle, test_df)
    return oof.astype(np.float32), test.astype(np.float32)
'''

# 1) replace gru00001 content
c = next(c for c in nb['cells'] if c.get('id') == 'gru00001')
c['source'] = (body + SHIM).splitlines(keepends=True)
print('replaced gru00001 with gru_offline module + shim')

# 2) insert offline-training cell after main() cell (id 209071f6)
OFFLINE = '''# ===== Option C: offline GRU training (run ONCE on a GPU notebook) =====
# Builds train features, searches GRU hyperparameters, trains a K-fold ensemble,
# tunes the blend weight against a quick LGB OOF, and saves gru_bundle.pt to OUT.
# Gated: runs only when ROGII_GRU_TRAIN=1. Afterwards, save OUT/gru_bundle.pt as a
# Kaggle dataset and attach it; normal INFERENCE runs auto-load and blend it.
if os.environ.get('ROGII_GRU_TRAIN', '0') == '1':
    from sklearn.model_selection import GroupKFold as _GKF
    from lightgbm import LGBMRegressor as _LGB, early_stopping as _es, log_evaluation as _le
    _gtr = sorted(p.stem.replace('__horizontal_well', '') for p in (CFG.DATA/'train').glob('*__horizontal_well.csv'))
    _gte = sorted(p.stem.replace('__horizontal_well', '') for p in (CFG.DATA/'test').glob('*__horizontal_well.csv'))
    print(f'[GRU-train] building features: {len(_gtr)} train / {len(_gte)} test wells', flush=True)
    init_imputers(_gtr)
    _trdf = add_likpf_features(build_features(_gtr, 'train', is_train=True), build_likpf(_gtr, 'train'))
    _tedf = add_likpf_features(build_features(_gte, 'test', is_train=False), build_likpf(_gte, 'test'))
    _feats = [c for c in _trdf.columns if c not in {'well', 'id', 'target'}
              and not (c.startswith('likpf_scale_') or c == 'likpf_mean') and c in _tedf.columns]
    _cv = _GKF(CFG.n_splits)
    _X = _trdf[_feats].values.astype(np.float32); _y = _trdf['target'].values.astype(np.float32); _gg = _trdf['well'].values
    _dev, _ = _device()
    _lgb_oof = np.zeros(len(_trdf))
    for _tr, _va in _cv.split(_X, _y, groups=_gg):
        _m = _LGB(**lgb_configs(_dev)[0])
        _m.fit(_X[_tr], _y[_tr], eval_set=[(_X[_va], _y[_va])], eval_metric='rmse',
               callbacks=[_es(200, verbose=False), _le(0)])
        _lgb_oof[_va] = _m.predict(_X[_va], num_iteration=_m.best_iteration_)
    print(f'[GRU-train] LGB base OOF RMSE={rmse(_y, _lgb_oof):.4f}', flush=True)
    fit_and_save(_trdf, _tedf, _feats, _cv, CFG.OUT/'gru_bundle.pt', base_oof=_lgb_oof,
                 do_search=(os.environ.get('ROGII_GRU_SEARCH', '1') == '1'), seed=CFG.seed, verbose=True)
    print('[GRU-train] done -> save OUT/gru_bundle.pt as a dataset and attach it for inference', flush=True)
else:
    print('Option C GRU offline training skipped (set ROGII_GRU_TRAIN=1 on a GPU notebook to build gru_bundle.pt)')
'''
idx = next(i for i, c in enumerate(nb['cells']) if c.get('id') == '209071f6')
nb['cells'].insert(idx + 1, {'cell_type': 'code', 'id': 'gruoff01', 'metadata': {},
                             'execution_count': None, 'outputs': [], 'source': OFFLINE.splitlines(keepends=True)})
print('inserted offline-training cell after index', idx)

# 3) blend bundle into meta_test in the INFERENCE branch
c33 = next(c for c in nb['cells'] if c.get('id') == '209071f6')
src = ''.join(c33['source'])
anchor = '        meta_test = np.mean([m.predict(Xt) for m in models], axis=0)'
assert src.count(anchor) == 1
blend = anchor + '''
        # option C: blend an offline-trained GRU bundle if one is attached
        try:
            _gp = _find_gru_bundle()
            if _gp is not None:
                _b = load_bundle(_gp)
                for _c in _b['features']:
                    if _c not in test_df.columns:
                        test_df[_c] = 0.0
                _gru_test = predict_bundle(_b, test_df)
                _w = float(os.environ.get('ROGII_GRU_W', _b.get('w_blend', 0.15)))
                meta_test = (1.0 - _w) * meta_test + _w * _gru_test
                print(f'GRU bundle blended (w={_w:.3f}) from {_gp}', flush=True)
            else:
                print('GRU bundle not found; LGB-only inference', flush=True)
        except Exception as _e:
            print('GRU bundle blend skipped:', _e, flush=True)'''
src = src.replace(anchor, blend)
c33['source'] = src.splitlines(keepends=True)
print('wired GRU bundle blend into INFERENCE branch')

json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', P)
