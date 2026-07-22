"""Assemble a standalone LGB (re)training notebook (train_lgb.ipynb).

Drops the GRU path entirely and re-trains the fleongg LGB models offline, saving
them as  lgb0.pkl, lgb1.pkl, ... + features.json  — the exact format the
submission notebook's fleongg _find_models()/INFERENCE branch already loads
(meta_test = mean of the lgb predictions). Attach the output as a dataset and the
submission runs fast INFERENCE on YOUR freshly-trained LGBs. No GRU bundle
attached => GRU auto-disables.

Runs ONLY the feature pipeline + LGB training (no sub_1/sub_2/fleongg-main/gold),
so it fits the time/RAM budget.
"""
import json, io

SRC = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(SRC, encoding='utf-8'))
def cell_src(cid):
    return ''.join(next(c for c in nb['cells'] if c.get('id') == cid)['source'])

FEATURE_CELLS = ['44e50dd0',   # imports, CFG, load_well, rmse
                 'c3ab97ce',   # PF/beam kernels
                 '83876170',   # lik_pf
                 'c9bba3da',   # imputers, helpers
                 'b515d5c6',   # build_well/init_imputers/build_likpf/build_features/add_likpf_features
                 'cc1a056a']   # _device, lgb_configs, cb_configs

HEADER = '''# Standalone LGB (re)trainer for ROGII — drops GRU, retrains the fleongg LGBs.
# Run on a Kaggle notebook with the competition data attached (GPU optional but
# faster for LightGBM). Output: lgb0.pkl, lgb1.pkl, ... + features.json.
# Publish that output as a dataset and attach it to the submission notebook; its
# fleongg INFERENCE branch will load & average these LGBs. Do NOT attach a
# gru_bundle (GRU stays off). Knobs:
#   ROGII_LGB_MAXWELLS=0   # train on fewer wells if RAM/time tight (0 = all)
import os
os.environ.setdefault("SHOW_FIGS", "0")
'''

DRIVER = '''# ===== re-train LGBs on the current features and save for fast inference =====
import gc, json, joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
test_wids  = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"test").glob("*__horizontal_well.csv"))
_MAX = int(os.environ.get("ROGII_LGB_MAXWELLS", "0"))
if _MAX:
    train_wids = train_wids[:_MAX]
if not train_wids:
    raise SystemExit(f"[data] No wells found under {CFG.DATA/'train'} — set os.environ['ROGII_DATA'] "
                     f"to the folder that contains train/ and test/ (current DATA={CFG.DATA})")
print(f"training wells: {len(train_wids)} | example test: {len(test_wids)}", flush=True)

def _feat_cache_name(_n):
    _g = "" if os.environ.get("ROGII_GRFILL", "1") == "1" else "_g0"
    _c = os.environ.get("ROGII_GRCAL", "off").lower()
    _c = "" if _c in ("off", "0") else f"_c{_c}"   # S1 mode changes likpf features -> separate cache key
    return f"train_features_v6_f{os.environ.get('ROGII_FEATS','1')}{_g}{_c}_w{_n}.parquet"

def load_or_build_train_features(train_wids):
    # ROGII_FEATCACHE: auto (use cache if present) | rebuild (force) | off (build, don't save)
    import glob as _g
    name = _feat_cache_name(len(train_wids)); mode = os.environ.get("ROGII_FEATCACHE", "auto")
    if mode != "rebuild":
        for _p in _g.glob(f"/kaggle/input/**/{name}", recursive=True) + [str(CFG.OUT / name)]:
            if os.path.exists(_p):
                print(f"[cache] loading train features from {_p}", flush=True)
                return pd.read_parquet(_p)
    print("[cache] building train features (slow ~2-3h)...", flush=True)
    _df = add_likpf_features(build_features(train_wids, "train", is_train=True), build_likpf(train_wids, "train"))
    if mode != "off":
        try:
            _df.to_parquet(CFG.OUT / name, index=False)
            print(f"[cache] saved -> {CFG.OUT / name}  (publish OUT as a dataset to reuse next runs)", flush=True)
        except Exception as _e:
            print("[cache] save skipped:", _e, flush=True)
    return _df

init_imputers(train_wids)   # cheap KDTrees; needed for test features regardless of cache
train_df = load_or_build_train_features(train_wids)
gc.collect()
test_df  = add_likpf_features(build_features(test_wids, "test", is_train=False),   build_likpf(test_wids, "test"))
gc.collect()

# feature list EXACTLY as the submission notebook's training path defines it
feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]
print(f"features: {len(feats)} | train rows: {len(train_df)}", flush=True)

dev, _ = _device()
X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values
cv = GroupKFold(CFG.n_splits)
dev = _resolve_lgb_device(X, y, dev)   # gpu(OpenCL) -> cuda -> cpu
cfgs = lgb_configs(dev)

# 1) GroupKFold OOF for the score, and capture best_iteration per config
oof = np.zeros((len(cfgs), len(train_df)))
best_iters = []
for ci, params in enumerate(cfgs):
    iters = []
    for tr, va in cv.split(X, y, groups=g):
        m = LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        oof[ci, va] = m.predict(X[va], num_iteration=m.best_iteration_)
        iters.append(int(m.best_iteration_ or params.get("n_estimators", 1000)))
        del m; gc.collect()
    nb_iter = int(np.mean(iters))
    best_iters.append(max(50, nb_iter))
    print(f"lgb{ci} OOF RMSE={rmse(y, oof[ci]):.4f}  (avg best_iter={nb_iter})", flush=True)
print(f"LGB ensemble (mean) OOF RMSE={rmse(y, oof.mean(0)):.4f}", flush=True)

# 2) refit each config on ALL train rows with the CV-chosen #trees, save for inference
outdir = CFG.OUT
for ci, params in enumerate(cfgs):
    p = dict(params); p["n_estimators"] = best_iters[ci]
    m = LGBMRegressor(**p)
    m.fit(X, y)
    joblib.dump(m, outdir / f"lgb{ci}.pkl")
    print(f"saved lgb{ci}.pkl (n_estimators={best_iters[ci]})", flush=True)
with open(outdir / "features.json", "w") as f:
    json.dump(feats, f)
print("saved features.json ->", outdir, flush=True)
print("DONE. Publish lgb*.pkl + features.json as a dataset; attach to the submission notebook.", flush=True)
'''

def mkcell(src, cid):
    return {'cell_type': 'code', 'id': cid, 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src.splitlines(keepends=True)}

cells = [mkcell(HEADER, 'hdrl0001')]
for cid in FEATURE_CELLS:
    cells.append(mkcell(cell_src(cid), 'featl_' + cid))
cells.append(mkcell(DRIVER, 'drvl0001'))

out_nb = dict(nb); out_nb['cells'] = cells
with io.open('train_lgb.py', 'w', encoding='utf-8') as _f:
    _f.write('#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# Auto-generated by build_lgb_notebook.py. Run on Kaggle/local with the data (and v5 feature-cache) attached.\n')
    for _c in cells:
        _s = ''.join(_c['source'])
        _f.write('\n# ' + '=' * 70 + '\n' + _s + ('' if _s.endswith('\n') else '\n'))
print('wrote train_lgb.py with', len(cells), 'cells')
