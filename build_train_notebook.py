"""Assemble a standalone, low-memory GRU-training notebook (train_gru.ipynb).
Pulls only the feature pipeline cells from the main notebook + the GRU module +
a memory-hardened training driver. Runs NONE of sub_1/sub_2/fleongg/gold, so RAM
stays small. Run this on a GPU Kaggle notebook with ROGII_GRU_TRAIN data attached
to produce gru_bundle.pt, then publish that as a dataset for the submission run.
"""
import json, io

SRC = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(SRC, encoding='utf-8'))
def cell_src(cid):
    return ''.join(next(c for c in nb['cells'] if c.get('id') == cid)['source'])

# feature-pipeline cells (fleongg section), in dependency order
FEATURE_CELLS = ['44e50dd0',  # imports, CFG(DATA/OUT/n_splits/seed), load_well, rmse
                 'c3ab97ce',  # PF/beam kernels, _grid, beam_search, run_pf_ancc/z, multi_scale_ncc
                 '83876170',  # _pf_lik_allseeds, lik_pf, JIT warmup
                 'c9bba3da',  # imputers, robust_slope, affine_cal, seg_b_well, FormationPlaneKNN, DenseANCCImputer
                 'cc1a056a']  # _device, lgb_configs, cb_configs, train_stack (need _device + lgb_configs)
# build_well / init_imputers / build_likpf / build_features / add_likpf_features
FEATURE_CELLS.insert(4, 'b515d5c6')   # CELL 30, must come before cc1a056a

# GRU module (updated, memory-hardened) — embed the .py body minus the smoke test
mod = io.open('gru_offline.py', encoding='utf-8').read()
gru_body = mod.split('# ----------------------------------------------------------------------------- smoke test')[0].rstrip() + '\n'

HEADER = '''# Standalone GRU bundle trainer for ROGII (Option C).
# Run on a GPU Kaggle notebook with the competition data attached.
# Output: gru_bundle.pt  -> publish as a dataset, attach to the submission notebook.
#
# Defaults are tuned to FIT the time budget (search OFF, 3 folds, 20 epochs,
# validate every 2 epochs, overlap 0.25). Feature building alone is ~2-3h.
# Speed / memory knobs (set before running if needed):
#   ROGII_GRU_MAXWELLS=500   # train on fewer wells (biggest time cut)
#   ROGII_GRU_FOLDS=3        # GRU ensemble folds (2 = faster, 5 = stronger)
#   ROGII_GRU_EPOCHS=20      # max epochs per fold (early-stops on patience)
#   ROGII_GRU_BATCH=8        # raise on GPU for speed if memory allows
#   ROGII_GRU_CHUNK=1000     # sequence window length
#   ROGII_GRU_OVERLAP=0.25   # window overlap (0 = fastest)
#   ROGII_GRU_EVAL_EVERY=2   # validate every N epochs
#   ROGII_GRU_SEARCH=0       # 1 = hyperparameter search (SLOW: each trial is a full K-fold)
import os
os.environ.setdefault("SHOW_FIGS", "0")
'''

DRIVER = '''# ===== train + save the GRU bundle (memory-hardened) =====
import gc
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
test_wids  = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"test").glob("*__horizontal_well.csv"))
_MAX = int(os.environ.get("ROGII_GRU_MAXWELLS", "0"))   # 0 = all wells; set e.g. 300 if RAM is tight
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

feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]
# downcast float64 -> float32 to halve RAM
for _c in feats + ["target"]:
    if _c in train_df and train_df[_c].dtype == np.float64:
        train_df[_c] = train_df[_c].astype(np.float32)
    if _c in test_df and test_df[_c].dtype == np.float64:
        test_df[_c] = test_df[_c].astype(np.float32)
print(f"features: {len(feats)} | train rows: {len(train_df)}", flush=True)

# quick LGB base OOF for blend-weight tuning
dev, _ = _device()
X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values
cv = GroupKFold(CFG.n_splits)
dev = _resolve_lgb_device(X, y, dev)   # gpu(OpenCL) -> cuda -> cpu
lgb_oof = np.zeros(len(train_df))
for tr, va in cv.split(X, y, groups=g):
    m = LGBMRegressor(**lgb_configs(dev)[0])
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
          callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
    lgb_oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
    del m; gc.collect()
del X; gc.collect()
print(f"LGB base OOF RMSE={rmse(y, lgb_oof):.4f}", flush=True)

# GRU K-fold ensemble -> tune blend -> save bundle.
# Search defaults OFF (each trial is a full K-fold and blows the time budget).
# Use a GRU-specific (smaller) fold count for speed; blend tuning is per-row so
# a 3-fold GRU OOF is still comparable to the 5-fold LGB OOF above.
_gru_folds = int(os.environ.get("ROGII_GRU_FOLDS", "3"))
gru_cv = GroupKFold(_gru_folds)
fit_and_save(train_df, test_df, feats, gru_cv, CFG.OUT/"gru_bundle.pt", base_oof=lgb_oof,
             do_search=(os.environ.get("ROGII_GRU_SEARCH", "0") == "1"), seed=CFG.seed, verbose=True)
print("DONE -> gru_bundle.pt saved to", CFG.OUT, flush=True)
'''

def mkcell(src, cid):
    return {'cell_type': 'code', 'id': cid, 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src.splitlines(keepends=True)}

cells = [mkcell(HEADER, 'hdr00001')]
for cid in FEATURE_CELLS:
    cells.append(mkcell(cell_src(cid), 'feat_' + cid))
cells.append(mkcell(gru_body, 'grumod01'))
cells.append(mkcell(DRIVER, 'drv00001'))

out_nb = dict(nb)  # copy top-level metadata (kernelspec etc.)
out_nb['cells'] = cells
with io.open('train_gru.py', 'w', encoding='utf-8') as _f:
    _f.write('#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# Auto-generated by build_train_notebook.py. Run on Kaggle/local with the data (and v5 feature-cache) attached.\n')
    for _c in cells:
        _s = ''.join(_c['source'])
        _f.write('\n# ' + '=' * 70 + '\n' + _s + ('' if _s.endswith('\n') else '\n'))
print('wrote train_gru.py with', len(cells), 'cells')
