"""Assemble a standalone FULL-STACK (re)training notebook (train_stack.ipynb).

Trains LGB×3 + CatBoost×2 with GroupKFold OOF, fits a Ridge meta on the base OOF,
refits base models on all data, and saves:
    lgb0/1/2.pkl, cb0/1.cbm, ridge.pkl, stack_meta.json, features.json
The submission notebook's INFERENCE branch (patched) detects stack_meta.json +
ridge.pkl and applies the FULL stack (base preds -> ridge) instead of averaging
LGBs — the notebook's own stronger configuration (~9.69 vs ~9.86 mean-LGB).

Runs ONLY the feature pipeline + stack training (no sub_1/sub_2/gold). Heavy but
one-time and on the 12h session budget; the SUBMISSION only loads & predicts.
"""
import json, io

SRC = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(SRC, encoding='utf-8'))
def cell_src(cid):
    return ''.join(next(c for c in nb['cells'] if c.get('id') == cid)['source'])

FEATURE_CELLS = ['44e50dd0', 'c3ab97ce', '83876170', 'c9bba3da', 'b515d5c6', 'cc1a056a']

HEADER = '''# Standalone FULL-STACK (re)trainer for ROGII — LGB×3 + CatBoost×2 + Ridge meta.
# Run on a GPU Kaggle notebook with the competition data attached.
# Output: lgb*.pkl, cb*.cbm, ridge.pkl, stack_meta.json, features.json.
# Publish as a dataset; attach to the submission notebook (DETACH any old lgb
# dataset so only this one is found). The patched INFERENCE branch applies the
# full stack. Do NOT attach a gru_bundle (GRU stays off).
# Time/RAM knobs (offline, 12h session budget):
#   ROGII_STACK_FOLDS=3      # OOF folds (default 5; 3 = ~40% faster)
#   ROGII_STACK_CB=1         # 0 = skip CatBoost (LGB+Ridge only, much faster)
#   ROGII_STACK_MAXWELLS=0   # 0 = all wells; set e.g. 500 if time/RAM tight
import os
os.environ.setdefault("SHOW_FIGS", "0")
os.environ.setdefault("ROGII_GRCAL", "blend")   # S1 blend ON by default (validated)
'''

DRIVER = '''# ===== full-stack retrain: LGB×3 + CatBoost×2 + Ridge meta, save for inference =====
import gc, json, joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor

train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
test_wids  = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"test").glob("*__horizontal_well.csv"))
_MAX = int(os.environ.get("ROGII_STACK_MAXWELLS", "0"))
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
train_df = add_alias_metafeats(train_df, "train")   # A5 (join-only, no cache rebuild)
gc.collect()
test_df  = add_likpf_features(build_features(test_wids, "test", is_train=False),   build_likpf(test_wids, "test"))
test_df = add_alias_metafeats(test_df, "test")   # A5
gc.collect()

feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]
print(f"features: {len(feats)} | train rows: {len(train_df)}", flush=True)

# optional feature pruning: from a prior feature_importance.csv or an explicit drop list
_impf = os.environ.get("ROGII_IMP_FILE", "")
_drop = set(x for x in os.environ.get("ROGII_DROP_FEATS", "").split(",") if x)
if _impf:
    import glob as _g2
    _cand = [_impf] if os.path.exists(_impf) else _g2.glob(f"/kaggle/input/**/{os.path.basename(_impf)}", recursive=True)
    if _cand:
        _imp = pd.read_csv(_cand[0]); _topn = int(os.environ.get("ROGII_TOP_FEATS", "0")); _minimp = float(os.environ.get("ROGII_MIN_IMP", "0"))
        _k = _imp.sort_values("gain", ascending=False)
        if _topn: _k = _k.head(_topn)
        if _minimp: _k = _k[_k["gain"] >= _minimp]
        _keep = set(_k["feature"]); feats = [c for c in feats if c in _keep and c not in _drop]
        print(f"[prune] kept {len(feats)} feats via {_cand[0]} (top={_topn}, min_imp={_minimp})", flush=True)
    else:
        print(f"[prune] ROGII_IMP_FILE not found: {_impf}", flush=True)
elif _drop:
    feats = [c for c in feats if c not in _drop]
    print(f"[prune] dropped {len(_drop)} -> {len(feats)} feats remain", flush=True)

dev, _ = _device()
X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values
# residual re-anchoring (ROGII_RESID=0 default — measured worse on LGB valid): opt-in via ROGII_RESID=1;
# inference adds the anchor back (stack_meta.json carries the flag). RMSE prints stay in delta space.
# A8: winsorize the TRAINING target at +-90ft (train p99.9=91.2, max 98.9 over ~1M
# rows -> ~0.1% of rows). All printed RMSEs still use the RAW y (comparable history).
if os.environ.get("ROGII_WINZ", "1") == "1":
    _nw = int((np.abs(y) > 90).sum())
    y_w = np.clip(y, -90.0, 90.0)
    print(f"[winz] training target winsorized at +-90ft ({_nw} rows affected)", flush=True)
else:
    y_w = y
_resid = os.environ.get("ROGII_RESID", "0") == "1"   # default OFF: residual target hurt LGB valid scores (measured 2026-07-09)
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y_w - anchor
if _resid:
    print(f"[resid] target re-anchored to likpf_mean_d (target std {y.std():.2f} -> residual std {y_fit.std():.2f})", flush=True)
_folds = int(os.environ.get("ROGII_STACK_FOLDS", str(CFG.n_splits)))
cv = GroupKFold(_folds)
use_cb = os.environ.get("ROGII_STACK_CB", "1") == "1"
# GPU probes: fall back to CPU per library if the GPU is not usable (e.g. LightGBM needs OpenCL)
lgb_dev = _resolve_lgb_device(X, y, dev)   # gpu(OpenCL) -> cuda -> cpu
cb_dev = dev
if cb_dev == "gpu":
    try:
        CatBoostRegressor(iterations=2, task_type="GPU", verbose=0).fit(X[:200], y[:200])
    except Exception as _e:
        print(f"[gpu] CatBoost GPU unavailable -> CPU ({str(_e)[:60]})", flush=True); cb_dev = "cpu"
lgb_cfgs = lgb_configs(lgb_dev); cb_cfgs = cb_configs(cb_dev)

base_names = []; oof_cols = {}; full_iters = {}
imp_sum = np.zeros(len(feats))   # accumulated LGB gain importance

# --- LGB base models: GroupKFold OOF + mean best_iteration ---
for ci, params in enumerate(lgb_cfgs):
    name = f"lgb{ci}"; oof = np.zeros(len(train_df)); iters = []
    for tr, va in cv.split(X, y, groups=g):
        m = LGBMRegressor(**params)
        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        iters.append(int(m.best_iteration_ or params.get("n_estimators", 1000)))
        imp_sum += m.booster_.feature_importance(importance_type="gain")
        del m; gc.collect()
    oof_cols[name] = oof; full_iters[name] = max(50, int(np.mean(iters))); base_names.append(name)
    print(f"{name} OOF RMSE={rmse(y, oof + anchor):.4f}  (avg best_iter={full_iters[name]})", flush=True)

# --- feature importance (aggregated LGB gain across folds+configs); pruning is OFF by default ---
imp_df = pd.DataFrame({"feature": feats, "gain": imp_sum}).sort_values("gain", ascending=False).reset_index(drop=True)
imp_df["gain_pct"] = (100.0 * imp_df["gain"] / max(imp_df["gain"].sum(), 1e-9)).round(3)
imp_df.to_csv(CFG.OUT / "feature_importance.csv", index=False)
print("[imp] saved feature_importance.csv (pruning is OFF unless ROGII_IMP_FILE/ROGII_DROP_FEATS is set)", flush=True)
print(f"[imp] full ranked feature importance (LGB gain), {len(imp_df)} features:\\n"
      + imp_df.to_string(index=True), flush=True)

# --- CatBoost base models ---
if use_cb:
    for ci, params in enumerate(cb_cfgs):
        name = f"cb{ci}"; oof = np.zeros(len(train_df)); iters = []
        for tr, va in cv.split(X, y, groups=g):
            m = CatBoostRegressor(**params)
            m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], use_best_model=True, early_stopping_rounds=250)
            oof[va] = m.predict(X[va])
            iters.append(int(m.get_best_iteration() or params.get("iterations", 1000)))
            del m; gc.collect()
        oof_cols[name] = oof; full_iters[name] = max(50, int(np.mean(iters))); base_names.append(name)
        print(f"{name} OOF RMSE={rmse(y, oof + anchor):.4f}  (avg best_iter={full_iters[name]})", flush=True)

# --- Ridge meta on the base OOF ---
OOF = np.column_stack([oof_cols[n] for n in base_names])
meta_oof = np.zeros(len(train_df))
for tr, va in cv.split(OOF, y_fit, groups=g):
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(OOF[tr], y_fit[tr]); meta_oof[va] = r.predict(OOF[va])
print(f"*** ridge-stack OOF RMSE={rmse(y, meta_oof + anchor):.4f}  (mean-LGB baseline was ~9.86; delta space) ***", flush=True)
ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True); ridge.fit(OOF, y_fit)
print("ridge coefs:", dict(zip(base_names, [round(float(c), 4) for c in ridge.coef_])),
      "| intercept", round(float(ridge.intercept_), 4), flush=True)

# --- refit base models on ALL data with CV-chosen sizes, save everything ---
outdir = CFG.OUT
for ci, params in enumerate(lgb_cfgs):
    p = dict(params); p["n_estimators"] = full_iters[f"lgb{ci}"]
    m = LGBMRegressor(**p); m.fit(X, y_fit); joblib.dump(m, outdir / f"lgb{ci}.pkl")
    print(f"saved lgb{ci}.pkl (n_estimators={p['n_estimators']})", flush=True)
if use_cb:
    for ci, params in enumerate(cb_cfgs):
        p = dict(params); p["iterations"] = full_iters[f"cb{ci}"]
        m = CatBoostRegressor(**p); m.fit(X, y_fit); m.save_model(str(outdir / f"cb{ci}.cbm"))
        print(f"saved cb{ci}.cbm (iterations={p['iterations']})", flush=True)
joblib.dump(ridge, outdir / "ridge.pkl")
json.dump({"base_names": base_names, "use_cb": use_cb, "ridge_oof_rmse": float(rmse(y, meta_oof + anchor)),
           "residual_anchor": ("likpf_mean_d" if _resid else None),
           "grcal": os.environ.get("ROGII_GRCAL", "off").lower()},
          open(outdir / "stack_meta.json", "w"))
json.dump(feats, open(outdir / "features.json", "w"))
print("DONE. saved lgb*.pkl, cb*.cbm, ridge.pkl, stack_meta.json, features.json ->", outdir, flush=True)
print("Publish as a dataset and attach to the submission notebook (detach old lgb dataset).", flush=True)
'''

def mkcell(src, cid):
    return {'cell_type': 'code', 'id': cid, 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src.splitlines(keepends=True)}

cells = [mkcell(HEADER, 'hdrs0001')]
for cid in FEATURE_CELLS:
    cells.append(mkcell(cell_src(cid), 'feats_' + cid))
cells.append(mkcell(DRIVER, 'drvs0001'))

out_nb = dict(nb); out_nb['cells'] = cells
with io.open('train_stack.py', 'w', encoding='utf-8') as _f:
    _f.write('#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# Auto-generated by build_stack_notebook.py. Run on Kaggle/local with the data (and v5 feature-cache) attached.\n')
    for _c in cells:
        _s = ''.join(_c['source'])
        _f.write('\n# ' + '=' * 70 + '\n' + _s + ('' if _s.endswith('\n') else '\n'))
print('wrote train_stack.py with', len(cells), 'cells')
