"""Assemble a FAST single-LGB OOF-check notebook (train_quick.ipynb).

Purpose: quickly measure the effect of a feature change or a parameter tweak,
WITHOUT the full LGB×3+CB×2+Ridge stack. Reuses the v5 feature cache (so no
~2.8h rebuild if the cache dataset is attached), trains ONE LightGBM with
GroupKFold, prints the OOF RMSE + feature importance. Does NOT save models.

Use the single-LGB OOF for RELATIVE comparison across experiments (A vs B); it
is higher than the full-stack ridge OOF (that's expected — this is a proxy).

Feature knobs:  ROGII_FEATS, ROGII_IMP_FILE(+ROGII_TOP_FEATS/ROGII_MIN_IMP), ROGII_DROP_FEATS
Param knobs:    ROGII_LGB_LEAVES, ROGII_LGB_LR, ROGII_LGB_TREES, ROGII_LGB_MCS,
                ROGII_LGB_FF (colsample), ROGII_LGB_BF (subsample),
                ROGII_LGB_L1, ROGII_LGB_L2, ROGII_LGB_FOLDS, ROGII_QUICK_MAXWELLS
"""
import json, io

SRC = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(SRC, encoding='utf-8'))
def cell_src(cid):
    return ''.join(next(c for c in nb['cells'] if c.get('id') == cid)['source'])

FEATURE_CELLS = ['44e50dd0', 'c3ab97ce', '83876170', 'c9bba3da', 'b515d5c6', 'cc1a056a']

HEADER = '''# FAST single-LGB OOF check for ROGII — feature/parameter experiments.
# Reuses the v5 feature cache (attach the cache dataset to skip the ~2.8h build).
# Trains ONE LightGBM (no CB/Ridge), prints OOF RMSE + feature importance.
# The single-LGB OOF is a PROXY — compare it RELATIVELY across runs (A vs B),
# not against the full-stack ridge OOF (which is lower).
#
# Feature knobs: ROGII_FEATS, ROGII_IMP_FILE(+ROGII_TOP_FEATS/MIN_IMP), ROGII_DROP_FEATS
# Param knobs:   ROGII_LGB_LEAVES(127) ROGII_LGB_LR(0.03) ROGII_LGB_TREES(4000)
#                ROGII_LGB_MCS(20) ROGII_LGB_FF(0.8) ROGII_LGB_BF(0.8)
#                ROGII_LGB_L1(0.1) ROGII_LGB_L2(5.0) ROGII_LGB_FOLDS(5)
#                ROGII_QUICK_MAXWELLS(150 = default; set 0 for all 773 wells)
# CatBoost:      ROGII_QUICK_CB(1=on) ROGII_CB_DEPTH(7) ROGII_CB_LR(0.03)
#                ROGII_CB_ITERS(4000) ROGII_CB_L2(3.0) ROGII_CB_MDL(20)
#                -> also prints CatBoost OOF + LGB+CB ridge-blend OOF
# Speed: fleongg PF/beam kernels are nogil -> feature build uses all cores.
import os
os.environ.setdefault("SHOW_FIGS", "0")
'''

DRIVER = '''# ===== quick single-LGB OOF (feature/param experiments) =====
import gc
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
# version-agnostic early stopping: callbacks (LightGBM>=3.3) or old fit kwargs (<3.3)
try:
    from lightgbm import early_stopping as _es, log_evaluation as _le
    _FIT_KW = dict(eval_metric="rmse", callbacks=[_es(250, verbose=False), _le(0)])
except Exception:
    _FIT_KW = dict(eval_metric="rmse", early_stopping_rounds=250, verbose=False)

train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
test_wids  = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"test").glob("*__horizontal_well.csv"))
_MAX = int(os.environ.get("ROGII_QUICK_MAXWELLS", "150"))   # default 150 wells for fast experiments; set 0 for all 773
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
    print("[cache] building train features (slow ~2-3h; attach the cache dataset to skip)...", flush=True)
    _df = add_likpf_features(build_features(train_wids, "train", is_train=True), build_likpf(train_wids, "train"))
    if mode != "off":
        try:
            _df.to_parquet(CFG.OUT / name, index=False)
            print(f"[cache] saved -> {CFG.OUT / name}", flush=True)
        except Exception as _e:
            print("[cache] save skipped:", _e, flush=True)
    return _df

init_imputers(train_wids)
train_df = add_alias_metafeats(load_or_build_train_features(train_wids), "train"); gc.collect()   # A5
test_df  = add_alias_metafeats(add_likpf_features(build_features(test_wids, "test", is_train=False), build_likpf(test_wids, "test")), "test"); gc.collect()

# ============================================================================
# CUSTOM QUICK FEATURES — add columns derived from EXISTING columns here to test
# a feature idea WITHOUT the ~2.8h rebuild (works whenever the new feature is a
# function of already-built columns). Add to BOTH frames; they auto-enter `feats`.
# Example:
#   for _df in (train_df, test_df):
#       _df["pfdelta_x_trkstd"] = _df["pf_ancc_delta"].astype("float32") * _df["trk_std"].astype("float32")
# Features needing raw GR/PF arrays must be added in build_well instead (then run
# once with ROGII_FEATCACHE=rebuild to refresh the cache).
# ============================================================================

feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]

# optional pruning (same knobs as train_stack)
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
elif _drop:
    feats = [c for c in feats if c not in _drop]
print(f"features: {len(feats)} | train rows: {len(train_df)}", flush=True)

dev, _ = _device()
X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values
_resid = os.environ.get("ROGII_RESID", "0") == "1"   # default OFF: residual target hurt LGB valid scores (measured 2026-07-09)   # train on (target - likpf_mean_d); prints stay in delta space
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y - anchor
if _resid:
    print(f"[resid] target re-anchored to likpf_mean_d (std {y.std():.2f} -> {y_fit.std():.2f})", flush=True)
folds = int(os.environ.get("ROGII_LGB_FOLDS", "5"))
cv = GroupKFold(folds)

# GPU probe: a machine can report a GPU (nvidia-smi) but LightGBM may lack the
# OpenCL device -> auto-fall back to CPU so local runs don't crash.
dev = _resolve_lgb_device(X, y, dev)   # gpu(OpenCL) -> cuda -> cpu

base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
if dev == "cuda":
    base.update(device_type="cuda")
elif dev == "gpu":
    base.update(device_type="gpu", gpu_use_dp=False)
params = dict(base,
    num_leaves=int(os.environ.get("ROGII_LGB_LEAVES", "127")),
    learning_rate=float(os.environ.get("ROGII_LGB_LR", "0.03")),
    n_estimators=int(os.environ.get("ROGII_LGB_TREES", "4000")),
    min_child_samples=int(os.environ.get("ROGII_LGB_MCS", "20")),
    subsample=float(os.environ.get("ROGII_LGB_BF", "0.8")), subsample_freq=1,
    colsample_bytree=float(os.environ.get("ROGII_LGB_FF", "0.8")),
    reg_alpha=float(os.environ.get("ROGII_LGB_L1", "0.1")),
    reg_lambda=float(os.environ.get("ROGII_LGB_L2", "5.0")),
    random_state=42)
print("LGB params:", {k: params[k] for k in ["num_leaves","learning_rate","n_estimators","min_child_samples","colsample_bytree","subsample","reg_alpha","reg_lambda"]}, "| folds:", folds, flush=True)

oof = np.zeros(len(train_df)); imp = np.zeros(len(feats)); iters = []
for fi, (tr, va) in enumerate(cv.split(X, y, groups=g)):
    m = LGBMRegressor(**params)
    m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], **_FIT_KW)
    oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
    imp += m.booster_.feature_importance(importance_type="gain")
    iters.append(int(m.best_iteration_ or params["n_estimators"]))
    print(f"  fold{fi}: best_iter={iters[-1]}", flush=True)
    del m; gc.collect()

print(f"*** QUICK single-LGB OOF RMSE = {rmse(y, oof + anchor):.4f}  (avg best_iter={int(np.mean(iters))}) ***", flush=True)
print("    (proxy — compare RELATIVELY across experiments, not vs the full-stack ridge OOF)", flush=True)

# ===== optional: also train ONE CatBoost for an LGB-vs-CB comparison =====
if os.environ.get("ROGII_QUICK_CB", "1") == "1":
    try:
        from catboost import CatBoostRegressor
        from sklearn.linear_model import Ridge
        cb_task = "GPU" if dev == "gpu" else "CPU"
        cb_params = dict(
            iterations=int(os.environ.get("ROGII_CB_ITERS", "4000")),
            depth=int(os.environ.get("ROGII_CB_DEPTH", "7")),
            learning_rate=float(os.environ.get("ROGII_CB_LR", "0.03")),
            l2_leaf_reg=float(os.environ.get("ROGII_CB_L2", "3.0")),
            min_data_in_leaf=int(os.environ.get("ROGII_CB_MDL", "20")),
            loss_function="RMSE", task_type=cb_task, od_type="Iter", od_wait=200,
            border_count=254, verbose=0, random_seed=42)
        print("CB params:", {k: cb_params[k] for k in ["depth","learning_rate","iterations","l2_leaf_reg","min_data_in_leaf","task_type"]}, flush=True)
        cb_oof = np.zeros(len(train_df)); cb_it = []
        for fi, (tr, va) in enumerate(cv.split(X, y, groups=g)):
            cm = CatBoostRegressor(**cb_params)
            try:
                cm.fit(X[tr], y_fit[tr], eval_set=(X[va], y_fit[va]), use_best_model=True, early_stopping_rounds=200)
            except Exception as _e:
                if cb_params["task_type"] == "GPU":
                    print(f"[cb] GPU failed -> CPU ({str(_e)[:60]})", flush=True)
                    cb_params["task_type"] = "CPU"; cm = CatBoostRegressor(**cb_params)
                    cm.fit(X[tr], y_fit[tr], eval_set=(X[va], y_fit[va]), use_best_model=True, early_stopping_rounds=200)
                else:
                    raise
            cb_oof[va] = cm.predict(X[va]); cb_it.append(int(cm.get_best_iteration() or cb_params["iterations"]))
        print(f"*** QUICK CatBoost  OOF RMSE = {rmse(y, cb_oof + anchor):.4f}  (avg best_iter={int(np.mean(cb_it))}) ***", flush=True)
        _O = np.column_stack([oof, cb_oof]); _meta = np.zeros(len(train_df))
        for tr, va in cv.split(_O, y_fit, groups=g):
            _r = Ridge(alpha=1.66, positive=True, fit_intercept=True); _r.fit(_O[tr], y_fit[tr]); _meta[va] = _r.predict(_O[va])
        print(f"*** LGB+CB ridge-blend OOF   = {rmse(y, _meta + anchor):.4f}  (vs LGB {rmse(y, oof + anchor):.4f}, CB {rmse(y, cb_oof + anchor):.4f}) ***", flush=True)
    except ImportError:
        print("[cb] catboost not installed; skipping (pip install catboost to enable, or ROGII_QUICK_CB=0)", flush=True)

imp_df = pd.DataFrame({"feature": feats, "gain": imp}).sort_values("gain", ascending=False).reset_index(drop=True)
imp_df["gain_pct"] = (100.0 * imp_df["gain"] / max(imp_df["gain"].sum(), 1e-9)).round(3)
imp_df.to_csv(CFG.OUT / "quick_feature_importance.csv", index=False)
print(f"[imp] {len(imp_df)} features (LGB gain):\\n" + imp_df.to_string(index=True), flush=True)
'''

def mkcell(src, cid):
    return {'cell_type': 'code', 'id': cid, 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src.splitlines(keepends=True)}

cells = [mkcell(HEADER, 'hdrq0001')]
for cid in FEATURE_CELLS:
    cells.append(mkcell(cell_src(cid), 'featq_' + cid))
cells.append(mkcell(DRIVER, 'drvq0001'))

out_nb = dict(nb); out_nb['cells'] = cells
# emit a flat runnable .py (same cells concatenated)
with io.open('train_quick.py', 'w', encoding='utf-8') as f:
    f.write('#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n')
    f.write('# FAST single-LGB OOF check for ROGII (feature/param experiments).\n')
    f.write('# Run on Kaggle (needs the competition data; attach the v5 feature-cache dataset to skip the ~2.8h build).\n')
    for c in cells:
        f.write('\n# ' + '=' * 70 + '\n')
        f.write(''.join(c['source']))
        if not ''.join(c['source']).endswith('\n'):
            f.write('\n')
print('wrote train_quick.py')
