# Smoke test for the NOTEBOOK-style gold pool extension (stage3 wiring preflight).
# Mirrors the exact wrapper that will be inserted into cell 52: _stride_track (notebook
# variant, eval-row TVT array return) + nbr_curve, fail-open, memoized.
import time
from pathlib import Path
import numpy as np
import pandas as pd
import _gold_port as G
import _stride_nb as SNB
import stage2_worker as W   # for the neighbor bank + nbr_curve (bank build ~20s)
import stride as ST

G._PF_PROCS = 1
t0 = time.time()

_g2_orig = G._gold_candidate_pool
_G2_MEMO = {}


def _g2_stride_full(wid, hw_m, tw, seg):
    key = (wid, int(hw_m["TVT_input"].notna().sum()), seg)
    if key in _G2_MEMO:
        return _G2_MEMO[key]
    out = None
    try:
        st = SNB._stride_track(hw_m, tw, seg_len=seg)
        ev_mask = hw_m["TVT_input"].isna().values
        if st is not None and len(st) == int(ev_mask.sum()) and np.all(np.isfinite(st)):
            full = np.full(len(hw_m), np.nan)
            full[ev_mask] = np.asarray(st, float)
            kn_v = hw_m["TVT_input"].values.astype(float)
            fin = np.isfinite(kn_v)
            full[fin] = kn_v[fin]
            out = full
    except Exception:
        out = None
    _G2_MEMO[key] = out
    return out


def _g2_pool_ext(wid, hw_m, tw, data_dir, variants, include_pf=True, n_seeds=24, n_particles=350):
    pool = _g2_orig(wid, hw_m, tw, data_dir, variants, include_pf=include_pf,
                    n_seeds=n_seeds, n_particles=n_particles)
    for name, seg in (("stride", 200.0), ("stride_stiff", 400.0), ("stride_loose", 100.0)):
        try:
            arr = _g2_stride_full(wid, hw_m, tw, seg)
            if arr is not None:
                pool[name] = arr
        except Exception:
            pass
    try:
        nc = W.nbr_curve(wid, hw_m)
        if nc is not None and np.isfinite(nc).mean() > 0.9:
            full = np.asarray(nc, float).copy()
            kn_v = hw_m["TVT_input"].values.astype(float)
            fin = np.isfinite(kn_v)
            full[fin] = kn_v[fin]
            pool["nbr_curve"] = full
    except Exception:
        pass
    return pool


G._gold_candidate_pool = _g2_pool_ext

wells = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                               columns=["well"])["well"].unique())[:2]
variants = G._gold_variant_grid()
for wid in wells:
    hw, tw = ST.load_well(wid, "train")
    pool = _g2_pool_ext(wid, hw, tw, Path("."), variants, include_pf=False)
    ev_n = int(hw["TVT_input"].isna().sum())
    print(f"[{time.time()-t0:.0f}s] {wid}: eval_rows={ev_n} pool_keys={sorted(pool.keys())}", flush=True)
    rep = G._gold_calibrate_well(wid, hw, tw, Path("."), variants)
    if rep is None:
        print("   calibrate -> None", flush=True)
        continue
    print(f"   status={rep.get('status')} best={rep.get('best_name')} "
          f"gain={rep.get('gain')} consistency={rep.get('consistency')}", flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
