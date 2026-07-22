# stage2_worker.py — per-well gold calibration worker (loky process target).
# Light import: _gold_port + stride + neighbor bank (~20s once per worker process).
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import _gold_port as G
import stride as ST

G._PF_PROCS = 1     # serial PF inside each worker (well-level processes provide the parallelism)

SPW = 60
bank_xy, bank_u, bank_wid = [], [], []
_wells = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet", columns=["well"])["well"].unique())
for _w in _wells:
    try:
        _df = pd.read_csv(f"train/{_w}__horizontal_well.csv", usecols=["X", "Y", "Z", "TVT"]).dropna()
    except Exception:
        continue
    if len(_df) == 0:
        continue
    _ix = np.linspace(0, len(_df) - 1, min(SPW, len(_df)), dtype=int)
    bank_xy.append(_df[["X", "Y"]].values[_ix])
    bank_u.append((_df["TVT"].values + _df["Z"].values)[_ix])
    bank_wid.extend([_w] * len(_ix))
bank_xy = np.vstack(bank_xy); bank_u = np.concatenate(bank_u); bank_wid = np.array(bank_wid)
TREE = cKDTree(bank_xy)

def nbr_curve(wid, hw_m):
    xy = hw_m[["X", "Y"]].values.astype(float)
    dd, ii = TREE.query(xy, k=12)
    mask = bank_wid[ii] != wid
    w = np.where(mask, 1.0 / np.maximum(dd, 1.0) ** 2, 0.0)
    ws = w.sum(1)
    if (ws <= 0).any():
        return None
    u = (w * bank_u[ii]).sum(1) / ws
    if float(np.median(np.where(mask, dd, np.inf).min(1))) > 600.0:
        return None
    kn = hw_m[hw_m["TVT_input"].notna()]
    if len(kn) < 30:
        return None
    off = float(np.median(kn["TVT_input"].values + kn["Z"].values - u[kn.index.values]))
    return u - hw_m["Z"].values.astype(float) + off

_orig_pool = G._gold_candidate_pool

def pool_base(wid, hw_m, tw, data_dir, variants, include_pf=True, n_seeds=24, n_particles=350):
    # Harness-only: drop contact-family candidates. Harness wells are train wells whose
    # own contact files reconstruct near-truth -> gold picks them 150/150 and the A/B
    # comparison degenerates. Hidden test wells have no contact files; this simulates them.
    pool = _orig_pool(wid, hw_m, tw, data_dir, variants, include_pf=include_pf,
                      n_seeds=n_seeds, n_particles=n_particles)
    for k in [k for k in pool if "contact" in str(k).lower()]:
        pool.pop(k, None)
    return pool

def pool_ext(wid, hw_m, tw, data_dir, variants, include_pf=True, n_seeds=24, n_particles=350):
    pool = pool_base(wid, hw_m, tw, data_dir, variants, include_pf=include_pf,
                     n_seeds=n_seeds, n_particles=n_particles)
    try:
        import sys as _sys
        _argv = _sys.argv
        _sys.argv = ["x"]
        import stride3 as S3
        _sys.argv = _argv
        S3.W_LEN = 0.5
        S3.SIG_P = 0.012
        pred = S3.decode(hw_m, tw)
        if pred is not None:
            full = np.full(len(hw_m), np.nan)
            ev_mask = hw_m["TVT_input"].isna().values
            if len(pred) == int(ev_mask.sum()) and np.isfinite(pred).all():
                full[ev_mask] = pred
                kn = hw_m["TVT_input"].values
                fin = np.isfinite(kn)
                full[fin] = kn[fin]
                pool["stride_v3"] = full
    except Exception:
        pass
    return pool


def gold_one(wid, arm):
    G._gold_candidate_pool = pool_ext if arm == "B" else pool_base
    variants = G._gold_variant_grid()
    try:
        hw, tw = ST.load_well(wid, "train")
        rep = G._gold_calibrate_well(wid, hw, tw, Path("."), variants)
        if rep is None or rep.get("status") != "ok":
            return wid, rep, None
        best = rep["best_name"]
        need_pf = str(best).startswith("pf|")
        pool_f = G._gold_candidate_pool(wid, hw, tw, Path("."), variants,
                                        include_pf=need_pf,
                                        n_seeds=G._GOLD_FINAL_SEEDS,
                                        n_particles=G._GOLD_PARTICLES)
        arr = pool_f.get(best)
        return wid, rep, (np.asarray(arr, float) if arr is not None else None)
    except Exception as e:
        return wid, {"status": "error", "well": wid, "err": str(e)[:60]}, None
