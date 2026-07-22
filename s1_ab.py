"""S1 4-way A/B: PF likelihood GR recalibration — off / affine / var / offset.
Same wells, same seeds. Reports selector pooled RMSE per arm + per-well deltas
split by band-miscalibration cohort (top-1/3 |mean-ratio − 1|).
Usage:  python3 s1_ab.py [--n 150] [--seeds 32] [--jobs 24] [--seed 7]
"""
import os, glob, time, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import cv_harness as H

def eval_well(wid, n_seeds, modes):
    try:
        hw, tw = H.load_well(wid)
    except Exception:
        return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10 or hw.TVT.isna().all() or ev.TVT.isna().any():
        return None
    y = ev.TVT.values.astype(float)
    last = float(kn.TVT_input.iloc[-1])
    out = {"wid": wid, "y": y}
    # band mis-calibration metric (for cohort split): |mean(hwGR)/mean(tw band) - 1|
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(float); tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    ktvt = kn.TVT_input.values.astype(float); kgr = kn.GR.values.astype(float)
    v = np.isfinite(kgr)
    band = tw_gr[(tw_tvt >= np.nanmin(ktvt) - 10) & (tw_tvt <= np.nanmax(ktvt) + 10)]
    out["miscal"] = abs(float(np.mean(kgr[v])) / max(float(np.mean(band)), 1e-6) - 1.0) if (v.sum() > 30 and len(band) > 20) else 0.0
    try:
        beam = H.run_beam_ensemble(hw, tw)[ev.index]
        variant = H.selector_well_code(hw)
        for mode in modes:
            pf, _ = H.lik_pf(hw, tw, n_seeds=n_seeds, grcal=mode)
            out[mode] = H.apply_variant(variant, pf, beam, last)
    except Exception:
        return None
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    MODES = ["off", "affine", "var", "offset"]
    wids = sorted(os.path.basename(f).split("__")[0] for f in glob.glob("train/*__horizontal_well.csv"))
    rng = np.random.default_rng(args.seed)
    samp = sorted(rng.choice(wids, min(args.n, len(wids)), replace=False).tolist())
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(eval_well)(w, args.seeds, MODES) for w in samp)
    res = [r for r in res if r is not None]
    print(f"evaluated {len(res)} wells in {time.time()-t0:.0f}s (seeds={args.seeds}, sample seed={args.seed})", flush=True)

    mis = np.array([r["miscal"] for r in res])
    thr = np.quantile(mis, 2 / 3)
    hi = [r for r in res if r["miscal"] >= thr]     # top-1/3 miscalibrated cohort
    lo = [r for r in res if r["miscal"] < thr]

    def pooled(rs, key):
        e = [ (rs_i[key] - rs_i["y"]) ** 2 for rs_i in rs ]
        return float(np.sqrt(np.mean(np.concatenate(e)))) if e else float("nan")

    print(f"\n{'mode':8s} {'pooled(all)':>12s} {'miscal-top1/3':>14s} {'rest':>10s}   wells worse-than-off")
    base_pw = {r["wid"]: float(np.sqrt(np.mean((r["off"] - r["y"]) ** 2))) for r in res}
    for m in MODES:
        pw = {r["wid"]: float(np.sqrt(np.mean((r[m] - r["y"]) ** 2))) for r in res}
        worse = sum(1 for w in pw if pw[w] > base_pw[w] + 0.05)
        print(f"{m:8s} {pooled(res, m):12.4f} {pooled(hi, m):14.4f} {pooled(lo, m):10.4f}   {worse if m != 'off' else '-'}")
    np.save(f"s1_ab_seed{args.seed}.npy", np.array(res, dtype=object), allow_pickle=True)
    print("\nsaved per-well results -> s1_ab_seed{}.npy".format(args.seed))
