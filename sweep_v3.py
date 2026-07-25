# STRIDE-v3 decoder hyper-parameters that were never swept: K_BEAM, TOP_AGG, LIK_W.
# WHY NOW: v3 is a deployed 0.10 blend pole whose constants live in a notebook cell, so a
# change costs a harness re-judge and nothing else — no rebuild, no retrain, no re-upload.
# WHY THESE THREE: wlen/sigp/trend-init were tuned when v3 was built; K/top_m/lik_w were
# inherited from v1 unexamined. lik_w especially — 0.1 was calibrated for v1's FIXED 200 ft
# segments, but v3's segments vary 100..760 ft, so the evidence per segment (and therefore
# the right damping) is a different quantity.
# Protocol as always: sweep seed7, confirm the winner on seed11, then the blend gate.
# RUN: python sweep_v3.py            (~40min CPU)
import sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed

CONFIGS = [
    ("deployed      K96 top32 lw0.10", dict()),
    ("beam narrow   K48 top32 lw0.10", dict(K_BEAM=48)),
    ("beam wide     K192 top32 lw0.10", dict(K_BEAM=192)),
    ("agg tight     K96 top8  lw0.10", dict(TOP_AGG=8)),
    ("agg broad     K96 top64 lw0.10", dict(TOP_AGG=64)),
    ("damp more     K96 top32 lw0.05", dict(LIK_W=0.05)),
    ("damp less     K96 top32 lw0.20", dict(LIK_W=0.20)),
]

def one(wid, over):
    import sys as _s
    _a = _s.argv; _s.argv = ["x", "--wlen", "0.5"]
    import stride3 as S3
    _s.argv = _a
    from stride import load_well
    # loky reuses workers across configs, so an override from a previous config would
    # leak into this one unless every swept knob is reset first (this bug contaminated
    # the first sweep: "lw0.20 = 10.1801" was really K192+top64+lw0.20).
    for k, v in dict(K_BEAM=96, TOP_AGG=32, LIK_W=0.1, TEMP=0.02, W_LEN=0.5, SIG_P=0.012).items():
        setattr(S3, k, v)
    for k, v in over.items():
        setattr(S3, k, v)
    try:
        hw, tw = load_well(wid, "train")
        pred = S3.decode(hw, tw)
        if pred is None:
            return None
        ev = hw[hw["TVT_input"].isna()]
        t = ev["TVT"].values.astype(float)
        m = np.isfinite(t) & np.isfinite(pred)
        if m.sum() < 5:
            return None
        return (float(((pred[m] - t[m]) ** 2).sum()), int(m.sum()),
                [f"{wid}_{i}" for i in ev.index.values], np.asarray(pred, np.float32))
    except Exception:
        return None

if __name__ == "__main__":
    import blend_eval as BE
    t0 = time.time()
    res7, _ = BE.selector_preds(7)
    wells7 = [r["wid"] for r in res7]
    best = None
    for tag, over in CONFIGS:
        outs = Parallel(n_jobs=5, backend="loky")(delayed(one)(w, over) for w in wells7)
        se = n = 0.0
        rows = []
        for o in outs:
            if o is None:
                continue
            se += o[0]; n += o[1]
            rows.append(pd.DataFrame({"id": o[2], "s3_tvt": o[3]}))
        r = (se / max(n, 1)) ** 0.5
        print(f"[{time.time()-t0:.0f}s] seed7  {tag}  standalone = {r:.4f}", flush=True)
        if best is None or r < best[0]:
            best = (r, tag, over, rows)
    print(f"\nseed7 winner: {best[1]}  ({best[0]:.4f})", flush=True)
    # confirm on seed11 + save the winner's predictions for the blend gate
    res11, _ = BE.selector_preds(11)
    wells11 = [r["wid"] for r in res11]
    for tag, over in [(best[1], best[2]), ("deployed      K96 top32 lw0.10", dict())]:
        outs = Parallel(n_jobs=5, backend="loky")(delayed(one)(w, over) for w in wells11)
        se = n = 0.0
        for o in outs:
            if o is not None:
                se += o[0]; n += o[1]
        print(f"[{time.time()-t0:.0f}s] seed11 {tag}  standalone = {(se/max(n,1))**0.5:.4f}", flush=True)
    if best[2]:
        allw = sorted(set(wells7) | set(wells11))
        outs = Parallel(n_jobs=5, backend="loky")(delayed(one)(w, best[2]) for w in allw)
        rows = [pd.DataFrame({"id": o[2], "s3_tvt": o[3]}) for o in outs if o is not None]
        pd.concat(rows, ignore_index=True).to_parquet("s3_preds_sweep.parquet", index=False)
        print(f"[{time.time()-t0:.0f}s] wrote s3_preds_sweep.parquet for the blend gate", flush=True)
    print("DONE", flush=True)
