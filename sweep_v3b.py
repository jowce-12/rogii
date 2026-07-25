# Follow-up on the lik_w finding (0.10 -> 0.20 gave 11.06 -> 10.18 on seed7).
# Two questions, one sweep:
#  (a) HOW FAR does it go? v3 decodes on a 10 ft grid while v1 decoded per row (~1 ft), so
#      most of the GR autocorrelation redundancy is already removed by subsampling and the
#      inherited 0.1 was damping twice. Try 0.30 / 0.50.
#  (b) WHICH mechanism? lik_w scales the whole score, so it simultaneously (i) makes the
#      evidence louder than the persistence/length priors and (ii) sharpens the softmax
#      over the top 32. Control: keep lik_w = 0.10 and halve TEMP instead. If that alone
#      reproduces the gain, the knob we actually want is the aggregation temperature.
import time
import numpy as np, pandas as pd
from joblib import Parallel, delayed

CONFIGS = [
    ("lw 0.20 (seed7 winner)   ", dict(LIK_W=0.20)),
    ("lw 0.30                  ", dict(LIK_W=0.30)),
    ("lw 0.50                  ", dict(LIK_W=0.50)),
    ("lw 0.10 + TEMP 0.010     ", dict(LIK_W=0.10, TEMP=0.010)),
    ("lw 0.10 + TEMP 0.005     ", dict(LIK_W=0.10, TEMP=0.005)),
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
    res11, _ = BE.selector_preds(11)
    w7 = [r["wid"] for r in res7]; w11 = [r["wid"] for r in res11]
    best = None
    for tag, over in CONFIGS:
        outs = Parallel(n_jobs=6, backend="loky")(delayed(one)(w, over) for w in w7)
        se = n = 0.0
        for o in outs:
            if o is not None:
                se += o[0]; n += o[1]
        r = (se / max(n, 1)) ** 0.5
        print(f"[{time.time()-t0:.0f}s] seed7  {tag} = {r:.4f}", flush=True)
        if best is None or r < best[0]:
            best = (r, tag, over)
    print(f"\nseed7 winner: {best[1]} ({best[0]:.4f})", flush=True)
    outs = Parallel(n_jobs=6, backend="loky")(delayed(one)(w, best[2]) for w in w11)
    se = n = 0.0
    for o in outs:
        if o is not None:
            se += o[0]; n += o[1]
    print(f"[{time.time()-t0:.0f}s] seed11 {best[1]} = {(se/max(n,1))**0.5:.4f}  "
          f"(deployed lw0.10 was 11.7417)", flush=True)
    allw = sorted(set(w7) | set(w11))
    outs = Parallel(n_jobs=6, backend="loky")(delayed(one)(w, best[2]) for w in allw)
    rows = [pd.DataFrame({"id": o[2], "s3_tvt": o[3]}) for o in outs if o is not None]
    pd.concat(rows, ignore_index=True).to_parquet("s3_preds_likw.parquet", index=False)
    print(f"[{time.time()-t0:.0f}s] wrote s3_preds_likw.parquet ({best[1]}) for the blend gate", flush=True)
    print("DONE", flush=True)
