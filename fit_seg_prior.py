# Fit the STRIDE-v3 segment-length prior from labels (shreygandhi design: log-normal on
# real segment lengths). Piecewise-linear segmentation of each train well's true
# dU = TVT + Z - anchor curve over MD via recursive max-deviation splitting (eps 1.5ft,
# min segment 40ft). Also refits the per-ft rate distribution and rate-persistence scale
# on the extracted segments (length-weighted). Outputs seg_prior.json.
# RUN from ~/rogii (~4min CPU).
import json
import time
import numpy as np
import pandas as pd
from stride import load_well

EPS = 1.5      # ft max deviation before a segment splits
MIN_LEN = 40.0  # ft


def segments(md, du):
    """Recursive piecewise-linear split; returns list of (len_ft, rate)."""
    out = []
    stack = [(0, len(md) - 1)]
    while stack:
        a, b = stack.pop()
        L = md[b] - md[a]
        if b - a < 4 or L < MIN_LEN * 2:
            out.append((max(L, 1.0), (du[b] - du[a]) / max(L, 1.0)))
            continue
        t = (md[a:b + 1] - md[a]) / max(L, 1e-9)
        line = du[a] + t * (du[b] - du[a])
        dev = np.abs(du[a:b + 1] - line)
        i = int(np.argmax(dev))
        if dev[i] <= EPS:
            out.append((L, (du[b] - du[a]) / L))
        else:
            m = a + max(2, min(i, b - a - 2))
            stack.append((a, m))
            stack.append((m, b))
    return out


t0 = time.time()
wells = sorted(pd.read_parquet("gru_rowmap.parquet", columns=["well"])["well"].unique())
lens, rates, dpairs = [], [], []
for k, wid in enumerate(wells, 1):
    try:
        hw, tw = load_well(wid, "train")
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) < 50:
            continue
        md = ev["MD"].values.astype(float)
        du = (ev["TVT"].values + ev["Z"].values).astype(float)
        fin = np.isfinite(du)
        md, du = md[fin], du[fin]
        if len(md) < 50:
            continue
        segs = segments(md, du)
        for j, (L, r) in enumerate(segs):
            lens.append(L)
            rates.append(r)
            if j > 0:
                dpairs.append((segs[j - 1][1], r, L))
    except Exception:
        continue
    if k % 200 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)}", flush=True)

lens = np.array(lens); rates = np.array(rates)
logl = np.log(np.clip(lens, 20, None))
prev = np.array([p for p, _r, _l in dpairs]); cur = np.array([r for _p, r, _l in dpairs])
dr = cur - prev
mad = float(np.median(np.abs(dr - np.median(dr))) * 1.4826)
prior = dict(
    n_segments=int(len(lens)),
    len_lognorm_mu=float(logl.mean()),
    len_lognorm_sigma=float(logl.std()),
    len_median_ft=float(np.exp(np.median(logl))),
    len_p10=float(np.quantile(lens, 0.10)), len_p90=float(np.quantile(lens, 0.90)),
    rate_std=float(rates.std()), rate_p99_abs=float(np.quantile(np.abs(rates), 0.99)),
    rate_persist_mad=mad,
)
json.dump(prior, open("seg_prior.json", "w"), indent=1)
print(json.dumps(prior, indent=1))
print(f"DONE [{time.time()-t0:.0f}s] -> seg_prior.json", flush=True)
