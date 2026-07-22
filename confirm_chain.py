"""Apply the seed7-chosen chain (FIXED configs) to seed11. Also report seed7 for
the same fixed configs. Chain: T1 S1-blend(w) -> T2 dense(wd=0.3, gate<=median
dist) -> T3 projection(PS-anchored deg4, blend .75) -> T4 gate(th1=.85,h=.35)."""
import numpy as np, pandas as pd
from offline_tests import load, pooled, b0, ba, project

def chain(res, w1, use_t4=True):
    med_dist = np.array([np.median(r["dense_dist"]) for r in res])
    thr = np.quantile(med_dist, 0.5)
    hf_med = np.nanmedian([r["tw_hf_std"] for r in res])
    out = []
    for r in res:
        p = (1-w1)*b0(r) + w1*ba(r)                                   # T1
        if np.median(r["dense_dist"]) <= thr:                          # T2
            p = 0.7*p + 0.3*r["tvt_dense"]
        p = project(r, p, deg=4, anchor_ps=True, blend=0.75)           # T3
        if use_t4:                                                     # T4
            exc = float(np.max(np.abs(p - r["last"])))
            if (np.nan_to_num(r["gr_corr"], nan=0) > 0.85) and (np.nan_to_num(r["tw_hf_std"], nan=99) < hf_med) and (exc < 15):
                p = 0.65*p + 0.35*r["last"]
        out.append(p)
    return out

for seed in (7, 11):
    res = load(seed)
    base = pooled(res, [b0(r) for r in res])
    print(f"=== seed {seed} ({len(res)} wells) | B0 = {base:.4f} ===")
    for w1 in (0.35, 0.4, 0.5):
        v4 = pooled(res, chain(res, w1, True))
        v3 = pooled(res, chain(res, w1, False))
        print(f"  w1={w1:.2f}: chain(no T4)={v3:.4f}  chain(+T4)={v4:.4f}")
