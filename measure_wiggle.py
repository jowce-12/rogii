# Measure writeup-inspired candidates on the saved eval data (both disjoint samples):
#   (1) warm-up ramp on the projection blend (their 5/5-fold pick: beta=0.75, 500 ft ramp)
#   (2) + savgol-51 smoothing after projection
#   (3) their free-intercept basis + ramp (vs our PS-anchored basis)
#   (4) T4 hold-gate contribution (their real-LB evidence: shrink-toward-last hedges are harmful)
import numpy as np, sys
from scipy.signal import savgol_filter
from offline_tests import load, pooled, b0, ba, robfit

S2_THR, S2_HF = 0.0083, 14.0112

def chain(res):
    P = []
    for r in res:
        p = 0.65 * b0(r) + 0.35 * ba(r)                       # T1
        if float(np.median(r["dense_dist"])) <= S2_THR:       # T2
            p = 0.7 * p + 0.3 * r["tvt_dense"]
        P.append(p)
    return P

def project_v(r, pred, deg=4, anchor_ps=True, blend=0.75, ramp_ft=0.0):
    anchor = r["last"] + r["z_ps"]
    du = pred + r["z"] - anchor
    s = (r["md"] - r["md_ps"]) / max(r["md"][-1] - r["md_ps"], 1e-6)
    fit = robfit(s, du, deg, anchor_ps=anchor_ps)
    if ramp_ft > 0:
        b = blend * np.clip((r["md"] - r["md_ps"]) / ramp_ft, 0.0, 1.0)
    else:
        b = blend
    du2 = (1 - b) * du + b * fit
    return (anchor + du2) - r["z"]

def t4(r, p, h=0.35):
    exc = float(np.max(np.abs(p - r["last"])))
    gate = (np.nan_to_num(r["gr_corr"], nan=0) > 0.85) and (np.nan_to_num(r["tw_hf_std"], nan=99) < S2_HF) and (exc < 15)
    return (1 - h) * p + h * r["last"] if gate else p

def sg51(p):
    n = len(p); wl = min(51, n if n % 2 == 1 else n - 1)
    return savgol_filter(p, wl, 3) if wl >= 5 else p

for seed in (7, 11):
    res = load(seed)
    P = chain(res)
    variants = {
        "P1 deployed: PS-deg4 flat 0.75":          [project_v(r, p) for p, r in zip(P, res)],
        "P2 + 500ft warm-up ramp":                 [project_v(r, p, ramp_ft=500.0) for p, r in zip(P, res)],
        "P2b ramp 300ft":                          [project_v(r, p, ramp_ft=300.0) for p, r in zip(P, res)],
        "P2c ramp 900ft":                          [project_v(r, p, ramp_ft=900.0) for p, r in zip(P, res)],
        "P4 their basis: free-int deg4 + ramp500": [project_v(r, p, anchor_ps=False, ramp_ft=500.0) for p, r in zip(P, res)],
    }
    variants["P3 = P2 + savgol51"] = [sg51(p) for p in variants["P2 + 500ft warm-up ramp"]]
    print(f"=== seed {seed} | {len(res)} wells | chain(no proj) = {pooled(res, P):.4f} ===")
    for name, q in variants.items():
        print(f"  {name:42s} {pooled(res, q):.4f}")
    # T4 on top of the deployed projection (as shipped in 7-159/7-129)
    base = variants["P1 deployed: PS-deg4 flat 0.75"]
    with_t4 = [t4(r, p) for p, r in zip(base, res)]
    print(f"  {'T4 effect: deployed proj -> +T4(h=0.35)':42s} {pooled(res, base):.4f} -> {pooled(res, with_t4):.4f}")
