"""Offline candidate tests on collected eval data. Tune on seed7, confirm the
chosen chain on seed11. Stages (composable, deployed order):
  B0  deployed-7.129 selector = mean(pf_scale_3,5,8) [off]
  T1  S1 blend on sub_2:  (1-w)*B0 + w*mean(aff_3,5,8)
  T2  S2-lite dense blend: toward neighbor-surface anchor, gated by dense_dist
  T3  A2 projection: deployed(deg3, IRLS, 0.75 fit) vs PS-anchored vs +excursion gate
  T4  A4 gate: hold-boost on high-gr_corr/low-hf/low-excursion wells
"""
import numpy as np, pandas as pd, sys

def load(seed):
    return np.load(f"eval_data_seed{seed}.npy", allow_pickle=True).tolist()

def pooled(res, preds):
    return float(np.sqrt(np.mean(np.concatenate([(p - r["y"])**2 for p, r in zip(preds, res)]))))

def b0(r):  return (r["o3"] + r["o5"] + r["o8"]) / 3.0
def ba(r):  return (r["a3"] + r["a5"] + r["a8"]) / 3.0

def robfit(s, du, deg, anchor_ps=False, iters=4):
    if len(s) < deg + 2: return du.copy()
    if anchor_ps:
        A = np.column_stack([s**k for k in range(1, deg+1)])   # no intercept -> dU(0)=0
    else:
        A = np.column_stack([s**k for k in range(0, deg+1)])
    w = np.ones(len(s))
    for _ in range(iters):
        W = w[:, None]
        try: c, *_ = np.linalg.lstsq(A*W, du*w, rcond=None)
        except Exception: return du.copy()
        fit = A @ c
        rres = du - fit
        sc = np.median(np.abs(rres)) * 1.4826 + 1e-6
        w = 1.0 / (1.0 + (rres / (2.0*sc))**2)
    return fit

def project(r, pred, deg=3, anchor_ps=False, blend=0.75, exc_gate=False):
    anchor = r["last"] + r["z_ps"]
    du = pred + r["z"] - anchor
    end = r["md"][-1]
    s = (r["md"] - r["md_ps"]) / max(end - r["md_ps"], 1e-6)
    fit = robfit(s, du, deg, anchor_ps=anchor_ps)
    b = blend
    if exc_gate:
        exc = float(np.max(np.abs(fit)))
        b = 0.85 if exc > 15 else 0.60
    du2 = (1-b)*du + b*fit
    return (anchor + du2) - r["z"]

def report(res, label, preds):
    print(f"  {label:42s} {pooled(res, preds):.4f}")
    return preds

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    res = load(seed)
    print(f"=== seed {seed} | {len(res)} wells ===")
    P = [b0(r) for r in res]
    report(res, "B0 deployed selector (avg 3/5/8, off)", P)

    print("-- T1: S1 blend on sub_2 --")
    best_w, best = 0.0, pooled(res, P)
    for w in [0.2, 0.3, 0.35, 0.4, 0.5]:
        q = [ (1-w)*b0(r) + w*ba(r) for r in res ]
        v = pooled(res, q); print(f"  w={w:.2f}  {v:.4f}")
        if v < best: best, best_w = v, w
    print(f"  -> T1 pick w={best_w}")
    P = [ (1-best_w)*b0(r) + best_w*ba(r) for r in res ]

    print("-- T2: S2-lite dense blend (gate by dense_dist percentile) --")
    med_dist = np.array([np.median(r["dense_dist"]) for r in res])
    base = pooled(res, P); best_cfg = (0.0, 1.0); best = base
    for pq in [0.5, 0.75, 1.01]:
        thr = np.quantile(med_dist, min(pq, 1.0)) if pq <= 1.0 else np.inf
        for wd in [0.1, 0.2, 0.3, 0.5]:
            q = [ ((1-wd)*p + wd*r["tvt_dense"]) if np.median(r["dense_dist"]) <= thr else p
                  for p, r in zip(P, res) ]
            v = pooled(res, q)
            print(f"  gate<=q{pq:.2f} wd={wd:.1f}  {v:.4f}")
            if v < best: best, best_cfg = v, (wd, pq)
    print(f"  -> T2 pick wd={best_cfg[0]}, gate_q={best_cfg[1]} (base {base:.4f})")
    if best_cfg[0] > 0:
        thr = np.quantile(med_dist, min(best_cfg[1], 1.0)) if best_cfg[1] <= 1.0 else np.inf
        P = [ ((1-best_cfg[0])*p + best_cfg[0]*r["tvt_dense"]) if np.median(r["dense_dist"]) <= thr else p
              for p, r in zip(P, res) ]

    print("-- T3: A2 projection variants --")
    variants = [
        ("none", None),
        ("deployed deg3 blend.75", dict(deg=3, anchor_ps=False, blend=0.75, exc_gate=False)),
        ("PS-anchored deg3 blend.75", dict(deg=3, anchor_ps=True, blend=0.75, exc_gate=False)),
        ("PS-anchored deg3 + exc gate", dict(deg=3, anchor_ps=True, blend=0.75, exc_gate=True)),
        ("PS-anchored deg4 blend.75", dict(deg=4, anchor_ps=True, blend=0.75, exc_gate=False)),
    ]
    best_v, best = "none", pooled(res, P)
    print(f"  none                                     {best:.4f}")
    for name, kw in variants[1:]:
        q = [ project(r, p, **kw) for p, r in zip(P, res) ]
        v = pooled(res, q); print(f"  {name:40s} {v:.4f}")
        if v < best: best, best_v = v, name
    print(f"  -> T3 pick: {best_v}")
    kw_pick = dict(variants)[best_v] if best_v != "none" else None
    if kw_pick: P = [ project(r, p, **kw_pick) for p, r in zip(P, res) ]

    print("-- T4: A4 hold-gate (gr_corr high & tw_hf low & low excursion) --")
    hf_med = np.nanmedian([r["tw_hf_std"] for r in res])
    base = pooled(res, P); best_cfg = None; best = base
    for th1 in [0.82, 0.85, 0.88]:
        for h in [0.15, 0.25, 0.35]:
            q = []
            for p, r in zip(P, res):
                exc = float(np.max(np.abs(p - r["last"])))
                gate = (np.nan_to_num(r["gr_corr"], nan=0) > th1) and (np.nan_to_num(r["tw_hf_std"], nan=99) < hf_med) and (exc < 15)
                q.append((1-h)*p + h*r["last"] if gate else p)
            v = pooled(res, q)
            print(f"  th1={th1} h={h:.2f}  {v:.4f}")
            if v < best: best, best_cfg = v, (th1, h)
    print(f"  -> T4 pick: {best_cfg} (base {base:.4f})")
    print(f"\n=== seed {seed} final chain pooled = {best:.4f} (started {pooled(res,[b0(r) for r in res]):.4f}) ===")
