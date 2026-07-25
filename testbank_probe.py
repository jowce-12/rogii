# TRANSDUCTIVE surface enrichment probe: at test time every test well's VISIBLE PREFIX
# gives true TVT at its own XY, i.e. ~200 extra surface samples that our neighbor bank
# (train wells only) currently ignores. If the test set sits on held-out pads, test wells
# are each other's nearest neighbours — exactly the isolated regime where our blend is
# weakest (dense_dist Q4: fleongg 7.65 vs GRU 5.25).
# Honest harness simulation (the 150/150 harness wells play "test"):
#   bank_A = train wells NOT in the harness, full trajectories        [= today's behaviour]
#   bank_B = bank_A + the OTHER harness wells' PREFIX-ONLY samples    [= proposed]
# Metric: per-well RMSE of the anchored surface prediction (u - Z + b) in the eval zone,
# where b is the median offset fitted on the well's own known prefix (deployment rule).
# RUN from ~/rogii: python testbank_probe.py       (~6min CPU)
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import blend_eval as BE
from stride import load_well

SPW_FULL = 60
SPW_PREFIX = 40
K = 20
t0 = time.time()

res7, _ = BE.selector_preds(7)
res11, _ = BE.selector_preds(11)
H = {7: sorted(r["wid"] for r in res7), 11: sorted(r["wid"] for r in res11)}
harness_all = sorted(set(H[7]) | set(H[11]))
print(f"[{time.time()-t0:.0f}s] harness wells {len(harness_all)}", flush=True)

# ---- sample banks -------------------------------------------------------------
full_xy, full_u, full_w = [], [], []
pre_xy, pre_u, pre_w = [], [], []
for p in sorted(Path("train").glob("*__horizontal_well.csv")):
    w = p.stem.replace("__horizontal_well", "")
    d = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
    if w in harness_all:
        kn = d[d["TVT_input"].notna()].dropna(subset=["X", "Y", "Z"])
        if len(kn) == 0:
            continue
        ix = np.linspace(0, len(kn) - 1, min(SPW_PREFIX, len(kn)), dtype=int)
        pre_xy.append(kn[["X", "Y"]].values[ix])
        pre_u.append((kn["TVT_input"].values + kn["Z"].values)[ix])
        pre_w.extend([w] * len(ix))
    else:
        dd = d.dropna(subset=["X", "Y", "Z", "TVT"])
        if len(dd) == 0:
            continue
        ix = np.linspace(0, len(dd) - 1, min(SPW_FULL, len(dd)), dtype=int)
        full_xy.append(dd[["X", "Y"]].values[ix])
        full_u.append((dd["TVT"].values + dd["Z"].values)[ix])
        full_w.extend([w] * len(ix))
FXY = np.vstack(full_xy); FU = np.concatenate(full_u); FW = np.array(full_w)
PXY = np.vstack(pre_xy); PU = np.concatenate(pre_u); PW = np.array(pre_w)
print(f"[{time.time()-t0:.0f}s] bank_A {len(FU)} samples / {len(set(FW))} wells | "
      f"prefix add-on {len(PU)} samples / {len(set(PW))} wells", flush=True)

BXY = np.vstack([FXY, PXY]); BU = np.concatenate([FU, PU]); BW = np.concatenate([FW, PW])
scale = np.where(FXY.std(0) < 1e-3, 1.0, FXY.std(0))
treeA = cKDTree(FXY / scale)
treeB = cKDTree(BXY / scale)


def surface_rmse(wid, tree, U, W):
    hw, _ = load_well(wid, "train")
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(kn) < 30 or len(ev) < 10:
        return None
    def idw(xy, self_w):
        fetch = min(K + SPW_FULL, len(U))
        dist, idx = tree.query(xy / scale, k=fetch, workers=-1)
        dist = np.where(W[idx] == self_w, np.inf, dist)
        take = np.argpartition(dist, K - 1, axis=1)[:, :K]
        dk = np.take_along_axis(dist, take, axis=1)
        ik = np.take_along_axis(idx, take, axis=1)
        valid = np.isfinite(dk)
        wgt = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
        sw = wgt.sum(1)
        if np.any(sw <= 0):
            return None, None
        return (U[ik] * wgt).sum(1) / sw, np.where(valid, dk, np.inf).min(1)
    u_kn, _ = idw(kn[["X", "Y"]].to_numpy(float), wid)
    u_ev, d_ev = idw(ev[["X", "Y"]].to_numpy(float), wid)
    if u_kn is None or u_ev is None:
        return None
    b = float(np.median(kn["TVT_input"].values + kn["Z"].values - u_kn))
    pred = u_ev - ev["Z"].to_numpy(float) + b
    truth = ev["TVT"].to_numpy(float)
    fin = np.isfinite(truth) & np.isfinite(pred)
    if fin.sum() < 10:
        return None
    return (float(np.sqrt(np.mean((pred[fin] - truth[fin]) ** 2))), int(fin.sum()),
            float(np.median(d_ev)))


rows = []
for k, wid in enumerate(harness_all, 1):
    a = surface_rmse(wid, treeA, FU, FW)
    b = surface_rmse(wid, treeB, BU, BW)
    if a is None or b is None:
        continue
    rows.append((wid, a[0], b[0], a[1], a[2]))
    if k % 60 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/{len(harness_all)}", flush=True)
R = pd.DataFrame(rows, columns=["well", "rmse_A", "rmse_B", "n", "nn_dist_A"])
R.to_parquet("testbank_probe.parquet", index=False)
pa = float(np.sqrt(np.average(R.rmse_A ** 2, weights=R.n)))
pb = float(np.sqrt(np.average(R.rmse_B ** 2, weights=R.n)))
print(f"\nsurface pooled RMSE  bank_A(train only) = {pa:.4f}   "
      f"bank_B(+test prefixes) = {pb:.4f}   delta = {pb - pa:+.4f}", flush=True)
print(f"wells improved: {(R.rmse_B < R.rmse_A).mean():.1%}", flush=True)
q = pd.qcut(R["nn_dist_A"], 4, labels=["Q1_near", "Q2", "Q3", "Q4_isolated"], duplicates="drop")
print(R.groupby(q, observed=True).agg(A=("rmse_A", "mean"), B=("rmse_B", "mean"),
                                      n=("well", "size")).round(3).to_string(), flush=True)
print("DONE", flush=True)
