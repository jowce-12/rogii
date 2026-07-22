# Clean OOF recompute + dip fusion for HETEROGENEOUS spatial legs (37-chan _s* and
# 41-chan _x* checkpoints in one ensemble). Uses gru_spatial2_fold*.parquet as the single
# spatial source (its base columns are bit-identical to v1 — verified in the spatial2
# smoke), building 6- or 10-channel spatial blocks per checkpoint as its chans demand.
# RUN: python gru_fusion_mixed.py --tags _xa,_xb,_xc --suffix _x3
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import _gru_infer as GI
from gru_fusion import fuse, _unit_test

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAMS = (64.0, 256.0, 1024.0)
TAGS = tuple(sys.argv[sys.argv.index("--tags") + 1].split(","))
SUF = sys.argv[sys.argv.index("--suffix") + 1]
S1 = ["dense_d", "dense_std", "dense_near", "pf_dense_gap", "dense_confidence", "dense_known_rmse"]
S2 = ["densex_med", "densex_spread", "densex_best", "densex_best_rmse"]
EXTRA_SURFACES = ["astnu", "astnl", "egfdu", "egfdl", "buda"]

_unit_test()
seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
RAW_CHANS = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
wells = sorted(seq["well"].unique())
fold_of = {}
for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = f

SPAT = {}
SRC_COLS = ["md", "spatial_u", "dense_std", "dense_dist"] + [f"u_{s}" for s in EXTRA_SURFACES]
for f in range(5):
    fr = pd.read_parquet(f"gru_spatial2_fold{f}.parquet")
    SPAT[f] = {w: {c: g[c].to_numpy() for c in SRC_COLS} for w, g in fr.groupby("well", sort=False)}
    del fr
print("spatial2 sources loaded", flush=True)


def spatial_block(x_raw, src, want2):
    """6 (want2=False) or 10 (True) spatial channels; math identical to train_gru2.py."""
    d_to_end = x_raw[:, RAW_CHANS.index("d_to_end")].astype(np.float64)
    tvt_rel = x_raw[:, RAW_CHANS.index("tvt_rel")].astype(np.float64)
    known = x_raw[:, RAW_CHANS.index("is_known")] > 0.5
    pf5 = x_raw[:, RAW_CHANS.index("pf5_d")].astype(np.float64)
    md_grid = float(src["md"][-1]) - d_to_end * 1000.0
    u = np.interp(md_grid, src["md"], src["spatial_u"])
    std_i = np.interp(md_grid, src["md"], src["dense_std"])
    dist_i = np.interp(md_grid, src["md"], src["dense_dist"])
    kp = np.flatnonzero(known)
    n_out = 10 if want2 else 6
    if len(kp) == 0:
        return np.zeros((len(x_raw), n_out), dtype=np.float32)
    cut_pos = int(kp[-1])
    u_rel = (u - u[cut_pos]) / 10.0
    bias = float(np.median(tvt_rel[known] - u_rel[known]))
    dense_d = np.clip(u_rel + bias, -12.0, 12.0)
    known_rmse = float(np.sqrt(np.mean((dense_d[known] - tvt_rel[known]) ** 2)))
    dense_std = np.clip(std_i / 20.0, 0.0, 6.0)
    dense_near = np.exp(-np.clip(dist_i, 0.0, None) / 0.02)
    cols = [dense_d, dense_std, dense_near, np.clip(pf5 - dense_d, -12.0, 12.0),
            np.exp(-np.clip(std_i, 0.0, None) / 30.0) * dense_near,
            np.full(len(x_raw), np.clip(known_rmse, 0.0, 6.0))]
    if want2:
        ds, rmses = [dense_d], [known_rmse]
        for s in EXTRA_SURFACES:
            u_s = src.get(f"u_{s}")
            if u_s is None:
                continue
            u_i = np.interp(md_grid, src["md"], u_s)
            if not np.isfinite(u_i).all():
                continue
            u_rel_i = (u_i - u_i[cut_pos]) / 10.0
            bias_i = float(np.median(tvt_rel[known] - u_rel_i[known]))
            d_i = np.clip(u_rel_i + bias_i, -12.0, 12.0)
            ds.append(d_i)
            rmses.append(float(np.sqrt(np.mean((d_i[known] - tvt_rel[known]) ** 2))))
        D = np.stack(ds, 0)
        best = int(np.argmin(rmses))
        cols += [np.median(D, 0),
                 np.clip(D.std(0), 0.0, 6.0) if len(ds) > 1 else np.zeros(len(x_raw)),
                 D[best],
                 np.full(len(x_raw), np.clip(rmses[best], 0.0, 6.0))]
    return np.column_stack(cols).astype(np.float32)


nets = {}
for tag in TAGS:
    for f in range(5):
        ck = torch.load(f"gru_fold{f}{tag}.pt", map_location="cpu")
        assert ck.get("dip") and ck.get("spatial"), f"{tag} fold{f}: expected dip+spatial ckpt"
        want2 = bool(ck.get("spatial2"))
        exp = RAW_CHANS + S1 + (S2 if want2 else [])
        assert ck["chans"] == exp, f"{tag} fold{f}: channel list mismatch"
        hid = ck["hid"]
        nchan = len(exp)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(nchan, hid)
                self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.25)
                self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 2))

            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.head(h)

        m = _Net(); m.load_state_dict(ck["state"]); m.eval().to(DEV)
        nets[(tag, f)] = (m, want2)
print(f"{len(nets)} ckpts loaded ({len(TAGS)} tags) | device={DEV}", flush=True)

cache = {}
t0 = time.time()
nat = seq[seq["cut"] == "nat"]
for k, (wid, g) in enumerate(nat.groupby("well"), 1):
    g = g.sort_values("step")
    f = fold_of[wid]
    tail = g["is_tail"].values.astype(bool)
    x_raw = g[RAW_CHANS].values.astype(np.float32)
    src = SPAT[f][wid]
    x37 = np.concatenate([x_raw, spatial_block(x_raw, src, False)], axis=1)
    x41 = (np.concatenate([x_raw, spatial_block(x_raw, src, True)], axis=1)
           if any(w2 for _m, w2 in nets.values()) else None)
    ps, ds = [], []
    with torch.no_grad():
        t37 = torch.from_numpy(x37[None]).to(DEV)
        t41 = torch.from_numpy(x41[None]).to(DEV) if x41 is not None else None
        for tag in TAGS:
            m, want2 = nets[(tag, f)]
            out = m(t41 if want2 else t37).cpu().numpy()[0]
            ps.append(out[:, 0] * 10.0)
            ds.append(out[:, 1] / 25.0 * 10.0 / GI.G_STEP)
    rm = rowmap[rowmap["well"] == wid]
    mg = (np.arange(len(g), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    cache[wid] = (np.mean(ps, 0), np.mean(ds, 0), tail, rm, mg)
    if k % 100 == 0:
        print(f"[{time.time()-t0:.0f}s] forward {k}/773", flush=True)

rm_y = rowmap[["id", "y"]].set_index("id")["y"]
def pooled(rows):
    j = pd.concat(rows, ignore_index=True).set_index("id")
    return float(np.sqrt(np.mean((j["gru_d"].values - rm_y.reindex(j.index).values) ** 2)))

base_rows = []
for wid, (p, d, tail, rm, mg) in cache.items():
    base_rows.append(pd.DataFrame({"id": rm["id"].values,
                                   "gru_d": np.interp(rm["md_rel"].values, mg, p).astype(np.float32)}))
print(f"== {SUF} unfused clean OOF = {pooled(base_rows):.4f}", flush=True)
best = (1e9, None, None)
for lam in LAMS:
    rows = []
    for wid, (p, d, tail, rm, mg) in cache.items():
        pf_ = p.copy()
        ti = np.where(tail)[0]
        if len(ti) > 2:
            pf_[ti] = fuse(p[ti], d[ti][:-1], lam)
        rows.append(pd.DataFrame({"id": rm["id"].values,
                                  "gru_d": np.interp(rm["md_rel"].values, mg, pf_).astype(np.float32)}))
    r = pooled(rows)
    print(f"   dip-fused lam={lam:4.0f}: {r:.4f}", flush=True)
    if r < best[0]:
        best = (r, lam, rows)
pd.concat(best[2], ignore_index=True).to_parquet(f"gru_oof_fused{SUF}.parquet", index=False)
print(f"BEST lam={best[1]} -> gru_oof_fused{SUF}.parquet ({best[0]:.4f})", flush=True)
print("DONE", flush=True)
