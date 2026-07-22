# Clean OOF recompute + dip fusion for the SPATIAL legs (_sa/_sb/_sc, 37-chan inputs =
# 31 raw + 6 fold-safe ANCC spatial channels rebuilt exactly as train_gru2.py does).
# Mirrors gru_fusion_dip.py. Outputs:
#   gru_oof_spatialclean.parquet (unfused clean OOF)  /  gru_oof_spatialfused.parquet (best lam)
# RUN from ~/rogii: python gru_fusion_spatial.py     (~4min GPU / ~1h CPU)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import _gru_infer as GI
from gru_fusion import fuse, _unit_test

import sys
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAMS = (64.0, 256.0, 1024.0)
TAGS = tuple(sys.argv[sys.argv.index("--tags") + 1].split(",")) if "--tags" in sys.argv \
    else ("_sa", "_sb", "_sc")
SUF = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
SPATIAL_CHANS = ["dense_d", "dense_std", "dense_near", "pf_dense_gap",
                 "dense_confidence", "dense_known_rmse"]

_unit_test()
seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
RAW_CHANS = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
CHANS = RAW_CHANS + SPATIAL_CHANS
wells = sorted(seq["well"].unique())
fold_of = {}
for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = f

# per-fold spatial sources (each fold parquet covers ALL wells)
SPAT = {}
for f in range(5):
    fr = pd.read_parquet(f"gru_spatial_fold{f}.parquet")
    SPAT[f] = {w: {c: g[c].to_numpy() for c in ("md", "spatial_u", "dense_std", "dense_dist")}
               for w, g in fr.groupby("well", sort=False)}
    del fr
print("spatial sources loaded", flush=True)


def make_spatial_features(sample, src):
    """Copied from train_gru2.py — MUST stay identical."""
    x = sample["x"]
    d_to_end = x[:, RAW_CHANS.index("d_to_end")].astype(np.float64)
    tvt_rel = x[:, RAW_CHANS.index("tvt_rel")].astype(np.float64)
    known = x[:, RAW_CHANS.index("is_known")] > 0.5
    pf5 = x[:, RAW_CHANS.index("pf5_d")].astype(np.float64)
    md_grid = float(src["md"][-1]) - d_to_end * 1000.0
    u = np.interp(md_grid, src["md"], src["spatial_u"])
    std_i = np.interp(md_grid, src["md"], src["dense_std"])
    dist_i = np.interp(md_grid, src["md"], src["dense_dist"])
    known_pos = np.flatnonzero(known)
    if len(known_pos) == 0:
        return np.zeros((len(x), len(SPATIAL_CHANS)), dtype=np.float32)
    cut_pos = int(known_pos[-1])
    u_rel = (u - u[cut_pos]) / 10.0
    bias = float(np.median(tvt_rel[known] - u_rel[known]))
    dense_d = np.clip(u_rel + bias, -12.0, 12.0)
    known_rmse = float(np.sqrt(np.mean((dense_d[known] - tvt_rel[known]) ** 2)))
    dense_std = np.clip(std_i / 20.0, 0.0, 6.0)
    dense_near = np.exp(-np.clip(dist_i, 0.0, None) / 0.02)
    dense_confidence = np.exp(-np.clip(std_i, 0.0, None) / 30.0) * dense_near
    return np.column_stack([
        dense_d,
        dense_std,
        dense_near,
        np.clip(pf5 - dense_d, -12.0, 12.0),
        dense_confidence,
        np.full(len(x), np.clip(known_rmse, 0.0, 6.0)),
    ]).astype(np.float32)


nets = {}
for tag in TAGS:
    for f in range(5):
        ck = torch.load(f"gru_fold{f}{tag}.pt", map_location="cpu")
        assert ck.get("dip") and ck.get("spatial"), f"{tag} fold{f}: expected dip+spatial ckpt"
        assert ck["chans"] == CHANS, f"{tag} fold{f}: channel list mismatch"
        hid = ck["hid"]

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(len(CHANS), hid)
                self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.25)
                self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 2))

            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.head(h)

        m = _Net(); m.load_state_dict(ck["state"]); m.eval().to(DEV)
        nets[(tag, f)] = m
print(f"15 ckpts loaded | device={DEV}", flush=True)

cache = {}
nat = seq[seq["cut"] == "nat"]
import time
t0 = time.time()
for k, (wid, g) in enumerate(nat.groupby("well"), 1):
    g = g.sort_values("step")
    f = fold_of[wid]
    tail = g["is_tail"].values.astype(bool)
    x_raw = g[RAW_CHANS].values.astype(np.float32)
    spat = make_spatial_features({"x": x_raw}, SPAT[f][wid])
    xf = np.concatenate([x_raw, spat], axis=1)
    ps, ds = [], []
    with torch.no_grad():
        x = torch.from_numpy(xf[None]).to(DEV)
        for tag in TAGS:
            out = nets[(tag, f)](x).cpu().numpy()[0]
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
pd.concat(base_rows, ignore_index=True).to_parquet(f"gru_oof_spatialclean{SUF}.parquet", index=False)
print(f"== spatial-3leg unfused clean OOF = {pooled(base_rows):.4f}  (dip-3leg ref: 8.1241 unfused / 8.1055 fused)", flush=True)
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
    print(f"   spatial dip-fused lam={lam:4.0f}: {r:.4f}", flush=True)
    if r < best[0]:
        best = (r, lam, rows)
pd.concat(best[2], ignore_index=True).to_parquet(f"gru_oof_spatialfused{SUF}.parquet", index=False)
print(f"BEST lam={best[1]} -> gru_oof_spatialfused{SUF}.parquet ({best[0]:.4f})", flush=True)
print("DONE", flush=True)
