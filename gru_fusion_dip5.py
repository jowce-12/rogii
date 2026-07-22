# Fusion with the TRAINED dip head (radiant's actual design). Uses _da/_db/_dc ckpts
# (2-output: [tvt/10, step-increment x25]). Pre-registered lambda grid {64, 256, 1024}.
# Prints unfused/fused clean OOF; saves best-config fused OOF for the blend judge.
# RUN (isic env): python gru_fusion_dip.py        (~4min GPU)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import _gru_infer as GI
from gru_fusion import fuse, _unit_test

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAMS = (1024.0, 4096.0)
TAGS = ("_da", "_db", "_dc", "_dd", "_de")

_unit_test()
seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
wells = sorted(seq["well"].unique())
fold_of = {}
for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = f
nets = {}
for tag in TAGS:
    for f in range(5):
        ck = torch.load(f"gru_fold{f}{tag}.pt", map_location="cpu")
        assert ck.get("dip"), f"{tag} fold{f} is not a dip-head checkpoint"
        chans, hid = ck["chans"], ck["hid"]

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(len(chans), hid)
                self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.25)
                self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 2))

            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.head(h)

        m = _Net(); m.load_state_dict(ck["state"]); m.eval().to(DEV)
        nets[(tag, f)] = (m, chans)

cache = {}
nat = seq[seq["cut"] == "nat"]
for k, (wid, g) in enumerate(nat.groupby("well"), 1):
    g = g.sort_values("step")
    f = fold_of[wid]
    tail = g["is_tail"].values.astype(bool)
    ps, ds = [], []
    with torch.no_grad():
        for tag in TAGS:
            m, chans = nets[(tag, f)]
            x = torch.from_numpy(g[chans].values.astype(np.float32)[None]).to(DEV)
            out = m(x).cpu().numpy()[0]
            ps.append(out[:, 0] * 10.0)                     # ft
            ds.append(out[:, 1] / 25.0 * 10.0 / GI.G_STEP)  # ft per ft
    rm = rowmap[rowmap["well"] == wid]
    mg = (np.arange(len(g), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    cache[wid] = (np.mean(ps, 0), np.mean(ds, 0), tail, rm, mg)
    if k % 200 == 0:
        print(f"forward {k}/773", flush=True)

rm_y = rowmap[["id", "y"]].set_index("id")["y"]
def pooled(rows):
    j = pd.concat(rows, ignore_index=True).set_index("id")
    return float(np.sqrt(np.mean((j["gru_d"].values - rm_y.reindex(j.index).values) ** 2)))

base_rows = []
for wid, (p, d, tail, rm, mg) in cache.items():
    base_rows.append(pd.DataFrame({"id": rm["id"].values,
                                   "gru_d": np.interp(rm["md_rel"].values, mg, p).astype(np.float32)}))
print(f"== dip-3leg unfused clean OOF = {pooled(base_rows):.4f}  (tvt-3leg ref: unfused 8.2689 / stride-fused 8.2421)", flush=True)
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
pd.concat(best[2], ignore_index=True).to_parquet("gru_oof_dipfused5.parquet", index=False)
print(f"BEST lam={best[1]} -> gru_oof_dipfused5.parquet ({best[0]:.4f})", flush=True)
print("DONE", flush=True)
