# Clean (unpadded, deployment-consistent) GRU OOF: per-sample forward, fold-matched
# model per leg, 3-leg mean -> gru_oof_clean.parquet. RUN (isic): python recompute_gru_oof.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import _gru_infer as GI

DEV = "cuda" if torch.cuda.is_available() else "cpu"
seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
CHANS = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
wells = sorted(seq["well"].unique())
fold_of = {}
for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = f

nets = {}
for tag in ("_pa", "_pb", "_pc"):
    for f in range(5):
        ck = torch.load(f"gru_fold{f}{tag}.pt", map_location="cpu")
        chans, hid = ck["chans"], ck["hid"]
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(len(chans), hid)
                self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.25)
                self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))
            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.head(h).squeeze(-1)
        m = _Net(); m.load_state_dict(ck["state"]); m.eval().to(DEV)
        nets[(tag, f)] = (m, chans)

rows = []
nat = seq[seq["cut"] == "nat"]
for k, (wid, g) in enumerate(nat.groupby("well"), 1):
    g = g.sort_values("step")
    f = fold_of[wid]
    rm = rowmap[rowmap["well"] == wid]
    md_rel_grid = (np.arange(len(g), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    preds = []
    with torch.no_grad():
        for tag in ("_pa", "_pb", "_pc"):
            m, chans = nets[(tag, f)]
            x = torch.from_numpy(g[chans].values.astype(np.float32)[None]).to(DEV)
            preds.append(m(x).cpu().numpy()[0] * 10.0)
    pred_ft = np.mean(preds, 0)
    rows.append(pd.DataFrame({"id": rm["id"].values,
                              "gru_d": np.interp(rm["md_rel"].values, md_rel_grid, pred_ft).astype(np.float32)}))
    if k % 150 == 0:
        print(f"{k}/773", flush=True)
oof = pd.concat(rows, ignore_index=True)
oof.to_parquet("gru_oof_clean3.parquet", index=False)
j = oof.merge(rowmap[["id", "y"]], on="id")
print(f"*** CLEAN ensemble pooled OOF = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft "
      f"(padded-eval was 8.2288) ***")
