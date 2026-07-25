# HONEST clean-OOF recompute using the PINNED training fold map (gru_folds.json, taken
# from the env that trained the checkpoints). Recomputing folds with a local sklearn is
# UNSAFE: sklearn 1.7.2 vs 1.8.0 disagree on 75.5% of wells, which silently evaluates a
# well with a model that trained on it (leakage worth ~1.7 ft).
# One pass over the wells produces BOTH deployment-relevant poles:
#   dip3 lam1024 (the LB-6.663 pole)  and  dip5 lam4096 (the patch48 pole)
# RUN (CPU, either env): python recompute_honest.py     (~25min)
import json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import _gru_infer as GI

torch.set_num_threads(8)
t0 = time.time()
FOLD = json.load(open("gru_folds.json"))
TAGS5 = ("_da", "_db", "_dc", "_dd", "_de")
seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
RAW = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
nets = {}
for t in TAGS5:
    for f in range(5):
        ck = torch.load(f"gru_fold{f}{t}.pt", map_location="cpu")
        hid, nch = ck["hid"], len(ck["chans"])
        class N(nn.Module):
            def __init__(s):
                super().__init__(); s.inp = nn.Linear(nch, hid)
                s.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.25)
                s.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 2))
            def forward(s, z):
                h, _ = s.gru(s.inp(z)); return s.head(h)
        m = N(); m.load_state_dict(ck["state"]); m.eval(); nets[(t, f)] = m
print(f"[{time.time()-t0:.0f}s] 25 ckpts loaded | folds pinned from gru_folds.json", flush=True)

rows3, rows5 = [], []
nat = seq[seq["cut"] == "nat"]
for k, (wid, g) in enumerate(nat.groupby("well"), 1):
    g = g.sort_values("step")
    f = FOLD[wid]
    tail = g["is_tail"].values.astype(bool)
    x = torch.from_numpy(g[RAW].values.astype(np.float32)[None])
    P, D = {}, {}
    with torch.no_grad():
        for t in TAGS5:
            out = nets[(t, f)](x).numpy()[0]
            P[t] = out[:, 0] * 10.0
            D[t] = out[:, 1] / 25.0 * 10.0 / GI.G_STEP
    rm = rowmap[rowmap["well"] == wid]
    mg = (np.arange(len(g), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    ti = np.where(tail)[0]
    for tags, lam, store in ((TAGS5[:3], 1024.0, rows3), (TAGS5, 4096.0, rows5)):
        p = np.mean([P[t] for t in tags], 0).copy()
        d = np.mean([D[t] for t in tags], 0)
        if len(ti) > 2:
            p[ti] = GI.gru_fuse(p[ti], d[ti][:-1], lam)
        store.append(pd.DataFrame({"id": rm["id"].values,
                                   "gru_d": np.interp(rm["md_rel"].values, mg, p).astype(np.float32)}))
    if k % 100 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/773", flush=True)

y = rowmap[["id", "y"]].set_index("id")["y"]
for name, store in (("gru_oof_dip3_honest.parquet", rows3), ("gru_oof_dip5_honest.parquet", rows5)):
    df = pd.concat(store, ignore_index=True)
    df.to_parquet(name, index=False)
    j = df.set_index("id")
    print(f"{name}: clean OOF = {float(np.sqrt(np.mean((j['gru_d'].values - y.reindex(j.index).values) ** 2))):.4f}",
          flush=True)
print("DONE", flush=True)
