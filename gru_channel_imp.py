# Which input channel does the GRU actually use? A bi-GRU has no split-gain importance,
# so we measure saliency: |d(tail prediction) / d(channel)| x channel std, averaged over
# the tail steps of 150 harness wells. Scaling by the channel's own std makes channels
# with different units comparable (it answers "how much does the output move when this
# channel moves by one of ITS standard deviations").
# Folds come from gru_folds.json (never re-derived — sklearn versions disagree).
import json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import blend_eval as BE

torch.set_num_threads(4)
t0 = time.time()
FOLD = json.load(open("gru_folds.json"))
seq = pd.read_parquet("gru_seq.parquet")
CH = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
nets = {}
for f in range(5):
    ck = torch.load(f"gru_fold{f}_da.pt", map_location="cpu")
    hid, nch = ck["hid"], len(ck["chans"])
    assert ck["chans"] == CH, "channel order mismatch"
    class N(nn.Module):
        def __init__(s):
            super().__init__(); s.inp = nn.Linear(nch, hid)
            s.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.25)
            s.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 2))
        def forward(s, x):
            h, _ = s.gru(s.inp(x)); return s.head(h)
    m = N(); m.load_state_dict(ck["state"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    nets[f] = m
res, _ = BE.selector_preds(7)
wells = [r["wid"] for r in res]
nat = seq[seq["cut"] == "nat"]
sal = np.zeros(len(CH)); n_ok = 0
for k, wid in enumerate(wells, 1):
    g = nat[nat["well"] == wid].sort_values("step")
    if not len(g):
        continue
    x = torch.tensor(g[CH].values.astype(np.float32)[None], requires_grad=True)
    tail = torch.tensor(g["is_tail"].values.astype(bool))
    out = nets[FOLD[wid]](x)
    out[0, tail, 0].mean().backward()
    sal += (x.grad[0].abs().mean(0) * x.detach()[0].std(0)).numpy()
    n_ok += 1
    if k % 50 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)}", flush=True)
sal /= max(n_ok, 1)
imp = pd.DataFrame({"channel": CH, "saliency": sal})
imp["pct"] = 100 * imp["saliency"] / imp["saliency"].sum()
imp = imp.sort_values("saliency", ascending=False).reset_index(drop=True)
imp.to_csv("gru_channel_importance.csv", index=False)
print(f"\nGRU channel importance ({n_ok} wells, leg _da, tail steps)\n")
print(imp.head(18).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print("\n--- bottom 6 ---")
print(imp.tail(6).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
FAM = {"GR texture": lambda c: c.startswith("grz") or c in ("gr_nan",),
       "typewell mismatch (mm*)": lambda c: c.startswith("mm"),
       "particle filter (pf*)": lambda c: c.startswith("pf"),
       "STRIDE decodes": lambda c: c.startswith("stride"),
       "geometry / position": lambda c: c in ("dzdmd", "z_rel", "d_from_cut", "d_to_end"),
       "prefix TVT path": lambda c: c in ("tvt_rel", "is_known")}
print("\n--- by family ---")
for name, f in FAM.items():
    s = imp[imp["channel"].map(f)]["pct"].sum()
    print(f"{name:26s} {s:6.1f}%  ({int(imp['channel'].map(f).sum())} channels)")
