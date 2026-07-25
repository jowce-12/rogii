# TEST-TIME SELF-SUPERVISION for the GRU, tested offline with data we already have.
# The GRU leans hardest on the prefix (dzdmd 21.6% + tvt_rel 12.5% of saliency), and a
# test well's prefix TVT is GIVEN — so we can cut that prefix artificially, let the model
# predict the hidden slice, and grade it against known TVT. Question: does the error the
# model makes on THAT slice predict the error it makes in the real eval zone?
# gru_seq.parquet already carries c40..c90 cut samples for every train well, so the whole
# idea is testable without recomputing a single particle filter.
import json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import blend_eval as BE

torch.set_num_threads(4)
CUT = "c40"                      # hide the last ~10% of the prefix
TAGS = ("_da", "_db", "_dc")
t0 = time.time()
FOLD = json.load(open("gru_folds.json"))
seq = pd.read_parquet("gru_seq.parquet")
CH = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
nets = {}
for tag in TAGS:
    for f in range(5):
        ck = torch.load(f"gru_fold{f}{tag}.pt", map_location="cpu")
        hid, nch = ck["hid"], len(ck["chans"])
        class N(nn.Module):
            def __init__(s):
                super().__init__(); s.inp = nn.Linear(nch, hid)
                s.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.25)
                s.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 2))
            def forward(s, x):
                h, _ = s.gru(s.inp(x)); return s.head(h)
        m = N(); m.load_state_dict(ck["state"]); m.eval()
        nets[(tag, f)] = m

res7, _ = BE.selector_preds(7); res11, _ = BE.selector_preds(11)
wells = sorted({r["wid"] for r in res7} | {r["wid"] for r in res11})
D_TO_END = CH.index("d_to_end")

def predict(g, f):
    x = torch.from_numpy(g[CH].values.astype(np.float32)[None])
    with torch.no_grad():
        return np.mean([nets[(t, f)](x).numpy()[0][:, 0] * 10.0 for t in TAGS], 0)

rows = []
for k, wid in enumerate(wells, 1):
    gc = seq[(seq["well"] == wid) & (seq["cut"] == CUT)].sort_values("step")
    gn = seq[(seq["well"] == wid) & (seq["cut"] == "nat")].sort_values("step")
    if not len(gc) or not len(gn):
        continue
    f = FOLD[wid]
    pc, pn = predict(gc, f), predict(gn, f)
    tc = gc["target"].values * 10.0
    tn = gn["target"].values * 10.0
    tail_c = gc["is_tail"].values.astype(bool)
    tail_n = gn["is_tail"].values.astype(bool)
    # absolute MD offsets: d_to_end is (end_md - md_grid)/1000 in both samples
    dte_c = gc[CH[D_TO_END]].values * 1000.0
    dte_n = gn[CH[D_TO_END]].values * 1000.0
    real_cut_dte = dte_n[tail_n][0]                 # distance-to-end at the REAL cut
    # self-supervised slice = hidden by the artificial cut but BEFORE the real cut
    ss = tail_c & (dte_c > real_cut_dte)
    if ss.sum() < 20 or tail_n.sum() < 20:
        continue
    rows.append(dict(well=wid,
                     bias_ss=float(np.mean(pc[ss] - tc[ss])),
                     rmse_ss=float(np.sqrt(np.mean((pc[ss] - tc[ss]) ** 2))),
                     n_ss=int(ss.sum()),
                     bias_real=float(np.mean(pn[tail_n] - tn[tail_n])),
                     rmse_real=float(np.sqrt(np.mean((pn[tail_n] - tn[tail_n]) ** 2)))))
    if k % 60 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)}", flush=True)
R = pd.DataFrame(rows)
R.to_parquet("gru_ttt_probe.parquet", index=False)
print(f"\nwells {len(R)} | self-supervised slice median {R.n_ss.median():.0f} steps", flush=True)
print(f"corr(bias on the known slice, bias in the real eval zone) = "
      f"{R[['bias_ss','bias_real']].corr().iloc[0,1]:.3f}", flush=True)
print(f"corr(|bias_ss|, rmse_real) = {R[['bias_ss','rmse_real']].apply(lambda c: c.abs()).corr().iloc[0,1]:.3f}",
      flush=True)
print(f"\nGRU pooled-ish RMSE in the real zone = {np.sqrt(np.average(R.rmse_real**2)):.3f}", flush=True)
for k_ in (0.25, 0.5, 0.75, 1.0):
    # what a per-well constant correction would have achieved (approximate, well-level)
    corrected = np.sqrt(np.average(R.rmse_real**2 - 2*k_*R.bias_ss*R.bias_real + (k_*R.bias_ss)**2))
    print(f"  minus {k_:.2f} x bias_ss -> {corrected:.3f}", flush=True)
print("DONE", flush=True)
