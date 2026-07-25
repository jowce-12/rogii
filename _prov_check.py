# Provenance/parity probe: recompute 3 wells' clean dip forwards and compare against the
# stored judgment artifacts. Run it in ANY env (CPU enforced) to prove numerical identity.
import numpy as np, pandas as pd, torch, torch.nn as nn, sys
from sklearn.model_selection import GroupKFold
import _gru_infer as GI
from gru_fusion import fuse
torch.set_num_threads(4)
print(f"env: python {sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
seq = pd.read_parquet("gru_seq.parquet"); rowmap = pd.read_parquet("gru_rowmap.parquet")
RAW = [c for c in seq.columns if c not in ("well","cut","step","is_tail","target")]
wells = sorted(seq["well"].unique()); fold_of = {}
for f,(_,va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va: fold_of[wells[i]] = f
S = {"dip5": (("_da","_db","_dc","_dd","_de"), 4096.0,
              pd.read_parquet("gru_oof_dipfused5.parquet").set_index("id")["gru_d"]),
     "dip3": (("_da","_db","_dc"), 1024.0,
              pd.read_parquet("gru_oof_dipfused.parquet").set_index("id")["gru_d"])}
nat = seq[seq["cut"]=="nat"]
worst = 0.0
for wid in wells[:3]:
    g = nat[nat["well"]==wid].sort_values("step"); f = fold_of[wid]
    tail = g["is_tail"].values.astype(bool)
    x = torch.from_numpy(g[RAW].values.astype(np.float32)[None])
    for nm,(tags,lam,store) in S.items():
        ps, ds = [], []
        for t in tags:
            ck = torch.load(f"gru_fold{f}{t}.pt", map_location="cpu")
            hid = ck["hid"]; nch = len(ck["chans"])
            class N(nn.Module):
                def __init__(s):
                    super().__init__(); s.inp=nn.Linear(nch,hid)
                    s.gru=nn.GRU(hid,hid,num_layers=2,batch_first=True,bidirectional=True,dropout=0.25)
                    s.head=nn.Sequential(nn.Linear(2*hid,hid),nn.GELU(),nn.Linear(hid,2))
                def forward(s,z):
                    h,_=s.gru(s.inp(z)); return s.head(h)
            m=N(); m.load_state_dict(ck["state"]); m.eval()
            with torch.no_grad(): out=m(x).numpy()[0]
            ps.append(out[:,0]*10.0); ds.append(out[:,1]/25.0*10.0/GI.G_STEP)
        p=np.mean(ps,0); d=np.mean(ds,0); ti=np.where(tail)[0]
        if len(ti)>2: p[ti]=fuse(p[ti], d[ti][:-1], lam)
        rm=rowmap[rowmap["well"]==wid]
        mg=(np.arange(len(g),dtype=np.float64)-GI.G_CTX)*GI.G_STEP
        diff=float(np.nanmax(np.abs(np.interp(rm["md_rel"].values, mg, p)
                                    - store.reindex(rm["id"].values).values)))
        worst=max(worst,diff)
        print(f"  {wid} {nm}: max|recompute - stored| = {diff:.6f} ft", flush=True)
print(f"WORST {worst:.6f} ft -> {'MATCH (same numerics)' if worst < 0.01 else 'MISMATCH'}")
