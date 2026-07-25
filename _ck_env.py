# Reproduce the Kaggle preview GRU checksum (11906.4454 / n=14151) in ANY env.
# Deployment path: _gru_infer.gru_channels on the TEST csvs + 25 dip ckpts + fuse lam4096.
import glob, numpy as np, pandas as pd, torch, torch.nn as nn, sys
from stride import grcal_tw, stride_track
import _t1_pf as PF
import _gru_infer as GI
torch.set_num_threads(4)
def stride_fn(hw, tw, seg):
    pred, _ = stride_track(hw, tw, seg_len=seg)
    if pred is None: return None
    kn = hw[hw["TVT_input"].notna()]
    return None, pred - float(kn["TVT_input"].iloc[-1]), None
cks = [torch.load(f, map_location="cpu") for f in sorted(glob.glob("gru_fold*_d[abcde].pt"))]
vals = []
for wid in ("000d7d20","00bbac68","00e12e8b"):
    hw = pd.read_csv(f"test/{wid}__horizontal_well.csv"); tw = pd.read_csv(f"test/{wid}__typewell.csv")
    ev = hw[hw["TVT_input"].isna()]
    pairs = PF._pf_seed_batch(hw, tw, 500, range(32))
    preds32 = np.stack([np.asarray(p, float)[ev.index.values] for p,_l in pairs], 0)
    liks32 = np.array([l for _p,l in pairs])
    ch, mg, cut_md, ev_idx, last = GI.gru_channels(hw, tw, (preds32, liks32), grcal_tw, stride_fn)
    p, dip = GI.gru_forward(ch, cks, torch, nn)
    ti = np.where(mg > cut_md)[0]
    p[ti] = GI.gru_fuse(p[ti], dip[ti][:-1], 4096.0)
    mrg = (np.arange(len(mg), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    ev_md = hw.loc[ev_idx, "MD"].values.astype(float) - cut_md
    vals.extend((np.interp(ev_md, mrg, p) + last).tolist())
print(f"env torch {torch.__version__} cuda={torch.cuda.is_available()} | "
      f"checksum mean={np.mean(vals):.4f} n={len(vals)}   (Kaggle preview: 11906.4454 n=14151)")
