
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from stride import load_well, grcal_tw, stride_track
import _t1_pf as PF
import _gru_infer as GI

def stride_fn(hw, tw, seg):
    pred, _ = stride_track(hw, tw, seg_len=seg)
    if pred is None:
        return None
    kn = hw[hw["TVT_input"].notna()]
    return None, pred - float(kn["TVT_input"].iloc[-1]), None

rowmap = pd.read_parquet("gru_rowmap.parquet")
ref = pd.read_parquet("gru_oof_dipfused.parquet").set_index("id")["gru_d"]
wells = sorted(rowmap["well"].unique())
fold_of = {}
for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = f
for wid in [wells[10], wells[400]]:
    hw, tw = load_well(wid, "train")
    ev = hw[hw["TVT_input"].isna()]
    pairs = PF._pf_seed_batch(hw, tw, 500, range(32))
    preds32 = np.stack([np.asarray(p, float)[ev.index.values] for p, _l in pairs], 0)
    liks32 = np.array([l for _p, l in pairs])
    ch, mg_grid, cut_md, ev_idx, last = GI.gru_channels(hw, tw, (preds32, liks32), grcal_tw, stride_fn)
    cks = [torch.load(f"gru_fold{fold_of[wid]}{t}.pt", map_location="cpu") for t in ("_da", "_db", "_dc")]
    p, dip = GI.gru_forward(ch, cks, torch, nn)
    ti = np.where(mg_grid > cut_md)[0]
    p[ti] = GI.gru_fuse(p[ti], dip[ti][:-1], 1024.0)
    mrg = (np.arange(len(mg_grid), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    rm = rowmap[rowmap["well"] == wid]
    vals = np.interp(rm["md_rel"].values, mrg, p)
    d = float(np.abs(vals - ref.reindex(rm["id"].values).values).max())
    print(f"{wid}: dip-fused parity max|diff| = {d:.4f} ft")
    assert d < 0.05, "PARITY FAILED"
print("PARITY OK")
