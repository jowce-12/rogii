# Parity proof: _gru_infer.gru_channels/forward vs training artifacts (gru_seq/gru_oof_a)
# on 3 train wells. Channels must match ~1e-5; forward vs OOF ~1e-2 ft (cuda/cpu fp).
# RUN: python parity_gru.py [--forward]   (--forward needs torch => isic env)
import sys
import numpy as np
import pandas as pd
from stride import load_well, grcal_tw, stride_track
import _t1_pf as PF
import _gru_infer as GI

def stride_fn(hw, tw, seg):
    pred, _ = stride_track(hw, tw, seg_len=seg)
    if pred is None:
        return None
    kn = hw[hw["TVT_input"].notna()]
    return None, pred - float(kn["TVT_input"].iloc[-1]), None

seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
wells = sorted(seq["well"].unique())
TEST_WELLS = [wells[0], wells[250], wells[600]]
CHANS = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]

for wid in TEST_WELLS:
    hw, tw = load_well(wid, "train")
    ev = hw[hw["TVT_input"].isna()]
    pairs = PF._pf_seed_batch(hw, tw, 500, range(32))
    preds32 = np.stack([np.asarray(p, float)[ev.index.values] for p, _ll in pairs], 0)
    liks32 = np.array([ll for _p, ll in pairs])
    r = GI.gru_channels(hw, tw, (preds32, liks32), grcal_tw, stride_fn)
    assert r is not None
    ch, md_grid, cut_md, ev_idx, last = r
    ref = seq[(seq["well"] == wid) & (seq["cut"] == "nat")].sort_values("step")
    assert len(ref) == len(md_grid), (len(ref), len(md_grid))
    worst = 0.0
    for c in CHANS:
        d = float(np.abs(np.asarray(ch[c], np.float32) - ref[c].values).max())
        worst = max(worst, d)
    print(f"{wid}: channel parity max|diff| = {worst:.2e} over {len(CHANS)} chans")
    assert worst < 1e-4, f"channel parity FAILED {worst}"

if "--forward" in sys.argv:
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    fold_of = {}
    for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
        for i in va:
            fold_of[wells[i]] = f
    oof_a = pd.read_parquet("gru_oof_a.parquet").set_index("id")["gru_d"]
    for wid in TEST_WELLS:
        hw, tw = load_well(wid, "train")
        ev = hw[hw["TVT_input"].isna()]
        pairs = PF._pf_seed_batch(hw, tw, 500, range(32))
        preds32 = np.stack([np.asarray(p, float)[ev.index.values] for p, _ll in pairs], 0)
        liks32 = np.array([ll for _p, ll in pairs])
        ch, md_grid, cut_md, ev_idx, last = GI.gru_channels(hw, tw, (preds32, liks32), grcal_tw, stride_fn)
        ck = torch.load(f"gru_fold{fold_of[wid]}_a.pt", map_location="cpu")
        pred_ft = GI.gru_forward(ch, [ck], torch, nn)
        rm = rowmap[rowmap["well"] == wid]
        md_rel_grid = (np.arange(len(md_grid), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
        row_pred = np.interp(rm["md_rel"].values, md_rel_grid, pred_ft)
        ref = oof_a.reindex(rm["id"].values).values
        d = float(np.abs(row_pred - ref).max())
        print(f"{wid}: forward parity max|diff| = {d:.4f} ft (fold {fold_of[wid]})")
        assert d < 0.05, f"forward parity FAILED {d}"
print("PARITY OK")
