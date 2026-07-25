import os
# Parity for the SPATIAL deployment path: live spatial_bank/spatial_query + gru_channels
# (spatial_src) + gru_forward(_sa/_sb/_sc) + fuse(1024) vs training artifacts
# (gru_spatial_fold{f}.parquet src values, gru_oof_spatialfused.parquet outputs).
# Uses FOLD-restricted banks so the comparison is exact; deployment uses the all-wells bank.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
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
ref = pd.read_parquet("gru_oof_spatialfused.parquet").set_index("id")["gru_d"]
wells = sorted(rowmap["well"].unique())
fold_of = {}
_FOLD_FILE = "gru_folds.json"
if os.path.exists(_FOLD_FILE):
    # PINNED: sklearn 1.7.2 vs 1.8.0 disagree on 75.5% of GroupKFold assignments, which
    # silently evaluates a well with a model that trained on it. The canonical map comes
    # from the env that trained the checkpoints; never re-derive it locally.
    import json as _fjson
    fold_of = {k: int(v) for k, v in _fjson.load(open(_FOLD_FILE)).items()}
    print(f"[folds] pinned from {_FOLD_FILE} ({len(fold_of)} wells)", flush=True)
else:
    for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
        for i in va:
            fold_of[wells[i]] = f
    import json as _fjson
    _fjson.dump(fold_of, open(_FOLD_FILE, "w"))
    print(f"[folds] derived and WROTE {_FOLD_FILE} ({len(fold_of)} wells)", flush=True)

for wid in [wells[10], wells[400]]:
    f = fold_of[wid]
    bank = GI.spatial_bank(Path("train"), wells={w for w in wells if fold_of[w] != f})
    hw, tw = load_well(wid, "train")
    src = GI.spatial_query(bank, wid, hw)
    # query-math parity vs the fold parquet the trainer consumed
    fr = pd.read_parquet(f"gru_spatial_fold{f}.parquet")
    fr = fr[fr["well"] == wid]
    for col, live in (("spatial_u", src["spatial_u"]), ("dense_std", src["dense_std"]),
                      ("dense_dist", src["dense_dist"])):
        d = float(np.abs(np.asarray(live, np.float32) - fr[col].values).max())
        print(f"{wid} src {col}: max|diff|={d:.6f}")
        assert d < 1e-3, f"src parity failed on {col}"
    ev = hw[hw["TVT_input"].isna()]
    pairs = PF._pf_seed_batch(hw, tw, 500, range(32))
    preds32 = np.stack([np.asarray(p, float)[ev.index.values] for p, _l in pairs], 0)
    liks32 = np.array([l for _p, l in pairs])
    ch, mg_grid, cut_md, ev_idx, last = GI.gru_channels(hw, tw, (preds32, liks32),
                                                        grcal_tw, stride_fn, spatial_src=src)
    cks = [torch.load(f"gru_fold{f}{t}.pt", map_location="cpu") for t in ("_sa", "_sb", "_sc")]
    p, dip = GI.gru_forward(ch, cks, torch, nn)
    ti = np.where(mg_grid > cut_md)[0]
    p[ti] = GI.gru_fuse(p[ti], dip[ti][:-1], 1024.0)
    mrg = (np.arange(len(mg_grid), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
    rm = rowmap[rowmap["well"] == wid]
    vals = np.interp(rm["md_rel"].values, mrg, p)
    d = float(np.abs(vals - ref.reindex(rm["id"].values).values).max())
    print(f"{wid}: spatial-fused parity max|diff| = {d:.4f} ft")
    assert d < 0.05, "PARITY FAILED"
print("PARITY OK")
