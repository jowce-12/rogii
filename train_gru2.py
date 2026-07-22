# Whole-well bi-GRU v2 (radiantallomancer recipe; NOT the dead Option-C train_gru.py —
# that one windowed rows with likpf features; this one is whole-well samples with
# prefix-cut augmentation, typewell-mismatch channels, and per-cut stride decodes).
# 5-fold GroupKFold by WELL (artificial cuts inherit the parent fold); loss on tail
# steps only; validation = natural-cut ORIGINAL eval rows, pooled RMSE in ft (grid
# predictions interpolated back to row MDs via gru_rowmap.parquet).
# Outputs: gru_oof.parquet (id, gru_d), gru_fold{k}.pt, gru_meta.json.
# RUN (isic env): python train_gru2.py          (~1-2h on GPU)
import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

import sys
def _arg(name, default, cast):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
SEED = _arg("--seed", 42, int)
HID = _arg("--hid", 128, int)
TAG = _arg("--tag", "", str)
DROPOUT = _arg("--dropout", 0.25, float)
DIP = _arg("--dip", 0, int)   # 1 = second head predicts per-step increment x25
SPATIAL = _arg("--spatial", 1, int)   # 1 = append fold-safe ANCC surface channels (default ON)
SPATIAL2 = _arg("--spatial2", 0, int)  # 1 = multi-surface consensus channels (needs gru_spatial2_fold*, implies --spatial 1)
WELLW = _arg("--wellw", 0.0, float)   # >0: upweight high-risk wells (1 + WELLW*ramp(likpf_ptstd wellmean, thr 3.39, /2))
CHDROP = _arg("--chdrop", 0.0, float)  # >0: per-sample prob of zeroing one evidence channel GROUP (stride/pf/spatial/mm)
DROPCUTS = set(x for x in _arg("--dropcuts", "", str).split(",") if x)  # e.g. "c40" to drop that aug cut
EPOCHS = 30
BATCH = 12
LR = 1e-3
CTX = 256          # must match build_gru_data.py
STEP = 4.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

t0 = time.time()
seq = pd.read_parquet("gru_seq.parquet")
rowmap = pd.read_parquet("gru_rowmap.parquet")
RAW_CHANS = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
if SPATIAL2:
    assert SPATIAL, "--spatial2 1 requires --spatial 1"
SPATIAL_CHANS = ["dense_d", "dense_std", "dense_near", "pf_dense_gap",
                 "dense_confidence", "dense_known_rmse"] if SPATIAL else []
SPATIAL2_CHANS = ["densex_med", "densex_spread", "densex_best", "densex_best_rmse"] if SPATIAL2 else []
EXTRA_SURFACES = ["astnu", "astnl", "egfdu", "egfdl", "buda"]
CHANS = RAW_CHANS + SPATIAL_CHANS + SPATIAL2_CHANS
print(f"[{time.time()-t0:.0f}s] {len(seq)} steps | {len(RAW_CHANS)}+{len(SPATIAL_CHANS)}+{len(SPATIAL2_CHANS)} chans | device={DEV}", flush=True)


def make_spatial_features(sample, src):
    """6 spatial channels for one sample, re-anchored on ITS known prefix (cut-honest).
    src = per-well dict from gru_spatial_fold{f}.parquet (md, spatial_u, dense_std, dense_dist)."""
    x = sample["x"]
    d_to_end = x[:, RAW_CHANS.index("d_to_end")].astype(np.float64)
    tvt_rel = x[:, RAW_CHANS.index("tvt_rel")].astype(np.float64)
    known = x[:, RAW_CHANS.index("is_known")] > 0.5
    pf5 = x[:, RAW_CHANS.index("pf5_d")].astype(np.float64)
    md_grid = float(src["md"][-1]) - d_to_end * 1000.0
    u = np.interp(md_grid, src["md"], src["spatial_u"])
    std_i = np.interp(md_grid, src["md"], src["dense_std"])
    dist_i = np.interp(md_grid, src["md"], src["dense_dist"])
    known_pos = np.flatnonzero(known)
    if len(known_pos) == 0:
        return np.zeros((len(x), len(SPATIAL_CHANS) + len(SPATIAL2_CHANS)), dtype=np.float32)
    cut_pos = int(known_pos[-1])
    u_rel = (u - u[cut_pos]) / 10.0
    bias = float(np.median(tvt_rel[known] - u_rel[known]))
    dense_d = np.clip(u_rel + bias, -12.0, 12.0)
    known_rmse = float(np.sqrt(np.mean((dense_d[known] - tvt_rel[known]) ** 2)))
    dense_std = np.clip(std_i / 20.0, 0.0, 6.0)
    dense_near = np.exp(-np.clip(dist_i, 0.0, None) / 0.02)
    dense_confidence = np.exp(-np.clip(std_i, 0.0, None) / 30.0) * dense_near
    cols = [
        dense_d,
        dense_std,
        dense_near,
        np.clip(pf5 - dense_d, -12.0, 12.0),
        dense_confidence,
        np.full(len(x), np.clip(known_rmse, 0.0, 6.0)),
    ]
    if SPATIAL2:
        # per-surface anchored deltas (ANCC included as a vote) -> consensus channels
        ds, rmses = [dense_d], [known_rmse]
        for s in EXTRA_SURFACES:
            u_s = src.get(f"u_{s}")
            if u_s is None:
                continue
            u_i = np.interp(md_grid, src["md"], u_s)
            if not np.isfinite(u_i).all():
                continue
            u_rel_i = (u_i - u_i[cut_pos]) / 10.0
            bias_i = float(np.median(tvt_rel[known] - u_rel_i[known]))
            d_i = np.clip(u_rel_i + bias_i, -12.0, 12.0)
            ds.append(d_i)
            rmses.append(float(np.sqrt(np.mean((d_i[known] - tvt_rel[known]) ** 2))))
        D = np.stack(ds, 0)
        best = int(np.argmin(rmses))
        cols += [
            np.median(D, 0),
            np.clip(D.std(0), 0.0, 6.0) if len(ds) > 1 else np.zeros(len(x)),
            D[best],
            np.full(len(x), np.clip(rmses[best], 0.0, 6.0)),
        ]
    return np.column_stack(cols).astype(np.float32)

samples = []
for (w, cut), g in seq.groupby(["well", "cut"], sort=True):
    g = g.sort_values("step")
    samples.append(dict(well=w, cut=cut,
                        x=g[RAW_CHANS].values.astype(np.float32),
                        y=g["target"].values.astype(np.float32),
                        tail=g["is_tail"].values.astype(bool)))
if not SPATIAL:
    for s in samples:
        s["xf"] = s["x"]
if DROPCUTS:
    n0 = len(samples)
    samples = [s for s in samples if s["cut"] not in DROPCUTS]
    print(f"[{time.time()-t0:.0f}s] dropcuts {sorted(DROPCUTS)}: {n0} -> {len(samples)} samples", flush=True)
wells = sorted({s["well"] for s in samples})
fold_of = {}
for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va:
        fold_of[wells[i]] = f
rm_by_well = {w: g for w, g in rowmap.groupby("well")}
if WELLW > 0:
    _risk = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                            columns=["well", "likpf_ptstd"]).groupby("well")["likpf_ptstd"].mean()
    for s in samples:
        _r = float(_risk.get(s["well"], np.nan))
        s["w"] = 1.0 + WELLW * float(np.clip((_r - 3.39) / 2.0, 0.0, 1.0)) if np.isfinite(_r) else 1.0
    print(f"[{time.time()-t0:.0f}s] wellw on: mean weight "
          f"{np.mean([s.get('w', 1.0) for s in samples]):.3f}", flush=True)
print(f"[{time.time()-t0:.0f}s] {len(samples)} samples, {len(wells)} wells", flush=True)


class GRUNet(nn.Module):
    def __init__(self, nin, hid=HID):
        super().__init__()
        self.inp = nn.Linear(nin, hid)
        self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                  nn.Linear(hid, 2 if DIP else 1))

    def forward(self, x, lengths=None):
        # NOTE (2026-07-20): pack_padded was tried and REVERTED — epoch time regressed in
        # this env and 6.734 LB was achieved without it. Batched eval therefore carries
        # bi-GRU padding contamination; the OFFICIAL OOF and deployment both use the
        # clean per-sample forward (recompute_gru_oof.py / notebook), so numbers stay honest.
        h, _ = self.gru(self.inp(x))
        out = self.head(h)
        return out if DIP else out.squeeze(-1)


rng = np.random.default_rng(SEED)
GR_IDX = [i for i, c in enumerate(CHANS) if c.startswith("grz")]
CH_GROUPS = [g for g in (
    [i for i, c in enumerate(CHANS) if c.startswith("stride")],
    [i for i, c in enumerate(CHANS) if c.startswith("pf") and c != "pf_dense_gap"],
    [i for i, c in enumerate(CHANS) if c.startswith(("dense", "densex")) or c == "pf_dense_gap"],
    [i for i, c in enumerate(CHANS) if c.startswith("mm")],
) if g]


def batches(idxs, shuffle):
    order = sorted(idxs, key=lambda i: len(samples[i]["y"]))
    chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        T = max(len(samples[i]["y"]) for i in ch)
        x = np.zeros((len(ch), T, len(CHANS)), np.float32)
        y = np.zeros((len(ch), T), np.float32)
        m = np.zeros((len(ch), T), bool)
        lens = np.zeros(len(ch), np.int64)
        wv = np.ones(len(ch), np.float32)
        for b, i in enumerate(ch):
            s = samples[i]; L = len(s["y"])
            x[b, :L] = s["xf"]; y[b, :L] = s["y"]; m[b, :L] = s["tail"]; lens[b] = L
            wv[b] = s.get("w", 1.0)
        yield (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m),
               torch.from_numpy(lens), torch.from_numpy(wv), ch)


def eval_rows(model, idxs):
    """Natural-cut original-row pooled squared error (ft) via grid->row interp."""
    model.eval()
    se = n = 0.0
    preds = []
    with torch.no_grad():
        for x, y, m, lens, _wv, ch in batches(idxs, shuffle=False):
            _o = model(x.to(DEV)).float().cpu().numpy()
            p = _o[..., 0] if DIP else _o
            for b, i in enumerate(ch):
                s = samples[i]
                rm = rm_by_well[s["well"]]
                L = len(s["y"])
                md_rel_grid = (np.arange(L, dtype=np.float64) - CTX) * STEP
                pred_ft = np.interp(rm["md_rel"].values, md_rel_grid, p[b, :L] * 10.0)
                se += float(((pred_ft - rm["y"].values) ** 2).sum()); n += len(rm)
                preds.append(pd.DataFrame({"id": rm["id"].values,
                                           "gru_d": pred_ft.astype(np.float32)}))
    return (se / max(n, 1)) ** 0.5, preds


oof_rows = []
meta = dict(chans=CHANS, hid=HID, seed=SEED, tag=TAG, epochs=EPOCHS, ctx=CTX, step=STEP, folds={})
for fold in range(5):
    if SPATIAL:
        spatial_path = (f"gru_spatial2_fold{fold}.parquet" if SPATIAL2
                        else f"gru_spatial_fold{fold}.parquet")
        if not os.path.exists(spatial_path):
            raise FileNotFoundError(f"{spatial_path} not found; run "
                                    f"{'make_spatial2.py' if SPATIAL2 else 'make_spatial_features.py'} first")
        spatial_frame = pd.read_parquet(spatial_path)
        src_cols = ["md", "spatial_u", "dense_std", "dense_dist"]
        if SPATIAL2:
            src_cols += [f"u_{s}" for s in EXTRA_SURFACES]
        spatial_by_well = {
            w: {c: g[c].to_numpy() for c in src_cols}
            for w, g in spatial_frame.groupby("well", sort=False)
        }
        del spatial_frame
        # fold-specific channels: rebuild the concatenated input for EVERY sample this fold
        for s in samples:
            s["xf"] = np.concatenate([s["x"], make_spatial_features(s, spatial_by_well[s["well"]])],
                                     axis=1)
        del spatial_by_well
        print(f"[{time.time()-t0:.0f}s] fold{fold}: spatial features ready", flush=True)
    tr_idx = [i for i, s in enumerate(samples) if fold_of[s["well"]] != fold]
    va_idx = [i for i, s in enumerate(samples) if fold_of[s["well"]] == fold and s["cut"] == "nat"]
    model = GRUNet(len(CHANS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    best = (1e9, None)
    for ep in range(EPOCHS):
        model.train()
        tot = cnt = 0.0
        for x, y, m, lens, wv, _ in batches(tr_idx, shuffle=True):
            x, y, m, wv = x.to(DEV), y.to(DEV), m.to(DEV), wv.to(DEV)
            if GR_IDX:
                x[:, :, GR_IDX] = x[:, :, GR_IDX] * (1 + 0.05 * torch.randn(x.shape[0], 1, 1, device=DEV))                                   + 0.10 * torch.randn(x.shape[0], 1, 1, device=DEV)
            if CHDROP > 0:
                for _b in range(x.shape[0]):
                    if float(torch.rand(1)) < CHDROP:
                        _gi = CH_GROUPS[int(torch.randint(len(CH_GROUPS), (1,)))]
                        x[_b, :, _gi] = 0.0
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if DEV == "cuda" else "cpu",
                                enabled=DEV == "cuda"):
                out = model(x)
                mw = m * wv[:, None]
                if DIP:
                    p = out[..., 0]
                    dpred = out[..., 1]
                    dip_t = (y[:, 1:] - y[:, :-1]) * 25.0
                    dip_mw = (m[:, 1:] & m[:, :-1]) * wv[:, None]
                    loss = (((p - y) ** 2 * mw).sum() / mw.sum().clamp(min=1e-6)
                            + ((dpred[:, :-1] - dip_t) ** 2 * dip_mw).sum() / dip_mw.sum().clamp(min=1e-6))
                else:
                    p = out
                    loss = ((p - y) ** 2 * mw).sum() / mw.sum().clamp(min=1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()) * float(m.sum()); cnt += float(m.sum())
        sched.step()
        rmse, _ = eval_rows(model, va_idx)
        if rmse < best[0]:
            best = (rmse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if ep % 3 == 2 or ep == EPOCHS - 1:
            print(f"[{time.time()-t0:.0f}s] fold{fold} ep{ep+1} "
                  f"train={((tot/max(cnt,1))**0.5)*10:.3f}ft val={rmse:.4f}ft best={best[0]:.4f}", flush=True)
    model.load_state_dict(best[1])
    torch.save({"state": best[1], "chans": CHANS, "hid": HID, "ctx": CTX, "step": STEP,
                "dip": bool(DIP), "spatial": bool(SPATIAL), "spatial2": bool(SPATIAL2)},
               f"gru_fold{fold}{TAG}.pt")
    meta["folds"][str(fold)] = best[0]
    _, preds = eval_rows(model, va_idx)
    oof_rows.extend(preds)
    print(f"fold{fold} DONE best={best[0]:.4f}ft", flush=True)

oof = pd.concat(oof_rows, ignore_index=True)
oof.to_parquet(f"gru_oof{TAG}.parquet", index=False)
j = oof.merge(rowmap[["id", "y"]], on="id")
pooled = float(np.sqrt(np.mean((j["gru_d"] - j["y"]) ** 2)))
print(f"\n*** GRU pooled OOF (natural-cut original rows) = {pooled:.4f} ft ***", flush=True)
meta["pooled_oof"] = pooled
json.dump(meta, open(f"gru_meta{TAG}.json", "w"), indent=1)
print(f"DONE [{time.time()-t0:.0f}s] -> gru_oof.parquet, gru_fold*.pt, gru_meta.json", flush=True)
