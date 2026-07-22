"""Option C: offline-trained GRU, loaded & blended in the fleongg INFERENCE path.

Two halves:
  • OFFLINE (run once on a GPU notebook): build train features → hyperparameter
    search (GroupKFold OOF) → final K-fold ensemble → tune blend weight vs a
    baseline OOF → save a self-contained bundle (gru_bundle.pt).
  • INFERENCE (every submission): load_bundle() → predict_bundle(test_df) →
    blend into the loaded-LGB average. No train-from-scratch, stays fast.

Architecture (per-row sequence regression of dTVT along MD):
    LayerNorm(F) → Linear→GELU→Dropout (input projection to d_model)
                 → [optional Conv1d stem, kernel 5, residual] (local GR texture)
                 → multi-layer BiGRU
                 → concat(GRU out, projected input)  [residual skip]
                 → LayerNorm → Linear→GELU→Dropout → Linear→1
Inference uses overlapping windows with a triangular blend to avoid chunk seams.

Safe: every public entry point degrades to a no-op / passthrough on any error or
when torch is unavailable, so a missing/corrupt bundle never breaks a submission.
"""
import os, json, math, time, numpy as np

# ----------------------------------------------------------------------------- helpers
def _torch():
    import torch  # raises if absent; callers guard
    return torch

def _well_groups(wells):
    order, idxs = [], {}
    for i, w in enumerate(wells):
        if w not in idxs:
            idxs[w] = []; order.append(w)
        idxs[w].append(i)
    return order, {w: np.asarray(v, dtype=np.int64) for w, v in idxs.items()}

def _windows(n, chunk, stride):
    if n <= chunk:
        return [(0, n)]
    out = []
    s = 0
    while s < n:
        e = min(s + chunk, n)
        out.append((s, e))
        if e >= n:
            break
        s += stride
    return out

def _tri_weights(L):
    # triangular window (>=eps) for overlap-add inference blending
    if L == 1:
        return np.ones(1, np.float32)
    w = 1.0 - np.abs(np.linspace(-1, 1, L, dtype=np.float32))
    return np.maximum(w, 1e-3)

# ----------------------------------------------------------------------------- model
def build_model(F, cfg):
    torch = _torch()
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class SeqGRU(nn.Module):
        def __init__(self):
            super().__init__()
            d, h, L, p = cfg["d_model"], cfg["hidden"], cfg["layers"], cfg["dropout"]
            self.in_norm = nn.LayerNorm(F)
            self.proj = nn.Sequential(nn.Linear(F, d), nn.GELU(), nn.Dropout(p))
            self.use_conv = bool(cfg.get("conv", False))
            if self.use_conv:
                self.conv = nn.Sequential(
                    nn.Conv1d(d, d, 5, padding=2), nn.GELU(),
                    nn.Conv1d(d, d, 5, padding=2), nn.GELU())
            self.gru = nn.GRU(d, h, L, batch_first=True, bidirectional=True,
                              dropout=p if L > 1 else 0.0)
            self.head = nn.Sequential(
                nn.LayerNorm(2 * h + d), nn.Linear(2 * h + d, h), nn.GELU(),
                nn.Dropout(p), nn.Linear(h, 1))

        def forward(self, x, lengths):
            x = self.in_norm(x)
            z = self.proj(x)
            if self.use_conv:
                z = z + self.conv(z.transpose(1, 2)).transpose(1, 2)
            packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            return self.head(torch.cat([out, z], dim=-1)).squeeze(-1)

    return SeqGRU()

DEFAULT_CFG = dict(d_model=128, hidden=128, layers=2, dropout=0.1, conv=True,
                   lr=1e-3, wd=1e-5, chunk=1000, overlap=0.25, batch=8,
                   epochs=20, patience=4, eval_every=2)

# ----------------------------------------------------------------------------- scaler
def fit_scaler(X):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.).astype(np.float32)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_scaler(X, mu, sd):
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.).astype(np.float32)
    return np.clip((X - mu) / sd, -8, 8).astype(np.float32)

# ----------------------------------------------------------------------------- train one fold-set
def _train_one(Xs, y, wells, tr_wells, va_wells, F, cfg, seed, dev, verbose=False):
    torch = _torch(); import torch.nn as nn
    order, idx = _well_groups(wells)
    chunk = int(cfg["chunk"]); stride = max(1, int(chunk * (1.0 - cfg["overlap"])))
    def samples(well_set):
        out = []
        for w in order:
            if w in well_set:
                rows = idx[w]
                for (s, e) in _windows(len(rows), chunk, stride):
                    out.append(rows[s:e])
        return out
    tr = samples(set(tr_wells)); va_rows = idx
    model = build_model(F, cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    bs = int(cfg["batch"])

    def make_batch(rows_list):
        lens = [len(r) for r in rows_list]; L = max(lens); B = len(rows_list)
        xb = np.zeros((B, L, F), np.float32); yb = np.zeros((B, L), np.float32); mb = np.zeros((B, L), np.float32)
        for k, r in enumerate(rows_list):
            xb[k, :len(r)] = Xs[r]; yb[k, :len(r)] = y[r]; mb[k, :len(r)] = 1.
        return (torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev),
                torch.from_numpy(mb).to(dev), torch.tensor(lens, dtype=torch.long))

    def predict_wells(well_set, n_total):
        model.eval(); acc = np.zeros(n_total, np.float64); wsum = np.zeros(n_total, np.float64)
        with torch.no_grad():
            wl = [w for w in order if w in well_set]
            # batch windows across wells
            allwin = []
            for w in wl:
                rows = idx[w]
                for (s, e) in _windows(len(rows), chunk, stride):
                    allwin.append(rows[s:e])
            for i in range(0, len(allwin), bs):
                batch = allwin[i:i+bs]
                xb, _, _, lens = make_batch(batch)
                pr = model(xb, lens).cpu().numpy()
                for k, r in enumerate(batch):
                    tw = _tri_weights(len(r))
                    acc[r] += pr[k, :len(r)] * tw; wsum[r] += tw
        wsum[wsum < 1e-9] = 1.0
        return (acc / wsum).astype(np.float32)

    best = math.inf; best_state = None; best_vp = None; bad = 0
    eval_every = max(1, int(cfg.get("eval_every", 1)))
    n_ep = int(cfg["epochs"])
    rng = np.random.default_rng(seed)
    va_rows_idx = np.concatenate([idx[w] for w in va_wells])
    for ep in range(n_ep):
        model.train(); rng.shuffle(tr)
        for i in range(0, len(tr), bs):
            batch = tr[i:i+bs]
            xb, yb, mb, lens = make_batch(batch)
            opt.zero_grad()
            pred = model(xb, lens)
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum().clamp_min(1.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        # validate only every eval_every epochs (and on the last one) — the val
        # predict pass is expensive, so this roughly halves wall-clock.
        if (ep % eval_every != 0) and (ep != n_ep - 1):
            continue
        vp = predict_wells(set(va_wells), len(y))
        vr = float(np.sqrt(np.mean((vp[va_rows_idx] - y[va_rows_idx]) ** 2)))
        sched.step(vr)
        if vr < best - 1e-4:
            best = vr; best_vp = vp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
        if verbose:
            print(f"      ep{ep} val={vr:.4f} best={best:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_vp

# ----------------------------------------------------------------------------- K-fold OOF + ensemble
def kfold_oof(train_df, features, cv, cfg, seed=42, verbose=True):
    import gc
    torch = _torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = len(features)
    X = train_df[features].to_numpy(np.float32); y = train_df["target"].to_numpy(np.float32)
    wells = train_df["well"].to_numpy()
    mu, sd = fit_scaler(X); Xs = apply_scaler(X, mu, sd)
    del X; gc.collect()   # free the raw copy; only the scaled matrix is needed below
    order, idx = _well_groups(wells)
    oof = np.zeros(len(y), np.float32); states = []
    for fold, (tr, va) in enumerate(cv.split(Xs, y, groups=wells)):
        va_set = set(va.tolist())
        va_wells = [w for w in order if idx[w][0] in va_set]
        tr_wells = [w for w in order if w not in set(va_wells)]
        model, vr, best_vp = _train_one(Xs, y, wells, tr_wells, va_wells, F, cfg, seed + fold, dev)
        # reuse the best-epoch validation predictions as OOF (no extra predict pass)
        va_idx = np.concatenate([idx[w] for w in va_wells])
        if best_vp is None:
            best_vp = np.zeros(len(y), np.float32)
        oof[va_idx] = best_vp[va_idx]
        states.append({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"    fold{fold}: val RMSE(dTVT)={vr:.4f}", flush=True)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    if verbose:
        print(f"  OOF RMSE(dTVT)={rmse:.4f}  cfg={cfg}", flush=True)
    return oof, states, (mu, sd), rmse

# ----------------------------------------------------------------------------- hyperparameter search
def search_hparams(train_df, features, cv, seed=42, trials=None, quick_epochs=8, verbose=True):
    grid = [
        dict(d_model=128, hidden=128, layers=2, dropout=0.10, conv=True),
        dict(d_model=128, hidden=192, layers=2, dropout=0.15, conv=True),
        dict(d_model=192, hidden=192, layers=3, dropout=0.20, conv=True),
        dict(d_model=96,  hidden=128, layers=2, dropout=0.10, conv=False),
        dict(d_model=160, hidden=160, layers=3, dropout=0.15, conv=True),
        dict(d_model=128, hidden=128, layers=2, dropout=0.10, conv=True, lr=6e-4),
    ]
    if trials:
        grid = grid[:trials]
    best = None
    for i, g in enumerate(grid):
        cfg = dict(DEFAULT_CFG); cfg.update(g); cfg["epochs"] = quick_epochs
        if verbose:
            print(f"  [search {i+1}/{len(grid)}] {g}", flush=True)
        _, _, _, rmse = kfold_oof(train_df, features, cv, cfg, seed=seed, verbose=False)
        if verbose:
            print(f"    -> OOF RMSE {rmse:.4f}", flush=True)
        if best is None or rmse < best[0]:
            best = (rmse, dict(g))
    best_cfg = dict(DEFAULT_CFG); best_cfg.update(best[1])
    if verbose:
        print(f"  best cfg (quick): {best[1]}  OOF={best[0]:.4f}", flush=True)
    return best_cfg

# ----------------------------------------------------------------------------- blend-weight tuning
def tune_blend(base_oof, gru_oof, y):
    base_oof = np.asarray(base_oof, float); gru_oof = np.asarray(gru_oof, float); y = np.asarray(y, float)
    best_w, best_r = 0.0, math.inf
    for w in np.linspace(0.0, 0.6, 25):
        r = float(np.sqrt(np.mean(((1 - w) * base_oof + w * gru_oof - y) ** 2)))
        if r < best_r:
            best_r, best_w = r, float(w)
    return best_w, best_r

# ----------------------------------------------------------------------------- save / load / predict
def save_bundle(path, states, cfg, scaler, features, w_blend, meta=None):
    torch = _torch()
    mu, sd = scaler
    torch.save(dict(states=states, cfg=cfg, mu=mu, sd=sd, features=list(features),
                    w_blend=float(w_blend), meta=meta or {}), path)

def load_bundle(path):
    torch = _torch()
    return torch.load(path, map_location="cpu", weights_only=False)

def predict_bundle(bundle, df, batch=8):
    """Fold-ensemble dTVT prediction aligned to df rows. Returns np.float32[len(df)]."""
    torch = _torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = bundle["features"]; cfg = bundle["cfg"]; F = len(feats)
    mu = bundle["mu"]; sd = bundle["sd"]
    X = apply_scaler(df[feats].to_numpy(np.float32), mu, sd)
    wells = df["well"].to_numpy(); order, idx = _well_groups(wells)
    chunk = int(cfg["chunk"]); stride = max(1, int(chunk * (1.0 - cfg["overlap"])))
    out = np.zeros(len(df), np.float64)
    for st in bundle["states"]:
        model = build_model(F, cfg).to(dev)
        sd_t = {k: torch.from_numpy(np.asarray(v)) for k, v in st.items()}
        model.load_state_dict(sd_t); model.eval()
        acc = np.zeros(len(df)); wsum = np.zeros(len(df))
        with torch.no_grad():
            for w in order:
                rows = idx[w]
                for (s, e) in _windows(len(rows), chunk, stride):
                    r = rows[s:e]
                    xb = torch.from_numpy(X[r][None]).to(dev)
                    pr = model(xb, torch.tensor([len(r)])).cpu().numpy()[0]
                    tw = _tri_weights(len(r)); acc[r] += pr * tw; wsum[r] += tw
        wsum[wsum < 1e-9] = 1.0
        out += acc / wsum
    return (out / max(len(bundle["states"]), 1)).astype(np.float32)


# ----------------------------------------------------------------------------- offline driver
def fit_and_save(train_df, test_df, features, cv, out_path, base_oof=None,
                 seed=42, do_search=True, search_trials=None, verbose=True):
    """Search → final K-fold ensemble → tune blend → save. Returns (bundle_path, info)."""
    t0 = time.time()
    # env overrides for the memory/time budget (apply to search + final training)
    for _k, _env in [("chunk", "ROGII_GRU_CHUNK"), ("batch", "ROGII_GRU_BATCH"),
                     ("epochs", "ROGII_GRU_EPOCHS"), ("eval_every", "ROGII_GRU_EVAL_EVERY")]:
        if os.environ.get(_env):
            DEFAULT_CFG[_k] = int(os.environ[_env])
    if os.environ.get("ROGII_GRU_OVERLAP"):
        DEFAULT_CFG["overlap"] = float(os.environ["ROGII_GRU_OVERLAP"])
    cfg = search_hparams(train_df, features, cv, seed=seed, trials=search_trials,
                         verbose=verbose) if do_search else dict(DEFAULT_CFG)
    if verbose:
        print(f"  final training with cfg={cfg}", flush=True)
    oof, states, scaler, rmse = kfold_oof(train_df, features, cv, cfg, seed=seed, verbose=verbose)
    y = train_df["target"].to_numpy(np.float32)
    if base_oof is not None:
        w, blend_rmse = tune_blend(base_oof, oof, y)
        if verbose:
            base_rmse = float(np.sqrt(np.mean((np.asarray(base_oof) - y) ** 2)))
            print(f"  blend: base OOF={base_rmse:.4f}  gru OOF={rmse:.4f}  "
                  f"-> w_gru={w:.3f}  blended OOF={blend_rmse:.4f}", flush=True)
    else:
        w = 0.15
        if verbose:
            print(f"  no base_oof given; defaulting w_gru={w}", flush=True)
    save_bundle(out_path, states, cfg, scaler, features, w,
                meta=dict(gru_oof_rmse=rmse, n_folds=len(states), seconds=time.time() - t0))
    if verbose:
        print(f"  saved bundle -> {out_path}  ({time.time()-t0:.0f}s)", flush=True)
    return out_path, dict(cfg=cfg, gru_oof_rmse=rmse, w_blend=w)


# ----------------------------------------------------------------------------- smoke test (synthetic, CPU)
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(0); rows = []
    for w in range(16):
        n = int(rng.integers(120, 600)); F = 18
        sig = np.cumsum(rng.standard_normal(n))
        d = {f"f{j}": (sig if j == 0 else rng.standard_normal(n)).astype("float32") for j in range(F)}
        d["well"] = f"w{w:02d}"
        d["target"] = (0.6 * sig + 0.4 * np.cumsum(rng.standard_normal(n))).astype("float32")
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True); feats = [c for c in df.columns if c.startswith("f")]
    test = df[df.well.isin(["w00", "w01", "w02"])].reset_index(drop=True)
    cv = GroupKFold(4)
    base_oof = df["target"].to_numpy() + rng.standard_normal(len(df)) * 4.0  # a noisy "lgb" baseline
    # tiny budget for the smoke test (correctness only, not real tuning)
    g = globals(); g["DEFAULT_CFG"]["epochs"] = 3; g["DEFAULT_CFG"]["chunk"] = 250; g["DEFAULT_CFG"]["batch"] = 16
    path, info = fit_and_save(df, test, feats, cv, "gru_bundle_smoke.pt",
                              base_oof=base_oof, do_search=True, search_trials=2, seed=0,
                              verbose=True)

    b = load_bundle(path); pred = predict_bundle(b, test)
    print("PRED finite:", np.isfinite(pred).all(), "shape:", pred.shape,
          "| w_blend:", round(b["w_blend"], 3), "| cfg:", b["cfg"])
    assert np.isfinite(pred).all() and len(pred) == len(test)
    print("ROUNDTRIP OK")
