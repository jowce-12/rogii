"""GRU base learner for the ROGII fleongg stack (CELL 31).

The eval zone of each well is a sequence along MD; a bidirectional GRU consumes
the per-row feature vectors (the same `features` the GBMs use, all known at
predict time) and outputs the dTVT sequence. Trained with the SAME GroupKFold
as the GBMs so its OOF/test predictions slot into the Ridge meta-stack as an
extra column ("gru0"). All inputs are known for eval rows, so bidirectional is
valid. Long wells are split into non-overlapping chunks to bound memory/time.

Safe by construction: if torch is missing or ROGII_GRU=off, train_gru_oof is a
no-op (returns None) and the stack is unchanged.
"""
import os, numpy as np

def _torch_ok():
    try:
        import torch  # noqa
        return True
    except Exception:
        return False

def _well_groups(wells):
    order, idxs = [], {}
    for i, w in enumerate(wells):
        if w not in idxs:
            idxs[w] = []; order.append(w)
        idxs[w].append(i)
    return order, {w: np.asarray(v, dtype=np.int64) for w, v in idxs.items()}

def _chunks(idx, chunk):
    if chunk <= 0 or len(idx) <= chunk:
        return [idx]
    return [idx[i:i+chunk] for i in range(0, len(idx), chunk)]

def train_gru_oof(train_df, test_df, features, cv, seed=42,
                  hidden=96, layers=2, drop=0.1, lr=1e-3, epochs=40,
                  patience=6, batch_wells=8, chunk=1200, verbose=True):
    """Return (oof[len train], test[len test]) dTVT predictions, or None if disabled."""
    mode = os.environ.get("ROGII_GRU", "add").strip().lower()
    if mode == "off" or not _torch_ok():
        if verbose:
            print(f"  GRU disabled (ROGII_GRU={mode}, torch_ok={_torch_ok()})")
        return None
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    # On a CPU-only run a BiGRU over ~773 long wells can blow the 9h budget; skip
    # unless the user explicitly forces it (ROGII_GRU_FORCE_CPU=1).
    if not torch.cuda.is_available() and os.environ.get("ROGII_GRU_FORCE_CPU", "0") != "1":
        if verbose:
            print("  GRU skipped: no CUDA (set ROGII_GRU_FORCE_CPU=1 to run on CPU)")
        return None

    # env-tunable budget knobs
    hidden = int(os.environ.get("ROGII_GRU_HIDDEN", hidden))
    epochs = int(os.environ.get("ROGII_GRU_EPOCHS", epochs))
    chunk = int(os.environ.get("ROGII_GRU_CHUNK", chunk))
    batch_wells = int(os.environ.get("ROGII_GRU_BATCH", batch_wells))

    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = len(features)

    X = train_df[features].to_numpy(np.float32)
    Xt = test_df[features].to_numpy(np.float32)
    y = train_df["target"].to_numpy(np.float32)
    g = train_df["well"].to_numpy()

    # standardize on train (robust to inf/nan)
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    Xt = np.nan_to_num(Xt, nan=0., posinf=0., neginf=0.)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd; Xt = (Xt - mu) / sd
    X = np.clip(X, -8, 8).astype(np.float32); Xt = np.clip(Xt, -8, 8).astype(np.float32)

    tr_order, tr_idx = _well_groups(g)
    te_order, te_idx = _well_groups(test_df["well"].to_numpy())
    # pre-chunk every well into samples (row-index arrays)
    tr_samples = {w: _chunks(tr_idx[w], chunk) for w in tr_order}
    te_samples = [(w, c) for w in te_order for c in _chunks(te_idx[w], chunk)]

    class GRUReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(F)
            self.gru = nn.GRU(F, hidden, layers, batch_first=True,
                              bidirectional=True, dropout=drop if layers > 1 else 0.)
            self.head = nn.Sequential(nn.Linear(2*hidden, 64), nn.GELU(), nn.Linear(64, 1))
        def forward(self, x, lengths):
            x = self.norm(x)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            return self.head(out).squeeze(-1)

    def make_batch(sample_rows):
        lens = [len(r) for r in sample_rows]; L = max(lens); B = len(sample_rows)
        xb = np.zeros((B, L, F), np.float32); yb = np.zeros((B, L), np.float32); mb = np.zeros((B, L), np.float32)
        for k, r in enumerate(sample_rows):
            xb[k, :len(r)] = X[r]; yb[k, :len(r)] = y[r]; mb[k, :len(r)] = 1.
        return (torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev),
                torch.from_numpy(mb).to(dev), torch.tensor(lens, dtype=torch.long))

    def predict(model, samples_rows, n_total):
        model.eval(); out = np.zeros(n_total, np.float32)
        with torch.no_grad():
            for i in range(0, len(samples_rows), batch_wells):
                batch = samples_rows[i:i+batch_wells]
                xb, _, mb, lens = make_batch(batch)
                pr = model(xb, lens).cpu().numpy()
                for k, r in enumerate(batch):
                    out[r] = pr[k, :len(r)]
        return out

    oof = np.zeros(len(train_df), np.float32)
    test = np.zeros(len(test_df), np.float32)
    te_rows_all = [c for (_, c) in te_samples]
    n_folds = 0
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=g)):
        va_set = set(va.tolist())
        va_wells = set(w for w in tr_order if tr_idx[w][0] in va_set)
        tr_wells = [w for w in tr_order if w not in va_wells]
        train_rows = [c for w in tr_wells for c in tr_samples[w]]
        val_rows = [c for w in va_wells for c in tr_samples[w]]
        model = GRUReg().to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        best_rmse = np.inf; best_state = None; bad = 0
        rng = np.random.default_rng(seed + fold)
        for ep in range(epochs):
            model.train(); rng.shuffle(train_rows)
            for i in range(0, len(train_rows), batch_wells):
                batch = train_rows[i:i+batch_wells]
                xb, yb, mb, lens = make_batch(batch)
                opt.zero_grad()
                pred = model(xb, lens)
                loss = (((pred - yb)**2) * mb).sum() / mb.sum().clamp_min(1.)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vp = predict(model, val_rows, len(train_df))
            vr = float(np.sqrt(np.mean((vp[va] - y[va])**2)))
            if vr < best_rmse - 1e-4:
                best_rmse = vr; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
            else:
                bad += 1
                if bad >= patience: break
        if best_state is not None:
            model.load_state_dict(best_state)
        oof[va] = predict(model, val_rows, len(train_df))[va]
        test += predict(model, te_rows_all, len(test_df))
        n_folds += 1
        if verbose:
            print(f"  GRU fold{fold}: val RMSE(dTVT)={best_rmse:.4f}", flush=True)
    test /= max(n_folds, 1)
    if verbose:
        print(f"  GRU OOF RMSE(dTVT)={float(np.sqrt(np.mean((oof - y)**2))):.4f}", flush=True)
    return oof, test


# ----------------- standalone smoke test (synthetic, CPU) -----------------
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(0)
    rows = []
    for w in range(14):
        n = int(rng.integers(80, 400)); F = 16
        gr = np.cumsum(rng.standard_normal(n))  # a smooth latent signal
        feat = rng.standard_normal((n, F)).astype(np.float32)
        feat[:, 0] = gr
        tgt = 0.7*gr + 0.3*np.cumsum(rng.standard_normal(n))  # sequential target
        d = {f"f{j}": feat[:, j] for j in range(F)}
        d["well"] = f"w{w:02d}"; d["target"] = tgt.astype(np.float32)
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    feats = [c for c in df.columns if c.startswith("f")]
    test = df[df["well"].isin(["w00", "w01", "w02"])].reset_index(drop=True)
    cv = GroupKFold(4)
    os.environ["ROGII_GRU"] = "add"
    out = train_gru_oof(df, test, feats, cv, epochs=8, hidden=32, layers=1, chunk=200, batch_wells=4)
    assert out is not None
    oof, tp = out
    print("oof shape", oof.shape, "finite", np.isfinite(oof).all(),
          "| test shape", tp.shape, "finite", np.isfinite(tp).all())
    base = float(np.sqrt(np.mean((df["target"].to_numpy() - 0)**2)))
    got = float(np.sqrt(np.mean((oof - df["target"].to_numpy())**2)))
    print(f"baseline(zero) RMSE={base:.3f}  GRU OOF RMSE={got:.3f}  (GRU should be lower)")
