# GRU test-time channel builder + ensemble forward. MUST stay numerically identical to
# build_gru_data.py (natural cut). Embedded verbatim into the submission notebooks; also
# imported by the offline parity test. Dependencies injected: grcal_fn (= grcal_tw /
# _s2_grcal), stride_fn(hw, tw, seg_len) -> (ev_index, pred_minus_last, _) per eval row.
import numpy as np
import pandas as pd

G_STEP = 4.0
G_CTX = 256
G_OFFS = np.arange(-48.0, 48.1, 8.0)
SP_SPW = 60
SP_K = 20
SP_ROW_STEP = 4
SPATIAL_CHANS = ("dense_d", "dense_std", "dense_near", "pf_dense_gap",
                 "dense_confidence", "dense_known_rmse")


def spatial_bank(train_dir, wells=None):
    """(X,Y)->ANCC IDW bank from train wells (60 samples/well; make_spatial_features.py
    parity). wells=None -> all train wells (deployment); a set -> restricted (fold parity)."""
    from scipy.spatial import cKDTree
    xs, us, ws = [], [], []
    for p in sorted(train_dir.glob("*__horizontal_well.csv")):
        w = p.stem.replace("__horizontal_well", "")
        if wells is not None and w not in wells:
            continue
        try:
            d = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
        except Exception:
            continue
        if len(d) == 0:
            continue
        ix = np.linspace(0, len(d) - 1, min(SP_SPW, len(d)), dtype=int)
        xs.append(d[["X", "Y"]].values[ix])
        us.append(d["ANCC"].values[ix])
        ws.extend([w] * len(ix))
    if not xs:
        return None
    xy = np.vstack(xs).astype(np.float64)
    scale = xy.std(0)
    scale = np.where(scale < 1e-3, 1.0, scale)
    return {"tree": cKDTree(xy / scale), "ancc": np.concatenate(us).astype(np.float64),
            "wid": np.array(ws), "scale": scale}


def spatial_query(bank, wid, hw):
    """Per-row spatial src (md, spatial_u, dense_std, dense_dist) for one well.
    Mirrors make_spatial_features.py query_surface: K=20 IDW 1/(d+1e-3), self-excluded
    (no-op when wid absent from the bank -> identical to the plain top-K query)."""
    pos = np.arange(0, len(hw), SP_ROW_STEP, dtype=int)
    if pos[-1] != len(hw) - 1:
        pos = np.r_[pos, len(hw) - 1]
    q = hw.iloc[pos]
    ancc, wids, scale = bank["ancc"], bank["wid"], bank["scale"]
    fetch = min(SP_K + SP_SPW, len(ancc))
    dist, idx = bank["tree"].query(q[["X", "Y"]].to_numpy(np.float64) / scale, k=fetch, workers=-1)
    if fetch == 1:
        dist, idx = dist[:, None], idx[:, None]
    dist = np.where(wids[idx] == wid, np.inf, dist)
    take = np.argpartition(dist, SP_K - 1, axis=1)[:, :SP_K]
    dk = np.take_along_axis(dist, take, axis=1)
    ik = np.take_along_axis(idx, take, axis=1)
    valid = np.isfinite(dk)
    weight = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
    sw = weight.sum(1)
    if np.any(sw <= 0):
        return None
    nb = ancc[ik]
    dense = (nb * weight).sum(1) / sw
    var = ((nb - dense[:, None]) ** 2 * weight).sum(1) / sw
    return {"md": q["MD"].to_numpy(np.float64),
            "spatial_u": -q["Z"].to_numpy(np.float64) + dense,
            "dense_std": np.sqrt(np.maximum(var, 0)),
            "dense_dist": np.where(valid, dk, np.inf).min(1)}


def gru_channels(hw, tw, pf_pairs, grcal_fn, stride_fn, spatial_src=None):
    """pf_pairs = (preds32[n_seed, n_ev] float, liks32[n_seed]) on NATURAL-cut eval rows.
    spatial_src = spatial_query() output (or None -> spatial channels omitted; only
    non-spatial checkpoints will run). Returns (chan_dict, md_grid, cut_md, ev_index,
    last_tvt) or None."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 10 or len(kn) < 100:
        return None
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    md_all = hw["MD"].values.astype(float)
    z_all = hw["Z"].values.astype(float)
    gr_raw = hw["GR"].values.astype(float)
    gr_nan = (~np.isfinite(gr_raw)).astype(float)
    gr_fill = pd.Series(gr_raw).interpolate(limit_direction="both")
    gr_fill = gr_fill.fillna(float(np.nanmean(gr_raw))).values

    mu = float(np.nanmean(kn["GR"].values)); sd = float(np.nanstd(kn["GR"].values)) + 1e-3
    tw_cal = grcal_fn(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
    last_tvt = float(kn["TVT_input"].iloc[-1])
    cut_i = int(kn.index[-1])
    cut_md = float(hw.loc[cut_i, "MD"])
    end_md = float(md_all[-1])
    md_grid = np.arange(cut_md - G_CTX * G_STEP, end_md + G_STEP / 2, G_STEP)
    tail = md_grid > cut_md
    if tail.sum() < 8:
        return None
    gr_g = np.interp(md_grid, md_all, gr_fill)
    grz = (gr_g - mu) / sd
    ch = {}
    ch["grz"] = grz
    ch["grz_s21"] = pd.Series(grz).rolling(21, center=True, min_periods=1).mean().values
    ch["grz_s101"] = pd.Series(grz).rolling(101, center=True, min_periods=1).mean().values
    ch["grz_std21"] = pd.Series(grz).rolling(21, center=True, min_periods=1).std().fillna(0).values
    ch["gr_nan"] = np.interp(md_grid, md_all, gr_nan)
    z_g = np.interp(md_grid, md_all, z_all)
    ch["dzdmd"] = np.clip(np.gradient(z_g, md_grid), -0.5, 0.5) * 20
    ch["z_rel"] = (z_g - z_g[tail.argmax()]) / 100.0
    ch["d_from_cut"] = np.maximum(md_grid - cut_md, 0) / 1000.0
    ch["d_to_end"] = (end_md - md_grid) / 1000.0
    kn_md = kn["MD"].values.astype(float)
    kn_tvt = kn["TVT_input"].values.astype(float)
    tvt_pre = np.interp(md_grid, kn_md, kn_tvt, left=float(kn_tvt[0]), right=last_tvt)
    ch["tvt_rel"] = np.where(~tail, (tvt_pre - last_tvt) / 10.0, 0.0)
    ch["is_known"] = (~tail).astype(float)
    # stride channels (builder-exact: interp pred over grid, left=anchor)
    md_ev = ev["MD"].values.astype(float)
    for name, seg in (("stride_d", 200.0), ("stride_stiff", 400.0), ("stride_loose", 100.0)):
        r = stride_fn(hw, tw, seg)
        if r is None:
            ch[name] = np.zeros(len(md_grid))
            continue
        pred = np.asarray(r[1], float) + last_tvt      # per eval row, TVT
        p = np.interp(md_grid, md_ev, pred, left=last_tvt, right=float(pred[-1]))
        ch[name] = np.clip((p - last_tvt) / 10.0, -12, 12)
    # PF channels (builder-exact from seed pairs)
    preds32, liks32 = pf_pairs
    liks = np.asarray(liks32, float); liks = liks - liks.max()
    pf5_path = np.full(len(md_grid), last_tvt, float)
    for k, scale in enumerate((3.0, 5.0, 8.0)):
        w = np.exp(liks / scale); w /= w.sum()
        path = (w[:, None] * np.asarray(preds32, float)).sum(0)
        p = np.interp(md_grid, md_ev, path, left=last_tvt, right=float(path[-1]))
        if k == 1:
            pf5_path = p.copy()
        ch[f"pf{int(scale)}_d"] = np.clip((p - last_tvt) / 10.0, -12, 12)
    sdv = np.asarray(preds32, float).std(0)
    ch["pf_sd"] = np.clip(np.interp(md_grid, md_ev, sdv, left=0, right=float(sdv[-1])) / 10.0, 0, 6)
    # mismatch profile around moving hypothesis (prefix=actual TVT, tail=pf5 path)
    hyp_all = np.where(~tail, tvt_pre, pf5_path)
    twz = (tw_cal - mu) / sd
    for j, off in enumerate(G_OFFS):
        ref = np.interp(np.clip(hyp_all + off, tw_tvt[0], tw_tvt[-1]), tw_tvt, twz)
        ch[f"mm{j}"] = np.clip(np.abs(grz - ref), 0, 6.0)
    # spatial channels (train_gru2.make_spatial_features parity; md_grid == its
    # reconstructed grid because d_to_end is anchored to the same well end)
    if spatial_src is not None:
        known = ~tail
        u = np.interp(md_grid, spatial_src["md"], spatial_src["spatial_u"])
        std_i = np.interp(md_grid, spatial_src["md"], spatial_src["dense_std"])
        dist_i = np.interp(md_grid, spatial_src["md"], spatial_src["dense_dist"])
        kp = np.flatnonzero(known)
        if len(kp) == 0:
            for c in SPATIAL_CHANS:
                ch[c] = np.zeros(len(md_grid))
        else:
            cut_pos = int(kp[-1])
            u_rel = (u - u[cut_pos]) / 10.0
            tvr = ch["tvt_rel"]
            bias = float(np.median(tvr[known] - u_rel[known]))
            dense_d = np.clip(u_rel + bias, -12.0, 12.0)
            krm = float(np.sqrt(np.mean((dense_d[known] - tvr[known]) ** 2)))
            ch["dense_d"] = dense_d
            ch["dense_std"] = np.clip(std_i / 20.0, 0.0, 6.0)
            ch["dense_near"] = np.exp(-np.clip(dist_i, 0.0, None) / 0.02)
            ch["pf_dense_gap"] = np.clip(ch["pf5_d"] - dense_d, -12.0, 12.0)
            ch["dense_confidence"] = np.exp(-np.clip(std_i, 0.0, None) / 30.0) * ch["dense_near"]
            ch["dense_known_rmse"] = np.full(len(md_grid), np.clip(krm, 0.0, 6.0))
    return ch, md_grid, cut_md, ev.index.values, last_tvt


def gru_fuse(p, dip, lam, step=G_STEP):
    """Quadratic path fusion: (I + lam*D^T D) x = p + lam*D^T(dip*step). Thomas solve.
    Keeps the path near the point predictions while matching increments to the predicted
    dip — soft constraint, no integration drift."""
    n = len(p)
    if n < 3:
        return p.copy()
    v = dip * step
    lower = np.full(n - 1, -lam)
    upper = np.full(n - 1, -lam)
    diag = np.full(n, 1.0 + 2.0 * lam)
    diag[0] = 1.0 + lam
    diag[-1] = 1.0 + lam
    rhs = p.copy()
    rhs[0] += lam * (-v[0])
    rhs[1:-1] += lam * (v[:-1] - v[1:])
    rhs[-1] += lam * v[-1]
    c = upper.copy(); d = rhs.copy(); b = diag.copy()
    for i in range(1, n):
        w = lower[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def gru_forward(ch, ckpts, torch, nn):
    """Mean (pred_ft, dip_ft_per_ft|None) over checkpoints. Supports 1-out (tvt) and
    2-out (tvt+dip x25 increments) heads via the ckpt's "dip" flag."""
    preds, dips = [], []
    n_skip = 0
    for ck in ckpts:
        chans = ck["chans"]; hid = ck["hid"]; is_dip = bool(ck.get("dip"))
        if any(c not in ch for c in chans):
            n_skip += 1          # e.g. spatial ckpt but no spatial_src -> skip, don't crash
            continue

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Linear(len(chans), hid)
                self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                                  bidirectional=True, dropout=0.25)
                self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(),
                                          nn.Linear(hid, 2 if is_dip else 1))

            def forward(self, x):
                h, _ = self.gru(self.inp(x))
                return self.head(h)

        m = _Net(); m.load_state_dict(ck["state"]); m.eval()
        x = np.stack([np.asarray(ch[c], np.float32) for c in chans], 1)[None]
        with torch.no_grad():
            out = m(torch.from_numpy(x)).numpy()[0]
        if is_dip:
            preds.append(out[:, 0] * 10.0)
            dips.append(out[:, 1] / 25.0 * 10.0 / G_STEP)
        else:
            preds.append(out[:, 0] * 10.0)
    if not preds:
        raise RuntimeError(f"no compatible checkpoints ({n_skip} skipped: channels missing)")
    return np.mean(preds, 0), (np.mean(dips, 0) if dips else None)
