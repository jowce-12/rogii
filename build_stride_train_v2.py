# STRIDE feature family v2 -> stride_join_v2.parquet. Superset of v1 (stride_d /
# stride_best_d byte-identical, same deployed decode) plus:
#   stride_stiff_d / stride_loose_d : seg_len 400 / 100 decodes (multi-stiffness family)
#   stride_sep / stride_minmass / stride_middelta / stride_bstd :
#       weighted 1D 2-means branch stats over the top-32 beam eval-mean levels
#       (the midhedge insight as FEATURES — the GBM learns when/how much to hedge)
#   stride_pfx_rmse : prefix backtest — cut the visible prefix at 60%, decode, score on
#       the held-out 40% (TVT_input only; identically computable at test time)
# RUN: python build_stride_train_v2.py [--limit N]
import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

WORKERS = 10
TOPM, TEMP = 32, 8.0


def _decode_paths(hw, tw, seg_len):
    """stride.stride_track replica that also returns (paths[order], w, z, ev) extras."""
    from stride import grcal_tw, _decode, _paths_from_rates
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 5 or len(kn) < 10:
        return None
    tw_gr = grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
    gmin = tw_tvt[0]
    gg = np.interp(np.arange(gmin, tw_tvt[-1] + 0.5, 0.5), tw_tvt, tw_gr)
    last = kn.iloc[-1]
    u0 = float(last["TVT_input"]) + float(last["Z"])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    s0 = float(np.clip(np.median((dt + dz)[m] / dm[m]) if m.sum() >= 3 else 0.0, -0.06, 0.06))
    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gam = float(np.clip(np.nanmedian(np.abs(kn["GR"].values.astype(float) - tw_at_k)), 5.0, 40.0))
    md = ev["MD"].values.astype(float); z = ev["Z"].values.astype(float)
    gr = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    bnds = [0]; cur = md[0]
    for i in range(len(md)):
        if md[i] >= cur + seg_len:
            bnds.append(i); cur = md[i]
    if bnds[-1] != len(md):
        bnds.append(len(md))
    bnds = np.array(bnds, dtype=np.int64)
    rates = np.arange(-0.06, 0.06 + 0.001, 0.002)
    msk = np.ones(len(md), np.int8)
    br, sc = _decode(md, z, gr, msk, gg, gmin, 0.5, u0, s0, gam, rates, bnds,
                     96, 0.012, 25.0, 0.0, 0.1)
    paths = _paths_from_rates(md, bnds, u0, rates, br)
    order = np.argsort(sc)[::-1][:min(TOPM, len(sc))]
    scc = sc[order]
    w = np.exp((scc - scc.max()) / TEMP); w /= w.sum()
    return dict(paths=paths[order], w=w, z=z, ev_index=ev.index.values,
                last=float(last["TVT_input"]))


def _branch_stats(levels, w):
    """Weighted 1D 2-means (public midhedge recipe) -> sep, minmass, middelta, bstd."""
    Lw = float((w * levels).sum())
    bstd = float(np.sqrt(max((w * (levels - Lw) ** 2).sum(), 0.0)))
    c1, c2 = float(np.quantile(levels, 0.1)), float(np.quantile(levels, 0.9))
    for _ in range(25):
        a = np.abs(levels - c1) <= np.abs(levels - c2)
        w1, w2 = float(w[a].sum()), float(w[~a].sum())
        if w1 <= 0 or w2 <= 0:
            return 0.0, 0.0, 0.0, bstd
        n1 = float((w[a] * levels[a]).sum() / w1)
        n2 = float((w[~a] * levels[~a]).sum() / w2)
        moved = abs(n1 - c1) + abs(n2 - c2)
        c1, c2 = n1, n2
        if moved < 1e-9:
            break
    a = np.abs(levels - c1) <= np.abs(levels - c2)
    w1, w2 = float(w[a].sum()), float(w[~a].sum())
    mid = 0.5 * (c1 + c2)
    return abs(c1 - c2), min(w1, w2), mid - Lw, bstd


def one_well(wid):
    try:
        from stride import load_well, stride_track
        hw, tw = load_well(wid, "train")
        base = _decode_paths(hw, tw, 200.0)
        if base is None:
            return wid, None, "too short"
        last = base["last"]; z = base["z"]; idx = base["ev_index"]
        tvt_mean = (base["w"][:, None] * base["paths"]).sum(0) - z
        tvt_best = base["paths"][0] - z
        lv = (base["paths"] - z[None, :]).mean(axis=1) - last     # per-beam eval-mean level (delta)
        sep, minmass, middelta, bstd = _branch_stats(lv, base["w"])
        out = pd.DataFrame({"id": [f"{wid}_{i}" for i in idx],
                            "stride_d": (tvt_mean - last).astype(np.float32),
                            "stride_best_d": (tvt_best - last).astype(np.float32)})
        for name, seg in (("stride_stiff_d", 400.0), ("stride_loose_d", 100.0)):
            r = _decode_paths(hw, tw, seg)
            out[name] = (((r["w"][:, None] * r["paths"]).sum(0) - r["z"] - last).astype(np.float32)
                         if r is not None else np.float32(np.nan))
        out["stride_sep"] = np.float32(sep); out["stride_minmass"] = np.float32(minmass)
        out["stride_middelta"] = np.float32(middelta); out["stride_bstd"] = np.float32(bstd)
        # prefix backtest: mask the last 40% of the known prefix, decode, score on it
        pfx = np.float32(np.nan)
        kn_idx = hw.index[hw["TVT_input"].notna()].values
        cut = int(round(len(kn_idx) * 0.6))
        if cut >= 50 and len(kn_idx) - cut >= 30:
            hw_m = hw.copy(deep=True)
            held = kn_idx[cut:]
            hw_m.loc[held, "TVT_input"] = np.nan
            pred, _ = stride_track(hw_m, tw)
            if pred is not None:
                ev_m = hw_m[hw_m["TVT_input"].isna()]
                pos = {int(r): k for k, r in enumerate(ev_m.index.values)}
                sel = [pos[int(r)] for r in held if int(r) in pos]
                truth = hw.loc[held, "TVT_input"].values.astype(float)
                if len(sel) >= 20:
                    pfx = np.float32(np.sqrt(np.mean((pred[sel] - truth[: len(sel)]) ** 2)))
        out["stride_pfx_rmse"] = pfx
        return wid, out, None
    except Exception as e:
        return wid, None, f"ERR {str(e)[:80]}"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    wells = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
    if limit:
        wells = wells[:limit]
    t0 = time.time()
    frames, skips = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (wid, df, err) in enumerate(ex.map(one_well, wells), 1):
            if df is None:
                skips.append((wid, err))
            else:
                frames.append(df)
            if k % 100 == 0 or k == len(wells):
                print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)} ({len(skips)} skipped)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet("stride_join_v2.parquet", index=False)
    # canonical trainer input = the ADOPTED columns only (split verdict 2026-07-19:
    # decode variants passed tune -0.024 / confirm -0.197; stats/delta cols REJECTED)
    out[["id", "stride_d", "stride_best_d", "stride_stiff_d", "stride_loose_d"]].to_parquet(
        "stride_join.parquet", index=False)
    print(f"DONE [{time.time()-t0:.0f}s] {len(out)} rows -> stride_join_v2.parquet (9 cols, research) "
          f"+ stride_join.parquet (4 cols, canonical) | skipped {len(skips)}: {skips[:6]}", flush=True)


if __name__ == "__main__":
    main()
