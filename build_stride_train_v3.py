# STRIDE v3 candidate columns -> stride_join_v3.parquet. Three PRE-REGISTERED groups
# (judged tune+confirm separately; adopt only both-pass groups):
#   T: stride_t2_d / stride_t32_d   — temperature family: same top-32 beams of the
#      deployed decode, posterior reweighted at temp 2 (sharp) / 32 (soft)
#   W: stride_lw05_d / stride_lw20_d — evidence-weight family: re-decode with
#      lik_w 0.05 (continuity-led) / 0.20 (GR-led); posterior = deployed top32/temp8
#   P: poly2_d / poly3_d            — GR-free geometric family: robust polynomial fit of
#      the visible surface U=TVT+Z over MD (known tail), extrapolated to the eval zone
# RUN: python build_stride_train_v3.py [--limit N]
import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

WORKERS = 10
TOPM = 32


def _decode_paths(hw, tw, seg_len=200.0, lik_w=0.1):
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
                     96, 0.012, 25.0, 0.0, lik_w)
    paths = _paths_from_rates(md, bnds, u0, rates, br)
    order = np.argsort(sc)[::-1][:min(TOPM, len(sc))]
    return dict(paths=paths[order], sc=sc[order], z=z,
                ev_index=ev.index.values, last=float(last["TVT_input"]))


def _pmean(r, temp):
    w = np.exp((r["sc"] - r["sc"].max()) / temp); w /= w.sum()
    return (w[:, None] * r["paths"]).sum(0) - r["z"] - r["last"]


def _poly_extrap(hw, deg):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(kn) < 120 or len(ev) < 5:
        return None
    tail = kn.tail(max(120, int(0.4 * len(kn))))
    md_k = tail["MD"].values.astype(float)
    u_k = (tail["TVT_input"].values + tail["Z"].values).astype(float)
    c, s = md_k.mean(), max(md_k.std(), 1e-6)
    try:
        cf = np.polyfit((md_k - c) / s, u_k, deg)
    except Exception:
        return None
    u_ev = np.polyval(cf, (ev["MD"].values.astype(float) - c) / s)
    last = float(kn["TVT_input"].iloc[-1])
    d = u_ev - ev["Z"].values.astype(float) - last
    return np.clip(d, -90.0, 90.0)


def one_well(wid):
    try:
        from stride import load_well
        hw, tw = load_well(wid, "train")
        base = _decode_paths(hw, tw)
        if base is None:
            return wid, None, "too short"
        idx = base["ev_index"]
        out = pd.DataFrame({"id": [f"{wid}_{i}" for i in idx]})
        out["stride_t2_d"] = _pmean(base, 2.0).astype(np.float32)
        out["stride_t32_d"] = _pmean(base, 32.0).astype(np.float32)
        for name, lw in (("stride_lw05_d", 0.05), ("stride_lw20_d", 0.20)):
            r = _decode_paths(hw, tw, lik_w=lw)
            out[name] = _pmean(r, 8.0).astype(np.float32) if r is not None else np.float32(np.nan)
        for name, deg in (("poly2_d", 2), ("poly3_d", 3)):
            p = _poly_extrap(hw, deg)
            out[name] = p.astype(np.float32) if p is not None else np.float32(np.nan)
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
    out.to_parquet("stride_join_v3.parquet", index=False)
    print(f"DONE [{time.time()-t0:.0f}s] {len(out)} rows, {out.shape[1]-1} cols "
          f"-> stride_join_v3.parquet | skipped {len(skips)}: {skips[:6]}", flush=True)


if __name__ == "__main__":
    main()
