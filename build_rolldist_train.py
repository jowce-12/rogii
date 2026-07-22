# Row-level rolling/distribution candidates -> rolldist_join.parquet (id-keyed).
# Two PRE-REGISTERED groups (tune+confirm double gate, judged separately):
#   S: stride_row_std — per-ROW weighted std across the top-32 beam paths (where the
#      decode wavers, unlike the rejected well-level bstd); stride_rate — local slope
#      of the posterior-mean surface (the decoder's dip estimate)
#   G: gr_skew101/201, gr_kurt101 — rolling GR shape stats (skew/kurt absent
#      from the current set)
# RUN: python build_rolldist_train.py [--limit N]
import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

WORKERS = 10
TOPM, TEMP = 32, 8.0


def _decode_paths(hw, tw):
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
        if md[i] >= cur + 200.0:
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
    return dict(paths=paths[order], w=w, md=md, ev_index=ev.index.values)


def one_well(wid):
    try:
        from stride import load_well
        hw, tw = load_well(wid, "train")
        r = _decode_paths(hw, tw)
        if r is None:
            return wid, None, "too short"
        idx = r["ev_index"]
        out = pd.DataFrame({"id": [f"{wid}_{i}" for i in idx]})
        # S: row-level beam spread + posterior-mean local slope
        u_mean = (r["w"][:, None] * r["paths"]).sum(0)
        dev = r["paths"] - u_mean[None, :]
        out["stride_row_std"] = np.sqrt((r["w"][:, None] * dev ** 2).sum(0)).astype(np.float32)
        out["stride_rate"] = np.clip(np.gradient(u_mean, r["md"]), -0.1, 0.1).astype(np.float32)
        # G: GR rolling shape stats over the full well, sliced to eval rows
        gr = hw["GR"].interpolate(limit_direction="both")
        gr = gr.fillna(float(gr.mean()))
        sk101 = gr.rolling(101, center=True, min_periods=20).skew()
        sk201 = gr.rolling(201, center=True, min_periods=40).skew()
        ku101 = gr.rolling(101, center=True, min_periods=20).kurt()
        # gr_pctl101 dropped: pandas rolling.rank(pct) proved version-quirky (values >1
        # in smoke) — a train/test pairing hazard on Kaggle's pandas. skew/kurt are stable.
        for name, s in (("gr_skew101", sk101), ("gr_skew201", sk201),
                        ("gr_kurt101", ku101)):
            out[name] = s.values[idx].astype(np.float32)
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
    out.to_parquet("rolldist_join.parquet", index=False)
    print(f"DONE [{time.time()-t0:.0f}s] {len(out)} rows, {out.shape[1]-1} cols "
          f"-> rolldist_join.parquet | skipped {len(skips)}: {skips[:6]}", flush=True)


if __name__ == "__main__":
    main()
