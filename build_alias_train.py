# Group Z (z-score features) -> alias_join.parquet (id-keyed). PRE-REGISTERED pair:
#   alias_cnt60/120 : (GR - rolling_mean101) / rolling_std101 — explicit normalized anomaly
#             (components exist as grm101/grs101 but trees cannot form ratios)
#   (matchability fraction in +-band around the likpf-predicted TVT) (GR - typewell_GR(at likpf-predicted TVT)) / gam — how anomalous the GR
#             match is UNDER the current best decode, in Cauchy-scale units (row-level
#             "the prediction is wrong here" signal); grcal'd typewell, same as stride
# RUN: python build_z_train.py [--limit N]
import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

WORKERS = 10


def one_well(args):
    wid, lik = args   # lik: DataFrame(id, likpf_mean_d) for this well
    try:
        from stride import load_well, grcal_tw
        hw, tw = load_well(wid, "train")
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) < 5 or len(kn) < 10:
            return wid, None, "too short"
        idx = ev.index.values
        ids = [f"{wid}_{i}" for i in idx]
        out = pd.DataFrame({"id": ids})
        gr_s = hw["GR"].interpolate(limit_direction="both")
        gr_s = gr_s.fillna(float(gr_s.mean()))
        tw_s = tw.sort_values("TVT")
        tw_tvt = tw_s["TVT"].values.astype(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
        tw_gr = grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
        gg_x = np.arange(tw_tvt[0], tw_tvt[-1] + 0.5, 0.5)
        gg = np.interp(gg_x, tw_tvt, tw_gr)
        tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
        gam = float(np.clip(np.nanmedian(np.abs(kn["GR"].values.astype(float) - tw_at_k)), 5.0, 40.0))
        last = float(kn["TVT_input"].iloc[-1])
        lk = pd.DataFrame({"id": ids}).merge(lik, on="id", how="left")["likpf_mean_d"].values
        tvt_pred = last + np.nan_to_num(lk.astype(float))
        gr_ev = gr_s.values[idx]
        for name, band in (("alias_cnt60", 60.0), ("alias_cnt120", 120.0)):
            cnt = np.zeros(len(idx), np.float32)
            for k in range(len(idx)):
                lo = np.searchsorted(gg_x, tvt_pred[k] - band)
                hi = np.searchsorted(gg_x, tvt_pred[k] + band)
                seg = gg[lo:hi]
                cnt[k] = float(np.mean(np.abs(seg - gr_ev[k]) < gam)) if len(seg) else np.nan
            out[name] = cnt
        return wid, out, None
    except Exception as e:
        return wid, None, f"ERR {str(e)[:80]}"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    lik_all = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                              columns=["id", "well", "likpf_mean_d"])
    wells = sorted(lik_all["well"].unique())
    if limit:
        wells = wells[:limit]
    groups = {w: g[["id", "likpf_mean_d"]] for w, g in lik_all.groupby("well") if w in set(wells)}
    t0 = time.time()
    frames, skips = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (wid, df, err) in enumerate(ex.map(one_well, [(w, groups[w]) for w in wells]), 1):
            if df is None:
                skips.append((wid, err))
            else:
                frames.append(df)
            if k % 100 == 0 or k == len(wells):
                print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)} ({len(skips)} skipped)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet("alias_join.parquet", index=False)
    print(f"DONE [{time.time()-t0:.0f}s] {len(out)} rows, {out.shape[1]-1} cols "
          f"-> alias_join.parquet | skipped {len(skips)}: {skips[:6]}", flush=True)


if __name__ == "__main__":
    main()
