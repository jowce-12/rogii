# Distribution-shape features: the two transform families ABSENT from the 262-col set —
# (a) skew/kurtosis (rolling + well-level, nowhere in the current features) and
# (b) known<->eval GR distribution-shift / calibration-support-departure signals
# (grcal + PF likelihood are fitted on known-zone GR; eval GR outside that support means
# extrapolation). Inputs only (GR, TVT_input, cached likpf path) -> leakage-safe.
# Output: grshape_join.parquet (id + 11 cols). RUN from ~/rogii (~3min CPU).
import time
import numpy as np
import pandas as pd

t0 = time.time()
fs = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     columns=["id", "well", "likpf_mean_d"])
fs["row_idx"] = fs["id"].str.rsplit("_", n=1).str[1].astype(int)

rows_out = []
for k, (wid, g) in enumerate(fs.groupby("well", sort=True), 1):
    try:
        hw = pd.read_csv(f"train/{wid}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        kn_mask = hw["TVT_input"].notna()
        kn_gr = hw.loc[kn_mask, "GR"].dropna()
        if len(kn_gr) < 30:
            continue
        mu, sd = float(kn_gr.mean()), float(kn_gr.std()) + 1e-6
        p1, p99 = float(kn_gr.quantile(0.01)), float(kn_gr.quantile(0.99))
        kn_sorted = np.sort(kn_gr.values)
        g = g.sort_values("row_idx")
        idx = g["row_idx"].values
        ev_gr = hw["GR"].iloc[idx]
        ev_val = ev_gr.dropna()
        # well-level shift/shape (constants)
        w_mu = (float(ev_val.mean()) - mu) / sd if len(ev_val) > 10 else np.nan
        w_sd = float(np.log((float(ev_val.std()) + 1e-6) / sd)) if len(ev_val) > 10 else np.nan
        w_skew = float(ev_val.skew() - kn_gr.skew()) if len(ev_val) > 10 else np.nan
        w_kurt = float(ev_val.kurt() - kn_gr.kurt()) if len(ev_val) > 10 else np.nan
        w_nov = float(((ev_val < p1) | (ev_val > p99)).mean()) if len(ev_val) > 10 else np.nan
        try:
            tw = pd.read_csv(f"train/{wid}__typewell.csv", usecols=["GR"])["GR"].dropna()
            t_skew, t_kurt = float(tw.skew()), float(tw.kurt())
        except Exception:
            t_skew = t_kurt = np.nan
        # row-level: quantile rank in the KNOWN distribution + rolling shape
        gr_fill = hw["GR"].interpolate(limit_direction="both")
        qrank = np.searchsorted(kn_sorted, gr_fill.iloc[idx].values) / len(kn_sorted)
        rskew = gr_fill.rolling(101, center=True, min_periods=20).skew().iloc[idx].values
        rkurt = gr_fill.rolling(101, center=True, min_periods=20).kurt().iloc[idx].values
        pfinc = pd.Series(np.diff(g["likpf_mean_d"].values, prepend=g["likpf_mean_d"].values[0]))
        pskew = pfinc.rolling(101, center=True, min_periods=20).skew().values
        n = len(g)
        rows_out.append(pd.DataFrame({
            "id": g["id"].values,
            "grshift_mu": np.full(n, w_mu, np.float32),
            "grshift_sd": np.full(n, w_sd, np.float32),
            "grshift_skew": np.full(n, w_skew, np.float32),
            "grshift_kurt": np.full(n, w_kurt, np.float32),
            "gr_novelty": np.full(n, w_nov, np.float32),
            "tw_skew": np.full(n, t_skew, np.float32),
            "tw_kurt": np.full(n, t_kurt, np.float32),
            "gr_qrank": qrank.astype(np.float32),
            "gr_rskew101": rskew.astype(np.float32),
            "gr_rkurt101": rkurt.astype(np.float32),
            "pfinc_skew101": pskew.astype(np.float32),
        }))
    except Exception as e:
        print(f"  {wid} skipped: {str(e)[:60]}", flush=True)
    if k % 200 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/773 wells", flush=True)

out = pd.concat(rows_out, ignore_index=True)
out.to_parquet("grshape_join.parquet", index=False)
print(f"DONE [{time.time()-t0:.0f}s] grshape_join.parquet: {len(out)} rows "
      f"({len(out)/len(fs):.1%} coverage)", flush=True)
