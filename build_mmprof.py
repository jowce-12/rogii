# Row-level MISMATCH-PROFILE (alias landscape) features around the likpf hypothesis —
# the GBM-consumable analogue of the GRU's 13-offset mm channels. Per eval row:
# windowed mean |grz - twz(hyp+off)| over offsets ±48ft step 8, windows 21/101 rows ->
#   mm{w}_0 (at offset 0), mm{w}_gap (at0 - min: how much a shifted hypothesis explains
#   GR better), mm{w}_argmin (signed offset of the min: local correction direction).
# Uses ONLY inputs (GR, typewell, cached likpf path) — leakage-safe on train eval rows.
# Output: mmprof_join.parquet (id + 6 cols). RUN from ~/rogii (~5min CPU).
import time
import numpy as np
import pandas as pd
from stride import load_well, grcal_tw

OFFS = np.arange(-48.0, 48.1, 8.0)
WINDOWS = (21, 101)
t0 = time.time()

fs = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     columns=["id", "well", "likpf_mean_d"])
fs["row_idx"] = fs["id"].str.rsplit("_", n=1).str[1].astype(int)
by_well = {w: g for w, g in fs.groupby("well", sort=True)}

rows_out = []
skipped = 0
for k, (wid, g) in enumerate(by_well.items(), 1):
    try:
        hw, tw = load_well(wid, "train")
        kn = hw[hw["TVT_input"].notna()]
        if len(kn) < 30 or len(tw) < 20:
            skipped += 1
            continue
        tw_s = tw.sort_values("TVT")
        tw_tvt = tw_s["TVT"].values.astype(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
        mu = float(np.nanmean(kn["GR"].values)); sd = float(np.nanstd(kn["GR"].values)) + 1e-3
        tw_cal = grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
        twz = (tw_cal - mu) / sd
        gr_fill = hw["GR"].interpolate(limit_direction="both")
        gr_fill = gr_fill.fillna(float(np.nanmean(hw["GR"].values))).values
        last_tvt = float(kn["TVT_input"].iloc[-1])
        idx = g["row_idx"].values
        hyp = last_tvt + g["likpf_mean_d"].values.astype(float)
        grz = (gr_fill[idx] - mu) / sd
        mm = np.empty((len(idx), len(OFFS)))
        for j, off in enumerate(OFFS):
            ref = np.interp(np.clip(hyp + off, tw_tvt[0], tw_tvt[-1]), tw_tvt, twz)
            mm[:, j] = np.clip(np.abs(grz - ref), 0, 6.0)
        out = {"id": g["id"].values}
        j0 = int(np.where(OFFS == 0.0)[0][0])
        for w in WINDOWS:
            sm = pd.DataFrame(mm).rolling(w, center=True, min_periods=1).mean().values
            at0 = sm[:, j0]
            mn = sm.min(1)
            out[f"mm{w}_0"] = at0.astype(np.float32)
            out[f"mm{w}_gap"] = (at0 - mn).astype(np.float32)
            out[f"mm{w}_argmin"] = OFFS[sm.argmin(1)].astype(np.float32)
        rows_out.append(pd.DataFrame(out))
    except Exception as e:
        skipped += 1
        print(f"  {wid} skipped: {str(e)[:60]}", flush=True)
    if k % 100 == 0:
        print(f"[{time.time()-t0:.0f}s] {k}/773 wells", flush=True)

out = pd.concat(rows_out, ignore_index=True)
out.to_parquet("mmprof_join.parquet", index=False)
cov = len(out) / len(fs)
print(f"DONE [{time.time()-t0:.0f}s] mmprof_join.parquet: {len(out)} rows "
      f"({cov:.1%} coverage, {skipped} wells skipped)", flush=True)
assert cov > 0.97, "coverage too low"
