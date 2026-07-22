# Cross-tracker disagreement features (STRIDE vs lik-PF vs dense anchor) — the pairs the
# GBM cannot form itself and that are NOT yet in the feature set (pf_vs_dense and
# spatial_vs_dense already exist; every STRIDE-side disagreement is new).
# Output: xtrk_join.parquet (id + 8 cols). Inputs only (no target) -> leakage-safe.
# RUN from ~/rogii (~1min CPU).
import time
import numpy as np
import pandas as pd

t0 = time.time()
fs = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                     columns=["id", "well", "likpf_mean_d", "tvt_dense_d"])
st = pd.read_parquet("stride_join.parquet")
df = fs.merge(st, on="id", how="left")
df["row_idx"] = df["id"].str.rsplit("_", n=1).str[1].astype(int)
df = df.sort_values(["well", "row_idx"]).reset_index(drop=True)

sxp = df["stride_d"] - df["likpf_mean_d"]
out = pd.DataFrame({"id": df["id"].values})
out["sxp"] = sxp.astype(np.float32)
out["sxp_abs"] = sxp.abs().astype(np.float32)
g = sxp.groupby(df["well"])
out["sxp_roll21"] = g.transform(lambda s: s.rolling(21, center=True, min_periods=1).std()).astype(np.float32)
out["sxp_roll101"] = g.transform(lambda s: s.rolling(101, center=True, min_periods=1).std()).astype(np.float32)
out["sxp_wellmean"] = sxp.abs().groupby(df["well"]).transform("mean").astype(np.float32)
tri = df[["stride_d", "stride_stiff_d", "stride_loose_d"]]
out["stride_span"] = (tri.max(axis=1) - tri.min(axis=1)).astype(np.float32)
out["sxb"] = (df["stride_best_d"] - df["stride_d"]).astype(np.float32)
out["sxd"] = (df["stride_d"] - df["tvt_dense_d"]).astype(np.float32)
out.to_parquet("xtrk_join.parquet", index=False)
cov = float(out["sxp"].notna().mean())
print(f"DONE [{time.time()-t0:.0f}s] xtrk_join.parquet: {len(out)} rows, stride coverage {cov:.1%}")
