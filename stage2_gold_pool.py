# Stage 2: gold candidate-pool A/B on the harness (seeds 7/11).
#   A = ported gold, ORIGINAL pool (poly/surface/PF)
#   B = original pool + STRIDE decodes (200/400/100) + NEIGHBOR full-curve transfer
# Application uses gold's own _gold_profile_output (balanced) on the final blended
# predictions (0.20/0.50/0.30 + gamma), so both arms share the exact deployed rule.
# RUN: python stage2_gold_pool.py [7|11]
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import _gold_port as G
import stride as ST
import blend_eval as BE
from offline_tests import pooled
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
t0 = time.time()

# ---- final blended predictions for the harness wells (deployed config) ----
gru = pd.read_parquet("gru_oof_dipfused5.parquet")
s3p = pd.read_parquet("s3_preds_tuned.parquet")
fs = pd.read_parquet("oof_stack.parquet").merge(
    pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                    columns=["id", "likpf_mean_d", "likpf_ptstd"]), on="id")
fy = fs["target"].values.astype(np.float64); fy_fit = np.clip(fy, -90, 90); fg = fs["well"].values
FX = np.column_stack([fs[c].values for c in ["lgb0","lgb1","lgb2","cb0","cb1"]]
                     + [np.nan_to_num(fs["likpf_mean_d"].values.astype(np.float64))])
meta = np.zeros(len(fs))
for tr, va in GroupKFold(5).split(FX, fy_fit, groups=fg):
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(FX[tr], fy_fit[tr]); meta[va] = r.predict(FX[va])
OLD5 = [f"old_{s}" for s in ["lightgbm-1","lightgbm-2","lightgbm-3","catboost-1","catboost-2"]]
sub1_fn = BE.make_ridge_fn(OLD5)
fl_s = pd.Series(meta, index=fs["id"].values)
gr_s = pd.Series(gru.set_index("id")["gru_d"])
s3_s = pd.Series(s3p.set_index("id")["s3_tvt"])
risk_s = pd.Series(np.nan_to_num(fs["likpf_ptstd"].values.astype(np.float64)), index=fs["id"].values)

res, SEL = BE.selector_preds(SEED)
wells = [r_["wid"] for r_ in res]
sub = BE.OOF[BE.OOF["well"].isin(wells)].copy()
sub["sub1_tvt"] = sub1_fn(sub)
sub["fl_tvt"] = sub["last_known_tvt"].values + fl_s.reindex(sub["id"].values).values
sub["gr_tvt"] = sub["last_known_tvt"].values + gr_s.reindex(sub["id"].values).values
sub["s3_tvt"] = s3_s.reindex(sub["id"].values).values
sub["risk"] = risk_s.reindex(sub["id"].values).values
by = {c: {w: g[c].values for w, g in sub.groupby("well")} for c in
      ("sub1_tvt", "fl_tvt", "gr_tvt", "s3_tvt", "last_known_tvt", "risk", "id")}
finals, ok_wells = {}, []
for r_, sel in zip(res, SEL):
    w = r_["wid"]
    s1, fl, gr = by["sub1_tvt"].get(w), by["fl_tvt"].get(w), by["gr_tvt"].get(w)
    if any(v is None or len(v) != len(sel) or np.isnan(v).any() for v in (s1, fl, gr)):
        finals[w] = np.asarray(sel, float)
        continue
    risk = float(np.mean(by["risk"][w])); last = float(by["last_known_tvt"][w][0])
    monster = np.isfinite(risk) and risk >= 3.39
    wsp, wfl, wgr = (0.20, 0.40, 0.40) if monster else (0.20, 0.50, 0.30)
    b = wsp * (0.3 * s1 + 0.7 * sel) + wfl * fl + wgr * gr
    sv = by["s3_tvt"].get(w)
    if sv is not None and len(sv) == len(sel) and np.isfinite(sv).all():
        b = 0.90 * b + 0.10 * sv
    if monster:
        b = last + 1.09 * (b - last)
    finals[w] = b
    ok_wells.append(w)
print(f"[{time.time()-t0:.0f}s] seed{SEED}: base finals ready ({len(ok_wells)} ok wells)", flush=True)

def run_gold(arm):
    import stage2_worker as W
    outs = Parallel(n_jobs=10, backend="loky")(delayed(W.gold_one)(w, arm) for w in wells)
    # build submission frame + candidate map, apply gold's own balanced profile rule
    rows, cand = [], {}
    for r_, sel in zip(res, SEL):
        w = r_["wid"]
        ids = by["id"].get(w)
        if ids is None or len(ids) != len(finals[w]):
            continue
        for i_, v in zip(ids, finals[w]):
            rows.append((i_, v))
    subdf = pd.DataFrame(rows, columns=["id", "tvt"])
    subdf = G._gold_split_ids(subdf)
    reports = {}
    n_ok = 0
    for wid, rep, arr in outs:
        if rep is not None and rep.get("status") == "ok" and arr is not None:
            gsub = subdf[subdf["well"] == wid]
            for rid, ri in zip(gsub["id"].astype(str).values, gsub["row_idx"].astype(int).values):
                if 0 <= ri < len(arr) and np.isfinite(arr[ri]):
                    cand[rid] = float(arr[ri])
            rep["final_candidate_available"] = True
            n_ok += 1
        if isinstance(rep, dict) and rep.get("well"):
            reports[rep["well"]] = rep
    out_sub, moves = G._gold_profile_output(subdf, cand, reports, "conservative")
    tvt_map = out_sub.set_index("id")["tvt"]
    B = []
    for r_, sel in zip(res, SEL):
        w = r_["wid"]
        ids = by["id"].get(w)
        if ids is None or len(ids) != len(finals[w]):
            B.append(finals[w]); continue
        B.append(tvt_map.reindex(ids).values)
    print(f"[{time.time()-t0:.0f}s] arm {arm}: ok_wells={n_ok} pooled={pooled(res, B):.4f}", flush=True)
    winners = {}
    for wid, rep, _ in outs:
        if rep and rep.get("status") == "ok":
            nm = str(rep.get("best_name"))
            key = nm.split("|")[0] if "|" in nm else nm
            winners[key] = winners.get(key, 0) + 1
    print(f"   winners: {winners}", flush=True)
    return pooled(res, [finals[r_["wid"]] for r_ in res]), None

base_pooled = pooled(res, [finals[r_["wid"]] for r_ in res])
print(f"seed{SEED} gold-OFF baseline: {base_pooled:.4f}", flush=True)
run_gold("A")
run_gold("B")
print("DONE", flush=True)
