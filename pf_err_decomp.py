# Decompose per-well error of the physics tracker (lik-PF + selector) by well traits.
# Usage: python3 pf_err_decomp.py [n_wells]
import os, sys, glob, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import cv_harness as H

N_SEEDS = 32

def well_row(wid):
    try:
        hw, tw = H.load_well(wid)
    except Exception:
        return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10 or hw.TVT.isna().all(): return None
    if ev.TVT.isna().any(): return None
    y = ev.TVT.values.astype(float)
    last = float(kn.TVT_input.iloc[-1])
    try:
        pf_by_scale, _ = H.lik_pf(hw, tw, n_seeds=N_SEEDS)
        beam = H.run_beam_ensemble(hw, tw)[ev.index]
        variant = H.selector_well_code(hw)
        pred = H.apply_variant(variant, pf_by_scale, beam, last)
    except Exception:
        return None
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rmse_last = float(np.sqrt(np.mean((last - y) ** 2)))
    err_end = float(abs(pred[-1] - y[-1]))

    # ---- well traits ----
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.values.astype(float)
    tw_gr_f = np.nan_to_num(tw_gr, nan=float(np.nanmean(tw_gr)))
    z_ev = ev.Z.values.astype(float)
    z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    tmin, tmax = float(np.nanmin(tw_tvt)), float(np.nanmax(tw_tvt))
    cov_escape = float(np.mean((y < tmin) | (y > tmax)))
    edge_margin = float(min(last - tmin, tmax - last))
    drift_end = float(abs(y[-1] - last))
    drift_max = float(np.max(np.abs(y - last)))
    # GR affine fit hwGR ~ a*twGR(TVT_input)+b on known zone (measured GR only)
    m = kn.GR.notna().values
    a = b = gr_r = gr_resid = np.nan
    if m.sum() >= 30:
        x = np.interp(kn.TVT_input.values[m], tw_tvt, tw_gr_f)
        yy = kn.GR.values[m].astype(float)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        gr_resid = float(np.std(yy - A @ coef))
        if np.std(x) > 1e-9 and np.std(yy) > 1e-9:
            gr_r = float(np.corrcoef(x, yy)[0, 1])
    # handoff dip rate (same recipe as lik_pf init)
    tail = kn.tail(30)
    dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values)
    mm = dm > 0
    ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    return dict(well=wid, variant=variant, rmse=rmse, rmse_last=rmse_last, err_end=err_end,
                eval_len=len(ev), known_len=len(kn),
                gr_nan_frac_eval=float(ev.GR.isna().mean()),
                gr_nan_frac_known=float(kn.GR.isna().mean()),
                z_span=z_span, drift_end=drift_end, drift_max=drift_max,
                cov_escape=cov_escape, edge_margin=edge_margin,
                gr_a=a, gr_b=b, gr_corr=gr_r, gr_resid_std=gr_resid,
                abs_dip_ir=abs(ir), tw_range=tmax - tmin)

def main():
    n_wells = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    wids = sorted(os.path.basename(f).split("__")[0] for f in glob.glob("train/*__horizontal_well.csv"))
    rng = np.random.default_rng(7)
    samp = sorted(rng.choice(wids, min(n_wells, len(wids)), replace=False).tolist())
    t0 = time.time()
    res = Parallel(n_jobs=16, prefer="threads")(delayed(well_row)(w) for w in samp)
    df = pd.DataFrame([r for r in res if r is not None])
    print(f"evaluated {len(df)}/{len(samp)} wells in {time.time()-t0:.0f}s  (n_seeds={N_SEEDS})")
    df.to_csv("pf_err_decomp_seed7.csv", index=False)

    pooled = float(np.sqrt(np.mean(df.rmse ** 2 * df.eval_len) / np.mean(df.eval_len)))
    print(f"pooled selector RMSE (eval-len weighted) = {pooled:.4f}")
    print(f"per-well RMSE: mean={df.rmse.mean():.3f} median={df.rmse.median():.3f} "
          f"p90={df.rmse.quantile(.9):.3f} max={df.rmse.max():.3f}")
    top10_share = float((df.rmse**2 * df.eval_len).nlargest(max(1, len(df)//10)).sum()
                        / (df.rmse**2 * df.eval_len).sum())
    print(f"top-10% wells' share of pooled squared error = {top10_share:.1%}")

    feats = ["eval_len","known_len","gr_nan_frac_eval","gr_nan_frac_known","z_span",
             "drift_end","drift_max","cov_escape","edge_margin","gr_a","gr_b","gr_corr",
             "gr_resid_std","abs_dip_ir","tw_range","rmse_last"]
    print("\n=== Spearman / Pearson(log rmse) correlation with per-well RMSE ===")
    rows = []
    lr = np.log1p(df.rmse.values)
    for f in feats:
        v = df[f].values.astype(float)
        ok = np.isfinite(v)
        sp = spearmanr(v[ok], df.rmse.values[ok]).correlation
        pe = np.corrcoef(v[ok], lr[ok])[0, 1] if np.nanstd(v[ok]) > 1e-12 else np.nan
        rows.append((f, sp, pe))
    ct = pd.DataFrame(rows, columns=["feature","spearman","pearson_logrmse"]).sort_values(
        "spearman", key=lambda s: s.abs(), ascending=False)
    print(ct.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\n=== Top-10% RMSE wells vs rest: median profile ===")
    thr = df.rmse.quantile(0.9)
    hi, lo = df[df.rmse >= thr], df[df.rmse < thr]
    prof = pd.DataFrame({"top10_median": hi[feats + ["rmse"]].median(),
                         "rest_median": lo[feats + ["rmse"]].median()})
    prof["ratio"] = prof.top10_median / prof.rest_median.replace(0, np.nan)
    print(prof.to_string(float_format=lambda x: f"{x: .3f}"))

    print("\nTop-10% wells detail:")
    cols = ["well","variant","rmse","rmse_last","drift_end","cov_escape","gr_nan_frac_eval",
            "gr_corr","gr_a","eval_len","z_span"]
    print(hi.sort_values("rmse", ascending=False)[cols].to_string(index=False,
          float_format=lambda x: f"{x: .3f}"))

    # simple variance decomposition: R^2 of log rmse on single traits and small combos
    print("\n=== univariate R^2 on log(1+rmse) ===")
    for f in ["drift_end","drift_max","cov_escape","rmse_last","gr_corr","gr_nan_frac_eval","z_span","eval_len"]:
        v = df[f].values.astype(float); ok = np.isfinite(v)
        if v[ok].std() < 1e-12: continue
        vv = np.log1p(np.abs(v[ok])) if f in ("drift_end","drift_max","rmse_last") else v[ok]
        r = np.corrcoef(vv, lr[ok])[0, 1]
        print(f"  {f:20s} R^2 = {r*r:.3f}")
    X = []
    combo = ["drift_end","cov_escape","gr_corr","gr_nan_frac_eval","z_span","eval_len","abs_dip_ir"]
    dfc = df.dropna(subset=combo)
    Xm = np.column_stack([np.log1p(dfc.drift_end), dfc.cov_escape, dfc.gr_corr,
                          dfc.gr_nan_frac_eval, dfc.z_span/100., dfc.eval_len/1000., dfc.abs_dip_ir*100])
    Xm = np.column_stack([Xm, np.ones(len(dfc))])
    yv = np.log1p(dfc.rmse.values)
    coef, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ coef
    r2 = 1 - resid.var() / yv.var()
    print(f"\nmulti-OLS (drift,escape,gr_corr,grnan,z_span,eval_len,dip) R^2 on log rmse = {r2:.3f}  (n={len(dfc)})")
    print("coefs:", dict(zip(combo + ["const"], np.round(coef, 3))))

if __name__ == "__main__":
    main()
