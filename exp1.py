"""Experiment 1: GR-calibrated + self-augmented reference log for the lik-PF.
Grounded in PPT slide 9: the horizontal well's own pre-PS GR has better
resolution and correlates better than the typewell. We map the typewell GR
into horizontal-GR units (affine cal from the known prefix) and overlay the
horizontal prefix self-log in the TVT band it covers, then run the same lik-PF.
"""
import os, glob, time, argparse, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import cv_harness as H
warnings.filterwarnings("ignore")

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., (float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.)
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    # guard against degenerate slope
    if not np.isfinite(a) or a < 0.2 or a > 5.0:
        return 1., float(np.nanmean(kgr)-np.nanmean(tw_at_k))
    return float(a), float(b)

def build_aug_grid(hw, tw, self_weight=0.6, step=0.2):
    """Return (gg, gmin, gst, gs) in HORIZONTAL GR units.
    gg = typewell GR mapped to horizontal units, with the horizontal prefix
    self-log overlaid in the band it covers."""
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]
    ktvt = kn.TVT_input.values.astype(float)
    kgr = kn.GR.interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).values.astype(float)
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    a, b = affine_cal(kgr, tw_at_k)
    # typewell mapped into horizontal units
    tw_gr_h = a * tw_gr + b
    # dense grid spanning typewell range
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    gg = np.interp(tvt_g, tw_tvt, tw_gr_h)
    # overlay horizontal prefix self-log where it has coverage
    klo, khi = float(np.nanmin(ktvt)), float(np.nanmax(ktvt))
    if khi - klo > 5.0 and len(ktvt) >= 30:
        order = np.argsort(ktvt)
        kt_s = ktvt[order]; kg_s = kgr[order]
        # average duplicate TVTs onto a monotone curve
        self_gr = np.interp(tvt_g, kt_s, kg_s, left=np.nan, right=np.nan)
        band = np.isfinite(self_gr)
        gg[band] = (1.0 - self_weight) * gg[band] + self_weight * self_gr[band]
    # GR sigma from calibrated residual (horizontal units)
    resid = kgr - (a * tw_at_k + b)
    gs = float(np.clip(np.nanstd(resid), 8., 60.))
    return gg.astype(np.float64), float(tmin), float(step), gs

def lik_pf_aug(hw, tw, n_seeds=64, scales=(3.,5.,8.,12.), self_weight=0.6, init_spr=4.5, n_particles=500):
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    gg, gmin, gst, gs = build_aug_grid(hw, tw, self_weight=self_weight)
    tail = kn.tail(30); dt=np.diff(tail.TVT_input.values); dz=np.diff(tail.Z.values); dm=np.diff(tail.MD.values); m=dm>0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.0
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = H._pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                     gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, 0,
                                     0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts/=wts.sum(); out[f"pf_scale_{sc:g}"] = (wts[:,None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    return out

def eval_well_aug(wid, n_seeds, self_weight, aug_seeds=48, aug_particles=400):
    try:
        hw, tw = H.load_well(wid)
    except Exception:
        return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev)==0 or len(kn)<10 or hw.TVT.isna().all() or ev.TVT.isna().any(): return None
    y = ev.TVT.values.astype(float); last = float(kn.TVT_input.iloc[-1])
    out = {"wid":wid, "y":y, "last":np.full(len(y),last)}
    try:
        # baseline PF (typewell only) — full notebook cost
        pf_base, _ = H.lik_pf(hw, tw, n_seeds=n_seeds)
        for k,v in pf_base.items(): out["base_"+k]=v
        beam = H.run_beam_ensemble(hw, tw)[ev.index]
        out["beam"]=beam
        variant = H.selector_well_code(hw)
        out["base_selector"] = H.apply_variant(variant, pf_base, beam, last)
        # augmented PF — CHEAP config (fewer seeds/particles) to fit the 9h budget
        pf_aug = lik_pf_aug(hw, tw, n_seeds=aug_seeds, n_particles=aug_particles, self_weight=self_weight)
        for k,v in pf_aug.items(): out["aug_"+k]=v
        out["aug_selector"] = H.apply_variant(variant, pf_aug, beam, last)
        out["blend_selector"] = 0.7*out["base_selector"]+0.3*out["aug_selector"]
    except Exception as e:
        return None
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sw", type=float, default=0.6)
    ap.add_argument("--aug_seeds", type=int, default=48)
    ap.add_argument("--aug_particles", type=int, default=400)
    args = ap.parse_args()
    wids = sorted(os.path.basename(f).split("__")[0] for f in glob.glob("train/*__horizontal_well.csv"))
    rng = np.random.default_rng(args.seed)
    samp = sorted(rng.choice(wids, min(args.n,len(wids)), replace=False).tolist())
    t0=time.time()
    res = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(eval_well_aug)(w,args.seeds,args.sw,args.aug_seeds,args.aug_particles) for w in samp)
    res=[r for r in res if r is not None]
    print(f"evaluated {len(res)} wells in {time.time()-t0:.0f}s | self_weight={args.sw}")
    def prmse(key):
        e=[ (r[key]-r["y"])**2 for r in res if key in r]
        return float(np.sqrt(np.mean(np.concatenate(e)))) if e else float('nan')
    for key in ["last","base_pf_scale_8","aug_pf_scale_8","base_selector","aug_selector","blend_selector"]:
        print(f"  {key:18s} {prmse(key):.4f}")
    # blend-weight sweep on saved selector predictions
    print("  -- blend weight sweep (w on aug) --")
    for w in [0.0,0.2,0.3,0.35,0.4,0.45,0.5,0.6]:
        e=[((1-w)*r["base_selector"]+w*r["aug_selector"]-r["y"])**2 for r in res]
        print(f"    w_aug={w:.2f}  {float(np.sqrt(np.mean(np.concatenate(e)))):.4f}")
    np.save(f"exp1_res_seed{args.seed}.npy", np.array(res,dtype=object), allow_pickle=True)
