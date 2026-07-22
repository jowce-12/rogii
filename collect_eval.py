"""Collect per-well eval data once, so ALL remaining improvement candidates
(S1-on-sub_2, S2-lite dense blend, A2 projection variants, A4 gate) can be
tested OFFLINE and tuned(seed7)/confirmed(seed11) without re-running PFs."""
import os, glob, time, argparse
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import cv_harness as H

FORMS = ["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]

class DenseANCC:
    def __init__(self, well_ids, spw=60):
        xs, ys, an, wd = [], [], [], []
        for wid in well_ids:
            try: df = pd.read_csv(f"train/{wid}__horizontal_well.csv", usecols=["X","Y","ANCC"]).dropna()
            except Exception: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([wid]*len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32); self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0)); self.tree = cKDTree(self.xy/self.scale)
    def impute(self, xy_q, self_wid=None, k=20, nfetch=3000):
        q = np.atleast_2d(xy_q)/self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        o = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, o, 1); ik = np.take_along_axis(idx, o, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.)
        sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw); a = self.ancc[ik]
        ap = (a*w).sum(1)/safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        return ap.astype(np.float64), np.where(vk, dk, np.inf).min(1).astype(np.float64)

def alias_stats(kn, tw_tvt, tw_gr):
    kt = kn.TVT_input.values.astype(float); kg = kn.GR.values.astype(float)
    ta = np.interp(kt, tw_tvt, tw_gr); v = np.isfinite(kg)
    if v.sum() < 100: return np.nan, np.nan
    a, b = kg[v], ta[v]
    gr_corr = float(np.corrcoef(a, b)[0,1]) if a.std()>1e-6 and b.std()>1e-6 else np.nan
    band = tw_gr[(tw_tvt >= np.nanmin(kt)-10) & (tw_tvt <= np.nanmax(kt)+10)]
    tw_hf = np.nan
    if len(band) >= 40:
        tb = pd.Series(band); tw_hf = float((tb - tb.rolling(101, center=True, min_periods=1).mean()).std())
    return gr_corr, tw_hf

def eval_well(wid, imp, n_seeds=32):
    try: hw, tw = H.load_well(wid)
    except Exception: return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 120 or hw.TVT.isna().all() or ev.TVT.isna().any(): return None
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    out = {"wid": wid, "y": ev.TVT.values.astype(float),
           "md": ev.MD.values.astype(float), "z": ev.Z.values.astype(float),
           "last": float(kn.TVT_input.iloc[-1]), "z_ps": float(kn.Z.iloc[-1]), "md_ps": float(kn.MD.iloc[-1])}
    try:
        off, _ = H.lik_pf(hw, tw, n_seeds=n_seeds)
        aff, _ = H.lik_pf(hw, tw, n_seeds=n_seeds, grcal="affine")
        for k in ("pf_scale_3","pf_scale_5","pf_scale_8","pf_scale_12"):
            out["o"+k[-2:].replace("_","")] = off[k]; out["a"+k[-2:].replace("_","")] = aff[k]
        # dense neighbor anchor (self-excluded)
        d_ev, dist_ev = imp.impute(ev[["X","Y"]].to_numpy(float), self_wid=wid)
        d_kn, _ = imp.impute(kn[["X","Y"]].to_numpy(float), self_wid=wid)
        b = float(np.median(kn.TVT_input.values + kn.Z.values - d_kn))
        out["tvt_dense"] = -ev.Z.values.astype(float) + d_ev + b
        out["dense_dist"] = dist_ev
        out["gr_corr"], out["tw_hf_std"] = alias_stats(kn, tw_tvt, tw_gr)
    except Exception:
        return None
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    wids = sorted(os.path.basename(f).split("__")[0] for f in glob.glob("train/*__horizontal_well.csv"))
    print("building dense imputer (773 wells)...", flush=True); t0=time.time()
    imp = DenseANCC(wids); print(f"imputer ready {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(args.seed)
    samp = sorted(rng.choice(wids, args.n, replace=False).tolist())
    t0 = time.time()
    res = Parallel(n_jobs=24, prefer="threads")(delayed(eval_well)(w, imp) for w in samp)
    res = [r for r in res if r is not None]
    print(f"collected {len(res)} wells in {time.time()-t0:.0f}s (seed {args.seed})", flush=True)
    np.save(f"eval_data_seed{args.seed}.npy", np.array(res, dtype=object), allow_pickle=True)
