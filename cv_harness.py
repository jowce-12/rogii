"""Local CV harness for ROGII wellbore TVT prediction.
Evaluates the physics-based trackers (lik-PF, beam, NCC, selector) on the
train wells' actual eval zones (TVT_input is NaN there but TVT is known).
Pure numpy/numba/scipy -> runs on CPU without the GBM stack.
"""
import os, sys, glob, time, warnings, argparse
import numpy as np, pandas as pd
from numba import njit
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

DATA = "."
FORMATIONS = ["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]

def load_well(wid, split="train"):
    hw = pd.read_csv(f"{DATA}/{split}/{wid}__horizontal_well.csv")
    tw = pd.read_csv(f"{DATA}/{split}/{wid}__typewell.csv").sort_values("TVT")
    return hw, tw

# ---------------- lik-PF (the workhorse) ----------------
@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _grcal_tw(tw_tvt, tw_gr, ktvt, kgr, mode):
    """S1: band-restricted recalibration of the typewell GR into hw-GR units."""
    ktvt = np.asarray(ktvt, float); kgr = np.asarray(kgr, float)
    v = np.isfinite(kgr)
    if v.sum() < 30:
        return tw_gr, 1.0, 0.0
    lo = float(np.nanmin(ktvt)) - 10.0; hi = float(np.nanmax(ktvt)) + 10.0
    band = tw_gr[(tw_tvt >= lo) & (tw_tvt <= hi)]
    if len(band) < 20:
        band = tw_gr
    bm = float(np.mean(band)); km = float(np.mean(kgr[v]))
    if mode == "offset":
        a, b = 1.0, km - bm
    elif mode == "var":
        sb = float(np.std(band)); sk = float(np.std(kgr[v]))
        a = float(np.clip(sk / sb, 0.2, 5.0)) if sb > 1e-6 else 1.0
        b = km - a * bm
    else:  # affine
        tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
        vv = v & np.isfinite(tw_at_k)
        if vv.sum() < 30 or np.std(tw_at_k[vv]) < 1e-6:
            a, b = 1.0, km - bm
        else:
            a, b = np.polyfit(tw_at_k[vv], kgr[vv], 1)
            if (not np.isfinite(a)) or a < 0.2 or a > 5.0:
                a, b = 1.0, km - bm
    return a * tw_gr + b, float(a), float(b)

def lik_pf(hw, tw, n_particles=500, n_seeds=128, scales=(3.,5.,8.,12.), init_spr=4.5, seed_base=0, grcal="off"):
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([])
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    if grcal in ("affine", "var", "offset"):
        tw_gr, _, _ = _grcal_tw(tw_tvt, tw_gr, kn.TVT_input.values, kn.GR.values, grcal)   # S1
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum(); out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    return out, ev.index.values

# ---------------- beam search ----------------
@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS*6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx+d
                if ni < 0 or ni >= nt: continue
                tot = cost+(gv-tw_gr[ni])**2/es+mc*(d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i+1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n-1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr): return len(arr)-1
    if i > 0 and abs(arr[i-1]-v) <= abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r*2+1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)

BEAM_CONFIGS = [(10,20.,144.,2),(10,8.,64.,2),(8,35.,220.,1),(10,14.,90.,5),(20,4.,36.,3),
    (12,12.,100.,3),(15,25.,180.,2),(20,30.,200.,2),(15,10.,80.,4),(25,6.,50.,3),
    (10,40.,300.,1),(12,18.,120.,5),(30,8.,70.,2),(10,50.,400.,0)]

def run_beam_ensemble(hw, tw):
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev)==0: return hw.TVT_input.values.astype(float).copy()
    last_tvt = float(kn.iloc[-1].TVT_input)
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    gr_all = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]
    res = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs,mc,es,r) for (bs,mc,es,r) in BEAM_CONFIGS]
    out = hw.TVT_input.values.astype(float).copy()
    out[list(ev.index)] = np.stack(res,0).mean(0)
    return out

# ---------------- selector ----------------
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {0:'pf_scale_5_hold_0.2',1:'pf_scale_3_hold_0.15',2:'pf_scale_12_beam_0.2_hold_0.15',
    3:'pf_scale_5_hold_0.15',4:'pf_scale_5_beam_0.05_hold_0.05',5:'pf_scale_12_beam_0.2_hold_0.05'}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'

def selector_well_code(hw):
    eval_mask = hw.TVT_input.isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask,'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval)-np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2*z_bin
    return SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)

def parse_variant(name):
    parts = name.split('_'); scale=float(parts[2]); beam=0.; hold=0.
    if 'beam' in parts: beam=float(parts[parts.index('beam')+1])
    if 'hold' in parts: hold=float(parts[parts.index('hold')+1])
    return scale,beam,hold

def apply_variant(name, pf_by_scale, tvt_beam, last_tvt):
    scale,beam,hold = parse_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}', pf_by_scale.get('pf_scale_8'))
    pred = (1.-beam)*base + beam*tvt_beam
    pred = (1.-hold)*pred + hold*last_tvt
    return pred

def _safe_key(s):
    return str(s).replace('.', 'p').replace('|', '_').replace('-', 'm')

def variant_grid():
    variants = set(SELECTOR_BIN_VARIANTS.values())
    variants.add(SELECTOR_GLOBAL_VARIANT)
    for scale in (3, 5, 8, 12):
        for hold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            variants.add(f'pf_scale_{scale:g}_hold_{hold:g}')
        for beam in (0.05, 0.10, 0.20, 0.30):
            for hold in (0.0, 0.05, 0.10, 0.15, 0.20):
                variants.add(f'pf_scale_{scale:g}_beam_{beam:g}_hold_{hold:g}')
    return sorted(variants)

# JIT warmup
_m=np.linspace(1,50,20);_z=np.zeros(20);_g=np.full(20,50.);_gg=np.linspace(45,55,100)
_pf_lik_allseeds(_m,_z,_g,_gg,45.,.1,20.,50.,0.,64,4,0,.998,.002,.005,.1,.001,.5,4.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)

def eval_well(wid, n_seeds=128):
    """Return dict of per-row preds for the eval zone + ground truth for each method."""
    try:
        hw, tw = load_well(wid)
    except Exception:
        return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev)==0 or len(kn)<10 or hw.TVT.isna().all(): return None
    if ev.TVT.isna().any(): return None
    y = ev.TVT.values.astype(float)
    last = float(kn.TVT_input.iloc[-1])
    out = {"wid":wid, "y":y, "last":np.full(len(y),last), "ev_idx":ev.index.values, "n_eval":len(ev)}
    try:
        pf_by_scale, ev_idx = lik_pf(hw, tw, n_seeds=n_seeds)
        for k,v in pf_by_scale.items(): out[k]=v
    except Exception as e:
        return None
    try:
        tvt_beam_full = run_beam_ensemble(hw, tw)
        out["beam"] = tvt_beam_full[ev.index]
    except Exception:
        out["beam"] = out["pf_scale_8"].copy()
    # selector and cheap blend candidates. These are non-leaky; oracle_* below is
    # only an upper-bound diagnostic for choosing future hard selectors.
    variant = selector_well_code(hw)
    out["selector"] = apply_variant(variant, pf_by_scale, out["beam"], last)
    if "pf_scale_12" in out:
        out["blend_pf12_beam20"] = 0.80*out["pf_scale_12"] + 0.20*out["beam"]
        out["blend_selector_pf12_beam"] = 0.70*out["selector"] + 0.20*out["pf_scale_12"] + 0.10*out["beam"]
        out["blend_selector_to_pf12_beam50"] = 0.50*out["selector"] + 0.50*out["blend_pf12_beam20"]
    variant_preds = {}
    for vname in variant_grid():
        try:
            pred = apply_variant(vname, pf_by_scale, out["beam"], last)
            key = "variant__" + _safe_key(vname)
            out[key] = pred
            variant_preds[vname] = pred
        except Exception:
            pass
    if variant_preds:
        best_name, best_pred, best_rmse = None, None, float("inf")
        for vname, pred in variant_preds.items():
            err = float(np.sqrt(np.mean((pred-y)**2)))
            if err < best_rmse:
                best_name, best_pred, best_rmse = vname, pred, err
        out["oracle_variant"] = best_pred
        out["oracle_variant_name"] = best_name
        out["oracle_variant_rmse"] = best_rmse
    return out

def pooled_rmse(results, key):
    errs=[];
    for r in results:
        if r is None or key not in r: continue
        val = r[key]
        if isinstance(val, str): continue
        errs.append((val-r["y"])**2)
    if not errs: return float('nan')
    return float(np.sqrt(np.mean(np.concatenate(errs))))

def per_well_rmse(results, keys):
    rows = []
    for r in results:
        if r is None: continue
        row = {"well": r.get("wid"), "n_eval": int(r.get("n_eval", len(r["y"]))) }
        for key in keys:
            if key in r and not isinstance(r[key], str):
                row[key] = float(np.sqrt(np.mean((r[key]-r["y"])**2)))
        if "oracle_variant_name" in r:
            row["oracle_variant_name"] = r["oracle_variant_name"]
            row["oracle_variant_rmse"] = float(r.get("oracle_variant_rmse", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)

def score_table(results, keys):
    rows = []
    for key in keys:
        score = pooled_rmse(results, key)
        if np.isfinite(score):
            rows.append({"method": key, "pooled_rmse": score})
    return pd.DataFrame(rows).sort_values("pooled_rmse").reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=128)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-variants", type=int, default=12)
    ap.add_argument("--save-preds", action="store_true", help="save full per-row prediction arrays to cv_results_*.npy")
    args = ap.parse_args()
    wids = sorted(os.path.basename(f).split("__")[0] for f in glob.glob(f"{DATA}/train/*__horizontal_well.csv"))
    rng = np.random.default_rng(args.seed)
    samp = sorted(rng.choice(wids, min(args.n,len(wids)), replace=False).tolist())
    t0=time.time()
    results = Parallel(n_jobs=args.jobs, prefer="threads")(delayed(eval_well)(w, args.seeds) for w in samp)
    results = [r for r in results if r is not None]
    print(f"evaluated {len(results)} wells in {time.time()-t0:.0f}s")
    base_keys = ["last","pf_mean","pf_scale_3","pf_scale_5","pf_scale_8","pf_scale_12","beam","selector",
                 "blend_pf12_beam20","blend_selector_pf12_beam","blend_selector_to_pf12_beam50","oracle_variant"]
    for key in base_keys:
        score = pooled_rmse(results, key)
        if np.isfinite(score):
            print(f"  {key:32s} pooled RMSE = {score:.4f}")

    variant_keys = sorted({k for r in results for k in r.keys() if k.startswith("variant__")})
    if variant_keys:
        vt = score_table(results, variant_keys)
        print("\nTop selector variants:")
        print(vt.head(args.top_variants).to_string(index=False, formatters={"pooled_rmse": "{:.4f}".format}))
        vt.to_csv(f"{DATA}/cv_variant_scores_{args.seed}.csv", index=False)

    report_keys = base_keys + (vt.head(args.top_variants)["method"].tolist() if variant_keys else [])
    per = per_well_rmse(results, report_keys)
    per.to_csv(f"{DATA}/cv_per_well_{args.seed}.csv", index=False)
    saved = f"saved cv_per_well_{args.seed}.csv" + (f", cv_variant_scores_{args.seed}.csv" if variant_keys else "")
    if args.save_preds:
        np.save(f"{DATA}/cv_results_{args.seed}.npy", np.array(results, dtype=object), allow_pickle=True)
        saved = "saved cv_results, " + saved
    print(saved)
