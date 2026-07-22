# Standalone FULL-STACK (re)trainer for ROGII ??LGB?? + CatBoost?? + Ridge meta.
# Run on a GPU Kaggle notebook with the competition data attached.
# Output: lgb*.pkl, cb*.cbm, ridge.pkl, stack_meta.json, features.json.
# Publish as a dataset; attach to the submission notebook (DETACH any old lgb
# dataset so only this one is found). The patched INFERENCE branch applies the
# full stack. Do NOT attach a gru_bundle (GRU stays off).
# Time/RAM knobs (offline, 12h session budget):
#   ROGII_STACK_FOLDS=3      # OOF folds (default 5; 3 = ~40% faster)
#   ROGII_STACK_CB=1         # 0 = skip CatBoost (LGB+Ridge only, much faster)
#   ROGII_STACK_MAXWELLS=0   # 0 = all wells; set e.g. 500 if time/RAM tight
#   ROGII_N_JOBS=4          # feature/PF build parallelism; lower if WSL kills the job
#   ROGII_BUILD_CHUNK=32    # wells per feature/PF batch; lower if RAM is tight
import os
os.environ.setdefault("SHOW_FIGS", "0")

import os, sys, glob, time, warnings, multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
os.environ.setdefault("SHOW_FIGS", "0")

# ---- environment / paths (Kaggle or local) -------------------------------------
def _find_data():
    for c in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction"]:
        if Path(c).exists() and (Path(c)/"train").exists():
            return Path(c)
    # fallback: find any mounted folder that contains a train/ directory
    for p in glob.glob("/kaggle/input/**/train", recursive=True):
        return Path(p).parent
    return Path(os.environ.get("ROGII_DATA", "."))   # local override for development

class CFG:
    DATA = _find_data()
    OUT  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    seed = 42
    n_splits = 5
    n_jobs = int(os.environ.get("ROGII_N_JOBS", os.environ.get("ROGII_NJOBS", str(min(16, multiprocessing.cpu_count())))))
    BUILD_CHUNK = int(os.environ.get("ROGII_BUILD_CHUNK", "64"))
    # lik-PF
    PF_SEEDS = 128
    PF_PARTICLES = 500
    PF_SCALES = (3., 5., 8., 12.)
    # FAST dev (local smoke test): limit train wells & trees
    FAST = bool(int(os.environ.get("FAST", "0")))
    N_TRAIN_WELLS = int(os.environ.get("N_TRAIN_WELLS", "0"))  # 0 = all
    USE_GPU = os.environ.get("USE_GPU", "auto")
    SHOW_FIGS = os.environ.get("SHOW_FIGS", "1") == "1"   # EDA plots (on in the notebook)

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
def _demo_well():
    """A train well with TVT + a sizable eval zone, for the EDA plots."""
    for w in sorted(p.stem.replace("__horizontal_well", "")
                    for p in (CFG.DATA/"train").glob("*__horizontal_well.csv")):
        try:
            d = pd.read_csv(CFG.DATA/"train"/f"{w}__horizontal_well.csv", usecols=["TVT", "TVT_input"])
        except Exception:
            continue
        if "TVT" in d and d.TVT.notna().any() and d.TVT_input.isna().sum() > 2000:
            return w
    return None
print("DATA:", CFG.DATA, "| OUT:", CFG.OUT, "| cores:", CFG.n_jobs, "| FAST:", CFG.FAST)

def load_well(wid, split="train"):
    base = CFG.DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float))**2)))

# ---- single particle filters (ANCC-anchored & Z-velocity-coupled), numba ---------
PF_N = 600; ANCC_N = 600
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
PF_GR_CAL_BAND = 10.; PF_GR_CAL_MIN = 20; PF_GR_CAL_A_MIN = 0.25; PF_GR_CAL_A_MAX = 2.5
PF_GR_WIN = 5; PF_GR_WT = 0.3; PF_RESAMP = 0.5; PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001

BEAMS = [(10,20.,144.,2,"cons"),(10,8.,64.,2,"loose"),(8,35.,220.,1,"vcons"),
         (10,14.,90.,5,"sm5"),(20,4.,36.,3,"vloose"),(12,12.,100.,3,"mid"),(15,25.,180.,2,"stiff")]

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1] = cum[j]+w[j]
    u0 = np.random.uniform(0., 1./N); np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0+j/N
        while ci < N-1 and cum[ci+1] < u: ci += 1
        np2[j] = pos[ci]+rp*np.random.randn(); na[j] = aux[ci]+rv*np.random.randn()
    return np2, na

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

@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ls+IS*np.random.randn(); rate[j] = ir+0.01*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA*rate[j]+RN*np.random.randn(); pos[j] += rate[j]*dm+PN*np.random.randn()
            tvt_j = pos[j]-z_v[i]; tvt_j = max(tvt_j, vmin-50.); tvt_j = min(tvt_j, vmin+len(gg)*step+50.)
            pos[j] = tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs
                lk = max(np.exp(-0.5*d*d) if d*d < 600. else 0., 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1./N
        tv = 0.
        for j in range(N): tv += w[j]*(pos[j]-z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i] = va**0.5; pm = md_v[i]
    return pts, std_

@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step, gs, ip, iv, beta, icpt, zsig, N,
         MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N)/N
    for j in range(N):
        pos[j] = ip+0.5*np.random.randn(); vel[j] = iv+0.02*np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0]-1.; pz = z_v[0]-1.
    for i in range(len(md_v)):
        dm = md_v[i]-pm; dm = max(dm, 1.); dzd = (z_v[i]-pz)/dm; ve = beta*dzd+icpt
        for j in range(N):
            vel[j] = MOM*vel[j]+VN*np.random.randn(); pos[j] += vel[j]*dm+PN*np.random.randn()
            pos[j] = max(pos[j], vmin-50.); pos[j] = min(pos[j], vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step); dp = (gr_v[i]-ep)/gs
                lp = max(np.exp(-0.5*dp*dp) if dp*dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step); ds = (gr_sm_v[i]-es)/(gs*1.5)
                    lsm = max(np.exp(-0.5*ds*ds) if ds*ds < 600. else 0., 1e-300); lk = (1.-GR_WT)*lp+GR_WT*lsm
                else: lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j]-ve)/max(zsig*2., 0.005); lz = max(np.exp(-0.5*dv*dv) if dv*dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1./N
        ne = 0.
        for j in range(N): ne += w[j]*w[j]
        if 1./ne < RESAMP*N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1./N
        wm = 0.
        for j in range(N): wm += w[j]*pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j]*(pos[j]-wm)**2
        std_[i] = va**0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def _band_mean_at(q_tvt, tw_tvt, tw_gr, band=PF_GR_CAL_BAND):
    q = np.asarray(q_tvt, dtype=float); x = np.asarray(tw_tvt, dtype=float); y = np.asarray(tw_gr, dtype=float)
    fallback = np.interp(q, x, y)
    if len(x) == 0 or band <= 0:
        return fallback
    lo = np.searchsorted(x, q - float(band), side="left")
    hi = np.searchsorted(x, q + float(band), side="right")
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cnt = hi - lo
    out = fallback.copy()
    ok = cnt > 0
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / cnt[ok]
    return out

def pf_gr_band_cal(hw, tw_tvt, tw_gr, band=PF_GR_CAL_BAND):
    kn = hw[hw.TVT_input.notna()]
    ktvt = kn.TVT_input.to_numpy(float)
    kgr = kn.GR.to_numpy(float)
    tw_band = _band_mean_at(ktvt, tw_tvt, tw_gr, band)
    a, b = affine_cal(kgr, tw_band, min_pts=PF_GR_CAL_MIN)
    if not np.isfinite(a): a = 1.0
    if not np.isfinite(b): b = 0.0
    a = float(np.clip(a, PF_GR_CAL_A_MIN, PF_GR_CAL_A_MAX))
    tw_gr_cal = (a * np.asarray(tw_gr, dtype=float) + b).astype(float)
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr_cal)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values.astype(float) - tw_at_k),
                       PF_GR_SIG_MIN, PF_GR_SIG_MAX))
    if not np.isfinite(gs):
        gs = float(PF_GR_SIG_DEF)
    return tw_gr_cal, gs, a, b

def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw.TVT_input.notna() & hw.GR.notna()]
    if len(kn) < 20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn.GR.values-np.interp(kn.TVT_input.values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))

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

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn.TVT_input.iloc[-1]+kn.Z.iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_fill = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))  # gr fill: feed filled GR to the PF (fewer skipped NaN steps)
    pts, std = _pf_ancc(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), gr_fill.loc[ev.index].values.astype(np.float64),
                        gg, gmin, gst, gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr); tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna.Z.values); dvt = np.diff(kna.TVT_input.values); dmd_k = np.diff(kna.MD.values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]; A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    else: beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2.TVT_input.values); dmd2 = np.diff(t2.MD.values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr); gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_fill = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))  # gr fill
    gr_sm = gr_fill.rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(ev.MD.values.astype(np.float64), ev.Z.values.astype(np.float64), gr_fill.loc[ev.index].values.astype(np.float64),
                     gr_sm.loc[ev.index].values.astype(np.float64), gg, gs2, gmin, gst, gs,
                     float(kna.TVT_input.iloc[-1]), iv, beta, icpt, zsig, N,
                     PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2*hw+1; nk = len(kgr); nh = len(hgr)
        if nk < win+1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk-win+1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None]+np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C-C.mean(1, keepdims=True))/(C.std(1, keepdims=True)+1e-6)
        hp = np.pad(hg, hw, mode="edge"); H = hp[np.arange(nh)[:, None]+np.arange(win)[None, :]].astype(np.float32)
        Hn = (H-H.mean(1, keepdims=True))/(H.std(1, keepdims=True)+1e-6)
        ncc = Hn@Cn.T/win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw, 0, nk-1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.*scores); sw /= sw.sum(1, keepdims=True)+1e-9
    return out, (tvts*sw).sum(1).astype(np.float32)

# ---- 128-seed likelihood-weighted particle filter (the workhorse), numba ---------
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

def lik_pf(hw, tw, n_particles=CFG.PF_PARTICLES, n_seeds=CFG.PF_SEEDS, scales=CFG.PF_SCALES,
           init_spr=4.5, seed_base=0, with_quality=False):
    """Likelihood-weighted PF ensemble. Returns ({pf_scale_X: pred_eval}, ev_index[, quality])."""
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0: return {}, np.array([]), {}
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    # S1: fit a known-zone, TVT +/-10ft band-limited affine GR correction and
    # feed the calibrated typewell log to the PF likelihood.
    tw_gr_lik, gs, gr_cal_a, gr_cal_b = pf_gr_band_cal(hw, tw_tvt, tw_gr)
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr_lik)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr_lik))).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                   gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, seed_base,
                                   0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln/float(sc)); wts /= wts.sum(); out[f"pf_scale_{sc:g}"] = (wts[:, None]*preds).sum(0)
    out["pf_mean"] = preds.mean(0)
    q = {}
    if with_quality:
        q = {"pf_best_ll": float(liks.max())/len(ev), "pf_ll_spread": float(liks.std()),
             "pf_pt_std": preds.std(0).astype(np.float32), "pf_gr_sig": gs,
             "pf_gr_cal_a": gr_cal_a, "pf_gr_cal_b": gr_cal_b, "pf_gr_cal_band": float(PF_GR_CAL_BAND)}
    return out, ev.index.values, q

# JIT warm-up so timings below are representative
_m = np.linspace(1, 50, 20); _z = np.zeros(20); _g = np.full(20, 50.); _gg = np.linspace(45, 55, 100)
_pf_ancc(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 8, .998, .002, .005, .3, .1, .001, .5)
_pf_z(_m, _z, _g, _g, _gg, _gg, 45., .1, 20., 50., 0., -1., 0., .1, 8, .993, .005, .01, .3, .2, .003, .5)
_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)
_pf_lik_allseeds(_m, _z, _g, _gg, 45., .1, 20., 50., 0., 64, 4, 0, .998, .002, .005, .1, .001, .5, 4.5)
print("trackers compiled.")

def fig_tracker_vs_truth(wid):
    import matplotlib.pyplot as plt
    hw, tw = load_well(wid); kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32); last = float(kn.TVT_input.iloc[-1])
    pf, _ = run_pf_ancc(hw, tw_tvt, tw_gr); out, _, _ = lik_pf(hw, tw, scales=(3.,))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ev.MD, ev.TVT, lw=2.2, color="black", label="True TVT", zorder=5)
    ax.plot(ev.MD, np.full(len(ev), last), lw=1.1, color="gray", ls=":", label="last-known baseline")
    ax.plot(ev.MD, pf, lw=1.0, color="tab:blue", alpha=.8, label="single particle filter")
    ax.plot(ev.MD, out["pf_scale_3"], lw=1.5, color="crimson", alpha=.9, label="128-seed lik-weighted PF")
    ax.set_xlabel("MD (ft)"); ax.set_ylabel("TVT (ft)"); ax.invert_yaxis(); ax.grid(alpha=.25)
    ax.set_title(f"Well {wid}: trackers vs ground truth 李????the lik-PF resists drift"); ax.legend(loc="best")
    plt.tight_layout(); plt.show()

PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

def robust_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])

def affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1); return float(a), float(b)

def seg_b_well(ktvt, kz, form_col):
    bv = ktvt+kz-form_col; n = len(bv); b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n-50):])) if n >= 5 else b_full
    t1, t2 = n//3, 2*n//3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1+1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02*np.arange(n)); w /= w.sum()
    return b_full, b_early, b_mid, b_late, float(np.dot(w, bv))

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y"]+FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS: row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x","y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy/self.scale); self.xa = self.df.x.to_numpy(); self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)
    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q/self.scale; nf = min(k+5, len(self.df)); dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w*xn; wy = w*yn
        A = np.zeros((len(q), 3, 3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs = np.stack([(wx[:,:,None]*fn).sum(1), (wy[:,:,None]*fn).sum(1), (w[:,:,None]*fn).sum(1)], 1)
        try: coef = np.linalg.solve(A, rhs)
        except:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq = xy_q[:,0]; Yq = xy_q[:,1]
        pred = (Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, an, wd = [], [], [], []
        for wid in well_ids:
            try: df = pd.read_csv(data_dir/f"{wid}__horizontal_well.csv", usecols=["X","Y","ANCC"]).dropna()
            except: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s.X.values); ys.append(s.Y.values); an.append(s.ANCC.values); wd.extend([wid]*len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(an).astype(np.float32); self.wids = np.array(wd)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0)); self.tree = cKDTree(self.xy/self.scale)
    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q/self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k-1, nf-1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1./(dk+1e-3), 0.); sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw)
        a = self.ancc[ik]; ap = (a*w).sum(1)/safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((a-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)

_FI = None; _DI = None
ANCH_OFFS = np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80], np.float32)
BEAM_OFFS = np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40], np.float32)
SC_OFFS = np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30], np.float32)
PF_OFFS = SC_OFFS.copy()

def add_derived_features(feats, hgr, tw_tvt, tw_gr, pf_use, pf_z, has_z, std_use,
                         beam_mean, sc_ens, tvt_dense, tvtF_ANCC, z_ev, dzdmd, dxdmd, dydmd,
                         last_tvt, a_cal, b_cal, md_since, known_len, nh, gr_isna_ev=None):
    """Physics/PPT-grounded derived features. All computable in the eval zone."""
    import os as _os
    if _os.environ.get("ROGII_FEATS", "1") != "1":
        return
    f32 = np.float32
    def _f(a):
        return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0., posinf=0., neginf=0.).astype(f32)
    tw_tvt = np.asarray(tw_tvt, np.float64); tw_gr = np.asarray(tw_gr, np.float64)
    hgr64 = np.asarray(hgr, np.float64); pf64 = np.asarray(pf_use, np.float64)
    # (1) local GR-match-quality profile -- rising = the track is drifting off the log
    tw_at_pf = np.interp(pf64, tw_tvt, tw_gr)
    mq = np.abs(hgr64 - tw_at_pf)
    s = pd.Series(mq)
    feats["mq_res"] = _f(mq)
    for w in (21, 51, 101):
        feats[f"mq_roll{w}"] = _f(s.rolling(w, center=True, min_periods=1).mean().values)
    feats["mq_roll51_std"] = _f(s.rolling(51, center=True, min_periods=1).std().fillna(0).values)
    feats["mq_cum"] = _f(s.expanding().mean().values)
    # (2) calibrated match offsets (slide-9: typewell GR -> horizontal GR units)
    for o in (-8, -4, 0, 4, 8):
        feats[f"tdpf_cal{int(o)}"] = _f(hgr64 - (a_cal * np.interp(pf64 + o, tw_tvt, tw_gr) + b_cal))
    for o in (-20, -5, 0, 5, 20):
        feats[f"tda_cal{int(o)}"] = _f(hgr64 - (a_cal * np.interp(last_tvt + o, tw_tvt, tw_gr) + b_cal))
    # (3) cross-tracker disagreement / rank of the PF within the panel
    cols = [pf64, (np.asarray(pf_z, np.float64) if has_z else pf64),
            np.asarray(beam_mean, np.float64), np.asarray(sc_ens, np.float64),
            np.asarray(tvt_dense, np.float64), np.asarray(tvtF_ANCC, np.float64)]
    trk = np.stack(cols, 1)
    med = np.median(trk, 1)
    feats["trk_std"] = _f(trk.std(1))
    feats["trk_iqr"] = _f(np.percentile(trk, 75, 1) - np.percentile(trk, 25, 1))
    feats["trk_med_d"] = _f(med - last_tvt)
    feats["pf_minus_med"] = _f(pf64 - med)
    # (4) GR-direction alignment (slides 6-7): does GR change match the expected log slope?
    d_gr = np.gradient(tw_gr); d_tvt = np.gradient(tw_tvt)
    tw_grad = d_gr / np.where(np.abs(d_tvt) < 1e-6, 1e-6, d_tvt)
    tw_grad_pf = np.interp(pf64, tw_tvt, tw_grad)
    grd = np.gradient(hgr64)
    dzc = np.nan_to_num(np.asarray(dzdmd, np.float64), nan=0.)
    feats["tw_grad_pf"] = _f(tw_grad_pf)
    feats["grdir_align"] = _f(grd * tw_grad_pf * dzc)
    # (6) normalized extrapolation distance + uncertainty interactions
    kl = max(float(known_len), 1.0)
    std64 = np.asarray(std_use, np.float64); ms = np.asarray(md_since, np.float64)
    feats["extrap_ratio"] = _f(ms / kl)
    feats["unc_x_dist"] = _f(std64 * ms)
    feats["unc_x_frac"] = _f(std64 * (np.arange(nh) / max(nh - 1, 1)))
    # ===== derived v2 batch =====
    # signed match residual (drift DIRECTION/bias; distinct from |mq|)
    ss = pd.Series(hgr64 - tw_at_pf)
    feats["sres_roll21"] = _f(ss.rolling(21, center=True, min_periods=1).mean().values)
    feats["sres_roll51"] = _f(ss.rolling(51, center=True, min_periods=1).mean().values)
    # best local shift over a small offset grid (is pf biased up/down here?)
    offs = np.array([-12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12], np.float64)
    rstack = np.stack([np.abs(hgr64 - np.interp(pf64 + o, tw_tvt, tw_gr)) for o in offs], 0)
    bi = rstack.argmin(0)
    feats["best_shift"] = _f(offs[bi])
    feats["best_absres"] = _f(rstack.min(0))
    # typewell local discriminability around pf (flat log => ambiguous => uncertain)
    tw_rstd = pd.Series(tw_gr).rolling(41, center=True, min_periods=1).std().fillna(0).values
    feats["twloc_std"] = _f(np.interp(pf64, tw_tvt, tw_rstd))
    # GR texture (hi/lo-freq energy ratio) + local slope
    hs = pd.Series(hgr64)
    g21 = hs.rolling(21, center=True, min_periods=1).std().fillna(0).values
    g101 = hs.rolling(101, center=True, min_periods=1).std().fillna(0).values
    feats["gr_hf_ratio"] = _f(g21 / (g101 + 1e-6))
    feats["gr_slope21"] = _f(np.gradient(hs.rolling(21, center=True, min_periods=1).mean().values))
    # PF trajectory velocity / acceleration (predicted TVT change rate)
    pv = np.gradient(pf64)
    feats["pf_vel"] = _f(pv)
    feats["pf_accel"] = _f(np.gradient(pv))
    # trajectory heading + inclination (slide 12: azimuth affects the experienced dip)
    dxm = np.nan_to_num(np.asarray(dxdmd, np.float64)); dym = np.nan_to_num(np.asarray(dydmd, np.float64))
    hyp = np.hypot(dxm, dym) + 1e-6
    feats["head_sin"] = _f(dym / hyp); feats["head_cos"] = _f(dxm / hyp)
    feats["incl"] = _f(dzc / hyp)
    # normalized extrapolation vs typewell range
    tw_rng = float(np.ptp(tw_tvt)) if np.ptp(tw_tvt) > 1e-6 else 1.0
    feats["extrap_vs_twrange"] = _f(ms / tw_rng)
    # ===== derived v3 batch: sequential signals not yet rolled (drift/steadiness) =====
    pfs = pd.Series(pf64)
    feats["pf_std21"] = _f(pfs.rolling(21, center=True, min_periods=1).std().fillna(0).values)   # PF jitter
    feats["pf_std51"] = _f(pfs.rolling(51, center=True, min_periods=1).std().fillna(0).values)
    dps = pd.Series(dzc)
    feats["dip_mean21"] = _f(dps.rolling(21, center=True, min_periods=1).mean().values)           # trajectory dip
    feats["dip_std51"] = _f(dps.rolling(51, center=True, min_periods=1).std().fillna(0).values)   # dip steadiness
    feats["grm201"] = _f(hs.rolling(201, center=True, min_periods=1).mean().values)               # slow GR trend
    feats["mq_roll201"] = _f(s.rolling(201, center=True, min_periods=1).mean().values)            # sustained mismatch
    feats["sres_cum"] = _f(ss.expanding().mean().values)                                          # systematic bias dir
    feats["gr_trend101"] = _f(np.gradient(hs.rolling(101, center=True, min_periods=1).mean().values))
    # ===== gr coverage / reliability (horizontal GR is ~28% NaN; flag interpolated regions) =====
    if gr_isna_ev is not None:
        miss = np.asarray(gr_isna_ev, np.float64)
        if len(miss) == nh:
            mser = pd.Series(miss)
            feats["gr_is_interp"] = _f(miss)                                           # per-point: GR was interpolated
            feats["gr_ev_valid_frac"] = _f(np.full(nh, 1.0 - float(miss.mean())))      # well-level eval GR coverage
            feats["gr_gap21"] = _f(mser.rolling(21, center=True, min_periods=1).mean().values)    # local missing density
            feats["gr_gap101"] = _f(mser.rolling(101, center=True, min_periods=1).mean().values)


def build_well(hw_path, tw_path, is_train, likpf_map=None):
    global _FI, _DI
    wid = Path(hw_path).stem.replace("__horizontal_well", "")
    try: hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except: return None
    if is_train and "TVT" not in hw.columns: return None
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw.TVT.isna().all(): return None
    tw_tvt = tw.TVT.to_numpy(np.float32); tw_gr = tw.GR.to_numpy(np.float32)
    if len(tw_tvt) < 3: return None
    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))
    lk = kn.iloc[-1]; last_tvt = float(lk.TVT_input)
    gr_full = hw.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32); kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)
    bpaths = {tag: beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r, tag) in BEAMS}
    beam_ref = (bpaths["cons"]+bpaths["sm5"])/2.
    ktvt = kn.TVT_input.to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]; sc_cons = (sc8+sc15+sc25)/3.
    sc_trust = float(np.clip(len(kn)/200., 0., 0.6)); hyb_ref = (1-sc_trust)*beam_ref+sc_trust*sc_ens
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32); a_cal, b_cal = affine_cal(kgr, tw_at_k)
    kmd = kn.MD.to_numpy(np.float32); kz = kn.Z.to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all = robust_slope(kmd, ktvt); slp_50 = robust_slope(kmd[-50:], ktvt[-50:]); slp_z = robust_slope(kz, ktvt)
    swid = wid if is_train else None
    xy_ev = ev[["X","Y"]].to_numpy(np.float64); xy_kn = kn[["X","Y"]].to_numpy(np.float64)
    form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid); form_kn, _ = _FI.impute(xy_kn, self_wid=swid)
    z_kn = kn.Z.to_numpy(np.float32); z_ev = ev.Z.to_numpy(np.float32)
    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev+form_ev[:, fi2]+b_full).astype(np.float32)
        tvt_fs[f"tvtF_{fn}"]=tvt_f; tvt_fs[f"tvtFw_{fn}"]=(-z_ev+form_ev[:,fi2]+b_wls).astype(np.float32)
        tvt_fs[f"tvtF50_{fn}"]=(-z_ev+form_ev[:,fi2]+b_late).astype(np.float32)
        tvt_fs[f"bw_{fn}"]=np.float32(b_full); tvt_fs[f"bww_{fn}"]=np.float32(b_wls); tvt_fs[f"bw50_{fn}"]=np.float32(b_late)
        tvt_fs[f"bw_early_{fn}"]=np.float32(b_early); tvt_fs[f"bw_mid_{fn}"]=np.float32(b_mid)
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2))); form_list.append(tvt_f)
    fs = np.stack(form_list, 1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32); form_std_d=fs.std(1).astype(np.float32); form_rng_d=(fs.max(1)-fs.min(1)).astype(np.float32)
    d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid); d_kn, d_std_kn, _ = _DI.impute(xy_kn, self_wid=swid)
    _, b_de, b_dm, b_dl, b_dw = seg_b_well(ktvt, z_kn, d_kn); b_d = float(np.median(ktvt+z_kn-d_kn))
    tvt_dense=(-z_ev+d_ancc+b_d).astype(np.float32); tvt_densew=(-z_ev+d_ancc+b_dw).astype(np.float32); tvt_dense50=(-z_ev+d_ancc+b_dl).astype(np.float32)
    res_kn = ktvt+z_kn-d_kn; d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))
    all_sigs=[pf_use]+list(bpaths.values())+[sc8,sc15,sc25,sc_ens,tvt_fs["tvtF_ANCC"],tvt_dense]
    sig_mat=np.stack(all_sigs,1); sig_std=sig_mat.std(1).astype(np.float32); sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)
    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1); rolls[f"grm{w}"]=r.mean().iloc[ev.index].values.astype(np.float32); rolls[f"grs{w}"]=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f"glag{lag}"]=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32); rolls[f"glead{lag}"]=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32); gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)).iloc[ev.index].values.astype(np.float32)
    hmd=ev.MD.to_numpy(np.float32); md_since=hmd-float(lk.MD)
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32); slp_b_50=(last_tvt+slp_50*md_since).astype(np.float32)
    mdd=hw.MD.diff().replace(0,np.nan)
    dzdmd=(hw.Z.diff()/mdd).iloc[ev.index].values.astype(np.float32); dxdmd=(hw.X.diff()/mdd).iloc[ev.index].values.astype(np.float32); dydmd=(hw.Y.diff()/mdd).iloc[ev.index].values.astype(np.float32)
    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)
    feats={"well":wid,"id":[f"{wid}_{i}" for i in ev.index],"last_known_tvt":sc(last_tvt),
        "pf_ancc":pf_use,"pf_ancc_std":std_use,"pf_ancc_delta":(pf_use-last_tvt).astype(np.float32),
        "pf_z":(pf_z.astype(np.float32) if has_z else sc(last_tvt)),"pf_z_delta":((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        "pf_vs_z":((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f"beam_{t}_d":(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        "beam_mean_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        "beam_std_d":np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        "beam_med_d":np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        "sc8_d":(sc8-np.float32(last_tvt)).astype(np.float32),"sc8_sc":sc8s,"sc15_d":(sc15-np.float32(last_tvt)).astype(np.float32),"sc15_sc":sc15s,
        "sc25_d":(sc25-np.float32(last_tvt)).astype(np.float32),"sc25_sc":sc25s,"sc_cons_d":(sc_cons-np.float32(last_tvt)).astype(np.float32),
        "sc_ens_d":(sc_ens-np.float32(last_tvt)).astype(np.float32),"sc_trust":sc(sc_trust),"hyb_d":(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        "sig_std":sig_std,"sig_mean_d":sig_mean,**tvt_fs,**{f"frm_rmse_{fn}":sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d":form_mean_d,"form_std_d":form_std_d,"form_rng_d":form_rng_d,
        "spatial_ancc_d":(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),"spatial_knn_dist":knn_d,
        "dense_ancc":d_ancc,"dense_std":d_std,"dense_dist":d_dist,"tvt_dense_d":(tvt_dense-last_tvt).astype(np.float32),
        "tvt_densew_d":(tvt_densew-last_tvt).astype(np.float32),"tvt_dense50_d":(tvt_dense50-last_tvt).astype(np.float32),
        "dense_rmse":sc(d_rmse),"dense_bias":sc(d_bias),"dense_nb_std":sc(d_nb_std),
        "pf_vs_spatial":(pf_use-tvt_fs["tvtF_ANCC"]).astype(np.float32),"pf_vs_dense":(pf_use-tvt_dense).astype(np.float32),
        "spatial_vs_dense":(tvt_fs["tvtF_ANCC"]-tvt_dense).astype(np.float32),"beam_vs_spatial":(bpaths["cons"]-tvt_fs["tvtF_ANCC"]).astype(np.float32),
        "sc_vs_beam":(sc_ens-bpaths["cons"]).astype(np.float32),"cal_a":sc(a_cal),"cal_b":sc(b_cal),
        "pfx_rmse":sc(pfx_rmse),"known_len":sc(len(kn)),"eval_len":sc(nh),"slp_all":sc(slp_all),"slp_50":sc(slp_50),"slp_z":sc(slp_z),
        "slp_b_d_all":(slp_b_all-last_tvt).astype(np.float32),"slp_b_d_50":(slp_b_50-last_tvt).astype(np.float32),
        "ktvt_range":sc(float(np.ptp(ktvt))),"ktvt_std":sc(float(ktvt.std())),"md_since":md_since,"frac":frac,"frac2":frac**2,"sqrt_frac":np.sqrt(frac),
        "z":z_ev,"dx":(ev.X-float(lk.X)).to_numpy(np.float32),"dy":(ev.Y-float(lk.Y)).to_numpy(np.float32),"dz":(z_ev-float(lk.Z)).astype(np.float32),
        "dxy":np.sqrt((ev.X-float(lk.X))**2+(ev.Y-float(lk.Y))**2).to_numpy(np.float32),"dzdmd":dzdmd,"dxdmd":dxdmd,"dydmd":dydmd,
        "gr":hgr,"gr_d1":gr_d1,"gr_d2":gr_d2,"gr_env":gr_env,"gr_nrg":gr_nrg,
        "gr_vs_tw_anc":hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),"gr_vs_slp_all":hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f"tda{int(o)}":hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f"tdbc{int(o)}":hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f"tdsc{int(o)}":hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f"tdpf{int(o)}":hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        "tw_range":sc(float(np.ptp(tw_tvt))),"tw_gr_mean":sc(float(tw_gr.mean()))}
    for k,v in rolls.items(): feats[k]=v
    add_derived_features(feats, hgr, tw_tvt, tw_gr, pf_use, pf_z, has_z, std_use,
                         np.stack(list(bpaths.values()), 1).mean(1).astype(np.float32),
                         sc_ens, tvt_dense, tvt_fs["tvtF_ANCC"], z_ev, dzdmd, dxdmd, dydmd,
                         last_tvt, a_cal, b_cal, md_since, len(kn), nh, ev["GR"].isna().to_numpy())
    res = pd.DataFrame(feats)
    if is_train: res["target"]=(ev.TVT.to_numpy(np.float32)-np.float32(last_tvt))
    return res

def init_imputers(train_wids):
    global _FI, _DI
    _FI = FormationPlaneKNN(train_wids, CFG.DATA/"train"); _DI = DenseANCCImputer(train_wids, CFG.DATA/"train")

def _likpf_rows(wid, split):
    hw, tw = load_well(wid, split)
    out, idx, _ = lik_pf(hw, tw)
    if not len(out): return None
    d = {"id": [f"{wid}_{i}" for i in idx]}
    for k, v in out.items():
        d["likpf_" + k.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = v.astype(np.float32)
    return pd.DataFrame(d)

def _chunks(seq, n):
    n = max(int(n), 1)
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def build_likpf(wids, split):
    # threads are safe here: the lik-PF numba kernel is compiled with nogil=True, so it
    # releases the GIL and parallelises across threads (no pickling of numba code needed).
    parts = []
    for chunk in _chunks(list(wids), CFG.BUILD_CHUNK):
        res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(delayed(_likpf_rows)(w, split) for w in chunk)
        parts.extend(r for r in res if r is not None)
        del res
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def build_features(wids, split, is_train):
    paths = [CFG.DATA/split/f"{w}__horizontal_well.csv" for w in wids]
    parts = []
    for chunk in _chunks(paths, CFG.BUILD_CHUNK):
        res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(
            delayed(build_well)(str(p), str(p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv"), is_train)
            for p in chunk if (p.parent/f"{p.stem.replace('__horizontal_well','')}__typewell.csv").exists())
        parts.extend(r for r in res if r is not None)
        del res
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = df[c].fillna(df["last_known_tvt"]); df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df

def _device():
    pref = str(CFG.USE_GPU).lower()
    if pref == "cpu": return "cpu", "CPU"
    if pref == "gpu": return "gpu", "GPU"
    if pref == "cuda": return "cuda", "CUDA"
    try:  # detect a real NVIDIA GPU (Kaggle GPU accelerator) via nvidia-smi
        import subprocess
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
            return "gpu", "GPU"
    except Exception:
        pass
    return "cpu", "CPU"

def lgb_configs(dev):
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if dev == "gpu": base.update(device_type="gpu", gpu_use_dp=False)
    elif dev == "cuda": base.update(device_type="cuda")
    n = 600 if CFG.FAST else 5000
    return [
        dict(**base, num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03, n_estimators=n, seed=123),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=0),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=29),
    ]

_LGB_DEVICE_CHOICE = None

def _lgb_device_order():
    pref = str(CFG.USE_GPU).lower()
    if pref == "cpu":
        return ["cpu"]
    if pref == "cuda":
        return ["cuda", "cpu"]
    if pref == "gpu":
        return ["gpu", "cuda", "cpu"]
    dev, _ = _device()
    return ["gpu", "cuda", "cpu"] if dev in {"gpu", "cuda"} else ["cpu"]

def _lgb_params_for_device(params, dev):
    p = dict(params)
    p.pop("device_type", None)
    p.pop("gpu_use_dp", None)
    if dev == "gpu":
        p.update(device_type="gpu", gpu_use_dp=False)
    elif dev == "cuda":
        p.update(device_type="cuda")
    return p

def fit_lgb_with_fallback(params, x_train, y_train, fit_kwargs=None, context="lgb"):
    from lightgbm import LGBMRegressor
    global _LGB_DEVICE_CHOICE
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    order = _lgb_device_order()
    if _LGB_DEVICE_CHOICE in order:
        order = [_LGB_DEVICE_CHOICE] + [d for d in order if d != _LGB_DEVICE_CHOICE]
    errors = []
    for i, dev in enumerate(order):
        p = _lgb_params_for_device(params, dev)
        try:
            m = LGBMRegressor(**p)
            m.fit(x_train, y_train, **fit_kwargs)
            if _LGB_DEVICE_CHOICE != dev:
                print(f"[lgb] {context}: using {dev.upper()} mode", flush=True)
            _LGB_DEVICE_CHOICE = dev
            return m
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else e}"
            errors.append(f"{dev} -> {msg}")
            if i + 1 < len(order):
                print(f"[lgb] {context}: {dev.upper()} failed ({msg}); trying {order[i+1].upper()}", flush=True)
            try:
                del m
            except Exception:
                pass
            import gc as _gc
            _gc.collect()
    raise RuntimeError("LightGBM failed on all device modes: " + " | ".join(errors))
def cb_configs(dev):
    tt = "GPU" if dev == "gpu" else "CPU"
    n = 800 if CFG.FAST else 8000
    return [
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),
    ]

def train_stack(train_df, test_df, features):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    dev, devname = _device(); print("device:", devname)
    X = train_df[features].values.astype(np.float32); y = train_df["target"].values.astype(np.float32)
    g = train_df["well"].values; Xt = test_df[features].values.astype(np.float32)
    cv = GroupKFold(CFG.n_splits); oof_cols = {}; test_cols = {}
    def run(name, make, fit_kw, is_lgb):
        # LightGBM: slice to best_iteration_ via num_iteration. CatBoost: use_best_model
        # already trims to the best tree, and its predict() takes no num_iteration kwarg.
        oof = np.zeros(len(train_df)); tp = np.zeros(len(test_df))
        for tr, va in cv.split(X, y, groups=g):
            if is_lgb:
                m = fit_lgb_with_fallback(make(), X[tr], y[tr],
                                          dict(eval_set=[(X[va], y[va])], **fit_kw),
                                          context=f"{name}/fold")
            else:
                m = make(); m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **fit_kw)
            if is_lgb:
                it = m.best_iteration_
                oof[va] = m.predict(X[va], num_iteration=it); tp += m.predict(Xt, num_iteration=it) / CFG.n_splits
            else:
                oof[va] = m.predict(X[va]); tp += m.predict(Xt) / CFG.n_splits
        oof_cols[name] = oof; test_cols[name] = tp
        print(f"  {name}: OOF RMSE={rmse(y, oof):.4f}", flush=True)
    for i, p in enumerate(lgb_configs(dev)):
        run(f"lgb{i}", lambda p=p: p,
            dict(eval_metric="rmse", callbacks=[early_stopping(250, verbose=False), log_evaluation(0)]), True)
    for i, p in enumerate(cb_configs(dev)):
        run(f"cb{i}", lambda p=p: CatBoostRegressor(**p),
            dict(early_stopping_rounds=250, use_best_model=True), False)
    # GRU base learner (sequence model over the eval zone); joins the meta-stack.
    # No-op unless torch + CUDA are present (ROGII_GRU=add by default).
    gru_out = train_gru_oof(train_df, test_df, features, cv)
    if gru_out is not None:
        oof_cols['gru0'], test_cols['gru0'] = gru_out
        print(f"  gru0: OOF RMSE={rmse(y, oof_cols['gru0']):.4f}")
    OOF = pd.DataFrame(oof_cols); TEST = pd.DataFrame(test_cols)
    rid = Ridge(alpha=1.66, positive=True, fit_intercept=True); meta = np.zeros(len(train_df))
    for tr, va in cv.split(OOF.values, y, groups=g):
        rid.fit(OOF.values[tr], y[tr]); meta[va] = rid.predict(OOF.values[va])
    rid.fit(OOF.values, y); meta_test = rid.predict(TEST.values)
    print(f"  ridge-stack OOF RMSE={rmse(y, meta):.4f}")
    return meta, meta_test, OOF, TEST

# ===== full-stack retrain: LGB?? + CatBoost?? + Ridge meta, save for inference =====
import gc, json, joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor

train_wids = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"train").glob("*__horizontal_well.csv"))
test_wids  = sorted(p.stem.replace("__horizontal_well", "") for p in (CFG.DATA/"test").glob("*__horizontal_well.csv"))
_MAX = int(os.environ.get("ROGII_STACK_MAXWELLS", "0"))
if _MAX:
    train_wids = train_wids[:_MAX]
if not train_wids:
    raise SystemExit(f"[data] No wells found under {CFG.DATA/'train'} ??set os.environ['ROGII_DATA'] "
                     f"to the folder that contains train/ and test/ (current DATA={CFG.DATA})")
print(f"training wells: {len(train_wids)} | example test: {len(test_wids)}", flush=True)

def _feat_cache_name(_n):
    return f"train_features_v6_s1gr_f{os.environ.get('ROGII_FEATS','1')}_w{_n}.parquet"

def load_or_build_train_features(train_wids):
    # ROGII_FEATCACHE: auto (use cache if present) | rebuild (force) | off (build, don't save)
    import glob as _g
    name = _feat_cache_name(len(train_wids)); mode = os.environ.get("ROGII_FEATCACHE", "auto")
    if mode != "rebuild":
        for _p in _g.glob(f"/kaggle/input/**/{name}", recursive=True) + [str(CFG.OUT / name)]:
            if os.path.exists(_p):
                print(f"[cache] loading train features from {_p}", flush=True)
                return pd.read_parquet(_p)
    print("[cache] building train features (slow ~2-3h)...", flush=True)
    _df = add_likpf_features(build_features(train_wids, "train", is_train=True), build_likpf(train_wids, "train"))
    if mode != "off":
        try:
            _df.to_parquet(CFG.OUT / name, index=False)
            print(f"[cache] saved -> {CFG.OUT / name}  (publish OUT as a dataset to reuse next runs)", flush=True)
        except Exception as _e:
            print("[cache] save skipped:", _e, flush=True)
    return _df

init_imputers(train_wids)   # cheap KDTrees; needed for test features regardless of cache
train_df = load_or_build_train_features(train_wids)
gc.collect()
test_df  = add_likpf_features(build_features(test_wids, "test", is_train=False),   build_likpf(test_wids, "test"))
gc.collect()

feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]
print(f"features: {len(feats)} | train rows: {len(train_df)}", flush=True)

# optional feature pruning: from a prior feature_importance.csv or an explicit drop list
_impf = os.environ.get("ROGII_IMP_FILE", "")
_drop = set(x for x in os.environ.get("ROGII_DROP_FEATS", "").split(",") if x)
if _impf:
    import glob as _g2
    _cand = [_impf] if os.path.exists(_impf) else _g2.glob(f"/kaggle/input/**/{os.path.basename(_impf)}", recursive=True)
    if _cand:
        _imp = pd.read_csv(_cand[0]); _topn = int(os.environ.get("ROGII_TOP_FEATS", "0")); _minimp = float(os.environ.get("ROGII_MIN_IMP", "0"))
        _k = _imp.sort_values("gain", ascending=False)
        if _topn: _k = _k.head(_topn)
        if _minimp: _k = _k[_k["gain"] >= _minimp]
        _keep = set(_k["feature"]); feats = [c for c in feats if c in _keep and c not in _drop]
        print(f"[prune] kept {len(feats)} feats via {_cand[0]} (top={_topn}, min_imp={_minimp})", flush=True)
    else:
        print(f"[prune] ROGII_IMP_FILE not found: {_impf}", flush=True)
elif _drop:
    feats = [c for c in feats if c not in _drop]
    print(f"[prune] dropped {len(_drop)} -> {len(feats)} feats remain", flush=True)

dev, _ = _device()
X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values
# residual re-anchoring (ROGII_RESID=0 default — measured worse on LGB valid): opt-in via ROGII_RESID=1;
# inference adds the anchor back (stack_meta.json carries the flag). RMSE prints stay in delta space.
_resid = os.environ.get("ROGII_RESID", "0") == "1"   # default OFF: residual target hurt LGB valid scores (measured 2026-07-09)
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y - anchor
if _resid:
    print(f"[resid] target re-anchored to likpf_mean_d (target std {y.std():.2f} -> residual std {y_fit.std():.2f})", flush=True)
_folds = int(os.environ.get("ROGII_STACK_FOLDS", str(CFG.n_splits)))
cv = GroupKFold(_folds)
use_cb = os.environ.get("ROGII_STACK_CB", "1") == "1"
lgb_cfgs = lgb_configs(dev); cb_cfgs = cb_configs(dev)

base_names = []; oof_cols = {}; full_iters = {}
imp_sum = np.zeros(len(feats))   # accumulated LGB gain importance

# --- LGB base models: GroupKFold OOF + mean best_iteration ---
for ci, params in enumerate(lgb_cfgs):
    name = f"lgb{ci}"; oof = np.zeros(len(train_df)); iters = []
    for tr, va in cv.split(X, y, groups=g):
        m = fit_lgb_with_fallback(params, X[tr], y_fit[tr],
                                  dict(eval_set=[(X[va], y_fit[va])], eval_metric="rmse",
                                       callbacks=[early_stopping(250, verbose=False), log_evaluation(0)]),
                                  context=f"{name}/fold")
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration_)
        iters.append(int(m.best_iteration_ or params.get("n_estimators", 1000)))
        imp_sum += m.booster_.feature_importance(importance_type="gain")
        del m; gc.collect()
    oof_cols[name] = oof; full_iters[name] = max(50, int(np.mean(iters))); base_names.append(name)
    print(f"{name} OOF RMSE={rmse(y, oof + anchor):.4f}  (avg best_iter={full_iters[name]})", flush=True)

# --- feature importance (aggregated LGB gain across folds+configs); pruning is OFF by default ---
imp_df = pd.DataFrame({"feature": feats, "gain": imp_sum}).sort_values("gain", ascending=False).reset_index(drop=True)
imp_df["gain_pct"] = (100.0 * imp_df["gain"] / max(imp_df["gain"].sum(), 1e-9)).round(3)
imp_df.to_csv(CFG.OUT / "feature_importance.csv", index=False)
print("[imp] saved feature_importance.csv (pruning is OFF unless ROGII_IMP_FILE/ROGII_DROP_FEATS is set)", flush=True)
print(f"[imp] full ranked feature importance (LGB gain), {len(imp_df)} features:\n"
      + imp_df.to_string(index=True), flush=True)

# --- CatBoost base models ---
if use_cb:
    for ci, params in enumerate(cb_cfgs):
        name = f"cb{ci}"; oof = np.zeros(len(train_df)); iters = []
        for tr, va in cv.split(X, y, groups=g):
            m = CatBoostRegressor(**params)
            m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], use_best_model=True, early_stopping_rounds=250)
            oof[va] = m.predict(X[va])
            iters.append(int(m.get_best_iteration() or params.get("iterations", 1000)))
            del m; gc.collect()
        oof_cols[name] = oof; full_iters[name] = max(50, int(np.mean(iters))); base_names.append(name)
        print(f"{name} OOF RMSE={rmse(y, oof + anchor):.4f}  (avg best_iter={full_iters[name]})", flush=True)

# --- Ridge meta on the base OOF ---
OOF = np.column_stack([oof_cols[n] for n in base_names])
meta_oof = np.zeros(len(train_df))
for tr, va in cv.split(OOF, y_fit, groups=g):
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(OOF[tr], y_fit[tr]); meta_oof[va] = r.predict(OOF[va])
print(f"*** ridge-stack OOF RMSE={rmse(y, meta_oof + anchor):.4f}  (mean-LGB baseline was ~9.86; delta space) ***", flush=True)
ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True); ridge.fit(OOF, y_fit)
print("ridge coefs:", dict(zip(base_names, [round(float(c), 4) for c in ridge.coef_])),
      "| intercept", round(float(ridge.intercept_), 4), flush=True)

# --- refit base models on ALL data with CV-chosen sizes, save everything ---
outdir = CFG.OUT
for ci, params in enumerate(lgb_cfgs):
    p = dict(params); p["n_estimators"] = full_iters[f"lgb{ci}"]
    m = fit_lgb_with_fallback(p, X, y_fit, context=f"lgb{ci}/full"); joblib.dump(m, outdir / f"lgb{ci}.pkl")
    print(f"saved lgb{ci}.pkl (n_estimators={p['n_estimators']})", flush=True)
if use_cb:
    for ci, params in enumerate(cb_cfgs):
        p = dict(params); p["iterations"] = full_iters[f"cb{ci}"]
        m = CatBoostRegressor(**p); m.fit(X, y_fit); m.save_model(str(outdir / f"cb{ci}.cbm"))
        print(f"saved cb{ci}.cbm (iterations={p['iterations']})", flush=True)
joblib.dump(ridge, outdir / "ridge.pkl")
json.dump({"base_names": base_names, "use_cb": use_cb, "ridge_oof_rmse": float(rmse(y, meta_oof + anchor)),
           "residual_anchor": ("likpf_mean_d" if _resid else None)},
          open(outdir / "stack_meta.json", "w"))
json.dump(feats, open(outdir / "features.json", "w"))
print("DONE. saved lgb*.pkl, cb*.cbm, ridge.pkl, stack_meta.json, features.json ->", outdir, flush=True)
print("Publish as a dataset and attach to the submission notebook (detach old lgb dataset).", flush=True)