# Cauchy emission likelihood in the lik-PF: selector-level A/B on the 2x150 harness.
# Gaussian arm = stored eval_data channels (collect_eval, 32 seeds) — parity-checked by
# recomputing a few wells with cv_harness.lik_pf. Cauchy arm = identical kernel except
# the emission: exp(-0.5*dd) -> 1/(1+0.5*dd) (same small-d curvature, heavy tail so a
# repeated bed cannot dominate; the STRIDE lesson applied to the PF).
import io, time
import numpy as np
from joblib import Parallel, delayed
import cv_harness as H
from offline_tests import load, pooled, b0, ba, robfit
import stride

# ---- build the cauchy kernel module by source transform of cv_harness ----
src = io.open("cv_harness.py", encoding="utf-8").read()
i0 = src.index("def _interp1")
i0 = src.rindex("@njit", 0, i0)
i1 = src.index("def _grcal_tw")
kern = src[i0:i1]
assert "def _pf_lik_allseeds" in kern and kern.count("lk = np.exp(-0.5*dd)") == 1
kern = kern.replace("def _pf_lik_allseeds", "def _pf_lik_allseeds_cauchy")
kern = kern.replace("""                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300""",
                    """                lk = 1.0/(1.0 + 0.5*dd)   # Cauchy emission (heavy tail)""")
io.open("_cauchy_kern.py", "w", encoding="utf-8").write(
    "import numpy as np\nfrom numba import njit\n" + kern)
import _cauchy_kern as CK

def lik_pf_cauchy(hw, tw, n_particles=500, n_seeds=32, scales=(3., 5., 8.), grcal="off"):
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]; ev = hw[hw.TVT_input.isna()]
    if len(ev) == 0:
        return {}, np.array([])
    last = kn.iloc[-1]; ls = float(last.TVT_input) + float(last.Z)
    if grcal in ("affine", "var", "offset"):
        tw_gr, _, _ = H._grcal_tw(tw_tvt, tw_gr, kn.TVT_input.values, kn.GR.values, grcal)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail.TVT_input.values); dz = np.diff(tail.Z.values); dm = np.diff(tail.MD.values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = H._grid(tw_tvt, tw_gr)
    gr_v = hw.GR.interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = CK._pf_lik_allseeds_cauchy(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                             gg, gmin, gst, gs, ls, ir, n_particles, n_seeds, 0,
                                             0.998, 0.002, 0.005, 0.1, 0.001, 0.5, 4.5)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln / float(sc)); wts /= wts.sum()
        out[f"{sc:g}"] = (wts[:, None] * preds).sum(0)
    return out, ev.index.values

S2_THR = 0.0083

def chain_from(o3, o5, o8, a3, a5, a8, rec):
    p = 0.65 * ((o3 + o5 + o8) / 3.0) + 0.35 * ((a3 + a5 + a8) / 3.0)
    if float(np.median(rec["dense_dist"])) <= S2_THR:
        p = 0.7 * p + 0.3 * rec["tvt_dense"]
    return p

def project(rec, pred):
    anchor = rec["last"] + rec["z_ps"]
    du = pred + rec["z"] - anchor
    s = (rec["md"] - rec["md_ps"]) / max(rec["md"][-1] - rec["md_ps"], 1e-6)
    fit = robfit(s, du, 4, anchor_ps=True)
    return (anchor + 0.25 * du + 0.75 * fit) - rec["z"]

def cauchy_well(rec):
    hw, tw = H.load_well(rec["wid"])
    off, _ = lik_pf_cauchy(hw, tw, grcal="off")
    aff, _ = lik_pf_cauchy(hw, tw, grcal="affine")
    if not off:
        return None
    return chain_from(off["3"], off["5"], off["8"], aff["3"], aff["5"], aff["8"], rec)

def stride_well(rec):
    hw, tw = stride.load_well(rec["wid"])
    st, _ = stride.stride_track(hw, tw)
    return st

if __name__ == "__main__":
    t0 = time.time()
    # gaussian parity: recomputing with cv_harness must reproduce the stored channels
    res7 = load(7)
    for rec in res7[:2]:
        hw, tw = H.load_well(rec["wid"])
        out, _ = H.lik_pf(hw, tw, n_seeds=32)
        d = float(np.abs(out["pf_scale_3"] - rec["o3"]).max())
        print(f"parity {rec['wid']}: max|fresh-stored o3| = {d:.2e}", flush=True)
        assert d < 1e-9, "gaussian parity FAILED — stored channels not reproducible"
    print(f"[{time.time()-t0:.0f}s] parity OK", flush=True)
    for seed in (7, 11):
        res = load(seed)
        G = [chain_from(r["o3"], r["o5"], r["o8"], r["a3"], r["a5"], r["a8"], r) for r in res]
        C = Parallel(n_jobs=24, prefer="threads")(delayed(cauchy_well)(r) for r in res)
        ST = Parallel(n_jobs=24, prefer="threads")(delayed(stride_well)(r) for r in res)
        ok = [i for i, c in enumerate(C) if c is not None]
        res = [res[i] for i in ok]; G = [G[i] for i in ok]; C = [C[i] for i in ok]; ST = [ST[i] for i in ok]
        print(f"=== seed {seed} | {len(res)} wells ===", flush=True)
        print(f"  chain(selector)   gauss={pooled(res, G):.4f}  cauchy={pooled(res, C):.4f}", flush=True)
        for tag, CH in [("gauss", G), ("cauchy", C)]:
            B = []
            for c, s_, r in zip(CH, ST, res):
                p = np.asarray(c, float)
                if s_ is not None and len(s_) == len(p) and np.all(np.isfinite(s_)):
                    p = 0.8 * p + 0.2 * s_
                B.append(p)
            P = [project(r, p) for p, r in zip(B, res)]
            print(f"  +stride+proj      {tag}: {pooled(res, P):.4f}", flush=True)
    print(f"done [{time.time()-t0:.0f}s]", flush=True)
