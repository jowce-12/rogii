# STRIDE v1 tuning/validation on the saved eval samples (tune on seed7, confirm on seed11).
# Reports: standalone pooled RMSE, error correlation vs deployed chain, blend sweep,
# and blended+projection (deployed P1) — the deployment-equivalent quantity.
import sys, time
import numpy as np
from joblib import Parallel, delayed
import stride
from offline_tests import load, pooled, b0, ba, robfit
from measure_wiggle import chain, project_v

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
res = load(SEED)
print(f"=== seed {SEED} | {len(res)} wells ===", flush=True)

def decode(rec, seg_len, K, sig_pers, jump_pen, rate_step=0.002):
    """Run one decode, return (paths_u_topM, scores_topM, z) aligned to rec rows."""
    hw, tw = stride.load_well(rec["wid"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    tw_gr = stride.grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
    gmin = tw_tvt[0]
    gg_x = np.arange(gmin, tw_tvt[-1] + 0.5, 0.5)
    gg = np.interp(gg_x, tw_tvt, tw_gr)
    last = kn.iloc[-1]
    u0 = float(last["TVT_input"]) + float(last["Z"])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    s0 = float(np.clip(np.median((dt + dz)[m] / dm[m]) if m.sum() >= 3 else 0.0, -0.06, 0.06))
    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gam = float(np.clip(np.nanmedian(np.abs(kn["GR"].values.astype(float) - tw_at_k)), 5.0, 40.0))
    md = ev["MD"].values.astype(float); z = ev["Z"].values.astype(float)
    gr = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    bnds = [0]; cur = md[0]
    for i in range(len(md)):
        if md[i] >= cur + seg_len:
            bnds.append(i); cur = md[i]
    if bnds[-1] != len(md):
        bnds.append(len(md))
    bnds = np.array(bnds, dtype=np.int64)
    rates = np.arange(-0.06, 0.06 + rate_step / 2, rate_step)
    br, sc = stride._decode(md, z, gr, gg, gmin, 0.5, u0, s0, gam, rates, bnds,
                            K, sig_pers, jump_pen, 0.0)
    paths = stride._paths_from_rates(md, bnds, u0, rates, br)
    order = np.argsort(sc)[::-1][:32]
    return paths[order], sc[order], z

def agg(paths, sc, z, mode):
    if mode == "best":
        return paths[0] - z
    if mode.startswith("top"):
        m = int(mode[3:])
        return paths[:m].mean(0) - z
    if mode.startswith("sm"):          # softmax with per-row-normalized temperature
        t = float(mode[2:])
        s = (sc - sc[0]) / max(len(z), 1)     # per-row score gap
        w = np.exp(s / max(t, 1e-9)); w /= w.sum()
        return (w[:, None] * paths).sum(0) - z

def run_cfg(seg_len, K, sig_pers, jump_pen, modes):
    raw = Parallel(n_jobs=24, prefer="threads")(
        delayed(decode)(r, seg_len, K, sig_pers, jump_pen) for r in res)
    out = {}
    for mode in modes:
        preds = [agg(p, s, z, mode) for (p, s, z) in raw]
        out[mode] = preds
    return out

t0 = time.time()
CH = chain(res)                                # deployed T1+T2 selector
print(f"chain selector pooled = {pooled(res, CH):.4f}")

# --- stage 1: aggregation mode (fixed decode) ---
MODES = ["best", "top4", "top8", "top16", "sm0.002", "sm0.005", "sm0.02"]
base_cfg = dict(seg_len=250.0, K=96, sig_pers=0.006, jump_pen=12.0)
outs = run_cfg(**base_cfg, modes=MODES)
print(f"-- stage1 aggregation (cfg {base_cfg}) [{time.time()-t0:.0f}s]")
best_mode, best_v = None, 1e9
for m in MODES:
    v = pooled(res, outs[m])
    print(f"  {m:8s} {v:.4f}")
    if v < best_v: best_v, best_mode = v, m
print(f"  -> {best_mode}")

# --- stage 2: decode params ---
print("-- stage2 decode params --")
best_cfg = dict(base_cfg)
for name, vals in [("sig_pers", [0.003, 0.006, 0.012]),
                   ("jump_pen", [6.0, 12.0, 25.0]),
                   ("seg_len", [150.0, 250.0, 400.0])]:
    best_pv = 1e9; best_val = best_cfg[name]
    for v in vals:
        cfg = dict(best_cfg); cfg[name] = v
        preds = run_cfg(**cfg, modes=[best_mode])[best_mode]
        pv = pooled(res, preds)
        print(f"  {name}={v}  {pv:.4f}", flush=True)
        if pv < best_pv: best_pv, best_val = pv, v
    best_cfg[name] = best_val
print(f"  -> tuned cfg: {best_cfg} ({best_pv:.4f})")

# --- final: standalone + correlation + blend + projection ---
ST = run_cfg(**best_cfg, modes=[best_mode])[best_mode]
print(f"STRIDE standalone pooled = {pooled(res, ST):.4f}")
e1 = np.concatenate([p - r["y"] for p, r in zip(CH, res)])
e2 = np.concatenate([p - r["y"] for p, r in zip(ST, res)])
print(f"error corr(chain, stride) = {np.corrcoef(e1, e2)[0,1]:.3f}")
print("-- blend sweep (selector level) --")
for w in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    B = [(1 - w) * c + w * s for c, s in zip(CH, ST)]
    P = [project_v(r, p) for p, r in zip(B, res)]     # deployed PS-deg4 projection
    print(f"  w={w:.2f}  blend={pooled(res, B):.4f}  +proj={pooled(res, P):.4f}")
print(f"done in {time.time()-t0:.0f}s")
