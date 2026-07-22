import numpy as np
import pandas as pd

def _s2_grcal(tw_tvt, tw_gr, ktvt, kgr):
    ktvt = np.asarray(ktvt, float); kgr = np.asarray(kgr, float)
    v = np.isfinite(kgr)
    if v.sum() < 30: return tw_gr
    band = tw_gr[(tw_tvt >= float(np.nanmin(ktvt)) - 10.0) & (tw_tvt <= float(np.nanmax(ktvt)) + 10.0)]
    if len(band) < 20: band = tw_gr
    bm = float(np.mean(band)); km = float(np.mean(kgr[v]))
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    vv = v & np.isfinite(tw_at_k)
    if vv.sum() < 30 or np.std(tw_at_k[vv]) < 1e-6:
        a, b = 1.0, km - bm
    else:
        a, b = np.polyfit(tw_at_k[vv], kgr[vv], 1)
        if (not np.isfinite(a)) or a < 0.2 or a > 5.0:
            a, b = 1.0, km - bm
    return a * tw_gr + b

def _stride_decode(md, z, gr, gg, gmin, gstep, u0, s0, gam, rates, bnds,
                   K, sig_pers, jump_pen, lik_w):
    n_seg = len(bnds) - 1; R = len(rates); ng = len(gg)
    beam_u = np.empty(K); beam_s = np.empty(K); beam_sc = np.empty(K)
    beam_rates = np.full((K, n_seg), -1, dtype=np.int16)
    beam_u[0] = u0; beam_s[0] = s0; beam_sc[0] = 0.0
    kb = 1
    cand_sc = np.empty(K * R); cand_u = np.empty(K * R)
    cand_par = np.empty(K * R, dtype=np.int32); cand_r = np.empty(K * R, dtype=np.int16)
    new_u = np.empty(K); new_s = np.empty(K); new_sc = np.empty(K)
    new_rates = np.full((K, n_seg), -1, dtype=np.int16)
    keys = np.empty(K, dtype=np.int64)
    for seg in range(n_seg):
        i0, i1 = bnds[seg], bnds[seg + 1]
        md0 = md[i0 - 1] if i0 > 0 else md[0] - (md[1] - md[0])
        nc = 0
        for b in range(kb):
            for ri in range(R):
                r = rates[ri]; ll = 0.0; ub = beam_u[b]
                for i in range(i0, i1):
                    tvt = ub + r * (md[i] - md0) - z[i]
                    gi = int((tvt - gmin) / gstep)
                    if gi < 0: gi = 0
                    elif gi >= ng: gi = ng - 1
                    d = (gr[i] - gg[gi]) / gam
                    ll += -np.log(1.0 + d * d)
                ds = (r - beam_s[b]) / sig_pers
                pen = ds * ds
                if pen > jump_pen: pen = jump_pen
                cand_sc[nc] = beam_sc[b] + lik_w * ll - pen
                cand_u[nc] = beam_u[b] + r * (md[i1 - 1] - md0)
                cand_par[nc] = b; cand_r[nc] = ri; nc += 1
        order = np.argsort(cand_sc[:nc])[::-1]
        nk = 0
        for oi in range(nc):
            c = order[oi]
            key = np.int64(np.round(cand_u[c] * 2.0)) * 1000 + cand_r[c]
            dup = False
            for j in range(nk):
                if keys[j] == key:
                    dup = True; break
            if dup: continue
            keys[nk] = key
            new_u[nk] = cand_u[c]; new_s[nk] = rates[cand_r[c]]; new_sc[nk] = cand_sc[c]
            p = cand_par[c]
            for s2 in range(seg):
                new_rates[nk, s2] = beam_rates[p, s2]
            new_rates[nk, seg] = cand_r[c]
            nk += 1
            if nk >= K: break
        kb = nk
        beam_u[:kb] = new_u[:kb]; beam_s[:kb] = new_s[:kb]; beam_sc[:kb] = new_sc[:kb]
        beam_rates[:kb] = new_rates[:kb]
    return beam_rates[:kb], beam_sc[:kb]

def _stride_paths(md, bnds, u0, rates, beam_rates):
    kb, n_seg = beam_rates.shape
    paths = np.empty((kb, len(md)))
    for k in range(kb):
        u = u0
        for seg in range(n_seg):
            i0, i1 = bnds[seg], bnds[seg + 1]
            md0 = md[i0 - 1] if i0 > 0 else md[0] - (md[1] - md[0])
            r = rates[beam_rates[k, seg]]
            for i in range(i0, i1):
                paths[k, i] = u + r * (md[i] - md0)
            u = u + r * (md[i1 - 1] - md0)
    return paths

def _stride_track(hw, tw, seg_len=200.0, K=96, sig_pers=0.012, jump_pen=25.0, lik_w=0.1):
    """Eval-zone TVT from a joint beam decode of the surface (deterministic)."""
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) < 5 or len(kn) < 10 or len(tw_tvt) < 20:
        return None
    tw_gr = _s2_grcal(tw_tvt, tw_gr, kn['TVT_input'].values, kn['GR'].values)
    gmin = tw_tvt[0]
    gg = np.interp(np.arange(gmin, tw_tvt[-1] + 0.5, 0.5), tw_tvt, tw_gr)
    _l = kn.iloc[-1]
    u0 = float(_l['TVT_input']) + float(_l['Z'])
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    s0 = float(np.clip(np.median((dt + dz)[m] / dm[m]) if m.sum() >= 3 else 0.0, -0.06, 0.06))
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gam = float(np.clip(np.nanmedian(np.abs(kn['GR'].values.astype(float) - tw_at_k)), 5.0, 40.0))
    md_v = ev['MD'].values.astype(float); z_v = ev['Z'].values.astype(float)
    gr_v = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)[ev.index]
    bnds = [0]; cur = md_v[0]
    for i in range(len(md_v)):
        if md_v[i] >= cur + seg_len:
            bnds.append(i); cur = md_v[i]
    if bnds[-1] != len(md_v):
        bnds.append(len(md_v))
    bnds = np.array(bnds, dtype=np.int64)
    rates = np.arange(-0.06, 0.06 + 0.001, 0.002)
    br, sc = _stride_decode(md_v, z_v, gr_v, gg, gmin, 0.5, u0, s0, gam, rates, bnds,
                            K, sig_pers, jump_pen, lik_w)
    paths = _stride_paths(md_v, bnds, u0, rates, br)
    order = np.argsort(sc)[::-1][:32]
    s = (sc[order] - sc[order][0]) / max(len(z_v), 1)
    w = np.exp(s / 0.02); w /= w.sum()
    return (w[:, None] * paths[order]).sum(0) - z_v
