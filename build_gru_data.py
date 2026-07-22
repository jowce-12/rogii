# Whole-well GRU data pipeline (radiantallomancer recipe adapted to our stack).
# Sample = (well, cut): natural cut + artificial prefix cuts at 60%/80% of the known
# zone (train wells have full TVT truth, so artificial tails extend to well end).
# 4-ft MD grid; channels: heel-normalized GR family, trajectory, position, typewell
# mismatch profile at 13 fixed TVT offsets (grcal'd), and per-cut STRIDE decodes
# (the one physics feature cheap enough to recompute per artificial cut).
# Targets: (TVT - anchor)/10 on tail steps. Validation mapping: natural-cut original
# eval-row (id, md) saved so the trainer can interp grid predictions back to rows.
# Outputs: gru_seq.parquet (long: well, cut, step, chan..., target, is_tail),
#          gru_rowmap.parquet (well, id, md) for natural-cut OOF scoring.
# RUN: python build_gru_data.py [--limit N]
import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

WORKERS = 10
STEP = 4.0
CTX = 256                     # prefix context steps fed to the GRU (256*4ft = 1024ft)
OFFS = np.arange(-48.0, 48.1, 8.0)   # typewell mismatch offsets (13)
CUT_FRACS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # dense prefix-cut augmentation (v2)


def _stride_channels(hw_m, tw, md_grid, last_tvt):
    """Per-cut STRIDE decodes on the 4ft grid. Returns (channels[N,3], hyp_tvt[N]) where
    hyp_tvt is the unclipped default-decode TVT path (anchor-held outside the tail)."""
    from stride import stride_track
    out = np.zeros((len(md_grid), 3), np.float32)
    hyp = np.full(len(md_grid), last_tvt, float)
    ev = hw_m[hw_m["TVT_input"].isna()]
    if len(ev) < 5:
        return out, hyp
    md_ev = ev["MD"].values.astype(float)
    for k, seg in enumerate((200.0, 400.0, 100.0)):
        pred, _ = stride_track(hw_m, tw, seg_len=seg)
        if pred is None:
            continue
        p = np.interp(md_grid, md_ev, pred, left=last_tvt, right=float(pred[-1]))
        if k == 0:
            hyp = p.copy()
        out[:, k] = np.clip((p - last_tvt) / 10.0, -12, 12)
    return out, hyp


PF_SEEDS = 32   # reduced-seed PF per cut: GRU input, not a final estimate


def _pf_channels(hw_m, md_grid, tw, last_tvt, ev_idx_md):
    """Per-cut PF ensemble (verbatim notebook PF, original typewell, seeds 0..31).
    Returns (channels[N,4] = pf3/pf5/pf8 anchor-rel /10 + across-seed std /10,
             pf5_path_tvt[N] for the mismatch hypothesis)."""
    import _t1_pf as PF
    out = np.zeros((len(md_grid), 4), np.float32)
    hyp = np.full(len(md_grid), last_tvt, float)
    ev_idx, md_ev = ev_idx_md
    if len(ev_idx) < 5:
        return out, hyp
    pairs = PF._pf_seed_batch(hw_m, tw, 500, range(PF_SEEDS))
    preds = np.stack([p[ev_idx] for p, _ll in pairs], 0)
    liks = np.array([ll for _p, ll in pairs]); liks = liks - liks.max()
    for k, scale in enumerate((3.0, 5.0, 8.0)):
        w = np.exp(liks / scale); w /= w.sum()
        path = (w[:, None] * preds).sum(0)
        p = np.interp(md_grid, md_ev, path, left=last_tvt, right=float(path[-1]))
        if k == 1:
            hyp = p.copy()
        out[:, k] = np.clip((p - last_tvt) / 10.0, -12, 12)
    sd = preds.std(0)
    out[:, 3] = np.clip(np.interp(md_grid, md_ev, sd, left=0, right=float(sd[-1])) / 10.0, 0, 6)
    return out, hyp


def one_well(wid):
    try:
        from stride import load_well, grcal_tw
        hw, tw = load_well(wid, "train")
        kn_all = hw[hw["TVT_input"].notna()]
        ev_nat = hw[hw["TVT_input"].isna()]
        if len(ev_nat) < 10 or len(kn_all) < 160 or not np.isfinite(hw["TVT"].values).all():
            return wid, None, None, "short/no-truth"
        tw_s = tw.sort_values("TVT")
        tw_tvt = tw_s["TVT"].values.astype(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
        md_all = hw["MD"].values.astype(float)
        z_all = hw["Z"].values.astype(float)
        tvt_true = hw["TVT"].values.astype(float)
        gr_raw = hw["GR"].values.astype(float)
        gr_nan = (~np.isfinite(gr_raw)).astype(float)
        gr_fill = pd.Series(gr_raw).interpolate(limit_direction="both")
        gr_fill = gr_fill.fillna(float(np.nanmean(gr_raw))).values

        kn_idx = kn_all.index.values
        cuts = [("nat", int(kn_idx[-1]))]
        for f in CUT_FRACS:
            c = int(round(len(kn_idx) * f))
            if c >= 120 and len(kn_idx) - c >= 60:
                cuts.append((f"c{int(f*100)}", int(kn_idx[c - 1])))

        frames = []
        rowmap = None
        for cut_name, cut_i in cuts:
            kn = hw.loc[:cut_i]
            kn = kn[kn["TVT_input"].notna()]
            if len(kn) < 100:
                continue
            # heel normalization + grcal'd typewell in heel units
            mu = float(np.nanmean(kn["GR"].values)); sd = float(np.nanstd(kn["GR"].values)) + 1e-3
            tw_cal = grcal_tw(tw_tvt, tw_gr, kn["TVT_input"].values, kn["GR"].values)
            last_tvt = float(kn["TVT_input"].iloc[-1])
            cut_md = float(hw.loc[cut_i, "MD"])
            end_md = float(md_all[-1])
            g0 = cut_md - CTX * STEP
            md_grid = np.arange(g0, end_md + STEP / 2, STEP)
            tail = md_grid > cut_md
            if tail.sum() < 8:
                continue
            gr_g = np.interp(md_grid, md_all, gr_fill)
            grz = (gr_g - mu) / sd
            grz_s21 = pd.Series(grz).rolling(21, center=True, min_periods=1).mean().values
            grz_s101 = pd.Series(grz).rolling(101, center=True, min_periods=1).mean().values
            grz_std21 = pd.Series(grz).rolling(21, center=True, min_periods=1).std().fillna(0).values
            nan_g = np.interp(md_grid, md_all, gr_nan)
            z_g = np.interp(md_grid, md_all, z_all)
            dzdmd = np.clip(np.gradient(z_g, md_grid), -0.5, 0.5)
            tvt_g = np.interp(md_grid, md_all, tvt_true)
            hw_m = hw.copy(deep=True)
            hw_m.loc[hw_m.index > cut_i, "TVT_input"] = np.nan
            sc, hyp = _stride_channels(hw_m, tw, md_grid, last_tvt)
            ev_m = hw_m[hw_m["TVT_input"].isna()]
            pfc, pf_hyp = _pf_channels(hw_m, md_grid, tw, last_tvt,
                                       (ev_m.index.values, ev_m["MD"].values.astype(float)))
            # prefix TVT_input path (the heel trend the net must extrapolate); 0 in tail
            kn_md = kn["MD"].values.astype(float)
            kn_tvt = kn["TVT_input"].values.astype(float)
            tvt_pre = np.interp(md_grid, kn_md, kn_tvt, left=float(kn_tvt[0]), right=last_tvt)
            tvt_rel = np.where(~tail, (tvt_pre - last_tvt) / 10.0, 0.0)
            # v2 mismatch: profile around a MOVING hypothesis — actual TVT_input in the
            # prefix (teaches what "aligned" looks like), stride decode in the tail.
            hyp_all = np.where(~tail, tvt_pre, pf_hyp)   # v3: PF path (his recipe), was stride
            twz = (tw_cal - mu) / sd
            mm = np.empty((len(md_grid), len(OFFS)), np.float32)
            for j, off in enumerate(OFFS):
                ref = np.interp(np.clip(hyp_all + off, tw_tvt[0], tw_tvt[-1]), tw_tvt, twz)
                mm[:, j] = np.clip(np.abs(grz - ref), 0, 6.0)
            n = len(md_grid)
            fr = {"well": wid, "cut": cut_name, "step": np.arange(n, dtype=np.int32),
                  "is_tail": tail.astype(np.int8),
                  "target": np.clip((tvt_g - last_tvt) / 10.0, -12, 12).astype(np.float32),
                  "grz": grz.astype(np.float32), "grz_s21": grz_s21.astype(np.float32),
                  "grz_s101": grz_s101.astype(np.float32), "grz_std21": grz_std21.astype(np.float32),
                  "gr_nan": nan_g.astype(np.float32), "dzdmd": (dzdmd * 20).astype(np.float32),
                  "z_rel": ((z_g - z_g[tail.argmax()]) / 100.0).astype(np.float32),
                  "d_from_cut": (np.maximum(md_grid - cut_md, 0) / 1000.0).astype(np.float32),
                  "d_to_end": ((end_md - md_grid) / 1000.0).astype(np.float32),
                  "tvt_rel": tvt_rel.astype(np.float32),
                  "is_known": (~tail).astype(np.float32),
                  "stride_d": sc[:, 0], "stride_stiff": sc[:, 1], "stride_loose": sc[:, 2],
                  "pf3_d": pfc[:, 0], "pf5_d": pfc[:, 1], "pf8_d": pfc[:, 2], "pf_sd": pfc[:, 3]}
            for j in range(len(OFFS)):
                fr[f"mm{j}"] = mm[:, j]
            frames.append(pd.DataFrame(fr))
            if cut_name == "nat":
                rowmap = pd.DataFrame({"well": wid,
                                       "id": [f"{wid}_{i}" for i in ev_nat.index],
                                       "md_rel": (ev_nat["MD"].values - cut_md).astype(np.float32),
                                       "y": (ev_nat["TVT"].values.astype(np.float32) - last_tvt),
                                       "anchor": np.float32(last_tvt)})
        if not frames:
            return wid, None, None, "no cuts"
        return wid, pd.concat(frames, ignore_index=True), rowmap, None
    except Exception as e:
        return wid, None, None, f"ERR {str(e)[:90]}"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    wells = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
    if limit:
        wells = wells[:limit]
    t0 = time.time()
    seqs, maps, skips = [], [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (wid, sq, rm, err) in enumerate(ex.map(one_well, wells), 1):
            if sq is None:
                skips.append((wid, err))
            else:
                seqs.append(sq)
                if rm is not None:
                    maps.append(rm)
            if k % 50 == 0 or k == len(wells):
                print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)} ({len(skips)} skipped)", flush=True)
    seq = pd.concat(seqs, ignore_index=True)
    seq.to_parquet("gru_seq.parquet", index=False)
    pd.concat(maps, ignore_index=True).to_parquet("gru_rowmap.parquet", index=False)
    n_samp = seq.groupby(["well", "cut"]).ngroups
    print(f"DONE [{time.time()-t0:.0f}s] {len(seq)} steps, {n_samp} samples "
          f"({len(set(seq['well']))} wells) -> gru_seq.parquet + gru_rowmap.parquet "
          f"| skipped {len(skips)}: {skips[:6]}", flush=True)


if __name__ == "__main__":
    main()
