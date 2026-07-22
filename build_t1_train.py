# T1 affine-PF channel as a feature -> t1_join.parquet (id, t1_d).
# Replicates the notebook's deployed T1 channel EXACTLY (cell 24 patch21 block):
#   tw_aff = typewell with GR affine-recalibrated on the visible prefix (grcal_tw)
#   pf = run_pf_lik_ensemble_scales(hw, tw_aff, n_particles=500, n_seeds=128)  [verbatim
#        notebook PF via _t1_pf.py, serial seeds 0..127 = identical results]
#   t1 = (pf_scale_3 + pf_scale_5 + pf_scale_8) / 3;  t1_d = t1[eval] - last_known
# RUN: python build_t1_train.py [--limit N]
import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

WORKERS = 10


def one_well(wid):
    try:
        from stride import load_well, grcal_tw   # grcal_tw == notebook _s2_grcal (verified)
        import _t1_pf as PF
        hw, tw = load_well(wid, "train")
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) < 5 or len(kn) < 30:
            return wid, None, "too short"
        tw_s = tw.sort_values("TVT")
        ttvt = tw_s["TVT"].values.astype(float)
        tgr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
        tw_aff = pd.DataFrame({"TVT": ttvt,
                               "GR": grcal_tw(ttvt, tgr, kn["TVT_input"].values, kn["GR"].values),
                               "Geology": np.nan})
        pf = PF.run_pf_lik_ensemble_scales(hw, tw_aff, n_particles=500, n_seeds=128)
        sel = (np.asarray(pf["pf_scale_3"], float) + np.asarray(pf["pf_scale_5"], float)
               + np.asarray(pf["pf_scale_8"], float)) / 3.0
        last = float(kn["TVT_input"].iloc[-1])
        idx = ev.index.values
        return wid, pd.DataFrame({"id": [f"{wid}_{i}" for i in idx],
                                  "t1_d": (sel[idx] - last).astype(np.float32)}), None
    except Exception as e:
        return wid, None, f"ERR {str(e)[:80]}"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    wells = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
    if limit:
        wells = wells[:limit]
    t0 = time.time()
    frames, skips = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (wid, df, err) in enumerate(ex.map(one_well, wells), 1):
            if df is None:
                skips.append((wid, err))
            else:
                frames.append(df)
            if k % 25 == 0 or k == len(wells):
                print(f"[{time.time()-t0:.0f}s] {k}/{len(wells)} ({len(skips)} skipped)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet("t1_join.parquet", index=False)
    print(f"DONE [{time.time()-t0:.0f}s] {len(out)} rows -> t1_join.parquet "
          f"| skipped {len(skips)}: {skips[:6]}", flush=True)


if __name__ == "__main__":
    main()
