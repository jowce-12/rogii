# STRIDE-v3 predictions for ALL 773 train wells -> s3_all.parquet (id, s3_d = pred - last
# known TVT, i.e. delta space to match the fleongg target convention). Feeds the
# "v3 as a fleongg feature" experiment. Deployment cost is zero: the submission already
# decodes v3 per well (patch48).
import sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed

def one(wid):
    import sys as _s
    _a = _s.argv; _s.argv = ["x", "--wlen", "0.5"]
    import stride3 as S3
    _s.argv = _a
    from stride import load_well
    try:
        hw, tw = load_well(wid, "train")
        pred = S3.decode(hw, tw)
        if pred is None:
            return None
        ev = hw[hw["TVT_input"].isna()]
        kn = hw[hw["TVT_input"].notna()]
        last = float(kn["TVT_input"].iloc[-1])
        return pd.DataFrame({"id": [f"{wid}_{i}" for i in ev.index.values],
                             "s3_d": (np.asarray(pred, float) - last).astype(np.float32)})
    except Exception:
        return None

if __name__ == "__main__":
    t0 = time.time()
    wells = sorted(pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                                   columns=["well"])["well"].unique())
    outs = Parallel(n_jobs=6, backend="loky")(delayed(one)(w) for w in wells)
    ok = [o for o in outs if o is not None]
    df = pd.concat(ok, ignore_index=True)
    df.to_parquet("s3_all.parquet", index=False)
    print(f"DONE [{time.time()-t0:.0f}s] s3_all.parquet: {len(df)} rows / {len(ok)}/{len(wells)} wells")
