import os, glob, time
import numpy as np
from joblib import Parallel, delayed
import s1_ab as S
if __name__ == "__main__":
    wids = sorted(os.path.basename(f).split("__")[0] for f in glob.glob("train/*__horizontal_well.csv"))
    rng = np.random.default_rng(11)
    samp = sorted(rng.choice(wids, 150, replace=False).tolist())
    t0 = time.time()
    res = Parallel(n_jobs=24, prefer="threads")(delayed(S.eval_well)(w, 32, ["off","affine"]) for w in samp)
    res = [r for r in res if r is not None]
    print(f"confirm: {len(res)} wells in {time.time()-t0:.0f}s (seed 11)")
    def pooled(key_fn):
        return float(np.sqrt(np.mean(np.concatenate([ (key_fn(r)-r['y'])**2 for r in res ]))))
    print(f"off            = {pooled(lambda r: r['off']):.4f}")
    print(f"affine         = {pooled(lambda r: r['affine']):.4f}")
    for w in [0.3,0.4,0.5,0.6]:
        print(f"blend w={w:.1f}     = {pooled(lambda r,w=w: (1-w)*r['off']+w*r['affine']):.4f}")
    np.save("s1_confirm_seed11.npy", np.array(res, dtype=object), allow_pickle=True)
