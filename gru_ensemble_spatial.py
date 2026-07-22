# Spatial dip-head 3-leg runner (mirrors the _da/_db/_dc configs; train_gru2.py has
# spatial default ON so these legs get the 6 ANCC surface channels). New tags _sa/_sb/_sc
# so the adopted deployment set (_da/_db/_dc) stays untouched.
# Requires gru_spatial_fold{0-4}.parquet (make_spatial_features.py — already built).
# RUN (isic env): python gru_ensemble_spatial.py       (~45min GPU)
import os, subprocess, sys
import numpy as np
import pandas as pd
LEGS = [("_sa", "42", "128", "0.25"), ("_sb", "202", "128", "0.25"), ("_sc", "777", "128", "0.4")]
for tag, seed, hid, drop in LEGS:
    if os.path.exists(f"gru_oof{tag}.parquet"):
        print(f"[skip] leg {tag} done")
        continue
    print(f"=== spatial dip leg {tag} (seed {seed}, drop {drop}) ===", flush=True)
    r = subprocess.run([sys.executable, "train_gru2.py", "--seed", seed, "--hid", hid,
                        "--dropout", drop, "--dip", "1", "--tag", tag])
    if r.returncode != 0:
        raise SystemExit(f"leg {tag} failed")
oofs = [pd.read_parquet(f"gru_oof{t}.parquet").set_index("id")["gru_d"] for t, _, _, _ in LEGS]
ens = pd.DataFrame({"id": oofs[0].index, "gru_d": np.mean([o.values for o in oofs], axis=0)})
ens.to_parquet("gru_oof_spatial3.parquet", index=False)
rm = pd.read_parquet("gru_rowmap.parquet")[["id", "y"]]
for (t, _, _, _), o in zip(LEGS, oofs):
    j = o.to_frame().join(rm.set_index("id"))
    print(f"leg {t}: {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft")
j = ens.set_index("id").join(rm.set_index("id"))
print(f"*** SPATIAL dip-head 3-leg ensemble (padded-eval) = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft ***")
print("NOTE: spatial ckpts are NOT notebook-deployable until _gru_infer grows the spatial builder;")
print("      judge via clean OOF recompute + blend harness first (as with every GRU leg).")
