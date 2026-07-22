# Multi-surface (spatial2) 3-leg runner: same seed/dropout diversification as the
# adopted _sa set, plus the 4 consensus channels from all six surfaces (41 chans total).
# Tags _xa/_xb/_xc; existing sets are never touched.
# Requires gru_spatial2_fold{0-4}.parquet (make_spatial2.py).
# RUN (isic env): python gru_ensemble_spatial2.py       (~45min GPU)
import os, subprocess, sys
import numpy as np
import pandas as pd
LEGS = [("_xa", "42", "128", "0.25"), ("_xb", "202", "128", "0.25"), ("_xc", "777", "128", "0.4")]
for tag, seed, hid, drop in LEGS:
    if os.path.exists(f"gru_oof{tag}.parquet"):
        print(f"[skip] leg {tag} done")
        continue
    print(f"=== spatial2 dip leg {tag} (seed {seed}, drop {drop}) ===", flush=True)
    r = subprocess.run([sys.executable, "train_gru2.py", "--seed", seed, "--hid", hid,
                        "--dropout", drop, "--dip", "1", "--spatial2", "1", "--tag", tag])
    if r.returncode != 0:
        raise SystemExit(f"leg {tag} failed")
oofs = [pd.read_parquet(f"gru_oof{t}.parquet").set_index("id")["gru_d"] for t, _, _, _ in LEGS]
rm = pd.read_parquet("gru_rowmap.parquet")[["id", "y"]]
for (t, _, _, _), o in zip(LEGS, oofs):
    j = o.to_frame().join(rm.set_index("id"))
    print(f"leg {t}: {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft")
ens = pd.DataFrame({"id": oofs[0].index, "gru_d": np.mean([o.values for o in oofs], axis=0)})
j = ens.set_index("id").join(rm.set_index("id"))
print(f"*** SPATIAL2 3-leg ensemble (padded-eval) = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft ***")
print("NOTE: padded-eval only; verdict via clean recompute + corr + both-seed gate (my side).")
print("      spatial2 ckpts need _gru_infer multi-surface support before ANY notebook attach.")
