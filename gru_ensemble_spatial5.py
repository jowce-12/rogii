# Spatial 5-leg extension runner: trains ONLY the two new legs _sd/_se (seeds 1301/555,
# dropouts 0.30/0.35 — same diversification lineage as the dip-era _pd/_pe), then prints
# the 5-leg padded-eval ensemble over _sa.._se. Existing _sa/_sb/_sc are never retrained
# (skip-guard on their OOF files).
# Judgment afterwards (my side): python gru_fusion_spatial.py --tags _sa,_sb,_sc,_sd,_se --suffix 5
# RUN (isic env): python gru_ensemble_spatial5.py       (~30min GPU)
import os, subprocess, sys
import numpy as np
import pandas as pd
NEW_LEGS = [("_sd", "1301", "128", "0.3"), ("_se", "555", "128", "0.35")]
ALL_TAGS = ["_sa", "_sb", "_sc", "_sd", "_se"]
for tag, seed, hid, drop in NEW_LEGS:
    if os.path.exists(f"gru_oof{tag}.parquet"):
        print(f"[skip] leg {tag} done")
        continue
    print(f"=== spatial dip leg {tag} (seed {seed}, drop {drop}) ===", flush=True)
    r = subprocess.run([sys.executable, "train_gru2.py", "--seed", seed, "--hid", hid,
                        "--dropout", drop, "--dip", "1", "--tag", tag])
    if r.returncode != 0:
        raise SystemExit(f"leg {tag} failed")
oofs = [pd.read_parquet(f"gru_oof{t}.parquet").set_index("id")["gru_d"] for t in ALL_TAGS]
rm = pd.read_parquet("gru_rowmap.parquet")[["id", "y"]]
for t, o in zip(ALL_TAGS, oofs):
    j = o.to_frame().join(rm.set_index("id"))
    print(f"leg {t}: {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft")
ens = pd.DataFrame({"id": oofs[0].index, "gru_d": np.mean([o.values for o in oofs], axis=0)})
j = ens.set_index("id").join(rm.set_index("id"))
print(f"*** SPATIAL 5-leg ensemble (padded-eval) = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft ***")
print("NOTE: padded-eval only. Adoption verdict comes from the clean recompute + corr +")
print("      both-seed blend gate (gru_fusion_spatial.py --tags _sa,_sb,_sc,_sd,_se --suffix 5).")
