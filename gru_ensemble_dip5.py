# Dip 5-leg widening runner: trains ONLY the two new NON-SPATIAL dip legs _dd/_de
# (seeds 1301/555, drops 0.30/0.35 — the diversification lineage that passed the gate on
# the spatial side before spatial died on LB). --spatial 0 is EXPLICIT because
# train_gru2.py now defaults spatial ON; these must match the deployed 31-chan recipe.
# Existing _da/_db/_dc are never touched (skip-guard).
# Judgment (my side): clean 5-leg recompute + corr + both-seed gate vs dip 3-leg.
# RUN (isic env): python gru_ensemble_dip5.py       (~30min GPU)
import os, subprocess, sys
import numpy as np
import pandas as pd
NEW_LEGS = [("_dd", "1301", "128", "0.3"), ("_de", "555", "128", "0.35")]
ALL_TAGS = ["_da", "_db", "_dc", "_dd", "_de"]
for tag, seed, hid, drop in NEW_LEGS:
    if os.path.exists(f"gru_oof{tag}.parquet"):
        print(f"[skip] leg {tag} done")
        continue
    print(f"=== dip leg {tag} (seed {seed}, drop {drop}, spatial OFF) ===", flush=True)
    r = subprocess.run([sys.executable, "train_gru2.py", "--seed", seed, "--hid", hid,
                        "--dropout", drop, "--dip", "1", "--spatial", "0", "--tag", tag])
    if r.returncode != 0:
        raise SystemExit(f"leg {tag} failed")
rm = pd.read_parquet("gru_rowmap.parquet")[["id", "y"]]
oofs = [pd.read_parquet(f"gru_oof{t}.parquet").set_index("id")["gru_d"] for t in ALL_TAGS]
for t, o in zip(ALL_TAGS, oofs):
    j = o.to_frame().join(rm.set_index("id"))
    print(f"leg {t}: {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft (padded)")
ens = pd.DataFrame({"id": oofs[0].index, "gru_d": np.mean([o.values for o in oofs], axis=0)})
j = ens.set_index("id").join(rm.set_index("id"))
print(f"*** DIP 5-leg ensemble (padded-eval) = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft ***")
print("NOTE: verify each new leg's gru_meta json shows 31 chans (spatial OFF).")
