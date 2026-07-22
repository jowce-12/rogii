# Experimental legs runner: two 1-leg experiments on the spatial+dip recipe, each a
# direct twin of _sa (seed 42, drop 0.25) so the single-leg read is controlled:
#   _ha : hard-well upweight (--wellw 1.0 -> monster wells' loss x2 at risk>=5.39, ramp from 3.39)
#   _ca : channel-group dropout (--chdrop 0.2 -> 20%/sample zero one of stride/pf/spatial/mm)
# Twin reference: _sa padded 7.8713. Judgment (my side) = clean recompute of 5-leg with the
# experimental leg swapped/added (gru_fusion_mixed.py) + both-seed gate.
# RUN (isic env): python gru_ensemble_exp.py       (~30min GPU)
import os, subprocess, sys
import numpy as np
import pandas as pd
LEGS = [("_ha", ["--wellw", "1.0"]), ("_ca", ["--chdrop", "0.2"])]
for tag, extra in LEGS:
    if os.path.exists(f"gru_oof{tag}.parquet"):
        print(f"[skip] leg {tag} done")
        continue
    print(f"=== exp leg {tag} ({' '.join(extra)}) ===", flush=True)
    r = subprocess.run([sys.executable, "train_gru2.py", "--seed", "42", "--hid", "128",
                        "--dropout", "0.25", "--dip", "1", "--tag", tag] + extra)
    if r.returncode != 0:
        raise SystemExit(f"leg {tag} failed")
rm = pd.read_parquet("gru_rowmap.parquet")[["id", "y"]]
for tag, _ in LEGS + [("_sa", None)]:
    o = pd.read_parquet(f"gru_oof{tag}.parquet").set_index("id")["gru_d"]
    j = o.to_frame().join(rm.set_index("id"))
    print(f"leg {tag}: {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft (padded)")
print("NOTE: _sa twin = controlled reference (same seed/drop). Verdict via clean gate (my side).")
