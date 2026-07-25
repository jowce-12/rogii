# WITHDRAWN 2026-07-25 — the rationale below rested on a CONTAMINATED comparison
# (CPU dip5 vs GPU dip3). Honest CPU-vs-CPU numbers: 3->5 legs is a SEED SPLIT
# (s7 6.9610->6.9983 worse, s11 5.7572->5.7412 better) and err-corr vs fleongg
# RISES with legs (0.7357 -> 0.7459; padded curve 1->5 legs: 0.760->0.800).
# Do not run without new evidence.
# Dip 7-leg widening: two more NON-SPATIAL dip legs (_df/_dg). Rationale from the 5-leg
# gate — going 3->5 legs barely moved standalone clean OOF (6.3516 -> 6.2905) but moved
# the BLEND a lot (8.0269 -> 7.3586 / 6.4734 -> 6.0188) because err-corr vs fleongg fell
# 0.792 -> 0.746: the gain is decorrelation (seed-averaging damps per-well blowups), so
# more legs may keep paying. --spatial 0 is explicit (trainer defaults spatial ON).
# Existing _da.._de are never retrained (skip-guard).
# RUN (isic env): python gru_ensemble_dip7.py       (~30min GPU)
import os, subprocess, sys
import numpy as np
import pandas as pd
NEW_LEGS = [("_df", "31", "128", "0.25"), ("_dg", "2027", "128", "0.35")]
ALL_TAGS = ["_da", "_db", "_dc", "_dd", "_de", "_df", "_dg"]
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
print(f"*** DIP 7-leg ensemble (padded-eval) = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft ***")
print("NOTE: verdict comes from the CPU clean recompute + corr + both-seed gate (my side).")
print("      Check each new gru_meta json shows 31 chans (spatial OFF).")
