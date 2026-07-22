# GRU ensemble runner: 3 legs (seed/width variants), checkpointed, then mean-merge.
# RUN (isic env): python gru_ensemble.py        (~45min GPU total)
import os, subprocess, sys
import numpy as np
import pandas as pd

# v2 (packed): 6 legs, ALL retrained with the pack_padded fix (old _a/_b/_c were
# trained with padding contamination -> new tags force full retrain). hid stays 128
# everywhere (160 hit a 6x cuDNN slow path); diversity = seed + dropout.
LEGS = [("_pa", "42", "128", "0.25"), ("_pb", "202", "128", "0.25"),
        ("_pc", "777", "128", "0.4"),  ("_pd", "1301", "128", "0.25"),
        ("_pe", "555", "128", "0.3"),  ("_pf", "31", "128", "0.35")]
for tag, seed, hid, drop in LEGS:
    if os.path.exists(f"gru_oof{tag}.parquet"):
        print(f"[skip] leg {tag} done")
        continue
    print(f"=== leg {tag} (seed {seed}, hid {hid}) ===", flush=True)
    r = subprocess.run([sys.executable, "train_gru2.py", "--seed", seed, "--hid", hid,
                    "--dropout", drop, "--tag", tag])
    if r.returncode != 0:
        raise SystemExit(f"leg {tag} failed")
oofs = [pd.read_parquet(f"gru_oof{t}.parquet").set_index("id")["gru_d"] for t, _, _, _ in LEGS]
ens = pd.DataFrame({"id": oofs[0].index, "gru_d": np.mean([o.values for o in oofs], axis=0)})
ens.to_parquet("gru_oof.parquet", index=False)
rm = pd.read_parquet("gru_rowmap.parquet")[["id", "y"]]
for t, o in zip([t for t, _, _, _ in LEGS], oofs):
    j = o.to_frame().join(rm.set_index("id"))
    print(f"leg {t}: {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft")
j = ens.set_index("id").join(rm.set_index("id"))
print(f"*** ENSEMBLE pooled OOF = {float(np.sqrt(np.mean((j.gru_d - j.y) ** 2))):.4f} ft -> gru_oof.parquet ***")
