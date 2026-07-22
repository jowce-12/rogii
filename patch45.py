# patch45: weight-probe candidates (apply ONE at a time, AFTER the patch44 LB reading).
# 5-leg harness (both seeds): current 0.15/0.45/0.40 = 6.8201/5.6897
#   A 0.05/0.45/0.50 = 6.5870/5.4706   (both hedges kept — recommended first probe)
#   B 0.05/0.40/0.55 = 6.4706/5.3547
#   C 0.05/0.35/0.60 = 6.3673/5.2467   (grid still monotone at 0.65 — climb ONLY as LB confirms)
# RUN: python patch45.py A --check | python patch45.py A | python patch45.py --undo
#      (same for B; --undo restores 0.15/0.45/0.40 from whichever candidate is applied)
import json, ast, sys

BASE = ("0.15", "0.45", "0.40", "spatial")
CANDS = {"A": ("0.05", "0.45", "0.50", "w-probe A"), "B": ("0.05", "0.40", "0.55", "w-probe B"),
         "C": ("0.05", "0.35", "0.60", "w-probe C")}

def line(w):
    return (f"_w3 = {w[0]} * _m3['tvt_sp45'] + {w[1]} * _m3['tvt_fleongg'] + {w[2]} * _m3['tvt_gru']",
            f"weights {w[0]}/{w[1]}/{w[2]} ({w[3]}); ")

UNDO = "--undo" in sys.argv
CHECK = "--check" in sys.argv
cand = next((a for a in sys.argv[1:] if a in CANDS), None)
assert UNDO or cand, "usage: patch45.py A|B [--check]  or  patch45.py --undo"

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    hit = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "_w3 = " not in s or "tvt_gru" not in s:
            continue
        if UNDO:
            src = next((CANDS[k] for k in CANDS if line(CANDS[k])[0] in s), None)
            assert src is not None, f"{nb_path}: no candidate weights found to undo"
            old_l, old_p = line(src)
            new_l, new_p = line(BASE)
        else:
            old_l, old_p = line(BASE)
            new_l, new_p = line(CANDS[cand])
            assert old_l in s, f"{nb_path}: baseline weights not found (another candidate applied? --undo first)"
        if CHECK:
            print(f"{nb_path}: cell {i} anchor OK (not applied)")
        else:
            new = s.replace(old_l, new_l).replace(old_p, new_p)
            assert new_l in new and old_l not in new
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            tag = "restored 0.15/0.45/0.40" if UNDO else f"candidate {cand} applied"
            print(f"{nb_path}: {tag} (cell {i}), ast OK")
        hit = True
        break
    assert hit, f"{nb_path}: blend cell not found"
print("CHECK OK" if CHECK else "DONE")
