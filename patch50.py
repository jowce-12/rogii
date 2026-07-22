# patch50 (PREPARED, apply AFTER the patch48+49 LB reading): normal-tier weights
# 0.20/0.50/0.30 -> 0.15/0.45/0.40 (N2). Full-base both-seed grid (monster tier fixed
# 0.20/0.40/0.40, ws3 0.10, gamma on):
#   base 7.0962/5.8095  N2 6.9983/5.7412  (edge still monotone at X3 0.10/0.40/0.50 =
#   6.9112/5.6867 — climb further ONLY as LB confirms each rung)
# RUN: python patch50.py --check | python patch50.py | python patch50.py --undo
import json, ast, sys

OLD_W = "    _w3 = 0.20 * _m3['tvt_sp45'] + 0.50 * _m3['tvt_fleongg'] + 0.30 * _m3['tvt_gru']"
NEW_W = "    _w3 = 0.15 * _m3['tvt_sp45'] + 0.45 * _m3['tvt_fleongg'] + 0.40 * _m3['tvt_gru']"
OLD_P = "weights 0.20/0.50/0.30 (probe B); "
NEW_P = "weights 0.15/0.45/0.40 normal-tier (N2); "

CHECK = "--check" in sys.argv
UNDO = "--undo" in sys.argv
for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    hit = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "tvt_2way" not in s or "patch49" not in s:
            continue
        applied = NEW_W in s
        if CHECK:
            print(f"{nb_path}: cell {i} patch50 {'APPLIED' if applied else 'not applied (anchor ' + ('OK' if OLD_W in s else 'MISSING') + ')'}")
        elif UNDO:
            assert applied
            new = s.replace(NEW_W, OLD_W, 1).replace(NEW_P, OLD_P, 1)
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch50 undone (cell {i})")
        else:
            assert not applied and OLD_W in s, f"{nb_path}: anchor state unexpected"
            new = s.replace(OLD_W, NEW_W, 1).replace(OLD_P, NEW_P, 1)
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch50 applied (cell {i})")
        hit = True
        break
    assert hit, f"{nb_path}: blend cell not found"
print("DONE")
