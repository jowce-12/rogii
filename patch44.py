# patch44 (PREPARED, apply AFTER the spatial LB reading): contact-only gold.
# Harness verdict (spatial base, contact-excluded hidden regime, both seeds):
#   seed7 gold-OFF 6.8346 vs balanced 7.4664 (+0.63) / seed11 5.7066 vs 6.4715 (+0.76)
# -> gold overrides on wells WITHOUT contact data are strongly net-negative on the new
# base. This patch skips gold calibration for wells with no train-side (contact) file:
# overlap wells keep the full balanced gold (LB-proven), hidden wells keep the blend.
# Side benefit: saves the per-well calibration cost (~42s/well) on all hidden wells.
# RUN: python patch44.py --check   (verify anchors only)
#      python patch44.py           (apply to BOTH notebooks)
import json, ast, sys

CHECK = "--check" in sys.argv
OLD = """            if not _hw_path.exists() or not _tw_path.exists():
                return dict(well=_wid, status='skip_missing_files'), [], {}
            _hw = _gold_pd.read_csv(_hw_path)"""
NEW = """            if not _hw_path.exists() or not _tw_path.exists():
                return dict(well=_wid, status='skip_missing_files'), [], {}
            # patch44: contact-only gold — hidden wells (no train-side file) keep the
            # blend untouched (harness: overrides there cost +0.63/+0.76 pooled).
            if not (_GOLD_DATA / 'train' / f'{_wid}__horizontal_well.csv').exists():
                return dict(well=_wid, status='skip_no_contact'), [], {}
            _hw = _gold_pd.read_csv(_hw_path)"""

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    hit = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "def _gold_one_well" not in s:
            continue
        assert "skip_no_contact" not in s, f"{nb_path}: already patched"
        assert OLD in s, f"{nb_path}: anchor missing"
        if CHECK:
            print(f"{nb_path}: cell {i} anchor OK (not applied)")
        else:
            new = s.replace(OLD, NEW, 1)
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch44 applied to cell {i}, ast OK")
        hit = True
        break
    assert hit, f"{nb_path}: gold worker cell not found"
print("CHECK OK" if CHECK else "ALL DONE")
