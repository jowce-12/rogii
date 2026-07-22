# patch47: ROLLBACK to the LB-proven 6.663 configuration after the spatial verdict
# (LB 7.140 with a mechanically-verified pipeline -> spatial harness gains do not
# transfer; radiant's grouped-CV-inflation warning confirmed).
#   (1) weights: 0.05/0.35/0.60 (w-probe C) -> 0.20/0.50/0.30 (probe B, the 6.663 run)
#   (2) remove patch44 (contact-only gold) — untested in isolation; final config purity
# Keeps patch46 (robust bank path — inert when dip ckpts are attached) and the spatial
# code paths (unused without spatial ckpts). USER SIDE: revert the GRU dataset to the
# dip 15 (gru_fold{0-4}_{da,db,dc}.pt) version.
import json, ast

W_OLD = "_w3 = 0.05 * _m3['tvt_sp45'] + 0.35 * _m3['tvt_fleongg'] + 0.60 * _m3['tvt_gru']"
W_NEW = "_w3 = 0.20 * _m3['tvt_sp45'] + 0.50 * _m3['tvt_fleongg'] + 0.30 * _m3['tvt_gru']"
P_OLD = "weights 0.05/0.35/0.60 (w-probe C); "
P_NEW = "weights 0.20/0.50/0.30 (probe B); "

G_OLD = """            # patch44: contact-only gold — hidden wells (no train-side file) keep the
            # blend untouched (harness: overrides there cost +0.63/+0.76 pooled).
            if not (_GOLD_DATA / 'train' / f'{_wid}__horizontal_well.csv').exists():
                return dict(well=_wid, status='skip_no_contact'), [], {}
"""

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    done_w = done_g = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if W_OLD in s:
            new = s.replace(W_OLD, W_NEW).replace(P_OLD, P_NEW)
            assert W_NEW in new and W_OLD not in new
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            done_w = True
        elif G_OLD in s:
            new = s.replace(G_OLD, "")
            assert "skip_no_contact" not in new and "def _gold_one_well" in new
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            done_g = True
    assert done_w and done_g, f"{nb_path}: weights={done_w} gold={done_g}"
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{nb_path}: weights -> 0.20/0.50/0.30, patch44 removed, ast OK")
print("ALL DONE")
