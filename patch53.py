# patch53: monster-tier weights, parameterised. The LB ladder pushed the NORMAL tier to
# 0.10/0.40/0.50, which left patch49's monster override (0.20/0.40/0.40) giving the
# high-risk 41% of rows LESS GRU than ordinary wells — the opposite of what patch49 was
# for (monsters wanted MORE GRU than the then-baseline 0.30). This restores the ordering.
# RUN: python patch53.py                 (apply 0.10/0.35/0.55)
#      python patch53.py 0.10 0.40 0.50  (make it identical to the normal tier)
#      python patch53.py --check
import json, ast, re, sys
args = [a for a in sys.argv[1:] if not a.startswith("--")]
W = tuple(f"{float(x):.2f}" for x in args) if len(args) == 3 else ("0.10", "0.35", "0.55")
CHECK = "--check" in sys.argv
PAT = re.compile(r"_w3_m = (\d\.\d\d) \* _m3\['tvt_sp45'\] \+ (\d\.\d\d) \* _m3\['tvt_fleongg'\] \+ "
                 r"(\d\.\d\d) \* _m3\['tvt_gru'\]")
LBL = re.compile(r"monster mix \d\.\d\d/\d\.\d\d/\d\.\d\d")
for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    hit = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        m = PAT.search(s)
        if not m:
            continue
        if CHECK:
            print(f"{nb_path}: cell {i} monster-tier weights = {'/'.join(m.groups())}")
        else:
            new = PAT.sub(f"_w3_m = {W[0]} * _m3['tvt_sp45'] + {W[1]} * _m3['tvt_fleongg'] + "
                          f"{W[2]} * _m3['tvt_gru']", s, count=1)
            new = LBL.sub(f"monster mix {'/'.join(W)}", new)
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: monster tier {'/'.join(m.groups())} -> {'/'.join(W)} (cell {i})")
        hit = True
        break
    assert hit, f"{nb_path}: monster block not found"
print("CHECK OK" if CHECK else "DONE")
