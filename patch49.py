# patch49: risk-tiered blend weights (APPLIED 2026-07-23 via inline run; this file is the
# canonical record + undo path). Monster wells (the gamma gate: AMP_RISK >= 3.39, no new
# threshold fitted) get 0.20/0.40/0.40 (0.10 fleongg->gru shift); others keep
# 0.20/0.50/0.30. ws3 stays 0.10 for both tiers (raising it hurt seed11).
# Both-seed gate on the patch48 base: 7.1980->7.0962 / 5.9541->5.8095.
# RUN: python patch49.py --check   (verify state)
#      python patch49.py           (apply; errors if already applied)
#      python patch49.py --undo    (restore the single-mix line)
import json, ast, sys

OLD = "    _w3 = 0.20 * _m3['tvt_sp45'] + 0.50 * _m3['tvt_fleongg'] + 0.30 * _m3['tvt_gru']"
NEW = """    _w3 = 0.20 * _m3['tvt_sp45'] + 0.50 * _m3['tvt_fleongg'] + 0.30 * _m3['tvt_gru']
    # patch49: risk-tiered mix — monster wells (gamma gate risk>=3.39) shift 0.10
    # fleongg -> gru (both-seed harness: 7.1980->7.0962 / 5.9541->5.8095)
    try:
        _t_risk = globals().get('AMP_RISK')
        if _t_risk is not None:
            _t_w = _m3['id'].astype(str).str.split('_', n=1).str[0]
            _t_r = _t_w.map(_t_risk)
            _t_m = _final_np.asarray((_t_r >= 3.39).fillna(False).values, bool)
            _w3_m = 0.20 * _m3['tvt_sp45'] + 0.40 * _m3['tvt_fleongg'] + 0.40 * _m3['tvt_gru']
            _w3 = _final_np.where(_t_m, _w3_m, _w3)
            print('[tier] monster mix 0.20/0.40/0.40 on %d rows (%d wells)'
                  % (int(_t_m.sum()), int(_t_w[_t_m].nunique())), flush=True)
        else:
            print('[tier] AMP_RISK unavailable -> single mix kept', flush=True)
    except Exception as _e:
        print('[tier] skipped: %s' % str(_e)[:60], flush=True)"""

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
        if "tvt_2way" not in s or "_w3 = 0.20" not in s:
            continue
        applied = "patch49" in s
        if CHECK:
            print(f"{nb_path}: cell {i} patch49 {'APPLIED' if applied else 'not applied (anchor ' + ('OK' if OLD in s else 'MISSING') + ')'}")
        elif UNDO:
            assert applied, f"{nb_path}: not applied"
            new = s.replace(NEW, OLD, 1)
            assert "patch49" not in new
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch49 removed (cell {i})")
        else:
            assert not applied, f"{nb_path}: already applied"
            assert OLD in s, f"{nb_path}: anchor missing"
            new = s.replace(OLD, NEW, 1)
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch49 applied (cell {i})")
        hit = True
        break
    assert hit, f"{nb_path}: blend cell not found"
print("DONE")
