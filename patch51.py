# patch51: expose STRIDE-v3 as a fleongg FEATURE column (s3_d) on the test side.
# Proxy evidence (150 wells, identical folds): v1 four cols 6.8736 | no stride 7.0509 |
# v3 single col 6.8241  -> the v3 column replaces the v1 family in TRAINING
# (train_stack.py already switched). Here we ADD s3_d while KEEPING the v1 columns, so
# the notebook stays compatible with BOTH artifact generations: inference selects columns
# by features.json, so old (v1-trained) models are unaffected and new (v3-trained) models
# get the column they expect. Values are reused from patch48's per-well decode (_S3_TVT),
# so the runtime cost is zero; a fresh decode is the fallback.
# RUN: python patch51.py --check | python patch51.py | python patch51.py --undo
import json, ast, sys

ANCHOR = """    if _srows:
        test_df = test_df.merge(pd.concat(_srows, ignore_index=True), on="id", how="left")
        print(f"[stride-feat] joined {sum(len(_f) for _f in _srows)} rows / {len(_srows)} wells", flush=True)"""

NEW = ANCHOR + """

    # patch51: STRIDE-v3 feature column (s3_d). Reuses the decode patch48 already ran for
    # the v3 pole (_S3_TVT, absolute TVT keyed by row id); falls back to a fresh decode.
    # v1 columns are kept so either artifact generation works.
    try:
        _v3rows = []
        _s3map = globals().get('_S3_TVT') or {}
        for _swid in test_wids:
            try:
                _shw, _stw = load_well(_swid, "test")
                _skn = _shw[_shw["TVT_input"].notna()]
                _sev = _shw[_shw["TVT_input"].isna()]
                if len(_skn) == 0 or len(_sev) == 0:
                    continue
                _slast = float(_skn["TVT_input"].iloc[-1])
                _ids = [f"{_swid}_{_si}" for _si in _sev.index]
                _vals = [_s3map.get(_i) for _i in _ids]
                if any(_v is None for _v in _vals):
                    _sp = _stride3_decode(_shw, _stw)
                    if _sp is None or len(_sp) != len(_sev):
                        continue
                    _vals = list(np.asarray(_sp, float))
                _v3rows.append(pd.DataFrame({"id": _ids,
                                             "s3_d": (np.asarray(_vals, float) - _slast).astype(np.float32)}))
            except Exception as _se:
                print(f"[s3-feat] {_swid} skipped: {str(_se)[:60]}", flush=True)
        if _v3rows:
            test_df = test_df.merge(pd.concat(_v3rows, ignore_index=True), on="id", how="left")
            print(f"[s3-feat] s3_d joined {sum(len(_f) for _f in _v3rows)} rows / "
                  f"{len(_v3rows)} wells (cache hits: {len(_s3map) > 0})", flush=True)
        else:
            print("[s3-feat] no s3_d rows -> column absent (v1-trained artifacts unaffected)", flush=True)
    except Exception as _se:
        print(f"[s3-feat] skipped: {str(_se)[:70]}", flush=True)"""

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
        if "[stride-feat] joined" not in s:
            continue
        applied = "patch51" in s
        if CHECK:
            print(f"{nb_path}: cell {i} patch51 "
                  f"{'APPLIED' if applied else 'not applied (anchor ' + ('OK' if ANCHOR in s else 'MISSING') + ')'}")
        elif UNDO:
            assert applied, f"{nb_path}: not applied"
            new = s.replace(NEW, ANCHOR, 1)
            assert "patch51" not in new
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch51 removed (cell {i})")
        else:
            assert not applied, f"{nb_path}: already applied"
            assert ANCHOR in s, f"{nb_path}: anchor missing"
            new = s.replace(ANCHOR, NEW, 1)
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{nb_path}: patch51 applied (cell {i})")
        hit = True
        break
    assert hit, f"{nb_path}: stride-feat cell not found"
print("CHECK OK" if CHECK else "DONE")
