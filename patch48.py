# patch48: STRIDE-v3 pole + lam4096 + GRU checksum (both notebooks).
#   (1) cell 24: embed the tuned v3 decoder (extracted from stride3.py, tokens renamed,
#       grcal -> _s2_grcal, seg-prior constants hardcoded) + per-well stash _S3_TVT
#   (2) cell 46: mix ws3=0.10 into _w3 before the 3way write (harness both-seed gate:
#       8.0269->7.8364 / 6.4734->6.3440)
#   (3) cell 43: gru_fuse lam 1024 -> 4096 (both-env both-seed micro-win) + checksum print
# Self-test: the renamed decoder is exec'd and compared against stride3.decode on one
# well (must match exactly) BEFORE any notebook is touched.
import ast
import json

import numpy as np

# ---- build the embedded decoder from stride3.py ----
src3 = open("stride3.py", encoding="utf-8").read()
tree = ast.parse(src3)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
body = ast.get_source_segment(src3, fn)
body = body.replace("def decode(hw, tw):", "def _stride3_decode(hw, tw):")
body = body.replace("    from stride import grcal_tw\n", "")
for old, new in (("grcal_tw", "_s2_grcal"), ("LEN_GRID", "_S3_LEN"), ("RATE_GRID", "_S3_RATE"),
                 ("LEN_LP", "_S3_LP"), ("W_LEN", "_S3_WLEN"), ("SIG_P", "_S3_SIGP"),
                 ("LIK_W", "_S3_LIKW"), ("K_BEAM", "_S3_K"), ("GSTEP", "_S3_G"),
                 ("TOP_AGG", "_S3_TOP"), ("TEMP", "_S3_TEMP")):
    body = body.replace(old, new)

HEADER = """
# ---- patch48: STRIDE-v3 pole (variable-length lattice DP; tuned wlen 0.5 / sigp 0.012,
# drilled-trend init; both-seed harness gate at ws3=0.10). Decoder extracted verbatim
# from stride3.py with renamed globals; seg-prior lognormal constants hardcoded. ----
_S3_TVT = {}
_S3_LEN = np.array([100.0, 160.0, 240.0, 360.0, 520.0, 760.0])
_S3_RATE = np.arange(-0.10, 0.1001, 0.005)
_S3_LP = -0.5 * ((np.log(_S3_LEN) - 5.75039929216328) / 0.9010370313922137) ** 2 - np.log(_S3_LEN)
_S3_WLEN = 0.5
_S3_SIGP = 0.012
_S3_LIKW = 0.1
_S3_K = 96
_S3_G = 10.0
_S3_TOP = 32
_S3_TEMP = 0.02

"""
DECODER = HEADER + body + "\n"

# ---- self-test: renamed decoder must reproduce stride3.decode exactly ----
import importlib.util
import sys as _sys
_argv = _sys.argv
_sys.argv = ["x", "--wlen", "0.5"]
spec = importlib.util.spec_from_file_location("s3", "stride3.py")
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)
_sys.argv = _argv
from stride import load_well, grcal_tw
ns = {"np": np, "_s2_grcal": grcal_tw}
exec(compile(DECODER, "<embed>", "exec"), ns)
hw, tw = load_well("000d7d20", "train")
a = s3.decode(hw, tw)
b = ns["_stride3_decode"](hw, tw)
d = float(np.abs(a - b).max())
assert d < 1e-9, f"embedded decoder mismatch: {d}"
print(f"self-test OK (max diff {d:.2e})")

WELL_BLOCK = """    # patch48: STRIDE-v3 pole — per-well decode stashed for the 3-way cell. Fail-open.
    try:
        _s3p = _stride3_decode(hw_te, tw_ref)
        if _s3p is not None and len(_s3p) == len(_ei2) and np.all(np.isfinite(_s3p)):
            for _si, _sv in zip(_ei2, _s3p):
                _S3_TVT['%s_%d' % (wid, _si)] = float(_sv)
            print('  STRIDE-v3 OK (%d rows)' % len(_s3p))
    except Exception as _e:
        print(f'  STRIDE-v3 skipped: {_e}')
"""

MIX_OLD = "    _m3['tvt'] = _final_np.where(_m3['tvt_gru'].notna(), _w3, _m3['tvt_2way'])"
MIX_NEW = """    try:
        _s3map = globals().get('_S3_TVT') or {}
        if _s3map:
            _s3v = _m3['id'].astype(str).map(_s3map)
            _w3 = _final_np.where(_s3v.notna(), 0.90 * _w3 + 0.10 * _s3v.values, _w3)
            print('[s3] stride-v3 pole mixed at 0.10 on %d rows' % int(_s3v.notna().sum()), flush=True)
        else:
            print('[s3] no stride-v3 predictions -> mix skipped', flush=True)
    except Exception as _e:
        print('[s3] mix skipped: %s' % str(_e)[:60], flush=True)
    _m3['tvt'] = _final_np.where(_m3['tvt_gru'].notna(), _w3, _m3['tvt_2way'])"""

LOOP_ANCHOR = "for i, wid in enumerate(test_wells):"
SKIP_ANCHOR = "        print(f'  STRIDE skipped: {_e}')\n"
LAM_OLD = "gru_fuse(_gpred[_gti], _gdip[_gti][:-1], 1024.0)"
LAM_NEW = "gru_fuse(_gpred[_gti], _gdip[_gti][:-1], 4096.0)"
CK_OLD = "        print('[gru] DONE: %d wells, %d skipped, %d rows, %.0fs'"
CK_NEW = """        if GRU_TVT:
            print('[gru] checksum mean=%.4f n=%d' % (sum(GRU_TVT.values()) / len(GRU_TVT),
                                                     len(GRU_TVT)), flush=True)
        print('[gru] DONE: %d wells, %d skipped, %d rows, %.0fs'"""

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    done = set()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if LOOP_ANCHOR in s and SKIP_ANCHOR in s:
            assert "patch48" not in s, f"{nb_path}: cell {i} already patched"
            s = s.replace(LOOP_ANCHOR, DECODER + "\n" + LOOP_ANCHOR, 1)
            s = s.replace(SKIP_ANCHOR, SKIP_ANCHOR + WELL_BLOCK, 1)
            ast.parse(s)
            nb["cells"][i]["source"] = s.splitlines(keepends=True)
            done.add("loop")
        elif MIX_OLD in s:
            s = s.replace(MIX_OLD, MIX_NEW, 1)
            ast.parse(s)
            nb["cells"][i]["source"] = s.splitlines(keepends=True)
            done.add("mix")
        elif LAM_OLD in s:
            s = s.replace(LAM_OLD, LAM_NEW, 1).replace(CK_OLD, CK_NEW, 1)
            assert "4096.0" in s and "checksum" in s
            ast.parse(s)
            nb["cells"][i]["source"] = s.splitlines(keepends=True)
            done.add("gru")
    assert done == {"loop", "mix", "gru"}, f"{nb_path}: applied={done}"
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{nb_path}: v3 pole + lam4096 + checksum applied, ast OK")
print("ALL DONE")
