# patch43: SPATIAL GRU deployment (both notebooks).
#   (1) Regenerate the GRU cell: re-embed the updated _gru_infer.py (spatial_bank/
#       spatial_query/spatial channels/ckpt skip-guard) and patch the driver to build
#       the ANCC bank once and pass spatial_src per well.
#   (2) Blend weights 0.20/0.50/0.30 -> 0.15/0.45/0.40 (spatial-judge recommendation).
import json, ast

GI_SRC = open("_gru_infer.py", encoding="utf-8").read()

DRIVER_OLD_LOAD = """        _gru_ckpts = [_gru_torch.load(_f, map_location='cpu') for _f in _gru_files]
        _n_dip = sum(1 for _c in _gru_ckpts if _c.get('dip'))
        print('[gru] %d checkpoints loaded (%d dip-head)' % (len(_gru_ckpts), _n_dip), flush=True)"""
DRIVER_NEW_LOAD = """        _gru_ckpts = [_gru_torch.load(_f, map_location='cpu') for _f in _gru_files]
        _n_dip = sum(1 for _c in _gru_ckpts if _c.get('dip'))
        _n_sp = sum(1 for _c in _gru_ckpts if _c.get('spatial'))
        print('[gru] %d checkpoints loaded (%d dip-head, %d spatial)'
              % (len(_gru_ckpts), _n_dip, _n_sp), flush=True)
        _gru_bank = None
        if _n_sp:
            try:
                _gru_bank = spatial_bank(CFG.dataset_path / 'train')
                print('[gru] spatial bank: %d samples' % len(_gru_bank['ancc']), flush=True)
            except Exception as _ge:
                print('[gru] spatial bank FAILED (%s) -> spatial ckpts will be skipped'
                      % str(_ge)[:60], flush=True)"""

DRIVER_OLD_CH = """                _gr = gru_channels(_ghw, _gtw, _GRU_PF[_gwid], _s2_grcal, _gru_stride_fn)"""
DRIVER_NEW_CH = """                _gsrc = None
                if _gru_bank is not None:
                    try:
                        _gsrc = spatial_query(_gru_bank, _gwid, _ghw)
                    except Exception:
                        _gsrc = None
                _gr = gru_channels(_ghw, _gtw, _GRU_PF[_gwid], _s2_grcal, _gru_stride_fn,
                                   spatial_src=_gsrc)"""

W_OLD = "_w3 = 0.20 * _m3['tvt_sp45'] + 0.50 * _m3['tvt_fleongg'] + 0.30 * _m3['tvt_gru']"
W_NEW = "_w3 = 0.15 * _m3['tvt_sp45'] + 0.45 * _m3['tvt_fleongg'] + 0.40 * _m3['tvt_gru']"
P_OLD = "weights 0.20/0.50/0.30 (probe B); "
P_NEW = "weights 0.15/0.45/0.40 (spatial); "

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    done_gru = done_w = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "checkpoints loaded" in s and "import glob as _gru_glob" in s:
            drv = s[s.index("import glob as _gru_glob"):]
            assert DRIVER_OLD_LOAD in drv and DRIVER_OLD_CH in drv, f"{nb_path}: driver anchors missing"
            drv = drv.replace(DRIVER_OLD_LOAD, DRIVER_NEW_LOAD).replace(DRIVER_OLD_CH, DRIVER_NEW_CH)
            new = ("# GRU pole (patch43: spatial channels + dip fusion) — _gru_infer.py embedded "
                   "verbatim below, then the driver.\n" + GI_SRC + "\n\n" + drv)
            ast.parse(new)
            assert "spatial_bank(CFG.dataset_path" in new and "spatial_src=_gsrc" in new
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            done_gru = True
        elif W_OLD in s:
            new = s.replace(W_OLD, W_NEW).replace(P_OLD, P_NEW)
            assert W_NEW in new
            ast.parse(new)
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
            done_w = True
    assert done_gru and done_w, f"{nb_path}: gru={done_gru} weights={done_w}"
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{nb_path}: GRU cell regenerated + weights 0.15/0.45/0.40, ast OK")
print("ALL DONE")
