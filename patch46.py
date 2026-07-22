# patch46: FIX the spatial-bank path bug that killed the GRU pole in the patch43
# submission (LB 7.x root cause). The fleongg section shadows the global CFG with its own
# class (attr .DATA, not .dataset_path), so `spatial_bank(CFG.dataset_path/'train')`
# raised -> all spatial ckpts skipped -> 2-way fallback for EVERY well.
# Fix: resolve the train dir through an attribute chain + hardcoded competition path
# fallback, and verify it actually contains well CSVs before building the bank.
import json, ast

OLD = """        _gru_bank = None
        if _n_sp:
            try:
                _gru_bank = spatial_bank(CFG.dataset_path / 'train')
                print('[gru] spatial bank: %d samples' % len(_gru_bank['ancc']), flush=True)
            except Exception as _ge:
                print('[gru] spatial bank FAILED (%s) -> spatial ckpts will be skipped'
                      % str(_ge)[:60], flush=True)"""
NEW = """        _gru_bank = None
        if _n_sp:
            try:
                import pathlib as _gru_pl
                _gru_tr = None
                for _cand in (getattr(CFG, 'dataset_path', None), getattr(CFG, 'DATA', None),
                              getattr(CFG, 'data_path', None),
                              '/kaggle/input/competitions/rogii-wellbore-geology-prediction',
                              '/kaggle/input/rogii-wellbore-geology-prediction', '.'):
                    if _cand is None:
                        continue
                    _p = _gru_pl.Path(_cand) / 'train'
                    try:
                        if next(_p.glob('*__horizontal_well.csv'), None) is not None:
                            _gru_tr = _p
                            break
                    except Exception:
                        continue
                if _gru_tr is None:
                    raise RuntimeError('no train dir with well CSVs found')
                _gru_bank = spatial_bank(_gru_tr)
                print('[gru] spatial bank: %d samples (from %s)'
                      % (len(_gru_bank['ancc']), _gru_tr), flush=True)
            except Exception as _ge:
                print('[gru] spatial bank FAILED (%s) -> spatial ckpts will be skipped'
                      % str(_ge)[:80], flush=True)"""

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    hit = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "spatial_bank(CFG.dataset_path / 'train')" not in s:
            continue
        assert OLD in s, f"{nb_path}: bank-block anchor missing"
        new = s.replace(OLD, NEW, 1)
        ast.parse(new)
        assert "spatial_bank(_gru_tr)" in new and "CFG.dataset_path / 'train'" not in new
        nb["cells"][i]["source"] = new.splitlines(keepends=True)
        json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{nb_path}: patch46 applied to cell {i}, ast OK")
        hit = True
        break
    assert hit, f"{nb_path}: GRU cell not found"
print("ALL DONE")
