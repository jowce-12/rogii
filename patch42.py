# patch42: stage3 wiring — extend the gold candidate pool in the submission notebooks.
# Inserts a fail-open wrapper into the gold cell (cell containing _gold_candidate_pool)
# right BEFORE the run block ("if not _GOLD_ENABLE:"), so the rebinding takes effect for
# calibration + final decode in the same cell. Applied to BOTH notebooks identically.
import json, ast, io

WRAPPER = '''
# ---- patch42 (stage3): gold candidate-pool extension ----
# Adds 3 STRIDE beam decodes (seg 200/400/100) + neighbor full-curve transfer (IDW k=12
# over train-well samples, known-zone offset anchor, 600ft gate) to the gold pool.
# Fail-open at every level: any failure leaves the original pool untouched.
_g2_orig_pool = _gold_candidate_pool
_G2_MEMO = {}
_G2_BANK = {'built': False, 'tree': None, 'u': None, 'wid': None}


def _g2_bank_build():
    if _G2_BANK['built']:
        return _G2_BANK['tree'] is not None
    _G2_BANK['built'] = True
    try:
        from scipy.spatial import cKDTree as _cKDTree
        xs, us, ws = [], [], []
        for _p in sorted((_GOLD_DATA / 'train').glob('*__horizontal_well.csv')):
            _w = _p.stem.replace('__horizontal_well', '')
            try:
                _df = _gold_pd.read_csv(_p, usecols=['X', 'Y', 'Z', 'TVT']).dropna()
            except Exception:
                continue
            if len(_df) == 0:
                continue
            _ix = _gold_np.linspace(0, len(_df) - 1, min(60, len(_df)), dtype=int)
            xs.append(_df[['X', 'Y']].values[_ix])
            us.append((_df['TVT'].values + _df['Z'].values)[_ix])
            ws.extend([_w] * len(_ix))
        if not xs:
            return False
        _G2_BANK['tree'] = _cKDTree(_gold_np.vstack(xs))
        _G2_BANK['u'] = _gold_np.concatenate(us)
        _G2_BANK['wid'] = _gold_np.array(ws)
        print(f"[g2] neighbor bank: {len(_G2_BANK['u'])} samples from train wells", flush=True)
        return True
    except Exception as _e:
        print(f"[g2] bank build failed (nbr_curve disabled): {_e}", flush=True)
        return False


def _g2_nbr_curve(wid, hw_m):
    if not _g2_bank_build():
        return None
    if not all(c in hw_m.columns for c in ('X', 'Y', 'Z')):
        return None
    xy = hw_m[['X', 'Y']].values.astype(float)
    dd, ii = _G2_BANK['tree'].query(xy, k=12)
    mask = _G2_BANK['wid'][ii] != wid
    w = _gold_np.where(mask, 1.0 / _gold_np.maximum(dd, 1.0) ** 2, 0.0)
    ws = w.sum(1)
    if (ws <= 0).any():
        return None
    u = (w * _G2_BANK['u'][ii]).sum(1) / ws
    if float(_gold_np.median(_gold_np.where(mask, dd, _gold_np.inf).min(1))) > 600.0:
        return None
    kn = hw_m[hw_m['TVT_input'].notna()]
    if len(kn) < 30:
        return None
    off = float(_gold_np.median(kn['TVT_input'].values + kn['Z'].values - u[kn.index.values]))
    return u - hw_m['Z'].values.astype(float) + off


def _g2_stride_full(wid, hw_m, tw, seg):
    key = (wid, int(hw_m['TVT_input'].notna().sum()), seg)
    if key in _G2_MEMO:
        return _G2_MEMO[key]
    out = None
    try:
        st = _stride_track(hw_m, tw, seg_len=seg)
        ev_mask = hw_m['TVT_input'].isna().values
        if st is not None and len(st) == int(ev_mask.sum()) and _gold_np.all(_gold_np.isfinite(st)):
            full = _gold_np.full(len(hw_m), _gold_np.nan)
            full[ev_mask] = _gold_np.asarray(st, float)
            kn_v = hw_m['TVT_input'].values.astype(float)
            fin = _gold_np.isfinite(kn_v)
            full[fin] = kn_v[fin]
            out = full
    except Exception:
        out = None
    _G2_MEMO[key] = out
    return out


def _g2_pool_ext(wid, hw_masked, tw, data_dir, variants, include_pf=True, n_seeds=24, n_particles=350):
    pool = _g2_orig_pool(wid, hw_masked, tw, data_dir, variants, include_pf=include_pf,
                         n_seeds=n_seeds, n_particles=n_particles)
    for _nm, _seg in (('stride', 200.0), ('stride_stiff', 400.0), ('stride_loose', 100.0)):
        try:
            _arr = _g2_stride_full(wid, hw_masked, tw, _seg)
            if _arr is not None:
                pool[_nm] = _arr
        except Exception:
            pass
    try:
        _nc = _g2_nbr_curve(wid, hw_masked)
        if _nc is not None and _gold_np.isfinite(_nc).mean() > 0.9:
            _full = _gold_np.asarray(_nc, float).copy()
            _kn = hw_masked['TVT_input'].values.astype(float)
            _fin = _gold_np.isfinite(_kn)
            _full[_fin] = _kn[_fin]
            pool['nbr_curve'] = _full
    except Exception:
        pass
    return pool


_gold_candidate_pool = _g2_pool_ext
print('[g2] gold pool extension wired (stride x3 + nbr_curve, fail-open)', flush=True)
# ---- end patch42 ----
'''

MARK = "if not _GOLD_ENABLE:"

for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",
                "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"):
    nb = json.load(open(nb_path, encoding="utf-8"))
    hit = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "def _gold_candidate_pool" in "".join(c["source"]):
            hit = i
            break
    assert hit is not None, f"{nb_path}: gold cell not found"
    src = "".join(nb["cells"][hit]["source"])
    assert "patch42" not in src, f"{nb_path}: already patched"
    lines = src.split("\n")
    at = next(i for i, ln in enumerate(lines) if ln.startswith(MARK))
    new_src = "\n".join(lines[:at]) + "\n" + WRAPPER + "\n" + "\n".join(lines[at:])
    ast.parse(new_src)              # whole-cell syntax check
    # sanity: exactly one rebinding, placed after the original def, before the run block
    assert new_src.index("def _gold_candidate_pool") < new_src.index("_g2_orig_pool = _gold_candidate_pool")
    assert new_src.index("_gold_candidate_pool = _g2_pool_ext") < new_src.index(MARK + "\n" if MARK + "\n" in new_src else MARK)
    nb["cells"][hit]["source"] = new_src.splitlines(keepends=True)
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{nb_path}: patched cell {hit} (insert at line {at}), ast OK")

# verify _gold_np exists in the gold cell (wrapper depends on it)
for nb_path in ("rogii-geology-aware-ensembling-lb-7-129.ipynb",):
    nb = json.load(open(nb_path, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code" and "def _gold_candidate_pool" in "".join(c["source"]):
            s = "".join(c["source"])
            assert "import numpy as _gold_np" in s or "_gold_np" in s.split("patch42")[0], "no _gold_np before wrapper"
            print("deps OK: _gold_np present before wrapper")
print("ALL DONE")
