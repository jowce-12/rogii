# patch38: GRU 4th pole deployment into the submission notebooks.
#   P1 cell7   run_pf_lik_ensemble_scales gains pairs_out (pure additive)
#   P2 cell24  selector main PF call caches seeds 0..31 eval-row preds per well
#   P3 new cell (after stride-join inference cell): embedded _gru_infer + driver
#   P4 blend cell: 3-way final (sp45 .30 / fleongg .40 / gru .30), 2-way fallback
import ast
import json

GI_SRC = open("_gru_infer.py", encoding="utf-8").read()

DRIVER = """

# === patch38 driver: GRU 4th-pole inference over test wells ===
# PF channels reuse seeds 0..31 CACHED from the selector's 192-seed pass (zero extra PF
# cost). Channels are the parity-proven replica of build_gru_data.py (bit-identical).
# Pre-registered blend weights: sp45 0.30 / fleongg 0.40 / gru 0.30 (harness: seed7
# 8.3158->8.1600, seed11 6.9974->6.6074). Missing ckpts/torch -> LOUD skip, 2-way fallback.
import glob as _gru_glob
import time as _gru_time
GRU_TVT = {}
try:
    import torch as _gru_torch
    import torch.nn as _gru_nn
    _gru_t0 = _gru_time.time()
    _gru_files = sorted(set(_gru_glob.glob('/kaggle/input/**/gru_fold*_*.pt', recursive=True)
                            + _gru_glob.glob('gru_fold*_*.pt')))
    if not _gru_files:
        print('[gru] NO checkpoints attached -> GRU pole DISABLED (2-way fallback)', flush=True)
    elif not _GRU_PF:
        print('[gru] PF seed cache empty -> GRU pole DISABLED (2-way fallback)', flush=True)
    else:
        _gru_ckpts = [_gru_torch.load(_f, map_location='cpu') for _f in _gru_files]
        print('[gru] %d checkpoints loaded' % len(_gru_ckpts), flush=True)

        def _gru_stride_fn(_h, _t, _seg):
            _r = _stride_feat(_h, _t, seg_len=_seg)
            if _r is None:
                return None
            return None, _r[1], None

        _gru_done = _gru_skip = 0
        for _gwid in sorted(_GRU_PF.keys()):
            try:
                _ghw, _gtw = load_well(_gwid, 'test')
                _gr = gru_channels(_ghw, _gtw, _GRU_PF[_gwid], _s2_grcal, _gru_stride_fn)
                if _gr is None:
                    _gru_skip += 1
                    continue
                _gch, _gmd_grid, _gcut_md, _gev_idx, _glast = _gr
                _gpred = gru_forward(_gch, _gru_ckpts, _gru_torch, _gru_nn)
                _gmrg = (np.arange(len(_gmd_grid), dtype=np.float64) - G_CTX) * G_STEP
                _gev_md = _ghw.loc[_gev_idx, 'MD'].values.astype(float) - _gcut_md
                _gvals = np.interp(_gev_md, _gmrg, _gpred) + _glast
                for _gi, _gv in zip(_gev_idx, _gvals):
                    GRU_TVT['%s_%d' % (_gwid, _gi)] = float(_gv)
                _gru_done += 1
                if _gru_done % 25 == 0:
                    print('[gru] %d wells (%.0fs)' % (_gru_done, _gru_time.time() - _gru_t0), flush=True)
            except Exception as _ge:
                _gru_skip += 1
                print('[gru] %s skipped: %s' % (_gwid, str(_ge)[:60]), flush=True)
        print('[gru] DONE: %d wells, %d skipped, %d rows, %.0fs'
              % (_gru_done, _gru_skip, len(GRU_TVT), _gru_time.time() - _gru_t0), flush=True)
except Exception as _ge:
    print('[gru] DISABLED (%s) -> 2-way fallback' % str(_ge)[:70], flush=True)
"""

GRU_CELL = ("# === patch38: GRU 4th pole - parity-proven channel builder (embedded _gru_infer) ===\n"
            + GI_SRC + DRIVER)
ast.parse(GRU_CELL)

BLEND_ADD = """

# === patch38: 3-way final blend with the GRU pole (pre-registered 0.30/0.40/0.30) ===
try:
    _g_map = GRU_TVT if isinstance(GRU_TVT, dict) else {}
except NameError:
    _g_map = {}
if _g_map:
    _m3 = _merged.copy()
    _m3['tvt_gru'] = _m3['id'].map(_g_map)
    _cov = int(_m3['tvt_gru'].notna().sum())
    _sel2 = _final_pd.read_csv(_WORK / 'submission.csv')[['id', 'tvt']]
    _m3 = _m3.merge(_sel2.rename(columns={'tvt': 'tvt_2way'}), on='id', how='left')
    _w3 = 0.30 * _m3['tvt_sp45'] + 0.40 * _m3['tvt_fleongg'] + 0.30 * _m3['tvt_gru']
    _m3['tvt'] = _final_np.where(_m3['tvt_gru'].notna(), _w3, _m3['tvt_2way'])
    assert _final_np.isfinite(_m3['tvt'].to_numpy(dtype=float)).all()
    _m3[['id', 'tvt']].to_csv(_WORK / 'submission_3way.csv', index=False)
    _m3[['id', 'tvt']].to_csv(_WORK / 'submission.csv', index=False)
    print('[3way] GRU pole applied: gru rows %d/%d; weights 0.30/0.40/0.30; '
          'submission.csv OVERWRITTEN' % (_cov, len(_m3)), flush=True)
else:
    print('[3way] no GRU predictions -> keeping 2-way submission', flush=True)
"""

CACHE_LINES = """        _gru_ev = hw_te[hw_te['TVT_input'].isna()]
        try:
            _GRU_PF[wid] = (np.stack([np.asarray(_p, float)[_gru_ev.index.values]
                                      for _p, _l in _gru_pairs[:32]], 0),
                            np.array([_l for _p, _l in _gru_pairs[:32]], float))
        except Exception as _ge:
            print('  [gru] pf cache skipped:', str(_ge)[:50], flush=True)
        del _gru_pairs"""

for nb in ["rogii-geology-aware-ensembling-lb-7-129.ipynb",
           "rogii-geology-aware-ensembling-lb-7-129-conservative.ipynb"]:
    d = json.load(open(nb, encoding="utf-8"))
    cells = d["cells"]
    hits = {"p1": 0, "p2": 0, "p4": 0}
    gru_cell_pos = None
    for idx, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        changed = False
        o1 = ("def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, n_seeds=128):\n"
              "    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)")
        if o1 in src:
            n1 = ("def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500, "
                  "n_seeds=128, pairs_out=None):\n"
                  "    pairs = _pf_all_seeds(hw, tw, n_particles, n_seeds)\n"
                  "    if pairs_out is not None:\n"
                  "        pairs_out.extend(pairs)   # patch38: expose per-seed (pred, loglik) for the GRU pole")
            src = src.replace(o1, n1, 1)
            hits["p1"] += 1
            changed = True
        if "pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=192" in src:
            lines = src.split("\n")
            k = next(i for i, ln in enumerate(lines) if ln.strip().startswith(
                "pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref, n_particles=500, n_seeds=192"))
            call = lines[k]
            assert "pairs_out" not in call
            new_call = call.replace("n_seeds=192)", "n_seeds=192, pairs_out=_gru_pairs)")
            assert new_call != call
            lines[k] = "        _gru_pairs = []   # patch38\n" + new_call + "\n" + CACHE_LINES
            j = next(i for i, ln in enumerate(lines) if ln.startswith("for i, wid in enumerate(test_wells)"))
            lines[j] = "_GRU_PF = {}   # patch38: per-well PF seed cache for the GRU pole\n" + lines[j]
            src = "\n".join(lines)
            hits["p2"] += 1
            changed = True
        if "_SELECTED_SP45_WEIGHT = 0.60" in src and "_merge_blend_inputs" in src:
            src = src.rstrip("\n") + "\n" + BLEND_ADD
            hits["p4"] += 1
            changed = True
        if "[stride-feat] joined" in src:
            gru_cell_pos = idx + 1
        if changed:
            ast.parse(src)
            c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + (
                [src.split("\n")[-1]] if src.split("\n")[-1] else [])
    assert gru_cell_pos is not None, nb
    tail = GRU_CELL.split("\n")[-1]
    cells.insert(gru_cell_pos, {"cell_type": "code", "execution_count": None, "metadata": {},
                                "outputs": [],
                                "source": [l + "\n" for l in GRU_CELL.split("\n")[:-1]] + ([tail] if tail else [])})
    assert all(v == 1 for v in hits.values()), (nb, hits)
    json.dump(d, open(nb, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(nb, "->", hits, "| GRU cell at", gru_cell_pos)
print("PATCH38 APPLIED")
