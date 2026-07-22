"""Patch 20 (option a): S1 blend mode + A5 alias meta-features.

S1 blend (validated on two disjoint 150-well samples: w=0.35 -> -0.24..-0.74):
  fleongg lik_pf gains ROGII_GRCAL=blend -> runs the PF twice (raw typewell +
  band-affine-calibrated typewell) and mixes predictions at ROGII_GRCAL_W (0.35).
  Cache key already separates via the _c{mode} suffix. stack_meta.json records
  the grcal mode; inference warns on train/inference mismatch.

A5 (alias well-level meta-features; join-only, NO cache rebuild):
  gr_corr / gr_corr_hf / tw_hf_std / alias_gap computed per well from raw CSVs
  (known-zone only, leak-free) + likpf_tspread = likpf_scale_3_d - likpf_scale_12_d
  from existing cache columns. Wired into: b515d5c6 (helpers), stack & quick
  drivers (train+test), and the submission notebook's fleongg main() test path
  (required — otherwise features.json mismatch would silently fill 0.0).
Idempotent."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))

# ---------------- 1) S1 blend in fleongg lik_pf (83876170) ----------------
c = next(x for x in nb['cells'] if x.get('id') == '83876170')
s = ''.join(c['source'])
if '"blend"' in s:
    print('lik_pf blend already applied')
else:
    o = '''    _gc = os.environ.get("ROGII_GRCAL", "off").lower()
    if _gc in ("affine", "var", "offset"):
        tw_gr, _, _ = _grcal_tw(tw_tvt, tw_gr, kn.TVT_input.values, kn.GR.values, _gc)   # S1'''
    n = '''    _gc = os.environ.get("ROGII_GRCAL", "off").lower()
    _tw_gr_cal = None
    if _gc == "blend":
        _tw_gr_cal, _, _ = _grcal_tw(tw_tvt, tw_gr, kn.TVT_input.values, kn.GR.values, "affine")   # S1 blend channel
    elif _gc in ("affine", "var", "offset"):
        tw_gr, _, _ = _grcal_tw(tw_tvt, tw_gr, kn.TVT_input.values, kn.GR.values, _gc)   # S1'''
    assert s.count(o) == 1; s = s.replace(o, n)

    o = '''    out["pf_mean"] = preds.mean(0)
    q = {}'''
    n = '''    out["pf_mean"] = preds.mean(0)
    if _tw_gr_cal is not None:
        # S1 blend: second PF pass on the affine-calibrated typewell, mix predictions.
        # Validated (150w x2 disjoint samples): w=0.35 improves pooled RMSE -0.24..-0.74.
        _ta = np.interp(kn.TVT_input.values, tw_tvt, _tw_gr_cal)
        _gs2 = float(np.clip(np.nanstd(kn.GR.fillna(0).values - _ta), 10., 60.))
        _gg2, _gm2, _gt2 = _grid(tw_tvt, _tw_gr_cal)
        _p2, _l2 = _pf_lik_allseeds(ev.MD.values.astype(float), ev.Z.values.astype(float), gr_v,
                                    _gg2, _gm2, _gt2, _gs2, ls, ir, n_particles, n_seeds, seed_base,
                                    0.998, 0.002, 0.005, 0.1, 0.001, 0.5, init_spr)
        _ln2 = _l2 - _l2.max()
        _wS1 = float(os.environ.get("ROGII_GRCAL_W", "0.35"))
        for sc in scales:
            _w2 = np.exp(_ln2 / float(sc)); _w2 /= _w2.sum()
            out[f"pf_scale_{sc:g}"] = (1 - _wS1) * out[f"pf_scale_{sc:g}"] + _wS1 * ((_w2[:, None] * _p2).sum(0))
        out["pf_mean"] = (1 - _wS1) * out["pf_mean"] + _wS1 * _p2.mean(0)
    q = {}'''
    assert s.count(o) == 1; s = s.replace(o, n)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print('S1 blend wired into fleongg lik_pf')

# ---------------- 2) A5 helpers in b515d5c6 ----------------
c = next(x for x in nb['cells'] if x.get('id') == 'b515d5c6')
s = ''.join(c['source'])
if 'add_alias_metafeats' in s:
    print('A5 helpers already present')
else:
    HELPERS = '''def _alias_stats(wid, split):
    """A5: well-level aliasing-risk stats from the known zone only (leak-free).
    gr_corr: raw hwGR-vs-twGR corr (probe: +0.32 spearman with tracker error,
    independent of drift). gr_corr_hf: rolling-101-detrended corr (localization
    signal). tw_hf_std: HF signal strength of the typewell in the drilled band."""
    try:
        hw = pd.read_csv(CFG.DATA / split / f"{wid}__horizontal_well.csv", usecols=["TVT_input", "GR"])
        tw = pd.read_csv(CFG.DATA / split / f"{wid}__typewell.csv", usecols=["TVT", "GR"]).sort_values("TVT")
    except Exception:
        return wid, np.nan, np.nan, np.nan
    kn = hw[hw.TVT_input.notna()]
    if len(kn) < 120:
        return wid, np.nan, np.nan, np.nan
    tw_tvt = tw.TVT.values.astype(float); tw_gr = tw.GR.ffill().bfill().values.astype(float)
    kt = kn.TVT_input.values.astype(float); kg = kn.GR.values.astype(float)
    ta = np.interp(kt, tw_tvt, tw_gr)
    v = np.isfinite(kg)
    if v.sum() < 100:
        return wid, np.nan, np.nan, np.nan
    a = kg[v]; b = ta[v]
    gr_corr = float(np.corrcoef(a, b)[0, 1]) if (a.std() > 1e-6 and b.std() > 1e-6) else np.nan
    sa = pd.Series(a); sb = pd.Series(b)
    ah = (sa - sa.rolling(101, center=True, min_periods=1).mean()).values
    bh = (sb - sb.rolling(101, center=True, min_periods=1).mean()).values
    gr_corr_hf = float(np.corrcoef(ah, bh)[0, 1]) if (ah.std() > 1e-6 and bh.std() > 1e-6) else np.nan
    band = tw_gr[(tw_tvt >= np.nanmin(kt) - 10) & (tw_tvt <= np.nanmax(kt) + 10)]
    if len(band) >= 40:
        tb = pd.Series(band)
        tw_hf_std = float((tb - tb.rolling(101, center=True, min_periods=1).mean()).std())
    else:
        tw_hf_std = np.nan
    return wid, gr_corr, gr_corr_hf, tw_hf_std

def add_alias_metafeats(df, split):
    """A5: join well-level alias stats + likpf temperature spread. Cache-free."""
    if "gr_corr" in df.columns:
        return df
    wids = df["well"].unique().tolist()
    rows = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(delayed(_alias_stats)(w, split) for w in wids)
    m = {r[0]: r[1:] for r in rows}
    gc_ = df["well"].map(lambda w: m.get(w, (np.nan,) * 3)[0]).astype(np.float32)
    gh_ = df["well"].map(lambda w: m.get(w, (np.nan,) * 3)[1]).astype(np.float32)
    th_ = df["well"].map(lambda w: m.get(w, (np.nan,) * 3)[2]).astype(np.float32)
    df["gr_corr"] = gc_; df["gr_corr_hf"] = gh_; df["tw_hf_std"] = th_
    df["alias_gap"] = (gc_ - gh_).astype(np.float32)
    if "likpf_scale_3_d" in df.columns and "likpf_scale_12_d" in df.columns:
        df["likpf_tspread"] = (df["likpf_scale_3_d"] - df["likpf_scale_12_d"]).astype(np.float32)
    return df

'''
    s = HELPERS + s
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print('A5 helpers added to b515d5c6')

# ---------------- 3) submission main(): test metafeats + grcal mismatch warn ----------------
c = next(x for x in nb['cells'] if x.get('id') == '209071f6')
s = ''.join(c['source'])
if 'add_alias_metafeats(test_df' in s:
    print('main() test metafeats already wired')
else:
    o = 'test_df = add_likpf_features(build_features(test_wids, "test", is_train=False), likpf_test).reset_index(drop=True)'
    n = ('test_df = add_likpf_features(build_features(test_wids, "test", is_train=False), likpf_test).reset_index(drop=True)\n'
         '    test_df = add_alias_metafeats(test_df, "test")   # A5: must match training features')
    assert s.count(o) == 1; s = s.replace(o, n)

    o = '_meta = json.load(open(_stack_meta)); _bn = _meta["base_names"]'
    n = ('''_meta = json.load(open(_stack_meta)); _bn = _meta["base_names"]
            _gtr = _meta.get("grcal", "off"); _gin = os.environ.get("ROGII_GRCAL", "off").lower()
            if _gtr != _gin:
                print(f"[WARN] grcal mismatch: model trained with '{_gtr}' but inference env is '{_gin}' "
                      f"— set ROGII_GRCAL={_gtr} in CELL 0 for consistent likpf features", flush=True)''')
    assert s.count(o) == 1; s = s.replace(o, n)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    print('main() wired: test metafeats + grcal mismatch warning')

json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------- 4) stack driver: metafeat joins + stack_meta grcal ----------------
f = 'build_stack_notebook.py'
t = io.open(f, encoding='utf-8').read()
if 'add_alias_metafeats' in t:
    print(f, 'already wired')
else:
    o = '''train_df = load_or_build_train_features(train_wids)
gc.collect()'''
    n = '''train_df = load_or_build_train_features(train_wids)
train_df = add_alias_metafeats(train_df, "train")   # A5 (join-only, no cache rebuild)
gc.collect()'''
    assert t.count(o) == 1; t = t.replace(o, n)
    o = '''test_df  = add_likpf_features(build_features(test_wids, "test", is_train=False),   build_likpf(test_wids, "test"))
gc.collect()'''
    n = '''test_df  = add_likpf_features(build_features(test_wids, "test", is_train=False),   build_likpf(test_wids, "test"))
test_df = add_alias_metafeats(test_df, "test")   # A5
gc.collect()'''
    assert t.count(o) == 1; t = t.replace(o, n)
    o = '"residual_anchor": ("likpf_mean_d" if _resid else None)},'
    n = '"residual_anchor": ("likpf_mean_d" if _resid else None),\n           "grcal": os.environ.get("ROGII_GRCAL", "off").lower()},'
    assert t.count(o) == 1; t = t.replace(o, n)
    io.open(f, 'w', encoding='utf-8').write(t)
    print(f, 'wired (A5 joins + stack_meta.grcal)')

# ---------------- 5) quick driver: metafeat joins ----------------
f = 'build_quick_notebook.py'
t = io.open(f, encoding='utf-8').read()
if 'add_alias_metafeats' in t:
    print(f, 'already wired')
else:
    o = 'train_df = load_or_build_train_features(train_wids); gc.collect()'
    n = 'train_df = add_alias_metafeats(load_or_build_train_features(train_wids), "train"); gc.collect()   # A5'
    assert t.count(o) == 1; t = t.replace(o, n)
    o = 'test_df  = add_likpf_features(build_features(test_wids, "test", is_train=False), build_likpf(test_wids, "test")); gc.collect()'
    n = 'test_df  = add_alias_metafeats(add_likpf_features(build_features(test_wids, "test", is_train=False), build_likpf(test_wids, "test")), "test"); gc.collect()'
    assert t.count(o) == 1; t = t.replace(o, n)
    io.open(f, 'w', encoding='utf-8').write(t)
    print(f, 'wired (A5 joins)')
