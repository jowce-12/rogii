# Patch 22 — deploy measured winners (measure_1234.log, 150w v5-cache proxy, midreg LGB 4-fold):
#   item 2: include likpf_scale_*_d features        11.4932 -> 11.0394  (-0.45)
#   item 1: + PF-quality features (ptstd/llspread/bestll/grsig)  -> 10.9329 with both (-0.56 total)
#   item 4: ridge meta + likpf_mean_d input          11.8783 -> 11.5818  (-0.30, anchor coef 0.45 = largest)
#   item 3: refit-iteration boost — NOISE (seed0 +0.001 / seed1 -0.02, direction disagrees) -> NOT applied
# Cache: v6 -> v7 (new likpf quality cols). Both trainers get a v6->v7 upgrade path that
# joins quality onto an existing v6 cache (~20-40min PF pass) instead of a ~2.8h full rebuild.
# Targets: train_stack.py, train_quick.py, both submission notebooks (+ .py mirror regen after).
import json, io, ast, shutil, sys

def apply(text, old, new, tag, where):
    n = text.count(old)
    assert n == 1, f"{where}: pattern '{tag}' matched {n} times (expected 1)"
    return text.replace(old, new)

# ---------------- shared replacements ----------------
R1_OLD = '''def _likpf_rows(wid, split):
    hw, tw = load_well(wid, split)
    out, idx, _ = lik_pf(hw, tw)
    if not len(out): return None
    d = {"id": [f"{wid}_{i}" for i in idx]}
    for k, v in out.items():
        d["likpf_" + k.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = v.astype(np.float32)
    return pd.DataFrame(d)'''
R1_NEW = '''LIKPF_QUALITY = ("likpf_ptstd", "likpf_llspread", "likpf_bestll", "likpf_grsig")

def _likpf_rows(wid, split):
    hw, tw = load_well(wid, split)
    out, idx, q = lik_pf(hw, tw, with_quality=True)
    if not len(out): return None
    d = {"id": [f"{wid}_{i}" for i in idx]}
    for k, v in out.items():
        d["likpf_" + k.replace("pf_scale_", "scale_").replace("pf_mean", "mean")] = v.astype(np.float32)
    # PF-uncertainty metrics: lets the GBM learn when to trust the tracker
    # (in-model version of the A4 alias gate; validated -0.11 OOF on top of scale_d)
    d["likpf_ptstd"] = q["pf_pt_std"].astype(np.float32)
    d["likpf_llspread"] = np.float32(q["pf_ll_spread"])
    d["likpf_bestll"] = np.float32(q["pf_best_ll"])
    d["likpf_grsig"] = np.float32(q["pf_gr_sig"])
    return pd.DataFrame(d)'''

R2_OLD = '''def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        df[c] = df[c].fillna(df["last_known_tvt"]); df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df'''
R2_NEW = '''def add_likpf_features(df, likpf):
    df = df.merge(likpf, on="id", how="left")
    for c in [c for c in likpf.columns if c != "id"]:
        if c in LIKPF_QUALITY:
            continue   # uncertainty metrics, not TVT-space: no last-known fill / delta (NaN is fine for GBMs)
        df[c] = df[c].fillna(df["last_known_tvt"]); df[c+"_d"] = (df[c]-df["last_known_tvt"]).astype(np.float32)
    return df'''

# feature filter — trainers (0-indent)
F1_OLD = '''feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]'''
F1_NEW = '''feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
         and not (c.startswith("likpf_scale_") and not c.endswith("_d"))   # keep scale DELTAS (validated -0.45 OOF)
         and c != "likpf_mean" and c in test_df.columns]'''
# feature filter — notebooks main() from-scratch path (8-indent)
F2_OLD = '''        feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
                 and not (c.startswith("likpf_scale_") or c == "likpf_mean") and c in test_df.columns]'''
F2_NEW = '''        feats = [c for c in train_df.columns if c not in {"well", "id", "target"}
                 and not (c.startswith("likpf_scale_") and not c.endswith("_d"))   # keep scale deltas
                 and c != "likpf_mean" and c in test_df.columns]'''

# ---------------- trainer-only: cache v7 + upgrade path ----------------
CN_OLD = '''    return f"train_features_v6_f{os.environ.get('ROGII_FEATS','1')}{_g}{_c}_w{_n}.parquet"'''
CN_NEW = '''    # v7: + PF-quality cols (likpf_ptstd/llspread/bestll/grsig) + scale-delta features kept
    return f"train_features_v7_f{os.environ.get('ROGII_FEATS','1')}{_g}{_c}_w{_n}.parquet"'''

QJ = '''def _quality_join(wids, split):
    # PF-uncertainty columns only. Forcing grcal=off reproduces exactly what a full v7
    # build stores: in blend mode the quality metrics come from the raw pass anyway.
    _prev = os.environ.get("ROGII_GRCAL")
    os.environ["ROGII_GRCAL"] = "off"
    try:
        def _qr(wid):
            hw, tw = load_well(wid, split)
            out, idx, q = lik_pf(hw, tw, with_quality=True)
            if not len(out): return None
            return pd.DataFrame({"id": [f"{wid}_{i}" for i in idx],
                                 "likpf_ptstd": q["pf_pt_std"].astype(np.float32),
                                 "likpf_llspread": np.float32(q["pf_ll_spread"]),
                                 "likpf_bestll": np.float32(q["pf_best_ll"]),
                                 "likpf_grsig": np.float32(q["pf_gr_sig"])})
        res = Parallel(n_jobs=CFG.n_jobs, prefer="threads")(delayed(_qr)(w) for w in wids)
        return pd.concat([r for r in res if r is not None], ignore_index=True)
    finally:
        if _prev is None: os.environ.pop("ROGII_GRCAL", None)
        else: os.environ["ROGII_GRCAL"] = _prev

def load_or_build_train_features(train_wids):'''

SEARCH_OLD = '''    if mode != "rebuild":
        for _p in _g.glob(f"/kaggle/input/**/{name}", recursive=True) + [str(CFG.OUT / name)]:
            if os.path.exists(_p):
                print(f"[cache] loading train features from {_p}", flush=True)
                return pd.read_parquet(_p)'''
SEARCH_NEW = SEARCH_OLD + '''
        # v6 -> v7 upgrade: join quality columns onto an existing v6 cache
        # (~20-40min lik-PF pass at 773 wells vs ~2.8h full rebuild)
        _v6 = name.replace("_v7_", "_v6_")
        for _p in _g.glob(f"/kaggle/input/**/{_v6}", recursive=True) + [str(CFG.OUT / _v6)]:
            if os.path.exists(_p):
                print(f"[cache] v6->v7 upgrade: joining PF-quality columns onto {_p}", flush=True)
                _df = pd.read_parquet(_p).merge(_quality_join(train_wids, "train"), on="id", how="left")
                if mode != "off":
                    try:
                        _df.to_parquet(CFG.OUT / name, index=False)
                        print(f"[cache] saved -> {CFG.OUT / name}", flush=True)
                    except Exception as _e:
                        print("[cache] save skipped:", _e, flush=True)
                return _df'''

# ---------------- stack-only: ridge meta anchor ----------------
M1_OLD = '''# --- Ridge meta on the base OOF ---
OOF = np.column_stack([oof_cols[n] for n in base_names])'''
M1_NEW = '''# --- Ridge meta on the base OOF + physics anchor as extra input (validated -0.30 proxy;
# anchor drew the largest coef — the meta learns per-regime GBM-vs-PF weighting) ---
META_EXTRA = ["likpf_mean_d"]
OOF = np.column_stack([oof_cols[n] for n in base_names]
                      + [np.nan_to_num(train_df[c].values.astype(np.float32)) for c in META_EXTRA])'''
M2_OLD = '''print("ridge coefs:", dict(zip(base_names, [round(float(c), 4) for c in ridge.coef_])),'''
M2_NEW = '''print("ridge coefs:", dict(zip(base_names + META_EXTRA, [round(float(c), 4) for c in ridge.coef_])),'''
M3_OLD = '''json.dump({"base_names": base_names, "use_cb": use_cb, "ridge_oof_rmse": float(rmse(y, meta_oof + anchor)),
           "residual_anchor": ("likpf_mean_d" if _resid else None),'''
M3_NEW = '''json.dump({"base_names": base_names, "use_cb": use_cb, "ridge_oof_rmse": float(rmse(y, meta_oof + anchor)),
           "meta_extra": META_EXTRA,
           "residual_anchor": ("likpf_mean_d" if _resid else None),'''

# ---------------- notebook-only: inference meta_extra + missing-feature warn ----------------
N1_OLD = '''            _cols = np.column_stack([_pred[_n] for _n in _bn if _n in _pred])
            meta_test = joblib.load(_ridge_pkl).predict(_cols)'''
N1_NEW = '''            _cols = np.column_stack([_pred[_n] for _n in _bn if _n in _pred])
            _mx = _meta.get("meta_extra") or []
            if _mx:
                # physics anchor joins the ridge inputs (must mirror training's META_EXTRA)
                _cols = np.hstack([_cols] + [np.nan_to_num(test_df[_c].values.astype(np.float32)).reshape(-1, 1)
                                             for _c in _mx])
                print(f"meta extra inputs: {_mx}", flush=True)
            meta_test = joblib.load(_ridge_pkl).predict(_cols)'''
N2_OLD = '''        feats = json.load(open(models_dir/"features.json"))
        for c in feats:
            if c not in test_df.columns: test_df[c] = 0.0'''
N2_NEW = '''        feats = json.load(open(models_dir/"features.json"))
        _missing = [c for c in feats if c not in test_df.columns]
        for c in _missing:
            test_df[c] = 0.0
        if _missing:
            print(f"[WARN] {len(_missing)} features.json features missing from the test build, zero-filled "
                  f"(notebook/model version mismatch?): {_missing[:8]}", flush=True)'''

# ================= trainers =================
for path, extra in [("train_stack.py", "stack"), ("train_quick.py", "quick")]:
    shutil.copy(path, path + ".bak22")
    t = io.open(path, encoding="utf-8").read()
    t = apply(t, R1_OLD, R1_NEW, "R1 likpf_rows", path)
    t = apply(t, R2_OLD, R2_NEW, "R2 add_likpf", path)
    t = apply(t, F1_OLD, F1_NEW, "F1 filter", path)
    t = apply(t, CN_OLD, CN_NEW, "cache v7", path)
    t = apply(t, "def load_or_build_train_features(train_wids):", QJ, "quality_join", path)
    t = apply(t, SEARCH_OLD, SEARCH_NEW, "v6->v7 upgrade", path)
    if extra == "stack":
        t = apply(t, M1_OLD, M1_NEW, "M1 meta extra", path)
        t = apply(t, M2_OLD, M2_NEW, "M2 coef names", path)
        t = apply(t, M3_OLD, M3_NEW, "M3 stack_meta", path)
    ast.parse(t)
    io.open(path, "w", encoding="utf-8").write(t)
    print(f"{path}: patched OK")

# ================= notebooks =================
NB = [("public-score-rogii-lb-7-159.ipynb", "public-score-rogii-lb-7-159.BACKUP22.ipynb"),
      ("rogii-geology-aware-ensembling-lb-7-129.ipynb", "rogii-geology-aware-ensembling-lb-7-129.BACKUP3.ipynb")]
for path, bak in NB:
    shutil.copy(path, bak)
    nb = json.load(io.open(path, encoding="utf-8"))
    hits = {k: 0 for k in ["R1", "R2", "F2", "N1", "N2"]}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"]); s0 = s
        for tag, old, new in [("R1", R1_OLD, R1_NEW), ("R2", R2_OLD, R2_NEW), ("F2", F2_OLD, F2_NEW),
                              ("N1", N1_OLD, N1_NEW), ("N2", N2_OLD, N2_NEW)]:
            if old in s:
                assert s.count(old) == 1, f"{path}: {tag} x{s.count(old)} in one cell"
                s = s.replace(old, new); hits[tag] += 1
        if s != s0:
            ast.parse(s)
            c["source"] = s.splitlines(keepends=True)
    assert all(v == 1 for v in hits.values()), f"{path}: replacement hits {hits} (each must be exactly 1)"
    json.dump(nb, io.open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{path}: patched OK {hits} (backup {bak})")

print("done")
