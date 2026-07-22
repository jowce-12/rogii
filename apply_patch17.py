"""Patch 17: residual target re-anchoring (A).
Train the stack on (target - likpf_mean_d) instead of (TVT - last_known); add the
anchor back at inference. All REPORTED RMSEs stay in delta space (comparable to
previous runs). Knob: ROGII_RESID (default 1; 0 = plain target).
Touches: build_stack_notebook.py driver, build_quick_notebook.py driver,
submission notebook FULL-STACK inference (adds anchor back via stack_meta flag).
"""
import json, io, sys, ast

# ---------------- 1) build_stack driver ----------------
f = 'build_stack_notebook.py'
t = io.open(f, encoding='utf-8').read()
if 'ROGII_RESID' in t:
    print(f, 'already patched')
else:
    o = 'X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values'
    n = (o + '''
# residual re-anchoring (ROGII_RESID=1 default): train on (target - likpf_mean_d);
# inference adds the anchor back (stack_meta.json carries the flag). RMSE prints stay in delta space.
_resid = os.environ.get("ROGII_RESID", "1") == "1"
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y - anchor
if _resid:
    print(f"[resid] target re-anchored to likpf_mean_d (target std {y.std():.2f} -> residual std {y_fit.std():.2f})", flush=True)''')
    assert t.count(o) == 1; t = t.replace(o, n)

    o = '''        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])'''
    n = '''        m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], eval_metric="rmse",
              callbacks=[early_stopping(250, verbose=False), log_evaluation(0)])'''
    assert t.count(o) == 1; t = t.replace(o, n)

    o = '            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], use_best_model=True, early_stopping_rounds=250)'
    n = '            m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], use_best_model=True, early_stopping_rounds=250)'
    assert t.count(o) == 1, ('cb fit', t.count(o)); t = t.replace(o, n)

    o = 'print(f"{name} OOF RMSE={rmse(y, oof):.4f}  (avg best_iter={full_iters[name]})", flush=True)'
    n = 'print(f"{name} OOF RMSE={rmse(y, oof + anchor):.4f}  (avg best_iter={full_iters[name]})", flush=True)'
    assert t.count(o) == 2, ('oof prints', t.count(o)); t = t.replace(o, n)

    o = '''for tr, va in cv.split(OOF, y, groups=g):
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(OOF[tr], y[tr]); meta_oof[va] = r.predict(OOF[va])
print(f"*** ridge-stack OOF RMSE={rmse(y, meta_oof):.4f}  (mean-LGB baseline was ~9.86) ***", flush=True)
ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True); ridge.fit(OOF, y)'''
    n = '''for tr, va in cv.split(OOF, y_fit, groups=g):
    r = Ridge(alpha=1.66, positive=True, fit_intercept=True); r.fit(OOF[tr], y_fit[tr]); meta_oof[va] = r.predict(OOF[va])
print(f"*** ridge-stack OOF RMSE={rmse(y, meta_oof + anchor):.4f}  (mean-LGB baseline was ~9.86; delta space) ***", flush=True)
ridge = Ridge(alpha=1.66, positive=True, fit_intercept=True); ridge.fit(OOF, y_fit)'''
    assert t.count(o) == 1; t = t.replace(o, n)

    o = 'm = LGBMRegressor(**p); m.fit(X, y); joblib.dump(m, outdir / f"lgb{ci}.pkl")'
    n = 'm = LGBMRegressor(**p); m.fit(X, y_fit); joblib.dump(m, outdir / f"lgb{ci}.pkl")'
    assert t.count(o) == 1; t = t.replace(o, n)

    o = 'm = CatBoostRegressor(**p); m.fit(X, y); m.save_model(str(outdir / f"cb{ci}.cbm"))'
    n = 'm = CatBoostRegressor(**p); m.fit(X, y_fit); m.save_model(str(outdir / f"cb{ci}.cbm"))'
    assert t.count(o) == 1; t = t.replace(o, n)

    o = 'json.dump({"base_names": base_names, "use_cb": use_cb, "ridge_oof_rmse": float(rmse(y, meta_oof))},'
    n = ('json.dump({"base_names": base_names, "use_cb": use_cb, "ridge_oof_rmse": float(rmse(y, meta_oof + anchor)),\n'
         '           "residual_anchor": ("likpf_mean_d" if _resid else None)},')
    assert t.count(o) == 1; t = t.replace(o, n)
    io.open(f, 'w', encoding='utf-8').write(t)
    print('patched', f)

# ---------------- 2) submission notebook FULL-STACK inference ----------------
P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == '209071f6')
s = ''.join(c['source'])
if 'residual_anchor' in s:
    print('inference already patched')
else:
    o = '''            meta_test = joblib.load(_ridge_pkl).predict(_cols)
            print(f"FULL-STACK inference: {[_n for _n in _bn if _n in _pred]} + ridge", flush=True)'''
    n = '''            meta_test = joblib.load(_ridge_pkl).predict(_cols)
            _ra = _meta.get("residual_anchor")
            if _ra:
                meta_test = meta_test + test_df[_ra].values.astype(float)   # add the residual anchor back
                print(f"residual anchor added back: {_ra}", flush=True)
            print(f"FULL-STACK inference: {[_n for _n in _bn if _n in _pred]} + ridge", flush=True)'''
    assert s.count(o) == 1, ('inference anchor', s.count(o))
    s = s.replace(o, n)
    ast.parse(s)
    c['source'] = s.splitlines(keepends=True)
    json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('patched submission inference (anchor add-back)')

# ---------------- 3) train_quick driver ----------------
f = 'build_quick_notebook.py'
t = io.open(f, encoding='utf-8').read()
if 'ROGII_RESID' in t:
    print(f, 'already patched')
else:
    o = 'X = train_df[feats].values.astype(np.float32); y = train_df["target"].values.astype(np.float32); g = train_df["well"].values'
    n = (o + '''
_resid = os.environ.get("ROGII_RESID", "1") == "1"   # train on (target - likpf_mean_d); prints stay in delta space
anchor = np.nan_to_num(train_df["likpf_mean_d"].values.astype(np.float32)) if _resid else np.zeros(len(train_df), np.float32)
y_fit = y - anchor
if _resid:
    print(f"[resid] target re-anchored to likpf_mean_d (std {y.std():.2f} -> {y_fit.std():.2f})", flush=True)''')
    assert t.count(o) == 1; t = t.replace(o, n)

    o = '    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], **_FIT_KW)'
    n = '    m.fit(X[tr], y_fit[tr], eval_set=[(X[va], y_fit[va])], **_FIT_KW)'
    assert t.count(o) == 1; t = t.replace(o, n)

    o = 'print(f"*** QUICK single-LGB OOF RMSE = {rmse(y, oof):.4f}  (avg best_iter={int(np.mean(iters))}) ***", flush=True)'
    n = 'print(f"*** QUICK single-LGB OOF RMSE = {rmse(y, oof + anchor):.4f}  (avg best_iter={int(np.mean(iters))}) ***", flush=True)'
    assert t.count(o) == 1; t = t.replace(o, n)

    o = 'cm.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True, early_stopping_rounds=200)'
    n = 'cm.fit(X[tr], y_fit[tr], eval_set=(X[va], y_fit[va]), use_best_model=True, early_stopping_rounds=200)'
    assert t.count(o) == 2, ('quick cb fits', t.count(o)); t = t.replace(o, n)

    o = 'print(f"*** QUICK CatBoost  OOF RMSE = {rmse(y, cb_oof):.4f}  (avg best_iter={int(np.mean(cb_it))}) ***", flush=True)'
    n = 'print(f"*** QUICK CatBoost  OOF RMSE = {rmse(y, cb_oof + anchor):.4f}  (avg best_iter={int(np.mean(cb_it))}) ***", flush=True)'
    assert t.count(o) == 1; t = t.replace(o, n)

    o = '''        for tr, va in cv.split(_O, y, groups=g):
            _r = Ridge(alpha=1.66, positive=True, fit_intercept=True); _r.fit(_O[tr], y[tr]); _meta[va] = _r.predict(_O[va])
        print(f"*** LGB+CB ridge-blend OOF   = {rmse(y, _meta):.4f}  (vs LGB {rmse(y, oof):.4f}, CB {rmse(y, cb_oof):.4f}) ***", flush=True)'''
    n = '''        for tr, va in cv.split(_O, y_fit, groups=g):
            _r = Ridge(alpha=1.66, positive=True, fit_intercept=True); _r.fit(_O[tr], y_fit[tr]); _meta[va] = _r.predict(_O[va])
        print(f"*** LGB+CB ridge-blend OOF   = {rmse(y, _meta + anchor):.4f}  (vs LGB {rmse(y, oof + anchor):.4f}, CB {rmse(y, cb_oof + anchor):.4f}) ***", flush=True)'''
    assert t.count(o) == 1; t = t.replace(o, n)
    io.open(f, 'w', encoding='utf-8').write(t)
    print('patched', f)
