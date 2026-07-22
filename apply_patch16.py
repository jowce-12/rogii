"""Patch 16: diversify the two dead stack slots (ridge coefs showed cb0=0.000,
lgb2=0.044 — both near-clones of cb1/lgb1). cb0 -> deep CatBoost (depth 9,
Bernoulli subsample). lgb2 -> shallow-extreme LGB (leaves 31, ff 0.4, stronger
reg). Also allow_writing_files=False on both CBs (no catboost_info dir).
Features unchanged -> cache stays v5. Idempotent."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'cc1a056a')
s = ''.join(c['source'])

if 'diversified per ridge-coef analysis' in s:
    print('Patch 16 already applied; aborting.')
    sys.exit(0)

# --- lgb2: seed-clone of lgb1 -> shallow-extreme diverse ---
old_lgb2 = '''        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.7, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0,
             learning_rate=0.02, n_estimators=min(2*n, 10000), random_state=29),
    ]'''
new_lgb2 = '''        # lgb2 diversified per ridge-coef analysis (was a seed-clone of lgb1, coef~0.04):
        # shallow-extreme + strong feature subsampling -> decorrelated errors for the meta.
        dict(**base, num_leaves=31, min_child_samples=60, subsample=0.6, subsample_freq=1,
             colsample_bytree=0.4, reg_lambda=60.0, reg_alpha=2.0,
             learning_rate=0.02, n_estimators=min(2*n, 10000), random_state=29),
    ]'''
assert s.count(old_lgb2) == 1, ('lgb2 anchor', s.count(old_lgb2))
s = s.replace(old_lgb2, new_lgb2)

# --- cb0: clone of cb1 -> deep diverse; both CBs stop writing catboost_info ---
old_cb = '''    return [
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),
    ]'''
new_cb = '''    return [
        # cb0 diversified (was a near-clone of cb1, ridge coef 0.000): deeper trees +
        # Bernoulli subsampling -> a genuinely different error profile.
        dict(iterations=n, depth=9, l2_leaf_reg=5.0, min_data_in_leaf=30, border_count=254,
             bootstrap_type="Bernoulli", subsample=0.7, allow_writing_files=False,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.02, random_seed=7),
        dict(iterations=n, depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
             allow_writing_files=False,
             loss_function="RMSE", task_type=tt, od_type="Iter", od_wait=300, verbose=0, learning_rate=0.03, random_seed=123),
    ]'''
assert s.count(old_cb) == 1, ('cb anchor', s.count(old_cb))
s = s.replace(old_cb, new_cb)

ast.parse(s)
c['source'] = s.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('diversified cb0 (depth9/Bernoulli) + lgb2 (leaves31/ff0.4); catboost_info disabled')
