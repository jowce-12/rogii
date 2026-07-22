"""Patch 13: retune lgb_configs (cell cc1a056a) per the 150-well sweep.
Old lgb0 (leaves255, l2=3) was UNDER-regularized (OOF 11.30); old lgb1/lgb2
(l2=95.75) were OVER-regularized (~11.14). Sweep optimum: moderate reg (l2~20-30,
ff=0.6, bf=0.7, mcs=40, lr=0.02). New configs keep diversity (deep vs shallow).
Features unchanged -> cache stays v5. Idempotent."""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'cc1a056a')
src = ''.join(c['source'])

if 'retuned per 150-well sweep' in src:
    print('Patch 13 already applied; aborting.')
    sys.exit(0)

old = '''def lgb_configs(dev):
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if dev == "gpu": base.update(device_type="gpu", gpu_use_dp=False)
    n = 600 if CFG.FAST else 5000
    return [
        dict(**base, num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05, learning_rate=0.03, n_estimators=n, seed=123),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=0),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.474, subsample_freq=1,
             colsample_bytree=0.393, reg_lambda=95.75, reg_alpha=10.79, min_child_weight=0.24,
             learning_rate=0.0093, n_estimators=min(2*n, 10000), random_state=29),
    ]'''

new = '''def lgb_configs(dev):
    base = dict(boosting_type="gbdt", objective="regression", verbose=-1, n_jobs=-1, max_bin=255)
    if dev == "gpu": base.update(device_type="gpu", gpu_use_dp=False)
    n = 600 if CFG.FAST else 5000
    # retuned per 150-well sweep: moderate reg (l2~20-30, ff=0.6, bf=0.7, mcs=40, lr=0.02)
    # beats old under-regularized deep (l2=3) and over-regularized shallow (l2=95.75).
    return [
        dict(**base, num_leaves=255, min_child_samples=40, subsample=0.7, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=20.0, reg_alpha=1.0, learning_rate=0.02, n_estimators=n, seed=123),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.7, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0, min_child_weight=0.24,
             learning_rate=0.02, n_estimators=min(2*n, 10000), random_state=0),
        dict(**base, num_leaves=64, min_child_samples=40, subsample=0.7, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=30.0, reg_alpha=1.0, min_child_weight=0.24,
             learning_rate=0.02, n_estimators=min(2*n, 10000), random_state=29),
    ]'''

assert src.count(old) == 1, ('anchor', src.count(old))
src = src.replace(old, new)
ast.parse(src)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('retuned lgb_configs (deep l2=20 / shallow l2=30, ff.6 bf.7 mcs40 lr.02)')
