"""Port the current fleongg pipeline into the public-best notebook
(rogii-geology-aware-ensembling-lb-7-129.ipynb) so the new train_stack model
dataset (lgb*.pkl + cb*.cbm + ridge.pkl + stack_meta.json + features.json,
trained with derived-feats + gr fill + A6 + S1 blend + A5) can be attached.

Replaced (Track B / fleongg only, matched by position + content signature):
  [32] imports/CFG            <- source 44e50dd0 (NJOBS, guards, _find_data)
  [35] PF kernels             <- source c3ab97ce (A6 beam resample, GRFILL gate, _grcal_tw)
  [36] lik-PF                 <- source 83876170 (S1 blend mode)
  [38] spatial imputers       <- source c9bba3da
  [39] build_well pipeline    <- source b515d5c6 (A5 helpers + 50 derived feats)
  [40] _device/configs        <- source cc1a056a (gpu cascade; configs only used from-scratch)
  [42] _find_models/main      <- source 209071f6 (FULL-STACK inference + anchor + grcal warn + A5 test join)
Modified in place:
  [ 2] CELL 0: + ROGII_GRCAL=blend (must match training)
  [41] PP kept AS-IS (7.129 tuning: alpha .985 / tau 120 / w_pf .05) + A8 clip line only
Preserved untouched (the 7.129 score-makers): scale-average selector, pp_params
0.985/120/0.05, deg-3 projection, sp45 0.60, gold balanced.
"""
import json, io, sys, ast

SRC = 'public-score-rogii-lb-7-159.ipynb'
TGT = 'rogii-geology-aware-ensembling-lb-7-129.ipynb'
src_nb = json.load(io.open(SRC, encoding='utf-8'))
tgt_nb = json.load(io.open(TGT, encoding='utf-8'))

def src_cell(cid):
    return ''.join(next(c for c in src_nb['cells'] if c.get('id') == cid)['source'])

# (target index, signature the target cell must start with, source cell id)
PLAN = [
    (32, 'import os, sys, glob, time, warnings, multiprocessing', '44e50dd0'),
    (35, '# ---- single particle filters', 'c3ab97ce'),
    (36, '# ---- 128-seed likelihood-weighted particle filter', '83876170'),
    (38, 'PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20', 'c9bba3da'),
    (39, 'def build_well(hw_path, tw_path, is_train', 'b515d5c6'),
    (40, 'def _device():', 'cc1a056a'),
    (42, 'def _find_models():', '209071f6'),
]

if any('add_alias_metafeats' in ''.join(c['source']) for c in tgt_nb['cells']):
    print('already ported; aborting'); sys.exit(0)

for idx, sig, cid in PLAN:
    cell = tgt_nb['cells'][idx]
    cur = ''.join(cell['source'])
    assert cell['cell_type'] == 'code' and cur.strip().startswith(sig), (idx, sig, cur[:60])
    new = src_cell(cid)
    ast.parse(new)
    cell['source'] = new.splitlines(keepends=True)
    print(f'[{idx:2d}] replaced with source {cid} ({len(cur)} -> {len(new)} chars)')

# CELL 0: add grcal env (must match training)
c0 = tgt_nb['cells'][2]
s0 = ''.join(c0['source'])
assert 'ROGII_GOLD_PROFILE' in s0
if 'ROGII_GRCAL' not in s0:
    s0 = s0.rstrip() + '\nos.environ["ROGII_GRCAL"] = "blend"   # must match train_stack (S1 blend)\n'
    ast.parse(s0)
    c0['source'] = s0.splitlines(keepends=True)
    print('[ 2] CELL 0: ROGII_GRCAL=blend added')

# PP cell [41]: keep 7.129 tuning, add A8 clip only
c41 = tgt_nb['cells'][41]
s41 = ''.join(c41['source'])
assert 'alpha = 0.985' in s41, 'PP tuning signature missing'
o = '''    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    pred = last + delta'''
if 'A8' not in s41:
    assert s41.count(o) == 1, ('PP clip anchor', s41.count(o))
    s41 = s41.replace(o, '''    delta = PP.w_sub1*sub1 + (1-PP.w_sub1)*lp
    delta = np.clip(delta, -110.0, 110.0)   # A8: train max|target|=98.9ft -> free runaway guard
    pred = last + delta''')
    ast.parse(s41)
    c41['source'] = s41.splitlines(keepends=True)
    print('[41] PP kept (0.985/120/0.05) + A8 clip added')

json.dump(tgt_nb, io.open(TGT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved', TGT)

# ---- verification ----
tgt = json.load(io.open(TGT, encoding='utf-8'))
a = '\n'.join(''.join(c['source']) for c in tgt['cells'])
checks = [
    ('FULL-STACK 추론', 'FULL-STACK inference'), ('잔차앵커 게이트', 'residual_anchor'),
    ('grcal mismatch 경고', 'grcal mismatch'), ('A5 join(test)', 'add_alias_metafeats(test_df'),
    ('A5 헬퍼', 'def add_alias_metafeats'), ('파생변수', 'def add_derived_features'),
    ('S1 blend', '"blend"'), ('A6 beam', 'A6: resample typewell'), ('CELL0 blend', 'os.environ["ROGII_GRCAL"] = "blend"'),
    ('[보존] selector 평균', 'USING AVERAGE OF SCALES'), ('[보존] PP 0.985', 'alpha = 0.985'),
    ('[보존] deg3 projection', '_robfit(_s, (_tvt + _Z) - _anchor, 3)'),
    ('[보존] sp45 0.60', '_SELECTED_SP45_WEIGHT = 0.60'),
    ('[보존] gold balanced', "submission_gold_prefix_balanced.csv'"),
]
for name, m in checks:
    print(('OK  ' if m in a else 'FAIL') + ' ' + name)
for c in tgt['cells']:
    if c['cell_type'] == 'code' and ''.join(c['source']).strip():
        ast.parse(''.join(c['source']))
print('all code cells parse OK')
