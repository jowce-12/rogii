"""Patch 2: add a GRU base learner to the fleongg stack (CELL 31).
1. Insert a cell defining train_gru_oof (module body) before CELL 31.
2. Wire it into train_stack so 'gru0' joins the Ridge meta-stack.
Idempotent.
"""
import json, io, sys

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))

joined = '\n'.join(''.join(c['source']) for c in nb['cells'])
if 'train_gru_oof' in joined:
    print('GRU patch already present; aborting.')
    sys.exit(0)

# --- module body (everything before the standalone smoke test) ---
mod = io.open('gru_stack.py', encoding='utf-8').read()
body = mod.split('# ----------------- standalone smoke test')[0].rstrip() + '\n'
# the notebook already imports numpy as np / os at the top; keep the `import os, numpy as np`
# line harmless (re-import is fine).

# 1) insert after CELL 30 (id b515d5c6) -> appears right before CELL 31
idx = next(i for i, c in enumerate(nb['cells']) if c.get('id') == 'b515d5c6')
cell = {'cell_type': 'code', 'id': 'gru00001', 'metadata': {},
        'execution_count': None, 'outputs': [], 'source': body.splitlines(keepends=True)}
nb['cells'].insert(idx + 1, cell)
print('inserted GRU module cell after index', idx)

# 2) wire into train_stack (CELL 31, id cc1a056a)
c31 = next(c for c in nb['cells'] if c.get('id') == 'cc1a056a')
src = ''.join(c31['source'])
anchor = "    OOF = pd.DataFrame(oof_cols); TEST = pd.DataFrame(test_cols)"
assert src.count(anchor) == 1, ('anchor count', src.count(anchor))
inject = (
    "    # GRU base learner (sequence model over the eval zone); joins the meta-stack.\n"
    "    # No-op unless torch + CUDA are present (ROGII_GRU=add by default).\n"
    "    gru_out = train_gru_oof(train_df, test_df, features, cv)\n"
    "    if gru_out is not None:\n"
    "        oof_cols['gru0'], test_cols['gru0'] = gru_out\n"
    "        print(f\"  gru0: OOF RMSE={rmse(y, oof_cols['gru0']):.4f}\")\n"
    + anchor
)
src = src.replace(anchor, inject)
c31['source'] = src.splitlines(keepends=True)
print('wired GRU into train_stack')

json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', P)
