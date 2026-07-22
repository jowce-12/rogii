"""Patch 11: add nogil=True to the fleongg (c3ab97ce) numba kernels that run
per-well under Parallel(prefer='threads') so they release the GIL and use all
cores. _beam_jit is deterministic (no RNG). _pf_ancc/_pf_z use np.random, so
under threads their random streams race -> feature values become slightly
non-reproducible during the BUILD (same pattern the lik-PF already uses; fine
for a stochastic PF, and the cache freezes one draw). Also apply to _resamp
(called inside the PF kernels) for consistency. Idempotent.
"""
import json, io, sys, ast

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))
c = next(x for x in nb['cells'] if x.get('id') == 'c3ab97ce')
src = ''.join(c['source'])

targets = ['_pf_ancc', '_pf_z', '_beam_jit', '_resamp']
changed = 0
for fn in targets:
    old = f'@njit(cache=True)\ndef {fn}('
    new = f'@njit(cache=True, nogil=True)\ndef {fn}('
    if new in src:
        continue
    assert src.count(old) == 1, (fn, src.count(old))
    src = src.replace(old, new)
    changed += 1

ast.parse(src)
c['source'] = src.splitlines(keepends=True)
json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'added nogil to {changed} kernels: {targets}')
