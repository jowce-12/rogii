"""Patch the competition notebook with the validated self-augmented PF blend.
1. Insert a new code cell (build_aug_typewell helper) right after CELL 4.
2. Inject the augmented-PF selector blend into CELL 17.
Idempotent: refuses to double-apply.
"""
import json, io, sys

P = 'public-score-rogii-lb-7-159.ipynb'
nb = json.load(io.open(P, encoding='utf-8'))

HELPER = '''# === Self-augmented typewell (PPT slide-9: horizontal pre-PS GR has higher
# resolution & better calibration than the typewell). Affine-map the typewell GR
# into the horizontal well's GR units (fit on the known prefix), then overlay the
# horizontal prefix self-log in the TVT band it covers. Returned typewell-shaped so
# the existing lik-PF / selector consume it unchanged. Validated on 773-well local
# CV: alone it is worse but decorrelated; blended at w_aug~0.30 it cuts the selector
# pooled-RMSE ~0.25-0.35 ft on two independent well samples. ===
def _affine_cal_gr(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, (float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0)
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    if (not np.isfinite(a)) or a < 0.2 or a > 5.0:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k))
    return float(a), float(b)

def build_aug_typewell(hw, tw, self_weight=0.6, step=0.2):
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]
    ktvt = kn['TVT_input'].values.astype(float)
    kgr = kn['GR'].interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr))).values.astype(float)
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    a, b = _affine_cal_gr(kgr, tw_at_k)
    tw_gr_h = a * tw_gr + b
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    gg = np.interp(tvt_g, tw_tvt, tw_gr_h)
    klo, khi = float(np.nanmin(ktvt)), float(np.nanmax(ktvt))
    if khi - klo > 5.0 and len(ktvt) >= 30:
        order = np.argsort(ktvt)
        self_gr = np.interp(tvt_g, ktvt[order], kgr[order], left=np.nan, right=np.nan)
        band = np.isfinite(self_gr)
        gg[band] = (1.0 - self_weight) * gg[band] + self_weight * self_gr[band]
    return pd.DataFrame({'TVT': tvt_g, 'GR': gg.astype(float), 'Geology': np.nan})
'''

# guard against double application
joined_all = '\n'.join(''.join(c['source']) for c in nb['cells'])
if 'build_aug_typewell' in joined_all:
    print('Patch already present; aborting to avoid duplicates.')
    sys.exit(0)

# 1) insert helper cell after CELL 4 (id ec5d18d6)
idx4 = next(i for i, c in enumerate(nb['cells']) if c.get('id') == 'ec5d18d6')
new_cell = {
    'cell_type': 'code',
    'id': 'augtw0001',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': HELPER.splitlines(keepends=True),
}
nb['cells'].insert(idx4 + 1, new_cell)
print('inserted helper cell after index', idx4)

# 2) inject blend into CELL 17 (id f9218767)
c17 = next(c for c in nb['cells'] if c.get('id') == 'f9218767')
src = ''.join(c17['source'])
anchor = "    tvt_selector = apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt)"
assert src.count(anchor) == 1, 'anchor not unique'
blend = anchor + '''

    # --- self-augmented PF blend (PPT slide-9 idea; +CV on the physics tracker) ---
    try:
        tw_aug = build_aug_typewell(hw_te, tw_ref)
        pf_by_scale_aug = run_pf_lik_ensemble_scales(hw_te, tw_aug, n_particles=500, n_seeds=128)
        tvt_selector_aug = apply_selector_variant(selector_variant, pf_by_scale_aug, tvt_beam, last_known_tvt)
        W_AUG = 0.30
        if len(tvt_selector_aug) == len(tvt_selector) and np.all(np.isfinite(tvt_selector_aug)):
            tvt_selector = (1.0 - W_AUG) * tvt_selector + W_AUG * tvt_selector_aug
            print(f'  Self-augmented PF blended (w_aug={W_AUG})')
        else:
            print('  Self-augmented PF skipped: shape/finite mismatch')
    except Exception as e:
        print(f'  Self-augmented PF skipped: {e}')'''
src = src.replace(anchor, blend)
c17['source'] = src.splitlines(keepends=True)
print('injected blend into CELL 17')

json.dump(nb, io.open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', P)
