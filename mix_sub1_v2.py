# sub_1 meta upgrades, all offline on saved OOFs:
#  A) all-10 ridge (old5+new5, positive ridge prunes)   B) + anchor columns
#  C) ridge alpha sweep on the best set
import glob, itertools
import numpy as np, pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

SLOTS = ['lightgbm-1','lightgbm-2','lightgbm-3','catboost-1','catboost-2']
def load_oof(root):
    return {s: np.asarray(joblib.load(sorted(glob.glob(f'{root}/models/{s}/*.pkl'))[0]).oof_preds, dtype=np.float64) for s in SLOTS}
old = load_oof('ravaghi'); new = load_oof('ravaghi_new')
try:
    meta = pd.read_parquet('ravaghi_train.parquet', columns=['well','target','tvt_dense_d','pf_ancc_delta','tvt_dense50_d'])
except Exception:
    meta = pd.read_csv('ravaghi/data/train.csv', usecols=['well','target','tvt_dense_d','pf_ancc_delta','tvt_dense50_d'], low_memory=False)
y = meta['target'].values.astype(np.float64); g = meta['well'].values
cv = GroupKFold(5)

def ridge_oof(cols, alpha=1.6602834637650032):
    X = np.column_stack(cols); oof = np.zeros(len(y))
    for tr_i, va_i in cv.split(X, y, groups=g):
        r = Ridge(random_state=42, alpha=alpha, tol=0.0005030247295617308, positive=True, fit_intercept=True)
        r.fit(X[tr_i], y[tr_i]); oof[va_i] = r.predict(X[va_i])
    return float(np.sqrt(np.mean((oof - y)**2)))

best5 = [old['lightgbm-1'], old['lightgbm-2'], old['lightgbm-3'], new['catboost-1'], new['catboost-2']]
all10 = [old[s] for s in SLOTS] + [new[s] for s in SLOTS]
anc = [np.nan_to_num(meta[c].values.astype(np.float64)) for c in ['tvt_dense_d','pf_ancc_delta','tvt_dense50_d']]
print('best5 (baseline)          :', round(ridge_oof(best5), 4), flush=True)
print('all10                     :', round(ridge_oof(all10), 4), flush=True)
print('best5 + tvt_dense_d       :', round(ridge_oof(best5 + [anc[0]]), 4), flush=True)
print('best5 + pf_ancc_delta     :', round(ridge_oof(best5 + [anc[1]]), 4), flush=True)
print('best5 + dense_d + dense50 :', round(ridge_oof(best5 + [anc[0], anc[2]]), 4), flush=True)
print('all10 + dense_d           :', round(ridge_oof(all10 + [anc[0]]), 4), flush=True)
print('all10 + dense_d + ancc    :', round(ridge_oof(all10 + [anc[0], anc[1]]), 4), flush=True)
print('all10 + all3 anchors      :', round(ridge_oof(all10 + anc), 4), flush=True)
for a in [0.5, 1.66, 5.0, 20.0, 100.0]:
    print(f'alpha={a:<6} all10+dense_d :', round(ridge_oof(all10 + [anc[0]], alpha=a), 4), flush=True)
