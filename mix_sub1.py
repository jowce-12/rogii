# Offline best-of-both composition: old(ravaghi) vs new(retrained) per slot, 32 combos,
# ridge OOF each (no retraining — Trainer pkls carry their oof_preds).
import glob, itertools, json
import numpy as np, pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

def load_oof(root):
    out = {}
    for slot in ['lightgbm-1','lightgbm-2','lightgbm-3','catboost-1','catboost-2']:
        p = sorted(glob.glob(f'{root}/models/{slot}/*.pkl'))
        tr = joblib.load(p[0])
        out[slot] = np.asarray(tr.oof_preds, dtype=np.float64)
    return out

old = load_oof('ravaghi')
new = load_oof('ravaghi_new')
try:
    meta = pd.read_parquet('ravaghi_train.parquet', columns=['well','target'])
except Exception:
    meta = pd.read_csv('ravaghi/data/train.csv', usecols=['well','target'], low_memory=False)
y = meta['target'].values.astype(np.float64); g = meta['well'].values
cv = GroupKFold(5)

def ridge_oof(cols):
    X = np.column_stack(cols)
    oof = np.zeros(len(y))
    for tr_i, va_i in cv.split(X, y, groups=g):
        r = Ridge(random_state=42, alpha=1.6602834637650032, tol=0.0005030247295617308,
                  positive=True, fit_intercept=True)
        r.fit(X[tr_i], y[tr_i]); oof[va_i] = r.predict(X[va_i])
    return float(np.sqrt(np.mean((oof - y)**2)))

slots = ['lightgbm-1','lightgbm-2','lightgbm-3','catboost-1','catboost-2']
res = []
for combo in itertools.product(['old','new'], repeat=5):
    cols = [ (old if c=='old' else new)[s] for c, s in zip(combo, slots) ]
    v = ridge_oof(cols)
    res.append((v, combo))
    print(f'{v:.4f}  ' + ' '.join(f'{s.split(chr(45))[0][:3]}{i+1}:{c}' for i,(c,s) in enumerate(zip(combo,slots))), flush=True)
res.sort()
print('BEST:', res[0], flush=True)
print('all-old:', [r for r in res if r[1]==('old',)*5][0][0], '| all-new:', [r for r in res if r[1]==('new',)*5][0][0], flush=True)
