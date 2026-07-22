# Export all sub_1 OOF columns (old5 + new5 + any candidates) to a portable parquet
import glob
import numpy as np, pandas as pd, joblib
cols = {}
for root, tag in [('ravaghi', 'old'), ('ravaghi_new', 'new')]:
    for p in sorted(glob.glob(f'{root}/models/*/')):
        slot = p.rstrip('/').split('/')[-1]
        pk = sorted(glob.glob(p + '*.pkl'))
        if not pk:
            continue
        tr = joblib.load(pk[0])
        cols[f'{tag}_{slot}'] = np.asarray(tr.oof_preds, np.float32)
try:
    meta = pd.read_parquet('ravaghi_train.parquet', columns=['well','id','last_known_tvt','target'])
except Exception:
    meta = pd.read_csv('ravaghi/data/train.csv', usecols=['well','id','last_known_tvt','target'], low_memory=False)
df = pd.DataFrame({'well': meta['well'], 'id': meta['id'],
                   'last_known_tvt': meta['last_known_tvt'].astype(np.float32),
                   'target': meta['target'].astype(np.float32)})
for k, v in cols.items():
    df[k] = v
df.to_parquet('sub1_oof.parquet', index=False)
print('exported', list(cols.keys()), len(df), 'rows')
