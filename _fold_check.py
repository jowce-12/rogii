import json, sys
import pandas as pd, sklearn
from sklearn.model_selection import GroupKFold
seq = pd.read_parquet("gru_seq.parquet", columns=["well"])
wells = sorted(seq["well"].unique())
fold_of = {}
for f,(_,va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
    for i in va: fold_of[wells[i]] = f
print(f"sklearn {sklearn.__version__} python {sys.version.split()[0]}")
print("first10:", [fold_of[w] for w in wells[:10]])
json.dump(fold_of, open(sys.argv[1], "w"))
