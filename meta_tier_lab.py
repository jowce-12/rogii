# Should the fleongg META (the ridge that combines the 5 bases) differ by risk tier?
# Blend-level tiering works because the POLES have different relative strengths per tier
# (measured: GRU-vs-fleongg gap 0.635 normal -> 0.040 monster). The meta combines five
# models of the SAME kind on the SAME features, so first check whether their relative
# ordering even moves across tiers; then test five tiering forms.
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

BASES = ["lgb0","lgb1","lgb2","cb0","cb1"]
oof = pd.read_parquet("oof_stack.parquet")
ctx = pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                      columns=["id","likpf_mean_d","likpf_ptstd"])
df = oof.merge(ctx, on="id")
y = df["target"].values.astype(np.float64); yf = np.clip(y,-90,90)
g = df["well"].values
risk = df.groupby("well")["likpf_ptstd"].transform("mean").values
MON = np.nan_to_num(risk) >= 3.39
B = np.column_stack([df[c].values.astype(np.float64) for c in BASES])
anchor = np.nan_to_num(df["likpf_mean_d"].values.astype(np.float64))
X0 = np.column_stack([B, anchor])
folds = list(GroupKFold(5).split(X0, yf, groups=g))
print(f"rows {len(df)} | monster rows {MON.mean():.1%}\n")

print("== do the bases even re-order across tiers? (RMSE per group) ==")
hdr = f"{'group':9s}" + "".join(f"{b:>9s}" for b in BASES) + f"{'anchor':>9s}"
print(hdr)
for mon,lab in ((False,"normal"),(True,"monster")):
    m = MON==mon
    vals = [float(np.sqrt(np.mean((B[m,i]-y[m])**2))) for i in range(len(BASES))]
    va = float(np.sqrt(np.mean((anchor[m]-y[m])**2)))
    print(f"{lab:9s}" + "".join(f"{v:9.3f}" for v in vals) + f"{va:9.3f}")
for mon,lab in ((False,"normal"),(True,"monster")):
    m = MON==mon
    vals = np.array([float(np.sqrt(np.mean((B[m,i]-y[m])**2))) for i in range(len(BASES))])
    print(f"{lab:9s} rank {list(np.argsort(vals))} | spread(best-worst) {vals.max()-vals.min():.3f}")

def score(oof_pred, tag):
    r = float(np.sqrt(np.mean((oof_pred-y)**2)))
    print(f"{tag:34s} fleongg OOF = {r:.4f}")
    return r

# M0 global ridge (deployed)
o = np.zeros(len(df))
for tr,va in folds:
    r=Ridge(alpha=1.66,positive=True,fit_intercept=True); r.fit(X0[tr],yf[tr]); o[va]=r.predict(X0[va])
m0 = score(o, "M0 global ridge (deployed)")

# T1 hard split at 3.39
o = np.zeros(len(df))
for tr,va in folds:
    for s in (False,True):
        t=tr[MON[tr]==s]; v=va[MON[va]==s]
        if len(v)==0: continue
        r=Ridge(alpha=1.66,positive=True,fit_intercept=True); r.fit(X0[t],yf[t]); o[v]=r.predict(X0[v])
score(o, "T1 hard tier split")

# T2 hard split, per-tier alpha search
o = np.zeros(len(df))
for tr,va in folds:
    for s in (False,True):
        t=tr[MON[tr]==s]; v=va[MON[va]==s]
        if len(v)==0: continue
        best=(1e18,None)
        for a in (0.5,1.66,10.0,100.0):
            rr=Ridge(alpha=a,positive=True,fit_intercept=True); rr.fit(X0[t],yf[t])
            e=float(np.mean((rr.predict(X0[t])-yf[t])**2))
            if e<best[0]: best=(e,rr)
        o[v]=best[1].predict(X0[v])
score(o, "T2 hard split + per-tier alpha")

# T3 risk quartiles
q = pd.qcut(pd.Series(np.nan_to_num(risk)).rank(method="first"), 4, labels=False).values
o = np.zeros(len(df))
for tr,va in folds:
    for s in range(4):
        t=tr[q[tr]==s]; v=va[q[va]==s]
        if len(v)==0: continue
        r=Ridge(alpha=1.66,positive=True,fit_intercept=True); r.fit(X0[t],yf[t]); o[v]=r.predict(X0[v])
score(o, "T3 risk quartiles (4 ridges)")

# T4 continuous: risk-interacted columns in ONE ridge (positive constraint dropped for
# the interaction block, so a separate unconstrained fit is used)
rz = (np.nan_to_num(risk) - np.nanmean(risk)) / (np.nanstd(risk)+1e-9)
X4 = np.column_stack([X0, B*rz[:,None]])
o = np.zeros(len(df))
for tr,va in folds:
    r=Ridge(alpha=1.66,fit_intercept=True); r.fit(X4[tr],yf[tr]); o[va]=r.predict(X4[va])
score(o, "T4 single ridge + risk interactions")

# T5 global ridge then per-tier linear residual correction
o = np.zeros(len(df))
for tr,va in folds:
    r=Ridge(alpha=1.66,positive=True,fit_intercept=True); r.fit(X0[tr],yf[tr])
    base_tr, base_va = r.predict(X0[tr]), r.predict(X0[va])
    o[va] = base_va
    for s in (False,True):
        t=tr[MON[tr]==s]; v=va[MON[va]==s]
        if len(v)==0 or len(t)<1000: continue
        c=Ridge(alpha=10.0,fit_intercept=True); c.fit(base_tr[MON[tr]==s].reshape(-1,1), yf[t])
        o[v]=c.predict(base_va[MON[va]==s].reshape(-1,1))
score(o, "T5 global + per-tier recalibration")
print(f"\nreference M0 = {m0:.4f} (lower is better)")
