# Re-judge the whole deployed ladder on HONEST (pinned-fold) GRU poles.
# LB fact from the user: raising the GRU weight helped -> the direction is confirmed;
# this fixes WHERE the optimum sits, since the previous grids used a leaked pole.
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import blend_eval as BE
from offline_tests import pooled

fs = pd.read_parquet("oof_stack.parquet").merge(
    pd.read_parquet("train_features_v7_f1_cblend_w773.parquet",
                    columns=["id","likpf_mean_d","likpf_ptstd"]), on="id")
fy=fs["target"].values.astype(np.float64); fyf=np.clip(fy,-90,90)
FX=np.column_stack([fs[c].values for c in ["lgb0","lgb1","lgb2","cb0","cb1"]]
                   +[np.nan_to_num(fs["likpf_mean_d"].values.astype(np.float64))])
meta=np.zeros(len(fs))
for tr,va in GroupKFold(5).split(FX,fyf,groups=fs["well"].values):
    r=Ridge(alpha=1.66,positive=True,fit_intercept=True); r.fit(FX[tr],fyf[tr]); meta[va]=r.predict(FX[va])
fl_s=pd.Series(meta,index=fs["id"].values)
risk_s=pd.Series(np.nan_to_num(fs["likpf_ptstd"].values.astype(np.float64)),index=fs["id"].values)
sub1_fn=BE.make_ridge_fn([f"old_{s}" for s in ["lightgbm-1","lightgbm-2","lightgbm-3","catboost-1","catboost-2"]])
POLES={"dip3": pd.Series(pd.read_parquet("gru_oof_dip3_honest.parquet").set_index("id")["gru_d"]),
       "dip5": pd.Series(pd.read_parquet("gru_oof_dip5_honest.parquet").set_index("id")["gru_d"])}
s3=pd.Series(pd.read_parquet("s3_preds_tuned.parquet").set_index("id")["s3_tvt"])
y_s=pd.Series(fy,index=fs["id"].values)
for nm,p in POLES.items():
    ids=p.index.intersection(y_s.index); e=p.reindex(ids).values-y_s.reindex(ids).values
    fe=fl_s.reindex(ids).values-y_s.reindex(ids).values; m=np.isfinite(e)&np.isfinite(fe)
    print(f"{nm} honest: clean OOF {np.sqrt(np.mean(e[m]**2)):.4f} | corr vs fleongg {np.corrcoef(e[m],fe[m])[0,1]:.4f}", flush=True)

DATA={}
for SEED in (7,11):
    res,SEL=BE.selector_preds(SEED)
    sub=BE.OOF[BE.OOF["well"].isin([r_["wid"] for r_ in res])].copy()
    sub["sub1_tvt"]=sub1_fn(sub); sub["fl_tvt"]=sub["last_known_tvt"].values+fl_s.reindex(sub["id"].values).values
    sub["s3_tvt"]=s3.reindex(sub["id"].values).values; sub["risk"]=risk_s.reindex(sub["id"].values).values
    DATA[SEED]=(res,SEL,sub)

def ev(seed, pole, wn, wm, ws3, gam=1.09):
    res,SEL,sub=DATA[seed]
    sub=sub.copy(); sub["gr_tvt"]=sub["last_known_tvt"].values+POLES[pole].reindex(sub["id"].values).values
    by={c:{w:g[c].values for w,g in sub.groupby("well")} for c in
        ("sub1_tvt","fl_tvt","gr_tvt","s3_tvt","last_known_tvt","risk")}
    finals=[]
    for r_,sel in zip(res,SEL):
        w=r_["wid"]; s1,fl,gr,sv=(by["sub1_tvt"].get(w),by["fl_tvt"].get(w),by["gr_tvt"].get(w),by["s3_tvt"].get(w))
        if any(v is None or len(v)!=len(sel) or np.isnan(np.asarray(v,float)).any() for v in (s1,fl,gr)):
            finals.append(np.asarray(sel,float)); continue
        risk=float(np.mean(by["risk"][w])); last=float(by["last_known_tvt"][w][0]); mon=np.isfinite(risk) and risk>=3.39
        wsp,wfl,wgr = wm if mon else wn
        b=wsp*(0.3*s1+0.7*sel)+wfl*fl+wgr*gr
        if ws3>0 and sv is not None and len(sv)==len(sel) and np.isfinite(sv).all(): b=(1-ws3)*b+ws3*sv
        if mon: b=last+gam*(b-last)
        finals.append(b)
    return pooled(res,finals)

N0=(0.20,0.50,0.30); M0=(0.20,0.50,0.30)
Nnow=(0.15,0.45,0.40); Mnow=(0.20,0.40,0.40)
print("\n== ladder (honest poles) ==", flush=True)
for tag,pole,wn,wm,ws3 in (("6663 config      ","dip3",N0,M0,0.0),
                           ("+dip5 pole       ","dip5",N0,M0,0.0),
                           ("+v3 pole 0.10    ","dip5",N0,M0,0.10),
                           ("+patch49 monster ","dip5",N0,Mnow,0.10),
                           ("+patch50 normal  ","dip5",Nnow,Mnow,0.10),
                           ("  (same, dip3)   ","dip3",Nnow,Mnow,0.10)):
    print(f"{tag}: s7 {ev(7,pole,wn,wm,ws3):.4f} | s11 {ev(11,pole,wn,wm,ws3):.4f}", flush=True)
print("\n== normal-tier grid (dip5, monster 0.20/0.40/0.40, ws3 0.10) ==", flush=True)
for wn in ((0.20,0.50,0.30),(0.15,0.50,0.35),(0.15,0.45,0.40),(0.10,0.45,0.45),(0.10,0.40,0.50)):
    print(f"  {wn}: s7 {ev(7,'dip5',wn,Mnow,0.10):.4f} | s11 {ev(11,'dip5',wn,Mnow,0.10):.4f}", flush=True)
print("\n== monster-tier grid (normal 0.15/0.45/0.40) ==", flush=True)
for wm in ((0.20,0.50,0.30),(0.20,0.45,0.35),(0.20,0.40,0.40),(0.15,0.40,0.45),(0.20,0.35,0.45)):
    print(f"  {wm}: s7 {ev(7,'dip5',Nnow,wm,0.10):.4f} | s11 {ev(11,'dip5',Nnow,wm,0.10):.4f}", flush=True)
print("\n== v3 pole weight (dip5, current tiers) ==", flush=True)
for ws3 in (0.0,0.05,0.10,0.15,0.20):
    print(f"  ws3={ws3:.2f}: s7 {ev(7,'dip5',Nnow,Mnow,ws3):.4f} | s11 {ev(11,'dip5',Nnow,Mnow,ws3):.4f}", flush=True)
print("\n== gamma (dip5, current tiers, ws3 0.10) ==", flush=True)
for g in (1.00,1.05,1.09,1.13):
    print(f"  gamma={g:.2f}: s7 {ev(7,'dip5',Nnow,Mnow,0.10,g):.4f} | s11 {ev(11,'dip5',Nnow,Mnow,0.10,g):.4f}", flush=True)
print("DONE", flush=True)
