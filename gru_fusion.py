# Quadratic path fusion (radiantallomancer step we skipped: 3-GRU+fusion 8.159->7.943).
#   min ||p - p_gru||^2 + lam * sum_i ((p[i+1]-p[i]) - dip_i*STEP)^2
# Tridiagonal solve per well (Thomas algorithm); dip is a SOFT constraint (no integration
# -> no drift accumulation, unlike the dead WARP). Dip sources (pre-registered, from
# existing channels, no retraining): stride_d path gradient / pf5_d path gradient.
# Outputs fused clean-OOF parquets per config + pooled prints.
# RUN (isic env): python gru_fusion.py            (~8min GPU)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import _gru_infer as GI

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAMS = (1024.0,)
DIPS = ("stride",)
SETS = {"3leg": ("_pa", "_pb", "_pc"), "6leg": ("_pa", "_pb", "_pc", "_pd", "_pe", "_pf")}


def fuse(p, dip, lam, step=GI.G_STEP):
    """Solve (I + lam*D^T D) x = p + lam*D^T(dip*step). Tridiagonal Thomas."""
    n = len(p)
    if n < 3:
        return p.copy()
    v = dip * step                      # target per-step increments, len n-1
    lower = np.full(n - 1, -lam)
    upper = np.full(n - 1, -lam)
    diag = np.full(n, 1.0 + 2.0 * lam)
    diag[0] = 1.0 + lam
    diag[-1] = 1.0 + lam
    rhs = p.copy()
    rhs[0] += lam * (-v[0])
    rhs[1:-1] += lam * (v[:-1] - v[1:])
    rhs[-1] += lam * v[-1]
    # Thomas
    c = upper.copy(); d = rhs.copy(); b = diag.copy()
    for i in range(1, n):
        w = lower[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _unit_test():
    rng = np.random.default_rng(0)
    p = np.cumsum(rng.normal(0, 0.3, 200))
    dip_exact = np.diff(p) / GI.G_STEP
    f = fuse(p, dip_exact, 50.0)
    assert float(np.abs(f - p).max()) < 1e-8, "fusion identity check failed"
    print("[unit] fusion identity OK (exact-dip -> unchanged path)")


def main():
    _unit_test()
    seq = pd.read_parquet("gru_seq.parquet")
    rowmap = pd.read_parquet("gru_rowmap.parquet")
    CHANS = [c for c in seq.columns if c not in ("well", "cut", "step", "is_tail", "target")]
    wells = sorted(seq["well"].unique())
    fold_of = {}
    for f, (_, va) in enumerate(GroupKFold(5).split(wells, groups=wells)):
        for i in va:
            fold_of[wells[i]] = f
    nets = {}
    for tag in SETS["6leg"]:
        for f in range(5):
            ck = torch.load(f"gru_fold{f}{tag}.pt", map_location="cpu")
            chans, hid = ck["chans"], ck["hid"]

            class _Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inp = nn.Linear(len(chans), hid)
                    self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True,
                                      bidirectional=True, dropout=0.25)
                    self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))

                def forward(self, x):
                    h, _ = self.gru(self.inp(x))
                    return self.head(h).squeeze(-1)

            m = _Net(); m.load_state_dict(ck["state"]); m.eval().to(DEV)
            nets[(tag, f)] = (m, chans)

    # pass 1: cache per-well grid preds per leg + dip fields + row mapping
    cache = {}
    nat = seq[seq["cut"] == "nat"]
    for k, (wid, g) in enumerate(nat.groupby("well"), 1):
        g = g.sort_values("step")
        f = fold_of[wid]
        tail = g["is_tail"].values.astype(bool)
        preds = {}
        with torch.no_grad():
            for tag in SETS["6leg"]:
                m, chans = nets[(tag, f)]
                x = torch.from_numpy(g[chans].values.astype(np.float32)[None]).to(DEV)
                preds[tag] = m(x).cpu().numpy()[0] * 10.0
        dips = {"stride": np.gradient(g["stride_d"].values.astype(float) * 10.0,
                                      GI.G_STEP * np.arange(len(g)) + 1e-9),
                "pf": np.gradient(g["pf5_d"].values.astype(float) * 10.0,
                                  GI.G_STEP * np.arange(len(g)) + 1e-9)}
        rm = rowmap[rowmap["well"] == wid]
        md_rel_grid = (np.arange(len(g), dtype=np.float64) - GI.G_CTX) * GI.G_STEP
        cache[wid] = (preds, dips, tail, rm, md_rel_grid)
        if k % 200 == 0:
            print(f"forward {k}/773", flush=True)

    rm_y = rowmap[["id", "y"]].set_index("id")["y"]
    def pooled(oof_rows):
        j = pd.concat(oof_rows, ignore_index=True).set_index("id")
        yy = rm_y.reindex(j.index).values
        return float(np.sqrt(np.mean((j["gru_d"].values - yy) ** 2)))

    for set_name, tags in SETS.items():
        base_rows = []
        for wid, (preds, dips, tail, rm, mg) in cache.items():
            p = np.mean([preds[t] for t in tags], 0)
            base_rows.append(pd.DataFrame({"id": rm["id"].values,
                                           "gru_d": np.interp(rm["md_rel"].values, mg, p).astype(np.float32)}))
        print(f"== {set_name} unfused clean OOF = {pooled(base_rows):.4f}", flush=True)
        for dip_name in DIPS:
            for lam in LAMS:
                rows = []
                for wid, (preds, dips, tail, rm, mg) in cache.items():
                    p = np.mean([preds[t] for t in tags], 0)
                    pf_ = p.copy()
                    ti = np.where(tail)[0]
                    pf_[ti] = fuse(p[ti], dips[dip_name][ti][:-1] if len(ti) > 1 else np.zeros(0), lam)
                    rows.append(pd.DataFrame({"id": rm["id"].values,
                                              "gru_d": np.interp(rm["md_rel"].values, mg, pf_).astype(np.float32)}))
                r = pooled(rows)
                print(f"   {set_name} dip={dip_name:6s} lam={lam:4.0f}: fused OOF = {r:.4f}", flush=True)
                if dip_name == "stride" and lam == 1024.0:
                    pd.concat(rows, ignore_index=True).to_parquet(f"gru_oof_fused_{set_name}.parquet", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
