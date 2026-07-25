# Figures explaining STRIDE v1 on a REAL well (044af7d1). Every panel is computed with
# the deployed code path / constants — no toy data.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stride import load_well, stride_track, grcal_tw, _decode, _paths_from_rates

WID = "044af7d1"
INK, PANEL = "#0f151b", "#161e26"
TXT, MUTED = "#dbe3ea", "#8fa0b0"
AMBER, TEAL, CRIMSON = "#d9a441", "#4fb3a8", "#e0625f"
GRIDC = "#2a3540"
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRIDC, "grid.color": GRIDC, "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "600", "figure.dpi": 130,
})

hw, tw = load_well(WID, "train")
kn = hw[hw["TVT_input"].notna()]
ev = hw[hw["TVT_input"].isna()]
tw_s = tw.sort_values("TVT")
tw_tvt = tw_s["TVT"].values.astype(float)
tw_gr_raw = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)
tw_gr = grcal_tw(tw_tvt, tw_gr_raw, kn["TVT_input"].values, kn["GR"].values)

md_ev = ev["MD"].values.astype(float)
z_ev = ev["Z"].values.astype(float)
gr_all = hw["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(hw["GR"]))).values.astype(float)
gr_ev = gr_all[ev.index]
truth = ev["TVT"].values.astype(float)
last = kn.iloc[-1]
u0 = float(last["TVT_input"]) + float(last["Z"])
cut_md = float(last["MD"])
tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
gam = float(np.clip(np.nanmedian(np.abs(kn["GR"].values.astype(float) - tw_at_k)), 5.0, 40.0))
pred, info = stride_track(hw, tw)
LAT = float(last["MD"]) - 700.0            # show only the lateral
knL = kn[kn["MD"] >= LAT]

def implied_gr(tvt_path):
    return np.interp(np.clip(tvt_path, tw_tvt[0], tw_tvt[-1]), tw_tvt, tw_gr)

def cauchy_ll(tvt_path):
    r = (gr_ev - implied_gr(tvt_path)) / gam
    return -np.log1p(r * r)

# ---------------------------------------------------------------- figure 1: the problem
fig, ax = plt.subplots(2, 1, figsize=(9.2, 5.4), sharex=True,
                       gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.14})
ax[0].plot(knL["MD"], knL["TVT_input"], color=TXT, lw=1.6, label="given: TVT on the visible prefix")
ax[0].plot(md_ev, truth, color=CRIMSON, lw=1.6, ls="--", label="hidden: the answer we must predict")
ax[0].axvline(cut_md, color=MUTED, lw=1, ls=":")
ax[0].annotate("cut", (cut_md, kn["TVT_input"].iloc[-1]), xytext=(10, -18),
               textcoords="offset points", color=MUTED, fontsize=10)
ax[0].invert_yaxis(); ax[0].grid(alpha=.25, lw=.6)
ax[0].set_ylabel("TVT  (ft)")
ax[0].legend(frameon=False, loc="lower left", fontsize=10)
ax[0].set_title(f"well {WID} — the lateral drifts {truth[-1]-float(last['TVT_input']):+.1f} ft "
                f"after the cut, and only GR is observed there", loc="left", fontsize=11.5)
_hl = hw[hw["MD"] >= LAT]
ax[1].plot(_hl["MD"], gr_all[_hl.index], color=AMBER, lw=.7)
ax[1].axvline(cut_md, color=MUTED, lw=1, ls=":")
ax[1].grid(alpha=.25, lw=.6); ax[1].set_ylabel("GR"); ax[1].set_xlabel("MD  (ft)")
fig.savefig("fig_stride_1.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- figure 2: the change of variables (the trick)
u_truth = truth + z_ev
u_kn = kn["TVT_input"].values + kn["Z"].values
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9), gridspec_kw={"wspace": 0.2})
ax[0].plot(knL["MD"], knL["TVT_input"], color=TXT, lw=1.2)
ax[0].plot(md_ev, truth, color=CRIMSON, lw=1.4, ls="--")
ax[0].invert_yaxis(); ax[0].grid(alpha=.25, lw=.6)
ax[0].set_title("TVT — the output we are scored on", loc="left", fontsize=11.5)
ax[0].set_xlabel("MD  (ft)"); ax[0].set_ylabel("TVT  (ft)")
ax[1].plot(knL["MD"], (knL["TVT_input"] + knL["Z"]).values, color=TXT, lw=1.2,
           label="visible prefix")
ax[1].plot(md_ev, u_truth, color=CRIMSON, lw=1.4, ls="--", label="hidden zone")
for b in np.arange(cut_md, md_ev[-1], 200.0):
    ax[1].axvline(b, color=GRIDC, lw=.6)
ax[1].invert_yaxis(); ax[1].grid(alpha=.25, lw=.6)
ax[1].set_title("U = TVT + Z — the surface STRIDE actually decodes", loc="left", fontsize=11.5)
ax[1].set_xlabel("MD  (ft)"); ax[1].set_ylabel("U  (ft)")
ax[1].legend(frameon=False, fontsize=9.5, loc="best")
fig.savefig("fig_stride_2.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- figure 3: how one candidate earns its score
win = slice(0, 700)
good, bad = -0.017, 0.030
fig, ax = plt.subplots(2, 1, figsize=(9.6, 5.0), sharex=True, gridspec_kw={"hspace": 0.28})
for a, r, col, lab in ((ax[0], good, TEAL, "a candidate close to the truth"),
                       (ax[1], bad, "#a86fb5", "a candidate drifting the wrong way")):
    tvt_c = (u0 + r * (md_ev - cut_md)) - z_ev
    imp = implied_gr(tvt_c)
    a.plot(md_ev[win], gr_ev[win], color=AMBER, lw=1.0, label="GR actually logged")
    a.plot(md_ev[win], imp[win], color=col, lw=1.2, label="GR this candidate implies")
    a.fill_between(md_ev[win], gr_ev[win], imp[win], color=col, alpha=.18, lw=0)
    ll = cauchy_ll(tvt_c)[win].sum()
    a.set_title(f"{lab}  (slope {r:+.3f})   —   Cauchy log-likelihood {ll:,.0f}", loc="left")
    a.grid(alpha=.25, lw=.6); a.set_ylabel("GR")
    a.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
ax[1].set_xlabel("MD  (ft)")
fig.savefig("fig_stride_3.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------- figure 4: why the evidence must be damped
grid = np.arange(-0.06, 0.0601, 0.002)
seg_end = int(np.searchsorted(md_ev, cut_md + 200.0))
ll = np.array([cauchy_ll((u0 + r * (md_ev - cut_md)) - z_ev)[:seg_end].sum() for r in grid])
true_rate = float(np.polyfit(md_ev[:seg_end] - cut_md, (truth + z_ev)[:seg_end] - u0, 1)[0])
fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9), gridspec_kw={"wspace": 0.22})
for a, w, ttl in ((ax[0], 1.0, "raw sum of per-row likelihood  (lik_w = 1.0)"),
                  (ax[1], 0.1, "damped for GR autocorrelation  (lik_w = 0.1)  ← deployed")):
    p = np.exp(w * (ll - ll.max())); p /= p.sum()
    a.fill_between(grid, p, color=TEAL, alpha=.35)
    a.plot(grid, p, color=TEAL, lw=1.4)
    a.axvline(true_rate, color=CRIMSON, lw=1.3, ls="--")
    a.annotate("true slope", (true_rate, p.max() * .92), xytext=(8, 0),
               textcoords="offset points", color=CRIMSON, fontsize=9.5)
    a.set_title(ttl, loc="left"); a.grid(alpha=.25, lw=.6)
    a.set_xlabel("segment slope  (ft of surface per ft of MD)"); a.set_ylabel("posterior")
fig.savefig("fig_stride_4.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- figure 5: the beam and its aggregation
gmin, gmax = tw_tvt[0], tw_tvt[-1]
gg_x = np.arange(gmin, gmax + 0.5, 0.5)
gg = np.interp(gg_x, tw_tvt, tw_gr)
tail = kn.tail(30)
dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
m = dm > 0
s0 = float(np.clip(np.median((dt + dz)[m] / dm[m]), -0.06, 0.06))
bnds = [0]; cur = md_ev[0]
for i in range(len(md_ev)):
    if md_ev[i] >= cur + 200.0:
        bnds.append(i); cur = md_ev[i]
if bnds[-1] != len(md_ev):
    bnds.append(len(md_ev))
bnds = np.array(bnds, dtype=np.int64)
rgrid = np.arange(-0.06, 0.06 + 0.001, 0.002)
beam_rates, scores = _decode(md_ev, z_ev, gr_ev, np.ones(len(md_ev), np.int8), gg, gmin, 0.5,
                             u0, s0, gam, rgrid, bnds, 96, 0.012, 25.0, 0.0, 0.1)
paths = _paths_from_rates(md_ev, bnds, u0, rgrid, beam_rates)
order = np.argsort(scores)[::-1][:32]
w = np.exp((scores[order] - scores[order].max()) / 8.0); w /= w.sum()
spread = float(np.mean(np.ptp(np.stack([paths[o] for o in order], 0), axis=0)))
fig, ax = plt.subplots(figsize=(9.6, 4.4))
for i, o in enumerate(order):
    ax.plot(md_ev, paths[o] - u_truth, color=TEAL, lw=.7, alpha=.30,
            label="each surviving beam, minus the truth" if i == 0 else None)
ax.axhline(0, color=CRIMSON, lw=1.4, ls="--", label="truth")
ax.plot(md_ev, ((pred + z_ev) - u_truth), color=TEAL, lw=2.0,
        label="likelihood-weighted mean of the 32 = STRIDE output")
ax.grid(alpha=.25, lw=.6)
ax.set_xlabel("MD  (ft)"); ax.set_ylabel("error vs truth  (ft)")
ax.legend(frameon=False, fontsize=10, loc="best")
ax.set_title(f"96 partial paths are kept alive; the best 32 disagree by {spread:.1f} ft on average "
             f"and their weighted mean lands at {np.sqrt(np.mean((pred-truth)**2)):.2f} ft RMSE",
             loc="left", fontsize=11.5)
fig.savefig("fig_stride_5.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_stride_1..5.png |",
      f"gam={gam:.2f} segs={info['n_seg']} rmse={np.sqrt(np.mean((pred-truth)**2)):.2f}")
