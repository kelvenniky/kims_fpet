"""
Builds every figure and the statistics used in the paper from the pilot summary.json files.

Usage (from the repo root):  python paper/scripts/make_figures.py
Writes paper/figures/*.pdf and paper/figures/stats.json.
"""
import collections
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

OUT = os.environ.get("FIG_OUT", "paper/figures")
DATASETS = ["cifar", "dtd", "smallnorb_azi", "dsprites_loc"]
NAMES = {"cifar": "CIFAR-100", "dtd": "DTD", "smallnorb_azi": "sNORB-Azim", "dsprites_loc": "dSpr-Loc"}
RUNS = ["adaptformer", "fpet-r2", "mcpet-r2", "fpet-r4", "mcpet-r4"]
SEEDS = [0, 1, 2]

# colour = method, marker = merge layer
GREY, ORANGE, BLUE = "#7a7974", "#eb6834", "#2a78d6"
COLOR = {"adaptformer": GREY, "fpet": ORANGE, "mcpet": BLUE}
LABEL = {"adaptformer": "AdaptFormer (no merging)", "fpet": "FPET", "mcpet": "MC-FPET (ours)"}
MARKER = {"2": "o", "4": "s"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.bbox": "tight",
})

summ = collections.defaultdict(dict)  # (dataset, run) -> seed -> summary
for f in glob.glob("*/pilot/*/seed*/*/summary.json"):
    j = json.load(open(f))
    summ[(j["dataset"], j["run"])][j["seed"]] = j


def acc(ds, run):
    return [summ[(ds, run)][s]["best_acc"] for s in SEEDS]


def family(run):
    return run.split("-")[0]


stats = {}

# ---------- paired statistics ----------
for r in ["2", "4"]:
    diffs = [m - f for ds in DATASETS for m, f in zip(acc(ds, f"mcpet-r{r}"), acc(ds, f"fpet-r{r}"))]
    t = sps.ttest_rel([m for ds in DATASETS for m in acc(ds, f"mcpet-r{r}")],
                      [f for ds in DATASETS for f in acc(ds, f"fpet-r{r}")])
    w = sps.wilcoxon(diffs)
    sign = sps.binomtest(sum(d > 0 for d in diffs), len(diffs), 0.5)
    stats[f"r{r}"] = {"mean_gain": st.mean(diffs), "sd_gain": st.stdev(diffs), "wins": sum(d > 0 for d in diffs),
                      "n": len(diffs), "t": t.statistic, "p_t": t.pvalue, "p_wilcoxon": w.pvalue,
                      "p_sign": sign.pvalue, "min": min(diffs), "max": max(diffs)}

# per dataset mean/std and averages
table = {}
for ds in DATASETS:
    for run in RUNS:
        a = acc(ds, run)
        table[f"{ds}/{run}"] = (st.mean(a), st.stdev(a))
for run in RUNS:
    per_seed_avg = [st.mean(summ[(ds, run)][s]["best_acc"] for ds in DATASETS) for s in SEEDS]
    table[f"avg/{run}"] = (st.mean(per_seed_avg), st.stdev(per_seed_avg))
stats["table"] = table

# recovered fraction of the merging loss
rec = {}
for r in ["2", "4"]:
    for ds in DATASETS:
        base, fp, mc = (st.mean(acc(ds, x)) for x in ["adaptformer", f"fpet-r{r}", f"mcpet-r{r}"])
        rec[f"{ds}/r{r}"] = {"fpet_drop": fp - base, "mc_drop": mc - base,
                             "recovered": (mc - fp) / (base - fp) if base - fp > 0 else None}
stats["recovered"] = rec

# cost
cost = {}
for run in RUNS:
    rows = [summ[(ds, run)][s] for ds in DATASETS for s in SEEDS]
    lat = [x["latency_ms_per_img"] for x in rows if x.get("latency_ms_per_img")]
    cost[run] = {"gflops": st.mean(x["gflops"] for x in rows), "latency_ms": st.mean(lat),
                 "latency_sd": st.stdev(lat), "mem_gb": st.mean(x["peak_train_mem_gb"] for x in rows),
                 # one AdaptFormer run (dsprites_loc, seed 0) ran in a slow session (4.52 h against ~1.8 h for
                 # the other seeds); it is left out of the mean training time, as stated in Table 5 / Table 4.7
                 "train_s": st.mean(x["total_train_time_s"] for x in rows
                                    if not (run == "adaptformer" and x["dataset"] == "dsprites_loc" and x["seed"] == 0)),
                 "params_trainable": rows[0]["params_trainable"], "params_total": rows[0]["params_total"]}
    # training time per dataset (mean over seeds)
    for ds in DATASETS:
        cost[run][f"train_s/{ds}"] = st.mean(summ[(ds, run)][s]["total_train_time_s"] for s in SEEDS)
stats["cost"] = cost

# merged vs full-token branch at the selected checkpoint
fb = {}
for run in ["fpet-r2", "mcpet-r2", "fpet-r4", "mcpet-r4"]:
    for ds in ["cifar", "dtd", "smallnorb_azi"]:
        pairs = [(summ[(ds, run)][s]["best_acc"], summ[(ds, run)][s]["best_acc_full_branch"]) for s in SEEDS]
        pairs = [p for p in pairs if p[1] is not None]
        fb[f"{ds}/{run}"] = {"merged": st.mean(p[0] for p in pairs), "full": st.mean(p[1] for p in pairs),
                             "gap": st.mean(p[0] - p[1] for p in pairs), "n": len(pairs)}
stats["full_branch"] = fb

# selected epoch
stats["best_epoch"] = {f"{ds}/{run}": [summ[(ds, run)][s]["best_epoch"] for s in SEEDS] for ds in DATASETS for run in RUNS}

json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)

# ---------- Fig: accuracy vs GFLOPs ----------
fig, ax = plt.subplots(figsize=(3.4, 2.5))
base_m, base_s = table["avg/adaptformer"]
ax.axhline(base_m, color=GREY, lw=1, ls="--", zorder=1)
ax.errorbar(cost["adaptformer"]["gflops"], base_m, yerr=base_s, fmt="D", ms=6, color=GREY, mec="white", mew=1,
            capsize=2, zorder=3, label=LABEL["adaptformer"])
for fam in ["fpet", "mcpet"]:
    xs = [cost[f"{fam}-r{r}"]["gflops"] + (-0.12 if fam == "fpet" else 0.12) for r in "24"]
    ys = [table[f"avg/{fam}-r{r}"][0] for r in "24"]
    es = [table[f"avg/{fam}-r{r}"][1] for r in "24"]
    ax.plot(xs, ys, color=COLOR[fam], lw=1.4, zorder=2)
    for x, y, e, r in zip(xs, ys, es, "24"):
        ax.errorbar(x, y, yerr=e, fmt=MARKER[r], ms=6, color=COLOR[fam], mec="white", mew=1, capsize=2, zorder=3)
    ax.plot([], [], color=COLOR[fam], marker="o", ms=5, label=LABEL[fam])
for r, dx in [("2", 10), ("4", 10)]:
    y = (table[f"avg/mcpet-r{r}"][0] + table[f"avg/fpet-r{r}"][0]) / 2
    ax.annotate(f"merge at\nlayer {r}", (cost[f"mcpet-r{r}"]["gflops"], y), xytext=(dx, 0),
                textcoords="offset points", ha="left", va="center", fontsize=6.5, color=MUTED)
ax.text(17.6, base_m - 0.12, "no merging", ha="center", va="top", fontsize=6.5, color=MUTED)
ax.set_xlabel("Inference GFLOPs per image")
ax.set_ylabel("Mean top-1 accuracy over 4 tasks (%)")
ax.set_xlim(9.7, 18.3)
ax.legend(loc="lower right", frameon=False)
fig.savefig(f"{OUT}/fig_tradeoff.pdf")
plt.close(fig)

# ---------- Fig: per-seed paired gains ----------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
for ax, r in zip(axes, "24"):
    ax.axhline(0, color=MUTED, lw=0.8)
    for i, ds in enumerate(DATASETS):
        d = [m - f for m, f in zip(acc(ds, f"mcpet-r{r}"), acc(ds, f"fpet-r{r}"))]
        ax.bar(i, st.mean(d), width=0.55, color=BLUE, alpha=0.35, edgecolor="none")
        ax.scatter([i - 0.12, i, 0.12 + i], d, s=14, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS])
    ax.set_title(f"Merge at layer {r}: mean {stats['r' + r]['mean_gain']:+.2f} pp,\n"
                 f"{stats['r' + r]['wins']} of 12 paired runs higher", color=INK)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("MC-FPET minus FPET (pp)")
fig.savefig(f"{OUT}/fig_gains.pdf")
plt.close(fig)

# ---------- Fig: test-accuracy curves, layer 2, mean and range over three seeds ----------
# Seed 0 histories come from the original runs. The histories of seeds 1-2 were not kept, so those runs
# were repeated with the same code, seed and settings (jobs.py phase rev-final, results/revision_final/).
reruns = collections.defaultdict(dict)
for f in glob.glob("results/revision_final/*/seed*/*/summary.json"):
    j = json.load(open(f))
    reruns[(j["dataset"], j["run"])][j["seed"]] = j


def history(ds, run, s):
    h = summ[(ds, run)].get(s, {}).get("history")
    return h if h else reruns[(ds, run)][s]["history"]


fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1))
for ax, ds in zip(axes, ["cifar", "dtd", "smallnorb_azi"]):
    late = []
    for run, fam in [("adaptformer", "adaptformer"), ("fpet-r2", "fpet"), ("mcpet-r2", "mcpet")]:
        hs = [[x for x in history(ds, run, s) if x["epoch"] >= 19] for s in SEEDS]  # epoch 10 is warm-up
        ep = [x["epoch"] + 1 for x in hs[0]]
        ys = [[100 * x["acc"] for x in h] for h in hs]
        mean = [st.mean(v) for v in zip(*ys)]
        ax.fill_between(ep, [min(v) for v in zip(*ys)], [max(v) for v in zip(*ys)], color=COLOR[fam], alpha=0.15, lw=0)
        ax.plot(ep, mean, color=COLOR[fam], ls="--" if fam == "adaptformer" else "-", marker="o", ms=3, label=LABEL[fam])
        late += [y for v in ys for y in v]
    ax.set_title(NAMES[ds], color=INK)
    ax.set_xlabel("Epoch")
    ax.set_ylim(min(late) - 1, max(late) + 1)
axes[0].set_ylabel("Test accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.savefig(f"{OUT}/fig_curves.pdf")
plt.close(fig)

# ---------- Fig: loss terms of MC-FPET, seed 0, layer 2 ----------
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0))
for ax, ds in zip(axes, ["cifar", "dtd", "smallnorb_azi"]):
    h = summ[(ds, "mcpet-r2")][0]["history"]
    ep = [x["epoch"] + 1 for x in h]
    ax.plot(ep, [x["loss_ce"] for x in h], color=BLUE, label="CE, merged path")
    ax.plot(ep, [x["loss_ce_full"] for x in h], color=GREY, ls="--", label="CE, full-token path")
    ax.plot(ep, [x["loss_kl"] for x in h], color=ORANGE, ls=":", lw=2, label="KL term (full to merged)")
    ax.set_title(NAMES[ds], color=INK)
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
axes[0].set_ylabel("Mean training loss")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.savefig(f"{OUT}/fig_losses.pdf")
plt.close(fig)

# ---------- Fig: training cost ----------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2))
order = ["adaptformer", "fpet-r4", "mcpet-r4", "fpet-r2", "mcpet-r2"]
labels = ["Adapt-\nFormer", "FPET\nlayer 4", "MC-FPET\nlayer 4", "FPET\nlayer 2", "MC-FPET\nlayer 2"]
for ax, key, yl in [(axes[0], "mem_gb", "Peak training memory (GB)"), (axes[1], "train_s", "Training time (h)")]:
    vals = [cost[r][key] / (3600 if key == "train_s" else 1) for r in order]
    ax.bar(range(5), vals, color=[COLOR[family(r)] for r in order], width=0.6, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(5), labels)
    ax.set_ylabel(yl)
    ax.grid(axis="x", visible=False)
fig.savefig(f"{OUT}/fig_cost.pdf")
plt.close(fig)

print(json.dumps({k: stats[k] for k in ["r2", "r4"]}, indent=1))
print(json.dumps({k: [round(x, 2) for x in v] for k, v in table.items() if k.startswith("avg")}, indent=0))
print(json.dumps({k: {a: round(b, 3) for a, b in v.items() if not a.startswith("train_s/")} for k, v in cost.items()}, indent=0))
