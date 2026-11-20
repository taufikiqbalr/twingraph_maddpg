#!/usr/bin/env python3
"""Recreate the reported comparison plots from bundled observations.

No policy inference, training, simulator, or private implementation is used.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/figures"
DPI = 600
POLICIES = ("maddpg_srdg_v2", "maddpg", "matd3", "mappo", "jcce_gat_maddpg", "jcce_centralized_ddqn")
SCALES, SEEDS = (3,5,7,9), (11,17,23)
DEVICES = {3:60,5:100,7:150,9:200}
NAMES = dict(zip(POLICIES,("TwinGraph-MADDPG","MADDPG","MATD3","MAPPO","GAT-MADDPG","Centralized DDQN")))
COLORS = ("#103F75", "#B98245", "#8D7CAF", "#4E957E", "#687F8D", "#B46368")
MARKERS = ("o", "s", "^", "D", "v", "P")
LINES = ("-", (0,(5,2)), (0,(3,2,1,2)), (0,(7,2)), (0,(2,2)), (0,(5,2,1,2,1,2)))
EVAL_METRICS = ("latency","communication","energy")
TRAIN_METRICS = ("reward","latency","fetches")
FILES = {8:"tdrive_evaluation",9:"ppp_evaluation",10:"tdrive_lookup_cost",
    11:"ppp_reward",12:"ppp_lookup_cost",13:"ppp_fetches",16:"tdrive_reward",17:"tdrive_fetches"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":17,"svg.fonttype":"none",
    "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"path.simplify":False})

def read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path,"rt",newline="") as stream:
        return list(csv.DictReader(stream))

def write_csv(path,rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path,"wt",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def export(fig,number):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for item in fig.findobj(match=Text):
        if item.get_visible() and item.get_text().strip():
            b=item.get_window_extent(renderer)
            if b.x0<-.5 or b.y0<-.5 or b.x1>width+.5 or b.y1>height+.5:
                raise ValueError("Label exceeds figure canvas: " + item.get_text())
    for extension in ("svg","pdf","png"):
        fig.savefig(OUTPUT/(FILES[number]+"."+extension),dpi=DPI,facecolor="white")
    plt.close(fig)
    return {"name":FILES[number],"formats":["svg","pdf","png"],"png_dpi":DPI}

def rolling20(values):
    values = np.asarray(values, dtype=float)
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - 20)
    prefix = np.r_[0, np.cumsum(values)]
    return (prefix[end] - prefix[start]) / (end - start)

def axes_style(axis):
    axis.grid(axis="y", color="#DFE5E9", lw=.7)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.spines[["left", "bottom"]].set_color("#85929E")

def common_legend(fig, y, markers=False):
    handles = [Line2D([], [], label=NAMES[p], color=COLORS[i],
        marker=MARKERS[i] if markers else None, linestyle=LINES[i],
        lw=2.9 if i == 0 else 1.8, markersize=7) for i, p in enumerate(POLICIES)]
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, y),
        ncol=3, fontsize=17, frameon=False, columnspacing=1.6, handlelength=2.6)
    legend.get_texts()[0].set_fontweight("semibold")

def evaluation(summary, mobility, number, limits):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.8))
    fig.subplots_adjust(left=.082, right=.985, bottom=.39, top=.79, wspace=.34)
    for column, (axis, metric) in enumerate(zip(axes, EVAL_METRICS)):
        for index, policy in enumerate(POLICIES):
            rows = sorted((r for r in summary if r["mobility"] == mobility and r["metric"] == metric
                and r["policy"] == policy), key=lambda r: r["scale"])
            assert [r["scale"] for r in rows] == list(SCALES)
            axis.errorbar(SCALES, [r["mean"] for r in rows], yerr=[r["seed_sd"] for r in rows],
                color=COLORS[index], marker=MARKERS[index], ls=LINES[index],
                lw=2.9 if index == 0 else 1.7, markersize=7, capsize=4,
                elinewidth=1.6 if index == 0 else 1.1, zorder=5 if index == 0 else 3)
        axis.set_xlim(2.7, 9.3)
        axis.set_ylim(limits[metric])
        axis.set_xticks(SCALES)
        axis.set_xlabel("Number of agents", fontsize=16, labelpad=9)
        axis.set_title(f"({'abc'[column]}) {('Lookup cost', 'Communication cost', 'Energy proxy')[column]}",
            fontsize=18, pad=13, fontweight="semibold")
        axis.set_ylabel("Weighted events / request" if metric == "communication" else "Proxy units / request",
            fontsize=15, labelpad=9)
        axes_style(axis)
    label = "T-Drive" if mobility == "tdrive" else "PPP"
    fig.suptitle(f"{label}: final-policy performance across deployment scales", y=.97,
        fontsize=22, fontweight="semibold")
    fig.text(.53, .895, "1,000 training episodes · 20 evaluation episodes per seed · 3 training seeds",
        ha="center", fontsize=16)
    common_legend(fig, .22, markers=True)
    fig.text(.53, .122, "Mean ± sample SD across trained seeds. Devices at 3 / 5 / 7 / 9 agents: 60 / 100 / 150 / 200.",
        ha="center", fontsize=13.5)
    note = ("Held-out vehicle traces; common logical-clock evaluation.\nTraining origins differ at 3 agents; all six methods use logical-clock training at 5 / 7 / 9 agents."
        if mobility == "tdrive" else "Common logical-clock evaluation on fresh synthetic scenarios.\nTwinGraph / MADDPG / MATD3 / MAPPO: archived wall-clock training; GAT / DDQN: logical-clock training.")
    fig.text(.53, .045, note, ha="center", fontsize=13, linespacing=1.4)
    return export(fig, number)

def training(curves, mobility, metric, number):
    selected = [r for r in curves if r["mobility"] == mobility and r["metric"] == metric]
    low = min(r["mean"] - r["seed_sd"] for r in selected)
    high = max(r["mean"] + r["seed_sd"] for r in selected)
    pad = (high - low) * .065
    limits = [low - pad, high + pad]
    if metric == "fetches":
        limits[0] = min(0, limits[0])
    if mobility == "tdrive" and metric in ("reward", "fetches"):
        ppp = [r for r in curves if r["mobility"] == "ppp" and r["metric"] == metric]
        p_low = min(r["mean"] - r["seed_sd"] for r in ppp)
        p_high = max(r["mean"] + r["seed_sd"] for r in ppp)
        p_pad = (p_high - p_low) * .065
        p_limits = [0 if metric == "fetches" else p_low - p_pad, p_high + p_pad]
        if p_limits[0] <= low and high <= p_limits[1]:
            limits = p_limits
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.975, bottom=.25, top=.84, hspace=.35, wspace=.17)
    ylabels = {"reward": "Global episodic reward", "latency": "Lookup cost (proxy units / request)",
        "fetches": "Cross-shard fetches per episode"}
    titles = {"reward": "global episodic reward", "latency": "global-cache lookup cost", "fetches": "cross-shard fetches"}
    groups = defaultdict(list)
    for row in selected:
        groups[row["scale"], row["policy"]].append(row)
    for panel, (axis, scale) in enumerate(zip(axes.flat, SCALES)):
        for index, policy in enumerate(POLICIES):
            rows = groups[scale, policy]
            x = np.array([r["episode"] for r in rows])
            mean = np.array([r["mean"] for r in rows])
            sd = np.array([r["seed_sd"] for r in rows])
            assert np.array_equal(x, np.arange(1, 1001))
            axis.fill_between(x, mean - sd, mean + sd, color=COLORS[index],
                alpha=.13 if index == 0 else .08, lw=0, zorder=1)
            axis.plot(x, mean, color=COLORS[index], ls=LINES[index],
                lw=2.8 if index == 0 else 1.7, zorder=5 if index == 0 else 3)
        axis.set_xlim(1, 1000)
        axis.set_ylim(limits)
        axis.set_xticks([1, 250, 500, 750, 1000])
        axis.set_title(f"({'abcd'[panel]}) {scale} agents / {DEVICES[scale]} devices",
            fontsize=18, fontweight="semibold", pad=13)
        if panel >= 2:
            axis.set_xlabel("Training episode", fontsize=17, labelpad=8)
        axes_style(axis)
    fig.supylabel(ylabels[metric], x=.025, y=.545, fontsize=16)
    label = "T-Drive" if mobility == "tdrive" else "PPP"
    fig.suptitle(f"{label}: {titles[metric]} during training", y=.975, fontsize=22, fontweight="semibold")
    fig.text(.53, .919, "1,000 episodes per run · 3 training seeds · exploration active", ha="center", fontsize=17)
    common_legend(fig, .158)
    fig.text(.53, .081, "Each seed: trailing 20-episode mean (shorter windows at startup). Lines: seed mean; bands: ±1 sample SD.",
        ha="center", fontsize=13.5)
    note = ("At 3 agents: archived four-method wall-clock training; GAT / DDQN use logical-clock training.\nAt 5 / 7 / 9 agents: all six methods use logical-clock training."
        if mobility == "tdrive" else "TwinGraph / MADDPG / MATD3 / MAPPO: archived wall-clock training.\nGAT-MADDPG / DDQN: logical-clock training. Training trajectories retain their original seed schedules.")
    fig.text(.53, .027, note, ha="center", fontsize=13, linespacing=1.4)
    return export(fig, number), limits

def aggregate(data):
    evaluation=read_csv(data/"evaluation_episodes.csv")
    training_rows=read_csv(data/"training_episodes.csv.gz")
    eval_groups,train_groups=defaultdict(list),defaultdict(list)
    for r in evaluation:
        if r["split"]=="eval":
            if int(r["checkpoint_episode"])!=1000: raise ValueError("Unexpected training budget")
            eval_groups[r["mobility"],int(r["scale"]),r["policy"],int(r["seed"])].append(r)
    for r in training_rows:
        train_groups[r["mobility"],int(r["scale"]),r["policy"],int(r["seed"])].append(r)
    expected=set(itertools.product(("ppp","tdrive"),SCALES,POLICIES,SEEDS))
    if set(eval_groups)!=expected or set(train_groups)!=expected: raise ValueError("Incomplete experiment grid")
    summaries,curves=[],[]
    for mobility,scale,policy in itertools.product(("ppp","tdrive"),SCALES,POLICIES):
        for metric in EVAL_METRICS:
            means=[]
            for seed in SEEDS:
                rows=eval_groups[mobility,scale,policy,seed]
                if len(rows)!=20 or len({r["evaluation_seed"] for r in rows})!=20:
                    raise ValueError("Expected 20 distinct evaluation scenarios")
                means.append(np.mean([float(r[metric]) for r in rows]))
            summaries.append(dict(mobility=mobility,scale=scale,policy=policy,metric=metric,
                mean=float(np.mean(means)),seed_sd=float(np.std(means,ddof=1))))
        for metric in TRAIN_METRICS:
            raw,smooth=[],[]
            for seed in SEEDS:
                rows=sorted(train_groups[mobility,scale,policy,seed],key=lambda r:int(r["episode"]))
                if [int(r["episode"]) for r in rows]!=list(range(1,1001)):
                    raise ValueError("Expected every episode from 1 to 1000")
                values=np.array([float(r[metric]) for r in rows])
                if not np.isfinite(values).all(): raise ValueError("Non-finite training value")
                raw.append(values);smooth.append(rolling20(values))
            raw,smooth=np.array(raw),np.array(smooth)
            for i,(mean,sd) in enumerate(zip(smooth.mean(axis=0),smooth.std(axis=0,ddof=1))):
                curves.append(dict(mobility=mobility,scale=scale,policy=policy,metric=metric,
                    episode=i+1,mean=float(mean),seed_sd=float(sd),
                    **{f"seed_{s}_raw":float(raw[j,i]) for j,s in enumerate(SEEDS)},
                    **{f"seed_{s}_rolling20":float(smooth[j,i]) for j,s in enumerate(SEEDS)}))
    if len(training_rows)!=144000 or len(curves)!=144000: raise ValueError("Incomplete training data")
    write_csv(OUTPUT/"evaluation_plot_values.csv",summaries)
    write_csv(OUTPUT/"training_plot_values.csv.gz",curves)
    return summaries,curves

def main():
    global OUTPUT,DPI
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"data")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--dpi",type=int,default=600,help="PNG resolution; publication default 600")
    args=parser.parse_args();OUTPUT=args.output.resolve();DPI=args.dpi
    if DPI<72: parser.error("DPI must be at least 72")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    summary,curves=aggregate(args.data.resolve())
    limits={}
    for metric in EVAL_METRICS:
        rows=[r for r in summary if r["metric"]==metric]
        low=min(r["mean"]-r["seed_sd"] for r in rows);high=max(r["mean"]+r["seed_sd"] for r in rows)
        pad=(high-low)*.12;limits[metric]=[low-pad,high+pad]
    files=[evaluation(summary,"tdrive",8,limits),evaluation(summary,"ppp",9,limits)]
    training_limits={}
    for number,mobility,metric in ((10,"tdrive","latency"),(11,"ppp","reward"),(12,"ppp","latency"),
                                  (13,"ppp","fetches"),(16,"tdrive","reward"),(17,"tdrive","fetches")):
        check,bound=training(curves,mobility,metric,number)
        files.append(check);training_limits[FILES[number]]=bound
    record=dict(status="PASS",training_runs=144,training_episodes=144000,training_seeds=list(SEEDS),
        figures=files,evaluation_y_limits=limits,training_y_limits=training_limits,
        uncertainty="sample SD across three seeds; training smoothing applied within each seed first",
        stage="recorded-data analysis; no training or policy inference")
    (OUTPUT/"plot_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"status":"PASS","figures":len(files),"output":str(OUTPUT)},indent=2))

if __name__=="__main__":main()
