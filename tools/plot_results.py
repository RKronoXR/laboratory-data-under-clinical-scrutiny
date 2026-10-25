#!/usr/bin/env python3
"""Plot the main fixed-threshold clinical summaries with between-seed SDs."""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=Path("results/descriptive/descriptive.csv"))
parser.add_argument("--out", type=Path, default=Path("results/figures"))
args = parser.parse_args()
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit("Install matplotlib 3.11.2 to generate the optional figures.") from error

with args.input.open(encoding="utf-8", newline="") as f:
    rows = {(r["architecture"], r["strategy"]): r for r in csv.DictReader(f) if r["threshold_mode"] == "fixed"}
strategies = ["clinical_only", "laboratory_pretraining", "knowledge_distillation", "pretraining_and_distillation"]
architectures = ["nnU-Net", "YOLO"]
metrics = [("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("accuracy", "Detection accuracy"),
           ("dice_positive", "Dice-Positive"), ("pixel_dice_micro", "Pixel micro-Dice"), ("mcc", "Matthews correlation coefficient")]
if len(rows) != 8:
    raise SystemExit("Expected the complete eight-strategy fixed-threshold summary.")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "savefig.facecolor": "white"})
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (metric, title) in zip(axes.flat, metrics):
    for j, (architecture, colour) in enumerate(zip(architectures, ["#2F5397", "#5AAAAF"])):
        means = [float(rows[architecture, s][metric + "_mean"]) for s in strategies]
        sds = [float(rows[architecture, s][metric + "_sd"]) for s in strategies]
        x = [i + (j - 0.5) * 0.36 for i in range(4)]
        ax.bar(x, means, width=0.34, color=colour, label=architecture, yerr=sds, capsize=3,
               error_kw={"elinewidth": 1, "capthick": 1})
    ax.set_title(title, color="#1A2E66", loc="left", fontweight="bold")
    ax.set_xticks(range(4), ["Clinical\nonly", "Pre-training", "Distillation", "Both"])
    ax.set_ylim(-0.15 if metric == "mcc" else 0, 1.02)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False)
fig.text(0.5, 0.015, "Fixed threshold 0.5. Bars show means; error bars show between-seed SD across three seeds.",
         ha="center", fontsize=10)
fig.tight_layout(rect=(0, 0.045, 1, 0.94))
args.out.mkdir(parents=True, exist_ok=True)
for extension in ["png", "svg"]:
    fig.savefig(args.out / f"clinical_performance.{extension}", dpi=300)
plt.close(fig)
print(args.out)
