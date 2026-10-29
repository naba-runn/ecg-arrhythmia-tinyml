"""Generate the report figures (vector PDF, IEEE column widths) from the data, saved results
and frozen models. Run from the project root: python report/make_figures.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wfdb  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import RR_SETS, load_split  # noqa: E402
from ecg.tflite_utils import TFLiteClassifier, _graph  # noqa: E402

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "outputs"
COL, FULL = 3.5, 7.16  # IEEE single / double column width, inches
PREVIEW_DIR = os.environ.get("FIGURE_PREVIEW_DIR")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CLS_COLORS = {"N": "#4C72B0", "SVEB": "#DD8452", "VEB": "#55A868", "F": "#C44E52", "Q": "#8172B3"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    if PREVIEW_DIR:  # PNG copies for checking figures without a PDF viewer
        fig.savefig(Path(PREVIEW_DIR) / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", name)


def fig_e3c():
    """Adherence of the 122 reviewed studies to the E3C criteria (data from Silva et al. [1])."""
    labels = ["AAMI-\ncompliant", "Inter-\npatient", "Embedded-\nfeasibility", "All three\n(E3C)"]
    counts = [68, 37, 38, 5]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    bars = ax.bar(labels, counts, color=["#4C72B0", "#4C72B0", "#4C72B0", "#C44E52"], width=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 1.5, f"{c}\n({100 * c / 122:.1f}%)", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Studies (of 122)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "e3c_adherence")


def fig_beats(x, y):
    """Mean +/- std beat waveform per class (training split)."""
    t = (np.arange(C.BEAT_LEN) - C.PRE_SAMPLES) / C.FS * 1000
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.45), sharey=True)
    for ax, (i, name) in zip(axes, enumerate(C.CLASSES[:4])):
        b = x["beat"][y == i, :, 0]
        m, s = b.mean(0), b.std(0)
        ax.fill_between(t, m - s, m + s, color=CLS_COLORS[name], alpha=0.25, linewidth=0)
        ax.plot(t, m, color=CLS_COLORS[name])
        ax.set_title(f"{name} (n = {len(b):,})")
        ax.set_xlabel("Time from R-peak (ms)")
        ax.axvline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Amplitude (z-score)")
    save(fig, "beat_morphology")


def fig_rr(x, y):
    """Prematurity ratio pre_rr / long_rr per class (training split)."""
    col = RR_SETS["base"].index("pre/long")
    bins = np.linspace(0.2, 2.0, 60)
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    for i, name in enumerate(C.CLASSES[:3]):
        ax.hist(x["rr"][y == i, col], bins=bins, density=True, histtype="step", color=CLS_COLORS[name],
                linewidth=1.1, label=name)
    ax.axvline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel(r"Prematurity ratio $RR_{\mathrm{pre}} / \overline{RR}_{\mathrm{5\,min}}$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "rr_prematurity")


def fig_class_distribution():
    counts = {}
    for split in ("train", "val", "test"):
        _, y, _ = load_split(split)
        counts[split] = np.bincount(y, minlength=5)
    names = {"train": "Train (17 rec.)", "val": "Validation (5 rec.)", "test": "Test, DS2 (22 rec.)"}
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    w = 0.26
    for k, split in enumerate(counts):
        ax.bar(np.arange(5) + (k - 1) * w, np.maximum(counts[split], 0.8), w, label=names[split],
               color=["#4C72B0", "#8DA0CB", "#DD8452"][k])
    ax.set_yscale("log")
    ax.set_xticks(range(5), C.CLASSES)
    ax.set_ylabel("Beats (log scale)")
    ax.set_ylim(0.8, 1e5)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "class_distribution")


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    for seed, color in zip((0, 1, 2), ("#4C72B0", "#DD8452", "#55A868")):
        log = json.loads((OUT / "runs" / "w1.0_aug_clip" / f"seed{seed}" / "train_log.json").read_text())
        ep = [h["epoch"] for h in log["history"]]
        f1 = [100 * h["val_selection_f1"] for h in log["history"]]
        ax.plot(ep, f1, color=color, label=f"seed {seed}")
        best = log["best_epoch"]
        ax.plot(best, f1[best - 1], "o", color=color, markersize=3.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val. F1 (N, SVEB, VEB) (%)")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "training_curves")


def fig_confusion(cm):
    cm = np.asarray(cm)[:4, :4]
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL * 0.78, COL * 0.68))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(4), C.CLASSES[:4])
    ax.set_yticks(range(4), C.CLASSES[:4])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    save(fig, "confusion_ds2")


def fig_quantization():
    rep = json.loads((OUT / "quant" / "report.json").read_text())["models"]
    models = [("fp32", "FP32", "#4C72B0"), ("ptq_int8", "INT8 PTQ", "#DD8452"), ("qat_int8", "INT8 QAT", "#55A868")]
    metrics = [("N", "Se"), ("SVEB", "Se"), ("SVEB", "+P"), ("VEB", "Se"), ("VEB", "+P")]
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    w = 0.26
    for k, (key, label, color) in enumerate(models):
        vals = [100 * rep[key]["test"]["per_class"][c][m] for c, m in metrics]
        ax.bar(np.arange(len(metrics)) + (k - 1) * w, vals, w, label=label, color=color)
    ax.set_xticks(range(len(metrics)), [f"{c} {m}" for c, m in metrics])
    ax.set_ylabel("DS2 (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "quantization")


def fig_memory_timeline():
    """Live activation bytes per op, FP32 vs INT8 (one ensemble member)."""
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    for path, label, color in [(OUT / "quant" / "fp32.tflite", "FP32", "#4C72B0"),
                               (OUT / "final" / "member_seed0_int8.tflite", "INT8", "#DD8452")]:
        interp, tensors, ops = _graph(path)
        inputs = {d["index"] for d in interp.get_input_details()}
        outputs = {d["index"] for d in interp.get_output_details()}
        acts = inputs | {t for op in ops for t in op["outputs"]}
        first, last = {t: 0 for t in inputs}, {t: len(ops) - 1 for t in outputs}
        for i, op in enumerate(ops):
            for t in op["outputs"]:
                first.setdefault(t, i)
            for t in op["inputs"]:
                if t in acts:
                    last[t] = max(last.get(t, i), i)
        nb = lambda t: int(np.prod(tensors[t]["shape"])) * np.dtype(tensors[t]["dtype"]).itemsize
        live = [sum(nb(t) for t in acts if first[t] <= i <= last.get(t, first[t])) / 1024 for i in range(len(ops))]
        ax.step(range(len(ops)), live, where="mid", color=color, label=f"{label} (peak {max(live):.1f} KB)")
    ax.set_xlabel("Operator index (execution order)")
    ax.set_ylabel("Live activations (KB)")
    ax.set_ylim(0, 22)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "memory_timeline")


def af_fraction(record):
    """Fraction of a record's beats annotated inside atrial fibrillation or flutter episodes."""
    ann = wfdb.rdann(str(C.RAW_DIR / record), "atr")
    rhythm, in_af, total = None, 0, 0
    for sym, aux in zip(ann.symbol, ann.aux_note):
        if sym == "+":
            rhythm = aux.strip("\x00 ")
        elif sym in C.AAMI_MAP:
            total += 1
            in_af += rhythm in ("(AFIB", "(AFL")
    return in_af / max(total, 1)


def fig_false_alarms():
    manifest = json.loads((OUT / "final" / "manifest.json").read_text())
    x, y, rec = load_split("test", manifest["inputs"]["rr"]["set"])
    probs = np.mean([TFLiteClassifier(OUT / "final" / m["file"]).predict(x) for m in manifest["members"]], axis=0)
    pred = probs.argmax(1)
    fa = (y == 0) & (pred == 1)
    records = sorted(set(rec), key=lambda r: -np.sum(fa & (rec == r)))[:10]
    counts = [int(np.sum(fa & (rec == r))) for r in records]
    af = [af_fraction(r) for r in records]
    labels = {"231": "2:1 AV block"}
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    colors = ["#C44E52" if a > 0.05 else ("#8172B3" if r in labels else "#8C8C8C") for r, a in zip(records, af)]
    ax.bar(records, counts, color=colors)
    for i, (c, a) in enumerate(zip(counts, af)):
        if a > 0.05:
            ax.text(i, c + 15, f"AF {100 * a:.0f}%", ha="center", fontsize=5.5, rotation=90, va="bottom")
    ax.set_xlabel("DS2 record")
    ax.set_ylabel("N beats called SVEB")
    ax.set_ylim(0, max(counts) * 1.35)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C44E52", label="AF/flutter episodes"), Patch(color="#8172B3", label="2:1 AV block"),
                       Patch(color="#8C8C8C", label="other")], frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "false_alarms_by_record")
    share = sum(counts[:5]) / fa.sum()
    print(f"  total false alarms {fa.sum()}, top-5 records share {100 * share:.1f}%")
    return {r: (c, a) for r, c, a in zip(records, counts, af)}


def fig_threshold():
    probs = np.load(OUT / "ensemble" / "w1.0_aug_clip" / "probs.npz")["int8_val"]
    _, y, _ = load_split("val")
    from ecg.metrics import summarize
    ts = np.round(np.arange(0.3, 1.0, 0.01), 2)
    se, ppv, f1 = [], [], []
    for t in ts:
        pred = probs.argmax(1)
        others = probs.copy()
        others[:, 1] = -1
        demote = (pred == 1) & (probs[:, 1] < t)
        pred[demote] = others[demote].argmax(1)
        s = summarize(y, pred)
        se.append(100 * s["per_class"]["SVEB"]["Se"])
        ppv.append(100 * s["per_class"]["SVEB"]["+P"])
        f1.append(100 * s["selection_f1"])
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    ax.plot(ts, se, color=CLS_COLORS["SVEB"], label="SVEB Se")
    ax.plot(ts, ppv, color="#4C72B0", label="SVEB +P")
    ax.plot(ts, f1, color="black", linestyle="--", label="F1 (N, SVEB, VEB)")
    ax.axvline(0.51, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("SVEB decision threshold $t$")
    ax.set_ylabel("Validation (%)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "threshold_sweep")


def fig_irregularity(x, y, rec):
    """Rhythm irregularity of DS1 training beats inside vs outside AF/flutter episodes."""
    col = RR_SETS["full"].index("rr_cv")
    in_af = np.zeros(len(y), bool)
    for r in np.unique(rec):
        ann = wfdb.rdann(str(C.RAW_DIR / r), "atr")
        rhythm, flags = None, []
        for sym, aux in zip(ann.symbol, ann.aux_note):
            if sym == "+":
                rhythm = aux.strip("\x00 ")
            elif sym in C.AAMI_MAP:
                flags.append(rhythm in ("(AFIB", "(AFL"))
        flags = np.array(flags[1:-1])  # build_dataset drops the first and last beat
        m = rec == r
        if len(flags) == m.sum():
            in_af[m] = flags
    groups = [("N, sinus", (y == 0) & ~in_af), ("SVEB, sinus", (y == 1) & ~in_af), ("N, AF/flutter", (y == 0) & in_af)]
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    bins = np.linspace(0, 0.6, 50)
    for (label, m), color in zip(groups, ("#4C72B0", "#DD8452", "#C44E52")):
        if m.sum():
            ax.hist(x["rr"][m, col], bins=bins, density=True, histtype="step", color=color, linewidth=1.1,
                    label=f"{label} (n = {m.sum():,})")
    ax.set_xlabel("RR coefficient of variation over the 10 preceding intervals")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "rr_irregularity")


def main():
    x, y, rec = load_split("train", "full")
    fig_e3c()
    fig_beats(x, y)
    fig_rr(x, y)
    fig_class_distribution()
    fig_training_curves()
    fig_confusion(json.loads((OUT / "final" / "ds2_results.json").read_text())["ds2"]["confusion_matrix"])
    fig_quantization()
    fig_memory_timeline()
    print("  false alarms:", fig_false_alarms())
    fig_threshold()
    fig_irregularity(x, y, rec)


if __name__ == "__main__":
    main()
