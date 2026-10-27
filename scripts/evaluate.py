"""Evaluate one training run on the validation hold-out and on DS2 (the inter-patient test set).

Every seed of the run is evaluated and reported as mean ± std. The seed with the best
validation score is the "final" model: its DS2 confusion matrix is plotted and it is copied
to outputs/final_model.keras for quantization.

Run this only once the configuration is chosen on validation: DS2 results must not feed
back into model choices.

Usage: python scripts/evaluate.py --tag <run>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.metrics import format_report, summarize  # noqa: E402

OUT_DIR = C.ROOT / "outputs"


def plot_confusion(cm, path, title):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def mean_std_table(summaries):
    """Per-class Se / +P / FPR as mean ± std across seeds."""
    pct = lambda v: f"{100 * np.nanmean(v):6.2f} ± {100 * np.nanstd(v):5.2f}"
    lines = [f"{'class':<5} {'Se':>15} {'+P':>15} {'FPR':>15}"]
    for c in C.CLASSES:
        vals = {m: [s["per_class"][c][m] for s in summaries] for m in ("Se", "+P", "FPR")}
        lines.append(f"{c:<5} {pct(vals['Se']):>15} {pct(vals['+P']):>15} {pct(vals['FPR']):>15}")
    acc = [s["accuracy"] for s in summaries]
    f1 = [s["macro_f1"] for s in summaries]
    lines.append(f"Accuracy {pct(acc).strip()}   macro-F1 (N,SVEB,VEB,F) {pct(f1).strip()}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    run_dir = OUT_DIR / "runs" / args.tag
    run_summary = json.loads((run_dir / "summary.json").read_text())
    seeds = run_summary["args"]["seeds"]
    best_seed = run_summary["best_seed"]

    data = {split: load_split(split) for split in ("val", "test")}
    results = {"tag": args.tag, "seeds": seeds, "best_seed": best_seed, "params": run_summary["params"],
               "val": {}, "test": {}}
    for seed in seeds:
        model = keras.models.load_model(run_dir / f"seed{seed}" / "model.keras")
        for split, (x, y, _) in data.items():
            results[split][seed] = summarize(y, model.predict(x, batch_size=1024, verbose=0).argmax(1))

    for split, title in [("val", "Validation (DS1 hold-out)"), ("test", "Test (DS2, inter-patient)")]:
        print(f"{title}, {len(seeds)} seeds, mean ± std")
        print(mean_std_table(list(results[split].values())) + "\n")

    print(format_report(results["test"][best_seed], f"Final model (seed {best_seed}, best on validation) on DS2"))

    plot_confusion(results["test"][best_seed]["confusion_matrix"], OUT_DIR / "confusion_ds2.png",
                   f"DS2 confusion matrix ({args.tag}, seed {best_seed})")
    shutil.copy(run_dir / f"seed{best_seed}" / "model.keras", OUT_DIR / "final_model.keras")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nParameters: {results['params']}")
    print(f"Saved {OUT_DIR / 'results.json'}, {OUT_DIR / 'confusion_ds2.png'}, {OUT_DIR / 'final_model.keras'}")


if __name__ == "__main__":
    main()
