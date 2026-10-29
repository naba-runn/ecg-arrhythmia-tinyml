"""Tune the SVEB decision threshold of the ensemble on the validation hold-out.

Decision rule: a beat is called SVEB only if the ensemble's SVEB probability is at least t;
otherwise it gets the most likely of the remaining classes. t = 0 reproduces plain argmax.
t is chosen on validation by the same score used for model selection (macro-F1 over N, SVEB,
VEB), separately for the float32 and INT8 ensembles, then applied unchanged to DS2.

Reads outputs/ensemble/<tag>/probs.npz (from ensemble.py). Writes threshold.json and
sveb_threshold_val.png to the same folder.

Usage: python scripts/tune_threshold.py [--tag <run>]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.metrics import format_report, summarize  # noqa: E402

OUT_DIR = C.ROOT / "outputs" / "ensemble"
SVEB = C.CLASSES.index("SVEB")
THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.01), 2)


def decide(probs, t):
    pred = probs.argmax(1)
    others = probs.copy()
    others[:, SVEB] = -1.0
    demote = (pred == SVEB) & (probs[:, SVEB] < t)
    pred[demote] = others[demote].argmax(1)
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", help="training run (default: the run evaluated in outputs/results.json)")
    args = p.parse_args()
    tag = args.tag or json.loads((C.ROOT / "outputs" / "results.json").read_text())["tag"]
    out_dir = OUT_DIR / tag

    probs = np.load(out_dir / "probs.npz")
    y = {split: load_split(split)[1] for split in ("val", "test")}
    out = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, kind in zip(axes, ("fp32", "int8")):
        sweep = [summarize(y["val"], decide(probs[f"{kind}_val"], t)) for t in THRESHOLDS]
        scores = np.array([s["selection_f1"] for s in sweep])
        best = int(np.flatnonzero(scores == scores.max())[0])  # lowest t among ties
        t = float(THRESHOLDS[best])

        print(f"\n{kind} ensemble, validation sweep (every 0.05)")
        print(f"{'t':>5} {'F1(N,S,V)':>10} {'Se SVEB':>8} {'+P SVEB':>8} {'F1 SVEB':>8} {'FPR N':>7}")
        for i in range(0, len(THRESHOLDS), 5):
            pc = sweep[i]["per_class"]
            print(f"{THRESHOLDS[i]:5.2f} {100 * scores[i]:10.2f} {100 * pc['SVEB']['Se']:8.2f} "
                  f"{100 * pc['SVEB']['+P']:8.2f} {100 * pc['SVEB']['F1']:8.2f} {100 * pc['N']['FPR']:7.2f}")
        print(f"chosen t = {t:.2f} (val F1(N,S,V) {100 * scores[best]:.2f}, argmax gives {100 * scores[0]:.2f})")

        test_argmax = summarize(y["test"], decide(probs[f"{kind}_test"], 0.0))
        test_tuned = summarize(y["test"], decide(probs[f"{kind}_test"], t))
        out[kind] = {"threshold": t, "val_argmax": sweep[0], "val_tuned": sweep[best],
                     "test_argmax": test_argmax, "test_tuned": test_tuned}

        se = [100 * s["per_class"]["SVEB"]["Se"] for s in sweep]
        ppv = [100 * s["per_class"]["SVEB"]["+P"] for s in sweep]
        ax.plot(THRESHOLDS, se, label="SVEB Se")
        ax.plot(THRESHOLDS, ppv, label="SVEB +P")
        ax.plot(THRESHOLDS, 100 * scores, label="F1 (N, SVEB, VEB)")
        ax.axvline(t, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{kind.upper()} ensemble, validation (t = {t:.2f})")
        ax.set_xlabel("SVEB threshold t")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir /"sveb_threshold_val.png", dpi=200)

    print("\nDS2 (threshold fixed on validation)")
    print(f"{'':<16} {'acc':>6} {'mF1':>6} {'Se N':>6} {'FPR N':>6} {'Se S':>6} {'+P S':>6} {'F1 S':>6} {'Se V':>6} {'+P V':>6}")
    for kind in ("fp32", "int8"):
        for label, s in [("argmax", out[kind]["test_argmax"]), (f"t={out[kind]['threshold']:.2f}", out[kind]["test_tuned"])]:
            pc = s["per_class"]
            print(f"{kind + ' ' + label:<16} {100 * s['accuracy']:6.2f} {100 * s['macro_f1']:6.2f} "
                  f"{100 * pc['N']['Se']:6.2f} {100 * pc['N']['FPR']:6.2f} {100 * pc['SVEB']['Se']:6.2f} "
                  f"{100 * pc['SVEB']['+P']:6.2f} {100 * pc['SVEB']['F1']:6.2f} {100 * pc['VEB']['Se']:6.2f} "
                  f"{100 * pc['VEB']['+P']:6.2f}")

    print("\n" + format_report(out["int8"]["test_tuned"], f"Final: INT8 ensemble, t = {out['int8']['threshold']:.2f}, DS2"))
    with open(out_dir /"threshold.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_dir /'threshold.json'} and {out_dir /'sveb_threshold_val.png'}")


if __name__ == "__main__":
    main()
