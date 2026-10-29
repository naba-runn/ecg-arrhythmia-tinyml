"""Ensemble the seeds of the chosen run by averaging their softmax outputs, in float32 and INT8.

Each seed model is post-training quantized to INT8 on its own (calibrated on the training split).
On a microcontroller the members run one after another in the same tensor arena, so peak
activation RAM is that of one member while weights and MACs add up.

Reports single-model results (mean ± std over seeds) next to the ensemble, on validation and DS2.
Outputs go to outputs/ensemble/<tag>/: seed<k>_int8.tflite, report.json and the ensemble's
averaged class probabilities in probs.npz (used by tune_threshold.py).

Usage: python scripts/ensemble.py [--tag <run>] [--val-only]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras  # noqa: E402
import numpy as np  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.metrics import format_report, summarize  # noqa: E402
from ecg.tflite_utils import (  # noqa: E402
    TFLiteClassifier, count_macs, keras_to_tflite, peak_activation_bytes, representative_dataset, weight_bytes,
)

OUT_DIR = C.ROOT / "outputs" / "ensemble"
ROW_METRICS = [("accuracy", None, None), ("macro_f1", None, None), ("Se", "N", None), ("Se", "SVEB", None),
               ("+P", "SVEB", None), ("Se", "VEB", None), ("+P", "VEB", None), ("FPR", "N", None)]


def metric(s, key, cls):
    return s[key] if cls is None else s["per_class"][cls][key]


def header():
    return f"{'':<22}" + "".join(f"{(k if c is None else f'{k} {c}'):>15}" for k, c, _ in ROW_METRICS)


def row(label, summaries):
    cells = []
    for key, cls, _ in ROW_METRICS:
        v = np.array([metric(s, key, cls) for s in summaries]) * 100
        cells.append(f"{v.mean():6.2f} ± {v.std():5.2f}" if len(v) > 1 else f"{v[0]:6.2f}")
    return f"{label:<22}" + "".join(f"{c:>15}" for c in cells)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", help="training run (default: the run evaluated in outputs/results.json)")
    p.add_argument("--val-only", action="store_true", help="skip DS2, for choosing between runs")
    args = p.parse_args()

    tag = args.tag or json.loads((C.ROOT / "outputs" / "results.json").read_text())["tag"]
    train_args = json.loads((C.ROOT / "outputs" / "runs" / tag / "summary.json").read_text())["args"]
    seeds, rr_set = train_args["seeds"], train_args.get("rr_set", "base")
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tr, _, _ = load_split("train", rr_set)
    splits = ("val",) if args.val_only else ("val", "test")
    data = {split: load_split(split, rr_set) for split in splits}

    probs = {(kind, split): [] for kind in ("fp32", "int8") for split in data}
    costs = []
    for seed in seeds:
        model = keras.models.load_model(C.ROOT / "outputs" / "runs" / tag / f"seed{seed}" / "model.keras")
        path = out_dir / f"seed{seed}_int8.tflite"
        path.write_bytes(keras_to_tflite(model, representative_dataset(x_tr)))
        costs.append({"file_bytes": path.stat().st_size, "weight_bytes": weight_bytes(path),
                      "macs": count_macs(path), "peak_activation_bytes": peak_activation_bytes(path)})
        int8 = TFLiteClassifier(path)
        for split, (x, _, _) in data.items():
            probs["fp32", split].append(model.predict(x, batch_size=1024, verbose=0))
            probs["int8", split].append(int8.predict(x))

    report = {"tag": tag, "seeds": seeds, "rr_set": rr_set, "results": {}}
    titles = {"val": "Validation (DS1 hold-out)", "test": "Test (DS2, inter-patient)"}
    for split in splits:
        title = titles[split]
        y = data[split][1]
        print(f"\n{title}\n{header()}")
        for kind in ("fp32", "int8"):
            singles = [summarize(y, p.argmax(1)) for p in probs[kind, split]]
            ens = summarize(y, np.mean(probs[kind, split], axis=0).argmax(1))
            report["results"][f"{split}_{kind}"] = {"single": singles, "ensemble": ens}
            print(row(f"{kind} single (n={len(seeds)})", singles))
            print(row(f"{kind} ensemble", [ens]))

    last = splits[-1]
    print("\n" + format_report(report["results"][f"{last}_int8"]["ensemble"], f"INT8 ensemble, {titles[last]}"))

    total = {
        "members": len(seeds),
        "file_bytes": sum(c["file_bytes"] for c in costs),
        "weight_bytes": sum(c["weight_bytes"] for c in costs),
        "macs": sum(c["macs"] for c in costs),
        # Members run sequentially and reuse one arena; add 3 x 5 bytes to keep their outputs.
        "peak_activation_bytes": max(c["peak_activation_bytes"] for c in costs) + len(seeds) * len(C.CLASSES),
    }
    report["cost"] = {"per_member": costs, "ensemble": total}
    print(f"\nINT8 ensemble cost: {total['members']} members, files {total['file_bytes'] / 1024:.1f} KB, "
          f"weights {total['weight_bytes'] / 1024:.1f} KB, {total['macs'] / 1e3:.0f}k MACs per beat, "
          f"peak activation RAM {total['peak_activation_bytes'] / 1024:.1f} KB")

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    np.savez(out_dir / "probs.npz", **{f"{kind}_{split}": np.mean(p, axis=0) for (kind, split), p in probs.items()})
    print(f"Saved {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
