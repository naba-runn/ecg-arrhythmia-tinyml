"""Final DS2 evaluation of the frozen model in outputs/final/.

Verifies every model file against the manifest's SHA-256, runs the INT8 members on DS2 with the
manifest's decision rule, and reports per-class Se, +P and FPR. It runs once: if results already
exist it refuses to run again unless --force is given, so the test set cannot quietly be reused.

Ablation runs (for example the causal-RR variant) can be reported in the same pass with
--ablations; they are labelled as such and play no part in the model choice.

Outputs outputs/final/ds2_results.json and outputs/final/confusion_ds2.png.

Usage: python scripts/final_eval.py [--ablations causal_gmp]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.metrics import format_report, summarize  # noqa: E402
from ecg.tflite_utils import TFLiteClassifier, count_macs, peak_activation_bytes, weight_bytes  # noqa: E402

FINAL_DIR = C.ROOT / "outputs" / "final"


def ensemble_predict(paths, x):
    return np.mean([TFLiteClassifier(p).predict(x) for p in paths], axis=0).argmax(1)


def cost(paths):
    return {"members": len(paths),
            "file_bytes": sum(p.stat().st_size for p in paths),
            "weight_bytes": sum(weight_bytes(p) for p in paths),
            "macs": sum(count_macs(p) for p in paths),
            "peak_activation_bytes": max(peak_activation_bytes(p) for p in paths) + len(paths) * len(C.CLASSES)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = FINAL_DIR / "ds2_results.json"
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists: the frozen model has already been evaluated on DS2")

    manifest = json.loads((FINAL_DIR / "manifest.json").read_text())
    paths = []
    for m in manifest["members"]:
        path = FINAL_DIR / m["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sha256"], f"{path} does not match the manifest"
        paths.append(path)
    print(f"Verified {len(paths)} frozen members of {manifest['run_tag']} against the manifest")

    from evaluate import plot_confusion  # scripts/ is on sys.path when run as a script

    x, y, _ = load_split("test", manifest["inputs"]["rr"]["set"])
    final = summarize(y, ensemble_predict(paths, x))
    results = {"model": manifest["run_tag"], "cost": cost(paths), "ds2": final, "ablations": {}}
    print("\n" + format_report(final, f"FINAL: frozen INT8 ensemble ({manifest['run_tag']}) on DS2"))
    plot_confusion(final["confusion_matrix"], FINAL_DIR / "confusion_ds2.png",
                   f"DS2, frozen INT8 ensemble ({manifest['run_tag']})")

    for tag in args.ablations:
        train_args = json.loads((C.ROOT / "outputs" / "runs" / tag / "summary.json").read_text())["args"]
        member_paths = [C.ROOT / "outputs" / "ensemble" / tag / f"seed{s}_int8.tflite" for s in train_args["seeds"]]
        xa, ya, _ = load_split("test", train_args.get("rr_set", "base"))
        s = summarize(ya, ensemble_predict(member_paths, xa))
        results["ablations"][tag] = {"rr_set": train_args.get("rr_set", "base"), "cost": cost(member_paths), "ds2": s}
        print("\n" + format_report(s, f"ABLATION (not used for selection): {tag} INT8 ensemble on DS2"))

    c = results["cost"]
    print(f"\nFrozen model cost: {c['members']} members, weights {c['weight_bytes'] / 1024:.1f} KB, "
          f"files {c['file_bytes'] / 1024:.1f} KB, {c['macs'] / 1e3:.0f}k MACs per beat, "
          f"peak activation RAM {c['peak_activation_bytes'] / 1024:.1f} KB")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path} and {FINAL_DIR / 'confusion_ds2.png'}")


if __name__ == "__main__":
    main()
