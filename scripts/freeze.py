"""Freeze the chosen INT8 ensemble as the final model.

Selection rule, fixed before the candidates' validation results were seen: among the candidate
runs, the INT8 ensemble with the highest validation F1 over N, SVEB and VEB wins. Candidates
within 0.5 points of the best are treated as tied; ties prefer a causal RR set (no wait for the
next beat), then fewer weights. Only validation results are read here.

Copies the ensemble members to outputs/final/ and writes outputs/final/manifest.json with the
preprocessing settings, decision rule, SHA-256 of every model file and the selection evidence.
Requires `ensemble.py --tag <run> --val-only` to have been run for every candidate.

Usage: python scripts/freeze.py --candidates w1.0_aug_clip full_gmp causal_gmp
"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg import config as C  # noqa: E402
from ecg.features import RR_SETS  # noqa: E402

FINAL_DIR = C.ROOT / "outputs" / "final"
TIE_MARGIN = 0.005


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--force", action="store_true", help="replace an existing frozen model")
    args = p.parse_args()

    if (FINAL_DIR / "manifest.json").exists() and not args.force:
        sys.exit(f"{FINAL_DIR} already holds a frozen model; pass --force to replace it")

    evidence = []
    for tag in args.candidates:
        report = json.loads((C.ROOT / "outputs" / "ensemble" / tag / "report.json").read_text())
        train_args = json.loads((C.ROOT / "outputs" / "runs" / tag / "summary.json").read_text())["args"]
        val = report["results"]["val_int8"]["ensemble"]
        evidence.append({
            "tag": tag,
            "rr_set": train_args.get("rr_set", "base"),
            "global_max": train_args.get("global_max", False),
            "val_int8_ensemble_selection_f1": val["selection_f1"],
            "val_int8_ensemble_macro_f1": val["macro_f1"],
            "val_SVEB_Se": val["per_class"]["SVEB"]["Se"],
            "val_SVEB_+P": val["per_class"]["SVEB"]["+P"],
            "weight_bytes": report["cost"]["ensemble"]["weight_bytes"],
        })

    best = max(e["val_int8_ensemble_selection_f1"] for e in evidence)
    tied = [e for e in evidence if e["val_int8_ensemble_selection_f1"] >= best - TIE_MARGIN]
    chosen = min(tied, key=lambda e: (e["rr_set"] != "causal", e["weight_bytes"]))

    print(f"{'candidate':<16} {'RR set':<7} {'max-pool':<8} {'val F1(N,S,V)':>13} {'SVEB Se':>8} {'SVEB +P':>8} {'weights':>8}")
    for e in evidence:
        mark = "  <- chosen" if e is chosen else ("  (tied)" if e in tied else "")
        print(f"{e['tag']:<16} {e['rr_set']:<7} {str(e['global_max']):<8} "
              f"{100 * e['val_int8_ensemble_selection_f1']:13.2f} {100 * e['val_SVEB_Se']:8.2f} "
              f"{100 * e['val_SVEB_+P']:8.2f} {e['weight_bytes'] / 1024:7.1f}K{mark}")

    tag = chosen["tag"]
    train_summary = json.loads((C.ROOT / "outputs" / "runs" / tag / "summary.json").read_text())
    src_dir = C.ROOT / "outputs" / "ensemble" / tag
    if FINAL_DIR.exists():
        shutil.rmtree(FINAL_DIR)
    FINAL_DIR.mkdir(parents=True)
    members = []
    for seed in train_summary["args"]["seeds"]:
        name = f"member_seed{seed}_int8.tflite"
        shutil.copy(src_dir / f"seed{seed}_int8.tflite", FINAL_DIR / name)
        members.append({"file": name, "seed": seed, "sha256": sha256(FINAL_DIR / name)})

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()
    rr_set = chosen["rr_set"]
    manifest = {
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_commit": commit,
        "run_tag": tag,
        "members": members,
        "decision_rule": "average the dequantized softmax outputs of all members, then argmax",
        "classes": C.CLASSES,
        "inputs": {
            "beat": {"lead": C.LEAD, "fs_hz": C.FS, "bandpass_hz": list(C.BANDPASS_HZ),
                     "samples_before_r": C.PRE_SAMPLES, "samples_after_r": C.POST_SAMPLES,
                     "normalization": "per-beat z-score"},
            "rr": {"set": rr_set, "features": RR_SETS[rr_set], "rr_clip_s": list(C.RR_CLIP_S),
                   "ratio_clip": list(C.RR_RATIO_CLIP), "irregularity_clip": list(C.IRREG_CLIP),
                   "local_rr_beats": C.LOCAL_RR_BEATS, "long_rr_seconds": C.LONG_RR_SECONDS,
                   "irregularity_rr_beats": C.IRREG_RR_BEATS},
        },
        "training": train_summary["args"],
        "selection": {"rule": __doc__.split("\n\n")[1].replace("\n", " "), "candidates": evidence},
    }
    with open(FINAL_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nFroze {tag} ({len(members)} members) to {FINAL_DIR}")


if __name__ == "__main__":
    main()
