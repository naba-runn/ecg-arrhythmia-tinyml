"""Build the train / val / test beat datasets from MIT-BIH.

train = DS1 minus the validation records, val = patient-wise hold-out from DS1, test = DS2.
Outputs data/processed/{train,val,test}.npz and prints per-class beat counts.

Usage: python scripts/build_dataset.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg import config as C  # noqa: E402
from ecg.data import build_split, class_counts  # noqa: E402


def main():
    assert not set(C.DS1) & set(C.DS2), "DS1 and DS2 must be patient-disjoint"
    assert not set(C.PACED) & set(C.DS1 + C.DS2), "paced records must be excluded"

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": C.TRAIN, "val": C.VAL, "test": C.DS2}

    print(f"{'split':<6} {'records':>7} {'beats':>7}  " + "  ".join(f"{c:>5}" for c in C.CLASSES))
    for name, records in splits.items():
        data = build_split(records)
        np.savez_compressed(C.PROCESSED_DIR / f"{name}.npz", **data)
        counts = class_counts(data["labels"])
        print(f"{name:<6} {len(records):>7} {len(data['labels']):>7}  "
              + "  ".join(f"{counts[c]:>5}" for c in C.CLASSES))

    print(f"\nBeat window: {C.BEAT_LEN} samples, RR features: {', '.join(C.RR_FEATURES)}")
    print(f"Saved to {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
