"""Print the validation summary of every training run side by side (validation only, no DS2).

Usage: python scripts/compare_runs.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg import config as C  # noqa: E402

RUNS_DIR = C.ROOT / "outputs" / "runs"


def main():
    cols = ["selection_f1", "macro_f1", "accuracy", "N_Se", "SVEB_Se", "SVEB_+P", "VEB_Se", "VEB_+P", "F_Se", "F_+P"]
    print(f"{'run':<18}" + "".join(f"{c:>16}" for c in cols))
    for path in sorted(RUNS_DIR.glob("*/summary.json")):
        val = json.loads(path.read_text())["val"]
        cells = [f"{100 * val[c]['mean']:6.2f} ± {100 * val[c]['std']:5.2f}" for c in cols]
        print(f"{path.parent.name:<18}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
