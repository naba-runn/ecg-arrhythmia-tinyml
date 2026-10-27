"""Compare the float model with its INT8 versions: size, MACs, peak activation RAM and accuracy.

Evaluates the Keras float model and every .tflite in outputs/quant on validation and DS2,
and checks the INT8 models against an ESP32-WROOM-32E memory budget.
Writes outputs/quant/report.json.

Usage: python scripts/compare_quant.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras  # noqa: E402
import numpy as np  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.metrics import summarize  # noqa: E402
from ecg.tflite_utils import TFLiteClassifier, count_macs, peak_activation_bytes  # noqa: E402

QUANT_DIR = C.ROOT / "outputs" / "quant"
MODELS = ["fp32", "ptq_int8", "qat_int8"]

# ESP32-WROOM-32E: 520 KB on-chip SRAM, 4 MB flash on the smallest module variant.
ESP32_SRAM = 520 * 1024
ESP32_FLASH = 4 * 1024 * 1024


def weight_bytes(path):
    """Bytes of constant tensors (weights, biases, normalization constants)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    produced = {t for op in interp._get_ops_details() for t in op["outputs"]}
    inputs = {d["index"] for d in interp.get_input_details()}
    return sum(int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize
               for t in interp.get_tensor_details() if t["index"] not in produced | inputs)


def main():
    data = {split: load_split(split) for split in ("val", "test")}
    keras_model = keras.models.load_model(C.ROOT / "outputs" / "final_model.keras")

    report = {"params": keras_model.count_params(), "models": {}}
    predictors = {"keras_fp32": lambda x: keras_model.predict(x, batch_size=1024, verbose=0)}
    for name in MODELS:
        path = QUANT_DIR / f"{name}.tflite"
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        predictors[name] = TFLiteClassifier(path).predict
        report["models"][name] = {
            "file_bytes": path.stat().st_size,
            "weight_bytes": weight_bytes(path),
            "macs": count_macs(path),
            "peak_activation_bytes": peak_activation_bytes(path),
        }

    for name, predict in predictors.items():
        entry = report["models"].setdefault(name, {})
        for split, (x, y, _) in data.items():
            entry[split] = summarize(y, predict(x).argmax(1))

    ref = report["models"]["keras_fp32"]["test"]
    print(f"Parameters: {report['params']}\n")
    print(f"{'model':<11} {'file':>8} {'weights':>8} {'MACs':>8} {'peak RAM':>9}   "
          f"{'DS2 acc':>7} {'Δacc':>6} {'mF1':>6}  {'Se N':>5} {'Se S':>5} {'+P S':>5} {'Se V':>5} {'+P V':>5}")
    for name, m in report["models"].items():
        t = m["test"]
        pc = t["per_class"]
        cost = (f"{m['file_bytes'] / 1024:7.1f}K {m['weight_bytes'] / 1024:7.1f}K {m['macs'] / 1e3:7.1f}k "
                f"{m['peak_activation_bytes'] / 1024:8.1f}K" if "macs" in m else f"{'-':>8} {'-':>8} {'-':>8} {'-':>9}")
        print(f"{name:<11} {cost}   {100 * t['accuracy']:7.2f} {100 * (t['accuracy'] - ref['accuracy']):+6.2f} "
              f"{100 * t['macro_f1']:6.2f}  {100 * pc['N']['Se']:5.1f} {100 * pc['SVEB']['Se']:5.1f} "
              f"{100 * pc['SVEB']['+P']:5.1f} {100 * pc['VEB']['Se']:5.1f} {100 * pc['VEB']['+P']:5.1f}")

    print("\nESP32-WROOM-32E budget (520 KB SRAM, 4 MB flash):")
    for name in ("ptq_int8", "qat_int8"):
        if "macs" not in report["models"].get(name, {}):
            continue
        m = report["models"][name]
        m["sram_fraction"] = m["peak_activation_bytes"] / ESP32_SRAM
        m["flash_fraction"] = m["file_bytes"] / ESP32_FLASH
        print(f"  {name}: activations {100 * m['sram_fraction']:.2f}% of SRAM, model {100 * m['flash_fraction']:.3f}% of flash")

    with open(QUANT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {QUANT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
