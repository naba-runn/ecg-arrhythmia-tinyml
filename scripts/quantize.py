"""Convert the final model to TFLite: a float32 reference and an INT8 post-training quantized model.

Calibration data for INT8 comes from the training split only.
Outputs outputs/quant/fp32.tflite and outputs/quant/ptq_int8.tflite.

Usage: python scripts/quantize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.tflite_utils import keras_to_tflite, representative_dataset  # noqa: E402

QUANT_DIR = C.ROOT / "outputs" / "quant"


def main():
    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(C.ROOT / "outputs" / "final_model.keras")
    x_tr, _, _ = load_split("train")

    for name, rep in [("fp32", None), ("ptq_int8", representative_dataset(x_tr))]:
        tflite = keras_to_tflite(model, rep)
        path = QUANT_DIR / f"{name}.tflite"
        path.write_bytes(tflite)
        print(f"Saved {path} ({len(tflite)} bytes)")


if __name__ == "__main__":
    main()
