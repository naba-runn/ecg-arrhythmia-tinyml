"""Quantization-aware fine-tuning of the final model, then conversion to a full-integer INT8 TFLite model.

The TF Model Optimization toolkit needs tf_keras (Keras 2) and has no Conv1D support, so the final
model is mirrored as the graph TFLite actually executes: each Conv1D + BatchNorm becomes a single
Conv2D with a (1 x k) kernel and the BatchNorm folded into its weights and bias. Fake quantization
during training therefore matches deployment. The model is fine-tuned on the training split with
the same class weights and augmentation as before; the best epoch is chosen on the validation
hold-out. DS2 is never loaded here.

Output: outputs/quant/qat_int8.tflite

Usage: python scripts/qat.py [--epochs 15] [--lr 1e-4]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_model_optimization as tfmot  # noqa: E402
import tf_keras  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import load_split  # noqa: E402
from ecg.metrics import format_report, summarize  # noqa: E402
from ecg.tflite_utils import convert, representative_dataset  # noqa: E402
from ecg.training import class_weights, make_dataset  # noqa: E402

QUANT_DIR = C.ROOT / "outputs" / "quant"


def fold_batch_norm(conv, bn):
    """Return (kernel, bias) of a conv with the following BatchNorm folded in."""
    gamma, beta, mean, var = bn.get_weights()
    scale = gamma / np.sqrt(var + bn.epsilon)
    (kernel,) = conv.get_weights()
    return kernel * scale, beta - mean * scale


def deployment_mirror(src):
    """Rebuild `src` in tf_keras as folded (1 x k) Conv2D layers and copy its weights over."""
    L = tf_keras.layers
    convs = [l for l in src.layers if isinstance(l, keras.layers.Conv1D)]
    bns = [l for l in src.layers if isinstance(l, keras.layers.BatchNormalization)]
    dense = [l for l in src.layers if isinstance(l, keras.layers.Dense)]
    dropout = next(l for l in src.layers if isinstance(l, keras.layers.Dropout))
    src_norm = src.get_layer("rr_norm")
    n_rr = src_norm.get_weights()[0].shape[-1]

    beat_in = tf_keras.Input(shape=(C.BEAT_LEN, 1), name="beat")
    rr_in = tf_keras.Input(shape=(n_rr,), name="rr")
    x = L.Reshape((1, C.BEAT_LEN, 1))(beat_in)
    new_convs = []
    for conv in convs:
        layer = L.Conv2D(conv.filters, (1, conv.kernel_size[0]), padding="same")
        new_convs.append(layer)
        x = L.MaxPooling2D((1, 2))(L.ReLU()(layer(x)))
    x = L.GlobalAveragePooling2D()(x)

    # Mean/variance go in as constructor constants: tf_keras caches adapted statistics at build
    # time, so ones restored with set_weights are lost when quantize_apply clones the model.
    mean, var, _ = src_norm.get_weights()
    norm = L.Normalization(mean=mean.ravel(), variance=var.ravel(), name="rr_norm")
    h = L.Concatenate()([x, norm(rr_in)])
    new_dense = [L.Dense(dense[0].units, activation="relu")]
    h = L.Dropout(dropout.rate)(new_dense[0](h))
    new_dense.append(L.Dense(dense[1].units, activation="softmax", name="aami"))
    dst = tf_keras.Model({"beat": beat_in, "rr": rr_in}, new_dense[1](h), name="ecg_1dcnn_qat")

    for conv, bn, layer in zip(convs, bns, new_convs):
        kernel, bias = fold_batch_norm(conv, bn)
        layer.set_weights([kernel[None], bias])
    for a, b in zip(dense, new_dense):
        b.set_weights(a.get_weights())
    return dst


def calibrate_ranges(model, x, steps, batch_size=256, seed=0):
    """Settle the fake-quant activation ranges before fine-tuning.

    The toolkit starts every range at [-6, 6] and only moves it by a slow moving average
    (decay 0.999) on training-mode forward passes, so without this the first epochs train
    against badly wrong ranges. Forward passes here update ranges but not weights.
    """
    ds = (tf.data.Dataset.from_tensor_slices(x).shuffle(len(x["rr"]), seed=seed).repeat()
          .batch(batch_size).take(steps))
    forward = tf.function(lambda b: model(b, training=True))
    for batch in ds:
        forward(batch)


def annotate(layer):
    """Fake-quantize the layers TFLite runs in INT8 kernels; the toolkit doesn't support Normalization."""
    if isinstance(layer, (tf_keras.layers.Conv2D, tf_keras.layers.Dense, tf_keras.layers.ReLU)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer.__class__.from_config(layer.get_config())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration-steps", type=int, default=5000)
    args = p.parse_args()

    run = json.loads((C.ROOT / "outputs" / "results.json").read_text())
    train_args = json.loads((C.ROOT / "outputs" / "runs" / run["tag"] / "summary.json").read_text())["args"]

    tf_keras.utils.set_random_seed(args.seed)
    x_tr, y_tr, _ = load_split("train")
    x_va, y_va, _ = load_split("val")

    src = keras.models.load_model(C.ROOT / "outputs" / "final_model.keras")
    float_model = deployment_mirror(src)
    diff = np.abs(src.predict(x_va, batch_size=1024, verbose=0) - float_model.predict(x_va, batch_size=1024, verbose=0))
    assert diff.max() < 1e-4, f"folded mirror disagrees with the original model ({diff.max():.2e})"

    annotated = tf_keras.models.clone_model(float_model, clone_function=annotate)
    annotated.set_weights(float_model.get_weights())
    model = tfmot.quantization.keras.quantize_apply(annotated)
    calibrate_ranges(model, x_tr, args.calibration_steps, seed=args.seed)
    model.compile(optimizer=tf_keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy")

    weights = class_weights(y_tr, train_args["weight_power"], train_args["max_class_weight"])
    ds = make_dataset(x_tr, y_tr, weights[y_tr].astype("float32"), args.batch_size, train_args["augment"], args.seed)

    def val_summary():
        return summarize(y_va, model.predict(x_va, batch_size=1024, verbose=0).argmax(1))

    s = val_summary()
    best_score, best_epoch, best_weights = s["selection_f1"], 0, model.get_weights()
    print(f"epoch   0 (fake-quant, before fine-tuning)  val F1(N,S,V) {100 * best_score:5.2f}")
    for epoch in range(1, args.epochs + 1):
        loss = model.fit(ds, epochs=1, verbose=0).history["loss"][0]
        s = val_summary()
        marker = ""
        if s["selection_f1"] > best_score:
            best_score, best_epoch, best_weights = s["selection_f1"], epoch, model.get_weights()
            marker = "  *"
        print(f"epoch {epoch:3d}  loss {loss:.4f}  val F1(N,S,V) {100 * s['selection_f1']:5.2f}  "
              + "  ".join(f"Se_{c} {100 * s['per_class'][c]['Se']:5.1f}" for c in C.CLASSES[:4]) + marker, flush=True)

    model.set_weights(best_weights)
    print("\n" + format_report(val_summary(), f"QAT model, validation (best epoch {best_epoch})"))

    QUANT_DIR.mkdir(parents=True, exist_ok=True)
    tflite = convert(tf.lite.TFLiteConverter.from_keras_model(model), representative_dataset(x_tr))
    path = QUANT_DIR / "qat_int8.tflite"
    path.write_bytes(tflite)
    print(f"Saved {path} ({len(tflite)} bytes)")


if __name__ == "__main__":
    main()
