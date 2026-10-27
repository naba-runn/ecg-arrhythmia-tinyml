"""TFLite conversion, batched inference and on-device cost estimates (MACs, peak activation RAM)."""

import numpy as np
import tensorflow as tf

INPUT_NAMES = ("beat", "rr")


def representative_dataset(inputs, n=2000, seed=0):
    """Calibration samples for INT8 quantization, drawn from the training split only."""
    idx = np.random.default_rng(seed).choice(len(inputs["rr"]), size=n, replace=False)

    def gen():
        for i in idx:
            yield {k: inputs[k][i:i + 1] for k in INPUT_NAMES}
    return gen


def convert(converter, rep_data=None):
    """Convert to TFLite: float32 if rep_data is None, otherwise full-integer INT8 (inputs and outputs too)."""
    if rep_data is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


class TFLiteClassifier:
    """Runs a .tflite model on whole arrays, quantizing inputs and dequantizing outputs when INT8."""

    def __init__(self, path, batch_size=1024):
        self.path, self.batch_size = path, batch_size

    def _interpreter(self, n):
        interp = tf.lite.Interpreter(model_path=str(self.path), num_threads=4)
        for d in interp.get_input_details():
            shape = list(d["shape"])
            shape[0] = n
            interp.resize_tensor_input(d["index"], shape)
        interp.allocate_tensors()
        return interp

    def predict(self, inputs):
        n_total = len(inputs["rr"])
        out = []
        for start in range(0, n_total, self.batch_size):
            batch = {k: inputs[k][start:start + self.batch_size] for k in INPUT_NAMES}
            interp = self._interpreter(len(batch["rr"]))
            for d in interp.get_input_details():
                name = next(k for k in INPUT_NAMES if k in d["name"])
                x = batch[name]
                if d["dtype"] == np.int8:
                    scale, zero = d["quantization"]
                    x = np.clip(np.round(x / scale + zero), -128, 127)
                interp.set_tensor(d["index"], x.astype(d["dtype"]))
            interp.invoke()
            od = interp.get_output_details()[0]
            y = interp.get_tensor(od["index"]).astype(np.float32)
            if od["dtype"] == np.int8:
                scale, zero = od["quantization"]
                y = (y - zero) * scale
            out.append(y)
        return np.concatenate(out)


def _graph(path):
    # Without the default XNNPACK delegate, so the op list is the plain builtin graph a microcontroller runs.
    interp = tf.lite.Interpreter(model_path=str(path),
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp.allocate_tensors()
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    ops = interp._get_ops_details()
    return interp, tensors, ops


def count_macs(path):
    """Multiply-accumulates per inference for the conv and fully-connected layers (batch size 1)."""
    _, tensors, ops = _graph(path)
    macs = 0
    for op in ops:
        if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            out_shape = tensors[op["outputs"][0]]["shape"]
            filt = tensors[op["inputs"][1]]["shape"]  # (out_ch, kh, kw, in_ch)
            per_output = np.prod(filt[1:]) if op["op_name"] == "CONV_2D" else np.prod(filt[1:3])
            macs += int(np.prod(out_shape) * per_output)
        elif op["op_name"] == "FULLY_CONNECTED":
            w = tensors[op["inputs"][1]]["shape"]  # (units, in)
            macs += int(np.prod(w))
    return macs


def peak_activation_bytes(path):
    """Peak memory of simultaneously live activation tensors, in execution order.

    A tensor is live from the op that produces it (or from the start, for model inputs)
    until the last op that reads it. Weights are excluded: on a microcontroller they stay in
    flash. This is a lower bound on the TFLite Micro tensor arena, which also adds alignment
    padding and interpreter bookkeeping.
    """
    interp, tensors, ops = _graph(path)
    inputs = {d["index"] for d in interp.get_input_details()}
    outputs = {d["index"] for d in interp.get_output_details()}
    produced = {t for op in ops for t in op["outputs"]}
    activations = inputs | produced

    first = {t: 0 for t in inputs}
    last = {t: len(ops) - 1 for t in outputs}
    for i, op in enumerate(ops):
        for t in op["outputs"]:
            first.setdefault(t, i)
        for t in op["inputs"]:
            if t in activations:
                last[t] = max(last.get(t, i), i)

    def nbytes(t):
        d = tensors[t]
        return int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize

    return max(sum(nbytes(t) for t in activations if first[t] <= i <= last.get(t, first[t]))
               for i in range(len(ops)))
