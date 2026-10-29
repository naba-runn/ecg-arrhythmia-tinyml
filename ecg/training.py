"""Training helpers shared by float training and quantization-aware training."""

import functools

import numpy as np
import tensorflow as tf

from . import config as C
from .augment import augment


def class_weights(labels, power, max_weight):
    """Inverse-frequency weights raised to `power` (1 = fully balanced, 0 = unweighted), then clipped."""
    counts = np.bincount(labels, minlength=len(C.CLASSES)).astype(float)
    w = np.where(counts > 0, len(labels) / (len(C.CLASSES) * np.maximum(counts, 1)), 0.0) ** power
    return np.minimum(w, max_weight)


def make_dataset(x, y, w, batch_size, use_augment, seed, n_abs=4):
    ds = tf.data.Dataset.from_tensor_slices((x, y, w)).shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if use_augment:
        ds = ds.map(functools.partial(augment, n_abs=n_abs), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
