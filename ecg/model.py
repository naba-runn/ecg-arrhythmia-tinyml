"""Lightweight 1D-CNN: beat morphology branch + RR-interval branch, five AAMI outputs."""

import keras

from . import config as C


def build_model(rr_train, n_rr, filters=(8, 16, 32), kernels=(7, 5, 5), dense_units=32, dropout=0.2,
                global_max=False):
    """Build the classifier.

    rr_train is used to fit the RR normalization layer, so the mean/std are stored in
    the model itself and the same preprocessing ships with it to the device.
    global_max=True summarizes each conv filter by its maximum as well as its average, which
    keeps short sharp features that averaging over the whole beat blurs out.
    """
    layers = keras.layers
    beat_in = keras.Input(shape=(C.BEAT_LEN, 1), name="beat")
    rr_in = keras.Input(shape=(n_rr,), name="rr")

    x = beat_in
    for f, k in zip(filters, kernels):
        x = layers.Conv1D(f, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    if global_max:
        x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    else:
        x = layers.GlobalAveragePooling1D()(x)

    rr_norm = layers.Normalization(name="rr_norm")
    rr_norm.adapt(rr_train)
    r = rr_norm(rr_in)

    h = layers.Concatenate()([x, r])
    h = layers.Dense(dense_units, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(len(C.CLASSES), activation="softmax", name="aami")(h)
    return keras.Model({"beat": beat_in, "rr": rr_in}, out, name="ecg_1dcnn")
