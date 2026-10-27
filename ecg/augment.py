"""Training-time augmentation, applied per batch inside the tf.data pipeline."""

import tensorflow as tf

from . import config as C


def augment(inputs, y, w, scale=(0.8, 1.2), noise_std=0.05, drift=0.2, max_shift=5, hr_scale=(0.85, 1.15)):
    beat, rr = inputs["beat"], inputs["rr"]
    b = tf.shape(beat)[0]

    # Amplitude scaling, additive noise, and a random linear baseline drift.
    beat = beat * tf.random.uniform([b, 1, 1], *scale)
    beat = beat + tf.random.normal(tf.shape(beat), stddev=noise_std)
    ramp = tf.linspace(-1.0, 1.0, C.BEAT_LEN)[None, :, None]
    beat = beat + ramp * tf.random.uniform([b, 1, 1], -drift, drift) + tf.random.uniform([b, 1, 1], -drift, drift)

    # Small time shift, mimicking R-peak detector jitter (edge samples are repeated).
    shift = tf.random.uniform([b], -max_shift, max_shift + 1, dtype=tf.int32)
    idx = tf.clip_by_value(tf.range(C.BEAT_LEN)[None, :] + shift[:, None], 0, C.BEAT_LEN - 1)
    beat = tf.gather(beat, idx, batch_dims=1)

    # Heart-rate change: scale the four absolute RR intervals together; the ratio features are unchanged.
    factor = tf.random.uniform([b, 1], *hr_scale)
    rr = tf.concat([tf.clip_by_value(rr[:, :4] * factor, *C.RR_CLIP_S), rr[:, 4:]], axis=1)

    return {"beat": beat, "rr": rr}, y, w
