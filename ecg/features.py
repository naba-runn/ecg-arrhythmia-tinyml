"""Model inputs derived from the processed datasets."""

import numpy as np

from . import config as C

RR_INPUTS = [
    "pre_rr", "post_rr", "local_rr", "long_rr",
    "pre/long", "post/long", "local/long", "pre/post",
]


def rr_inputs(rr):
    """Absolute RR intervals plus ratios to the patient's own long-term rhythm.

    The ratios make prematurity visible regardless of a patient's baseline heart rate,
    which is what lets SVEB beats generalize across patients.
    """
    pre, post, local, long = rr.T
    absolute = np.clip(np.stack([pre, post, local, long], axis=1), *C.RR_CLIP_S)
    ratios = np.clip(np.stack([pre / long, post / long, local / long, pre / post], axis=1), *C.RR_RATIO_CLIP)
    return np.concatenate([absolute, ratios], axis=1).astype(np.float32)


def load_split(name):
    """Return (inputs dict, labels, record ids) for one split saved by build_dataset.py."""
    d = np.load(C.PROCESSED_DIR / f"{name}.npz")
    inputs = {"beat": d["beats"][..., None], "rr": rr_inputs(d["rr"])}
    return inputs, d["labels"].astype(np.int64), d["records"]
