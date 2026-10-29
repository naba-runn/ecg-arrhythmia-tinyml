"""Model inputs derived from the processed datasets."""

import numpy as np

from . import config as C

# Named sets of RR inputs. Absolute intervals come first in each set (augmentation rescales them).
#   base:   the original eight inputs.
#   full:   base plus rhythm irregularity.
#   causal: no input that depends on the next beat (post_rr, post/long, pre/post), so a beat can
#           be classified as soon as it ends instead of one beat later.
RR_SETS = {
    "base": ["pre_rr", "post_rr", "local_rr", "long_rr", "pre/long", "post/long", "local/long", "pre/post"],
    "full": ["pre_rr", "post_rr", "local_rr", "long_rr", "pre/long", "post/long", "local/long", "pre/post",
             "rr_cv", "rr_succ_diff"],
    "causal": ["pre_rr", "local_rr", "long_rr", "pre/long", "local/long", "rr_cv", "rr_succ_diff"],
}
N_ABSOLUTE = {"base": 4, "full": 4, "causal": 3}
RR_INPUTS = RR_SETS["base"]


def rr_inputs(rr, rr_set="base"):
    """RR inputs of one named set: absolute intervals, ratios to the patient's own long-term
    rhythm, and rhythm irregularity.

    The ratios make prematurity visible regardless of a patient's baseline heart rate,
    which is what lets SVEB beats generalize across patients.
    """
    pre, post, local, long = (np.clip(c, *C.RR_CLIP_S) for c in rr[:, :4].T)
    raw_pre, raw_post, raw_long = rr[:, 0], rr[:, 1], rr[:, 3]
    ratio = lambda a, b: np.clip(a / b, *C.RR_RATIO_CLIP)
    cols = {
        "pre_rr": pre, "post_rr": post, "local_rr": local, "long_rr": long,
        "pre/long": ratio(raw_pre, raw_long), "post/long": ratio(raw_post, raw_long),
        "local/long": ratio(rr[:, 2], raw_long), "pre/post": ratio(raw_pre, raw_post),
    }
    if rr.shape[1] > 4:
        cols["rr_cv"] = np.clip(rr[:, 4], *C.IRREG_CLIP)
        cols["rr_succ_diff"] = np.clip(rr[:, 5], *C.IRREG_CLIP)
    return np.stack([cols[name] for name in RR_SETS[rr_set]], axis=1).astype(np.float32)


def load_split(name, rr_set="base"):
    """Return (inputs dict, labels, record ids) for one split saved by build_dataset.py."""
    d = np.load(C.PROCESSED_DIR / f"{name}.npz")
    inputs = {"beat": d["beats"][..., None], "rr": rr_inputs(d["rr"], rr_set)}
    return inputs, d["labels"].astype(np.int64), d["records"]
