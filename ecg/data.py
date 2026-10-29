"""Load MIT-BIH records, filter the signal, segment beats and compute RR-interval features."""

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

from . import config as C


def load_record(record_id):
    """Return the MLII signal and all annotations of one record."""
    path = str(C.RAW_DIR / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    # MLII is channel 0 in every record except 114, so look it up by name.
    lead = rec.sig_name.index(C.LEAD)
    return rec.p_signal[:, lead], np.asarray(ann.sample), np.asarray(ann.symbol)


def bandpass(signal, fs=C.FS, band=C.BANDPASS_HZ, order=4):
    """Zero-phase Butterworth band-pass to remove baseline wander and high-frequency noise."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def rr_features(r_peaks, fs=C.FS):
    """Per-beat RR features, using only past beats except for post_rr.

    Columns follow C.RR_FEATURES: four intervals in seconds, then two rhythm-irregularity
    measures over the C.IRREG_RR_BEATS intervals before the current one (the current beat's own
    pre_rr is left out, so a single premature beat does not count as an irregular rhythm).
    Rows for the first and last beat are NaN because pre_rr / post_rr are undefined there.
    """
    t = r_peaks / fs
    rr = np.diff(t)
    n = len(t)
    feats = np.full((n, len(C.RR_FEATURES)), np.nan)
    for i in range(1, n - 1):
        pre = rr[i - 1]
        post = rr[i]
        past = rr[:i]
        local = past[-C.LOCAL_RR_BEATS:].mean()
        long_window = past[t[1:i + 1] >= t[i] - C.LONG_RR_SECONDS]

        # Clipped so annotation gaps don't register as irregularity.
        context = np.clip(past[-(C.IRREG_RR_BEATS + 1):-1], *C.RR_CLIP_S)
        if len(context) >= 2:
            cv = context.std() / context.mean()
            succ_diff = np.abs(np.diff(context)).mean() / context.mean()
        else:
            cv = succ_diff = 0.0
        feats[i] = (pre, post, local, long_window.mean(), cv, succ_diff)
    return feats


def extract_beats(record_id):
    """Segment all AAMI-classifiable beats of one record.

    Returns a dict with beats (n, BEAT_LEN), rr (n, len(C.RR_FEATURES)), labels (n,) and the
    original symbols.
    """
    signal, samples, symbols = load_record(record_id)
    signal = bandpass(signal)

    # RR intervals are computed over every beat annotation, not just the ones we keep.
    is_beat = np.isin(symbols, list(C.AAMI_MAP))
    r_peaks, beat_symbols = samples[is_beat], symbols[is_beat]
    rr = rr_features(r_peaks)

    beats, rr_out, labels, kept_symbols = [], [], [], []
    for i, (r, sym) in enumerate(zip(r_peaks, beat_symbols)):
        start, end = r - C.PRE_SAMPLES, r + C.POST_SAMPLES
        if start < 0 or end > len(signal) or np.isnan(rr[i]).any():
            continue
        beat = signal[start:end]
        beats.append((beat - beat.mean()) / (beat.std() + 1e-8))
        rr_out.append(rr[i])
        labels.append(C.AAMI_MAP[sym])
        kept_symbols.append(sym)

    return {
        "beats": np.asarray(beats, dtype=np.float32),
        "rr": np.asarray(rr_out, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int8),
        "symbols": np.asarray(kept_symbols),
    }


def build_split(record_ids):
    """Concatenate beats from several records, keeping the record id of every beat."""
    parts = [extract_beats(r) for r in record_ids]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["records"] = np.concatenate([np.full(len(p["labels"]), r) for r, p in zip(record_ids, parts)])
    return out


def class_counts(labels):
    return {name: int((labels == i).sum()) for i, name in enumerate(C.CLASSES)}
