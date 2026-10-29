"""Dataset constants: record splits, AAMI class mapping and preprocessing parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "mitdb"
PROCESSED_DIR = ROOT / "data" / "processed"

FS = 360  # MIT-BIH sampling rate (Hz)
LEAD = "MLII"

# Records with paced beats, excluded as recommended by AAMI EC57.
PACED = ["102", "104", "107", "217"]

# Inter-patient split of de Chazal et al. (2004).
DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]

# Patient-wise validation hold-out taken from DS1, used only for model selection.
# Chosen so that SVEB, VEB and F beats all appear; 208 (most DS1 F beats) stays in training.
VAL = ["106", "118", "201", "220", "223"]
TRAIN = [r for r in DS1 if r not in VAL]

CLASSES = ["N", "SVEB", "VEB", "F", "Q"]

# MIT-BIH beat annotation symbol -> AAMI EC57 class index.
AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}

# Beat window around the annotated R-peak: 250 ms before, ~444 ms after.
PRE_SAMPLES = 90
POST_SAMPLES = 160
BEAT_LEN = PRE_SAMPLES + POST_SAMPLES

BANDPASS_HZ = (0.5, 40.0)

# RR features: local average over the previous N beats, long-term average over the previous window.
LOCAL_RR_BEATS = 10
LONG_RR_SECONDS = 300
RR_FEATURES = ["pre_rr", "post_rr", "local_rr", "long_rr", "rr_cv", "rr_succ_diff"]

# Rhythm irregularity (coefficient of variation and mean successive difference, both relative to
# the mean RR) over the intervals before the current beat. High in atrial fibrillation, low in a
# regular rhythm with an isolated premature beat.
IRREG_RR_BEATS = 10
IRREG_CLIP = (0.0, 1.0)

# Physiological limits for the model's RR inputs. Gaps in the beat annotations (unreadable signal
# stretches) otherwise produce "RR intervals" of up to 100 s, which distort the normalization and
# stretch the INT8 input range so far that the small ratio changes marking premature beats are lost.
RR_CLIP_S = (0.2, 2.0)  # 30-300 bpm
RR_RATIO_CLIP = (0.0, 3.0)
