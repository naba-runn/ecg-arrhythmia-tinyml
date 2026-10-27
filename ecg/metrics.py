"""AAMI EC57-style metrics: per-class sensitivity, positive predictivity and false positive rate."""

import numpy as np

from . import config as C

# Q has only a handful of beats in MIT-BIH (none in our validation set), so the
# reported summary score averages over the four clinical classes.
SCORED_CLASSES = [0, 1, 2, 3]
# The validation hold-out has only 16 F beats, too few to rank epochs on, so model
# selection uses N, SVEB and VEB; F is still reported.
SELECTION_CLASSES = [0, 1, 2]


def macro_f1(per_class, classes):
    return float(np.mean([per_class[C.CLASSES[i]]["F1"] for i in classes]))


def confusion_matrix(y_true, y_pred, n_classes=len(C.CLASSES)):
    """Rows are the reference class, columns the predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _ratio(num, den):
    return float(num / den) if den else float("nan")


def per_class_metrics(cm):
    total = cm.sum()
    out = {}
    for i, name in enumerate(C.CLASSES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        se, ppv = _ratio(tp, tp + fn), _ratio(tp, tp + fp)
        out[name] = {
            "support": int(cm[i].sum()),
            "Se": se,
            "+P": ppv,
            "FPR": _ratio(fp, fp + tn),
            "Sp": _ratio(tn, tn + fp),
            "F1": _ratio(2 * se * ppv, se + ppv) if tp else 0.0,
        }
    return out


def summarize(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class = per_class_metrics(cm)
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_f1": macro_f1(per_class, SCORED_CLASSES),
        "selection_f1": macro_f1(per_class, SELECTION_CLASSES),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_report(summary, title=""):
    pct = lambda v: "   -  " if np.isnan(v) else f"{100 * v:6.2f}"
    lines = [title] if title else []
    lines.append(f"Accuracy {100 * summary['accuracy']:.2f}%   macro-F1 (N,SVEB,VEB,F) {100 * summary['macro_f1']:.2f}%")
    lines.append(f"{'class':<5} {'n':>6}  {'Se':>6}  {'+P':>6}  {'FPR':>6}  {'F1':>6}")
    for name, m in summary["per_class"].items():
        lines.append(f"{name:<5} {m['support']:>6}  {pct(m['Se'])}  {pct(m['+P'])}  {pct(m['FPR'])}  {pct(m['F1'])}")
    cm = np.asarray(summary["confusion_matrix"])
    lines.append("Confusion matrix (rows = reference, cols = predicted):")
    lines.append("      " + "".join(f"{c:>7}" for c in C.CLASSES))
    for name, row in zip(C.CLASSES, cm):
        lines.append(f"{name:<6}" + "".join(f"{v:>7}" for v in row))
    return "\n".join(lines)
