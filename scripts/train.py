"""Train the 1D-CNN on DS1 (minus the validation records), selecting the best epoch on the
patient-wise validation hold-out. DS2 is never loaded here.

Each run is saved under outputs/runs/<tag>/seed<k>/, with a summary of validation metrics
(mean and std across seeds) in outputs/runs/<tag>/summary.json.

Usage: python scripts/train.py --tag base --seeds 0 1 2 [--weight-power 0.5] [--augment]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras  # noqa: E402

from ecg import config as C  # noqa: E402
from ecg.features import RR_INPUTS, load_split  # noqa: E402
from ecg.metrics import format_report, summarize  # noqa: E402
from ecg.model import build_model  # noqa: E402
from ecg.training import class_weights, make_dataset  # noqa: E402

RUNS_DIR = C.ROOT / "outputs" / "runs"


class SelectOnValidation(keras.callbacks.Callback):
    """Track the validation selection score each epoch, keep the best weights and stop when it plateaus."""

    def __init__(self, val_inputs, val_labels, patience):
        super().__init__()
        self.val_inputs, self.val_labels, self.patience = val_inputs, val_labels, patience
        self.best_score, self.best_epoch, self.best_weights = -1.0, 0, None
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.val_inputs, batch_size=1024, verbose=0).argmax(1)
        s = summarize(self.val_labels, pred)
        pc = s["per_class"]
        self.history.append({"epoch": epoch + 1, "loss": float(logs["loss"]), "val_selection_f1": s["selection_f1"],
                             **{f"val_Se_{c}": pc[c]["Se"] for c in C.CLASSES[:4]}})
        marker = ""
        if s["selection_f1"] > self.best_score:
            self.best_score, self.best_epoch = s["selection_f1"], epoch + 1
            self.best_weights = self.model.get_weights()
            marker = "  *"
        print(f"  epoch {epoch + 1:3d}  loss {logs['loss']:.4f}  val F1(N,S,V) {100 * s['selection_f1']:5.2f}  "
              + "  ".join(f"Se_{c} {100 * pc[c]['Se']:5.1f}" for c in C.CLASSES[:4]) + marker, flush=True)
        if epoch + 1 - self.best_epoch >= self.patience:
            self.model.stop_training = True


def train_one(args, seed, x_tr, y_tr, x_va, y_va, weights, out_dir):
    keras.utils.set_random_seed(seed)
    steps = int(np.ceil(len(y_tr) / args.batch_size)) * args.epochs
    lr = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=steps, alpha=0.05)

    model = build_model(x_tr["rr"], n_rr=len(RR_INPUTS))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")

    sample_w = weights[y_tr].astype(np.float32)
    ds = make_dataset(x_tr, y_tr, sample_w, args.batch_size, args.augment, seed)
    selector = SelectOnValidation(x_va, y_va, args.patience)
    model.fit(ds, epochs=args.epochs, callbacks=[selector], verbose=0)

    model.set_weights(selector.best_weights)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")

    pred = model.predict(x_va, batch_size=1024, verbose=0).argmax(1)
    summary = summarize(y_va, pred)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump({"seed": seed, "best_epoch": selector.best_epoch, "params": model.count_params(),
                   "val": summary, "history": selector.history}, f, indent=2)
    print(format_report(summary, f"  seed {seed}: validation at best epoch {selector.best_epoch}"))
    return summary, model.count_params()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--max-class-weight", type=float, default=25.0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    args = p.parse_args()

    run_dir = RUNS_DIR / args.tag
    x_tr, y_tr, _ = load_split("train")
    x_va, y_va, _ = load_split("val")

    weights = class_weights(y_tr, args.weight_power, args.max_class_weight)
    print(f"[{args.tag}] class weights:", {c: round(float(w), 2) for c, w in zip(C.CLASSES, weights)})

    per_seed = []
    for seed in args.seeds:
        print(f"[{args.tag}] seed {seed}")
        summary, params = train_one(args, seed, x_tr, y_tr, x_va, y_va, weights, run_dir / f"seed{seed}")
        per_seed.append(summary)

    def stat(get):
        v = np.array([get(s) for s in per_seed])
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}

    agg = {"selection_f1": stat(lambda s: s["selection_f1"]), "macro_f1": stat(lambda s: s["macro_f1"]),
           "accuracy": stat(lambda s: s["accuracy"])}
    for c in C.CLASSES[:4]:
        for m in ("Se", "+P"):
            agg[f"{c}_{m}"] = stat(lambda s: s["per_class"][c][m])
    best_seed = args.seeds[int(np.argmax([s["selection_f1"] for s in per_seed]))]

    with open(run_dir / "summary.json", "w") as f:
        json.dump({"args": vars(args), "class_weights": weights.tolist(), "params": params,
                   "best_seed": best_seed, "val": agg}, f, indent=2)
    print(f"[{args.tag}] val F1(N,S,V) {100 * agg['selection_f1']['mean']:.2f} ± {100 * agg['selection_f1']['std']:.2f}"
          f"  SVEB Se {100 * agg['SVEB_Se']['mean']:.1f}  F Se {100 * agg['F_Se']['mean']:.1f}  best seed {best_seed}")


if __name__ == "__main__":
    main()
