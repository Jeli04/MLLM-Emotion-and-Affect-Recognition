"""
Recompute and add Accuracy, AUROC, AUPRC, MCC, and F1 to result JSON files.

Usage:
  python results/add_metrics.py
  python results/add_metrics.py results/meld/*.json
  python results/add_metrics.py results/iemocap
"""
import glob
import json
import math
import sys
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

MELD_LABEL_NAMES = sorted(["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])
IEMOCAP_LABEL_NAMES = sorted(
    [
        "angry",
        "disgusted",
        "excited",
        "fearful",
        "frustrated",
        "happy",
        "neutral",
        "other",
        "sad",
        "surprised",
    ]
)


def _label_names_for(data, labels, preds, path):
    if data.get("dataset") == "iemocap" or "iemocap" in str(path):
        return IEMOCAP_LABEL_NAMES
    if data.get("dataset") == "meld" or set(labels).issubset(MELD_LABEL_NAMES):
        return MELD_LABEL_NAMES
    return sorted(set(labels) | set(preds))


def _safe_macro_auroc(y_true_bin, y_pred_bin):
    per_class_scores = []
    for i in range(y_true_bin.shape[1]):
        if len(set(y_true_bin[:, i])) < 2:
            continue
        per_class_scores.append(roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i]))
    if not per_class_scores:
        return None
    return float(sum(per_class_scores) / len(per_class_scores))


def _safe_macro_auprc(y_true_bin, y_pred_bin):
    per_class_scores = []
    for i in range(y_true_bin.shape[1]):
        if y_true_bin[:, i].sum() == 0:
            continue
        per_class_scores.append(average_precision_score(y_true_bin[:, i], y_pred_bin[:, i]))
    if not per_class_scores:
        return None
    return float(sum(per_class_scores) / len(per_class_scores))


def _format_metric(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def compute_metrics(data, path):
    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        return None

    labels, preds = [], []
    for p in predictions:
        if (
            p.get("valid")
            and not p.get("skipped")
            and p.get("prediction")
            and p.get("ground_truth")
        ):
            labels.append(p["ground_truth"])
            preds.append(p["prediction"])

    if not preds:
        return None

    label_names = _label_names_for(data, labels, preds, path)
    y_true_bin = label_binarize(labels, classes=label_names)
    y_pred_bin = label_binarize(preds, classes=label_names)

    return {
        "accuracy": accuracy_score(labels, preds),
        "auroc_macro_ovr": _safe_macro_auroc(y_true_bin, y_pred_bin),
        "auprc_macro_ovr": _safe_macro_auprc(y_true_bin, y_pred_bin),
        "mcc": matthews_corrcoef(labels, preds),
        "f1_macro": f1_score(labels, preds, labels=label_names, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            labels, preds, labels=label_names, average="weighted", zero_division=0
        ),
    }


def process_file(path):
    with open(path) as f:
        data = json.load(f)

    metrics = compute_metrics(data, path)
    if metrics is None:
        print(f"{path}: no predictions/valid predictions, skipping")
        return

    data.update(metrics)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"{path}")
    print(f"  Accuracy:    {_format_metric(metrics['accuracy'])}")
    print(f"  AUROC:       {_format_metric(metrics['auroc_macro_ovr'])}")
    print(f"  AUPRC:       {_format_metric(metrics['auprc_macro_ovr'])}")
    print(f"  MCC:         {_format_metric(metrics['mcc'])}")
    print(f"  F1 macro:    {_format_metric(metrics['f1_macro'])}")
    print(f"  F1 weighted: {_format_metric(metrics['f1_weighted'])}")


def default_paths():
    results_dir = Path(__file__).resolve().parent
    return sorted(str(p) for p in results_dir.glob("**/*.json"))


def expand_paths(args):
    if not args:
        return default_paths()

    paths = []
    for arg in args:
        matches = glob.glob(arg, recursive=True)
        if matches:
            for match in matches:
                path = Path(match)
                if path.is_dir():
                    paths.extend(str(p) for p in path.glob("**/*.json"))
                else:
                    paths.append(str(path))
        else:
            paths.append(arg)
    return sorted(dict.fromkeys(paths))


if __name__ == "__main__":
    paths = expand_paths(sys.argv[1:])
    for path in paths:
        process_file(path)
