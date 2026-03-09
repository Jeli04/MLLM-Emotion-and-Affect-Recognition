"""
Recompute and add AUROC, AUPRC, and MCC to existing results JSON files.
Usage: python add_metrics.py [results/*.json ...]
       python add_metrics.py  # defaults to all results/*.json
"""
import glob
import json
import sys

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import label_binarize

LABEL_NAMES = sorted(["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"])


def compute_metrics(predictions):
    labels, preds = [], []
    for p in predictions:
        if p.get("valid") and not p.get("skipped") and p["prediction"] and p["ground_truth"]:
            labels.append(p["ground_truth"])
            preds.append(p["prediction"])

    if not preds:
        return None, None, None, None

    y_true_bin = label_binarize(labels, classes=LABEL_NAMES)
    y_pred_bin = label_binarize(preds, classes=LABEL_NAMES)

    acc = accuracy_score(labels, preds)
    auroc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
    auprc = average_precision_score(y_true_bin, y_pred_bin, average="macro")
    mcc = matthews_corrcoef(labels, preds)
    return acc, auroc, auprc, mcc


def process_file(path):
    with open(path) as f:
        data = json.load(f)

    acc, auroc, auprc, mcc = compute_metrics(data["predictions"])
    if acc is None:
        print(f"{path}: no valid predictions, skipping")
        return

    data["accuracy"] = acc
    data["auroc_macro_ovr"] = auroc
    data["auprc_macro_ovr"] = auprc
    data["mcc"] = mcc

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"{path}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    print(f"  MCC:       {mcc:.4f}")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
    for path in paths:
        process_file(path)
