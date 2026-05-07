import argparse
import glob
import json
import math
import os
from collections import defaultdict


MELD_EMOS = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
CMU_SENT = ['positive', 'negative', 'neutral']


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(xs)] * len(xs)


def parse_text_pred(response, candidates):
    if not response:
        return None
    r = response.lower().strip()
    for label in candidates:
        if label.lower() in r:
            return label
    return None


def predict_from_record(rec, candidates):
    logits_dict = rec.get("emotion_logits")
    if logits_dict and all(c in logits_dict for c in candidates):
        logits = [logits_dict[c] for c in candidates]
        probs = softmax(logits)
        idx = max(range(len(candidates)), key=lambda i: logits[i])
        return candidates[idx], probs
    return parse_text_pred(rec.get("response", ""), candidates), None


def compute_meld(records):
    from sklearn.metrics import (
        f1_score as sklearn_f1,
        classification_report as sklearn_report,
        matthews_corrcoef,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    label2idx = {l: i for i, l in enumerate(MELD_EMOS)}

    preds = []
    gts = []
    all_probs = []
    invalid = []
    skipped = []
    per_sample = []

    for rec in records:
        if rec.get("error"):
            skipped.append(rec["name"])
            per_sample.append({
                "name": rec["name"],
                "ground_truth_idx": rec.get("ground_truth_idx"),
                "ground_truth_label": rec.get("ground_truth_label"),
                "predicted_idx": None,
                "predicted_label": None,
                "error": rec["error"],
            })
            continue
        gt = rec.get("ground_truth_label")
        if gt is None:
            continue
        pred, probs = predict_from_record(rec, MELD_EMOS)
        if pred is None:
            invalid.append({"name": rec["name"], "response": rec.get("response", "")})
        preds.append(pred)
        gts.append(gt)
        all_probs.append(probs)
        per_sample.append({
            "name": rec["name"],
            "ground_truth_idx": rec.get("ground_truth_idx"),
            "ground_truth_label": gt,
            "predicted_idx": label2idx[pred] if pred in label2idx else None,
            "predicted_label": pred,
            "probs": probs,
        })

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    correct = 0
    total = 0
    for p, g in zip(preds, gts):
        per_class_total[g] += 1
        total += 1
        if p == g:
            correct += 1
            per_class_correct[g] += 1

    accuracy = correct / total if total else 0.0
    per_class_acc = {
        emo: (per_class_correct[emo] / per_class_total[emo])
              if per_class_total[emo] else 0.0
        for emo in MELD_EMOS
    }

    inv = len(MELD_EMOS)
    y_true = [label2idx.get(g, inv) for g in gts]
    y_pred = [label2idx.get(p, inv) for p in preds]

    weighted_f1 = float(sklearn_f1(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(sklearn_f1(y_true, y_pred, average="macro", zero_division=0))
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    report = sklearn_report(
        y_true, y_pred,
        labels=list(range(len(MELD_EMOS))),
        target_names=MELD_EMOS,
        zero_division=0,
        output_dict=True,
    )

    auroc = auprc = None
    if all_probs and all(p is not None for p in all_probs):
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(len(MELD_EMOS))))
            probs_arr = [p for p in all_probs]
            auroc = float(roc_auc_score(y_true_bin, probs_arr,
                                        average="macro", multi_class="ovr"))
            auprc = float(average_precision_score(y_true_bin, probs_arr, average="macro"))
        except Exception as e:
            auroc = None
            auprc = None
            print(f"  (AUROC/AUPRC unavailable: {e})")

    return {
        "label_to_idx": label2idx,
        "idx_to_label": {i: l for l, i in label2idx.items()},
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "auroc_macro_ovr": auroc,
        "auprc_macro": auprc,
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "total_samples": total,
        "correct": correct,
        "invalid_predictions": len(invalid),
        "skipped_samples": len(skipped),
        "used_logits": all_probs and all(p is not None for p in all_probs),
        "per_sample": per_sample,
    }


def compute_cmumosei(records):
    from sklearn.metrics import f1_score as sklearn_f1
    preds = []
    gts = []
    for rec in records:
        if rec.get("error"):
            continue
        g = rec.get("ground_truth_label")
        if g is None:
            continue
        pred, _ = predict_from_record(rec, CMU_SENT)
        preds.append(pred)
        gts.append(g)

    non_zero = [i for i, g in enumerate(gts) if g != "neutral"]
    sent_correct = sum(1 for i in non_zero if preds[i] == gts[i])
    sent_total = len(non_zero)
    sent_acc = sent_correct / sent_total if sent_total else 0.0
    binary_correct = sum(
        1 for i in non_zero
        if (preds[i] == "positive") == (gts[i] == "positive")
    )
    binary_acc = binary_correct / sent_total if sent_total else 0.0

    label2idx = {l: i for i, l in enumerate(CMU_SENT)}
    y_true = [label2idx.get(gts[i], len(CMU_SENT)) for i in non_zero]
    y_pred = [label2idx.get(preds[i], len(CMU_SENT)) for i in non_zero]
    try:
        wf1 = float(sklearn_f1(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        wf1 = 0.0
    return {
        "binary_accuracy": binary_acc,
        "sentiment_accuracy": sent_acc,
        "weighted_f1": wf1,
        "total_samples": len(gts),
        "non_zero_samples": sent_total,
        "invalid_predictions": sum(1 for p in preds if p is None),
    }


def detect_dataset_from_path(path):
    name = os.path.basename(path).lower()
    if name.startswith("meld"):
        return "MELD"
    if name.startswith("cmumosei"):
        return "CMUMOSEI"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No matching files.")
        return

    for path in files:
        dataset = args.dataset or detect_dataset_from_path(path)
        if dataset is None:
            print(f"[skip] could not detect dataset for {path}")
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        metrics = compute_meld(records) if dataset == "MELD" else compute_cmumosei(records)

        out_path = args.out or path.rsplit(".jsonl", 1)[0] + ".metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if dataset == "MELD":
            extras = ""
            if metrics.get("auroc_macro_ovr") is not None:
                extras = (f"  AUROC={metrics['auroc_macro_ovr']:.4f}"
                          f"  AUPRC={metrics['auprc_macro']:.4f}")
            print(
                f"{path}\n"
                f"  acc={metrics['accuracy']:.4f}  "
                f"wF1={metrics['weighted_f1']:.4f}  "
                f"mF1={metrics['macro_f1']:.4f}  "
                f"MCC={metrics['mcc']:.4f}{extras}  "
                f"invalid={metrics['invalid_predictions']}  "
                f"used_logits={metrics['used_logits']}  "
                f"-> {out_path}"
            )
        else:
            print(
                f"{path}\n"
                f"  binAcc={metrics['binary_accuracy']:.4f}  "
                f"sentAcc={metrics['sentiment_accuracy']:.4f}  "
                f"wF1={metrics['weighted_f1']:.4f}  "
                f"-> {out_path}"
            )


if __name__ == "__main__":
    main()
