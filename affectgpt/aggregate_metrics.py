import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


MELD_EMOS = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
CMU_SENT = ['positive', 'negative', 'neutral']

CONDITION_ORDER = [
    "none", "text", "audio", "video",
    "text_audio", "text_video", "audio_video", "text_audio_video",
]


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


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def is_successful(rec, candidates):
    if rec.get("error"):
        return False
    if rec.get("ground_truth_label") is None:
        return False
    pred, _ = predict_from_record(rec, candidates)
    return pred is not None


def discover_conditions(root, requested=None):
    out = []
    seen = set()
    for cond in CONDITION_ORDER:
        p = os.path.join(root, cond, "meld.jsonl")
        if os.path.isfile(p):
            out.append((cond, p))
            seen.add(cond)
    for sub in sorted(os.listdir(root)):
        if sub in seen:
            continue
        p = os.path.join(root, sub, "meld.jsonl")
        if os.path.isfile(p):
            out.append((sub, p))
    if requested:
        keep = set(requested)
        out = [(c, p) for c, p in out if c in keep]
        missing = keep - {c for c, _ in out}
        if missing:
            print(f"WARNING: requested conditions missing meld.jsonl: "
                  f"{sorted(missing)}", file=sys.stderr)
    return out


def compute_meld_subset(records_by_name, names, label2idx):
    from sklearn.metrics import (
        f1_score as sklearn_f1,
        classification_report as sklearn_report,
        matthews_corrcoef,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import label_binarize

    preds, gts, all_probs = [], [], []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for name in names:
        rec = records_by_name.get(name)
        if rec is None:
            continue
        gt = rec["ground_truth_label"]
        pred, probs = predict_from_record(rec, MELD_EMOS)
        if pred is None:
            continue
        preds.append(pred)
        gts.append(gt)
        all_probs.append(probs)
        per_class_total[gt] += 1
        if pred == gt:
            per_class_correct[gt] += 1

    total = len(preds)
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
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
            auroc = float(roc_auc_score(y_true_bin, all_probs,
                                        average="macro", multi_class="ovr"))
            auprc = float(average_precision_score(y_true_bin, all_probs, average="macro"))
        except Exception as e:
            print(f"  (AUROC/AUPRC unavailable: {e})", file=sys.stderr)

    return {
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "auroc_macro_ovr": auroc,
        "auprc_macro": auprc,
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "used_logits": bool(all_probs and all(p is not None for p in all_probs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--dataset", default="MELD")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--require-all", action="store_true")
    args = ap.parse_args()

    if args.dataset.upper() != "MELD":
        print("Only MELD aggregation implemented.", file=sys.stderr)
        sys.exit(1)

    pairs = discover_conditions(args.root, args.conditions)
    if not pairs:
        print(f"No meld.jsonl found under {args.root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} conditions:")
    for cond, p in pairs:
        print(f"  {cond:<20s} {p}")
    print()

    per_cond = {}
    for cond, path in pairs:
        recs = load_jsonl(path)
        records_by_name = {r["name"]: r for r in recs if "name" in r}
        successful = {n for n, r in records_by_name.items()
                      if is_successful(r, MELD_EMOS)}
        errors = sum(1 for r in records_by_name.values() if r.get("error"))
        per_cond[cond] = {
            "records_by_name": records_by_name,
            "successful": successful,
            "all_names": set(records_by_name.keys()),
            "errors": errors,
        }
        print(f"  [{cond}] {len(records_by_name)} records, "
              f"{len(successful)} successful, {errors} errors")

    if args.require_all and any(len(v["records_by_name"]) == 0 for v in per_cond.values()):
        print("ERROR: at least one condition has 0 records.", file=sys.stderr)
        sys.exit(2)

    success_sets = [v["successful"] for v in per_cond.values()]
    intersection = set.intersection(*success_sets) if success_sets else set()
    union = set.union(*success_sets) if success_sets else set()
    all_seen = set.union(*[v["all_names"] for v in per_cond.values()]) \
        if per_cond else set()

    print()
    print(f"Intersection: {len(intersection)}")
    print(f"Union:        {len(union)}")
    print(f"Total names:  {len(all_seen)}")
    print()

    drop_breakdown = {}
    for cond, v in per_cond.items():
        dropped = v["all_names"] - intersection
        err = sum(1 for n in dropped if v["records_by_name"].get(n, {}).get("error"))
        invalid = len(dropped) - err - len(v["all_names"] - v["records_by_name"].keys())
        drop_breakdown[cond] = {
            "dropped_from_intersection": len(dropped),
            "dropped_due_to_error": err,
            "dropped_due_to_invalid_pred": max(0, invalid),
        }

    label2idx = {l: i for i, l in enumerate(MELD_EMOS)}
    sorted_intersection = sorted(intersection)

    per_cond_metrics = {}
    for cond, v in per_cond.items():
        if not sorted_intersection:
            per_cond_metrics[cond] = {
                "total_samples": 0,
                "accuracy": 0.0,
                "weighted_f1": 0.0,
                "macro_f1": 0.0,
                "mcc": 0.0,
            }
            continue
        m = compute_meld_subset(v["records_by_name"], sorted_intersection, label2idx)
        m["dropped"] = drop_breakdown[cond]
        per_cond_metrics[cond] = m

    cols = ["condition", "n", "acc", "wF1", "mF1", "MCC", "AUROC", "AUPRC", "errors"]
    widths = [22, 5, 7, 7, 7, 7, 7, 7, 7]
    header = " ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for cond, _ in pairs:
        m = per_cond_metrics[cond]

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, float) else (str(x) if x is not None else "—")

        row = [
            cond,
            str(m.get("total_samples", 0)),
            fmt(m.get("accuracy", 0.0)),
            fmt(m.get("weighted_f1", 0.0)),
            fmt(m.get("macro_f1", 0.0)),
            fmt(m.get("mcc", 0.0)),
            fmt(m.get("auroc_macro_ovr")),
            fmt(m.get("auprc_macro")),
            str(per_cond[cond]["errors"]),
        ]
        print(" ".join(s.ljust(w) for s, w in zip(row, widths)))

    out_json = args.out or os.path.join(args.root, "aggregate.json")
    out_csv = args.csv or os.path.join(args.root, "aggregate.csv")

    aggregate = {
        "root": os.path.abspath(args.root),
        "conditions": [c for c, _ in pairs],
        "intersection_size": len(intersection),
        "intersection_names": sorted_intersection,
        "union_size": len(union),
        "total_unique_names": len(all_seen),
        "per_condition_counts": {
            cond: {
                "total_records": len(v["records_by_name"]),
                "successful": len(v["successful"]),
                "errors": v["errors"],
            } for cond, v in per_cond.items()
        },
        "per_condition_metrics": per_cond_metrics,
    }
    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "n_intersection", "accuracy",
            "weighted_f1", "macro_f1", "mcc",
            "auroc_macro_ovr", "auprc_macro",
            "errors", "successful", "total_records",
        ])
        for cond, _ in pairs:
            m = per_cond_metrics[cond]
            counts = aggregate["per_condition_counts"][cond]
            w.writerow([
                cond,
                m.get("total_samples", 0),
                m.get("accuracy", 0.0),
                m.get("weighted_f1", 0.0),
                m.get("macro_f1", 0.0),
                m.get("mcc", 0.0),
                m.get("auroc_macro_ovr"),
                m.get("auprc_macro"),
                counts["errors"],
                counts["successful"],
                counts["total_records"],
            ])


if __name__ == "__main__":
    main()
