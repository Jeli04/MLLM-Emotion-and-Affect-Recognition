import argparse
import copy
import json
import os
import random
from collections import Counter


MELD_MAJORITY_DEFAULT = ["neutral", "joy"]
MELD_MINORITY_DEFAULT = ["anger", "disgust", "fear", "sadness", "surprise"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a DPO preference dataset to focus on minority classes "
                    "and synthesize anti-majority rejection pairs.",
    )
    parser.add_argument("--input", required=True,
                        help="Path to dpo_samples_*.json produced by build_dpo_dataset.py")
    parser.add_argument("--output", required=True,
                        help="Path to write the rebalanced dpo_samples_*.json")

  
    parser.add_argument("--minority", nargs="+", default=MELD_MINORITY_DEFAULT,
                        help="Class labels considered minority (kept by default). "
                             f"Default (MELD): {MELD_MINORITY_DEFAULT}")
    parser.add_argument("--anti_majority_targets", nargs="*", default=MELD_MAJORITY_DEFAULT,
                        help="Labels to inject as synthetic rejected for every kept "
                             "minority-chosen pair. Pass an empty list "
                             "(--anti_majority_targets) to disable synthesis. "
                             f"Default (MELD): {MELD_MAJORITY_DEFAULT}")
    parser.add_argument("--anchor_majority_fraction", type=float, default=0.0,
                        help="Fraction of original majority-chosen pairs to retain as "
                             "anchors (default 0.0 = drop them all). Range [0,1].")

  
    parser.add_argument("--anchor_seed", type=int, default=42,
                        help="RNG seed for sampling anchor majority pairs.")
    parser.add_argument("--no_synth_when_existing", action="store_true",
                        help="Skip synthesizing a pair if the source sample's existing "
                             "rejected already equals the synthesis target (default: skip).")
    args = parser.parse_args()
    if not 0.0 <= args.anchor_majority_fraction <= 1.0:
        raise SystemExit("--anchor_majority_fraction must be in [0, 1]")
    return args


def synthesize_pair(base_sample, new_rejected):
    new_sample = copy.deepcopy(base_sample)
    new_sample["rejected"] = new_rejected
    new_sample["confusion_pair"] = {
        "ground_truth": base_sample["chosen"],
        "rejected": new_rejected,
    }
    new_sample["selection_reason"] = "synthetic_anti_majority"
    rejected_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": new_rejected}],
    }
    new_sample["rejected_messages"] = list(base_sample["prompt_messages"]) + [rejected_message]
    return new_sample


def pair_key(sample):
    return (sample["sample_index"], sample["chosen"], sample["rejected"])


def main():
    args = parse_args()
    minority = set(args.minority)
    anti_targets = list(dict.fromkeys(args.anti_majority_targets))  

    with open(args.input) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"Loaded {len(samples)} pairs from {args.input}")
    print(f"  original chosen dist: {Counter(s['chosen'] for s in samples).most_common()}")
    print(f"  original rejected dist: {Counter(s['rejected'] for s in samples).most_common()}")

    minority_pairs = [s for s in samples if s["chosen"] in minority]
    majority_pairs = [s for s in samples if s["chosen"] not in minority]

    rng = random.Random(args.anchor_seed)
    n_anchor = int(round(args.anchor_majority_fraction * len(majority_pairs)))
    anchor_pairs = rng.sample(majority_pairs, n_anchor) if n_anchor > 0 else []

    kept = list(minority_pairs) + list(anchor_pairs)

    seen = {pair_key(s) for s in kept}
    synthesized = []
    skipped_dup = 0
    for s in minority_pairs:
        for tgt in anti_targets:
            if tgt == s["chosen"]:
                continue
            new_sample = synthesize_pair(s, tgt)
            key = pair_key(new_sample)
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)
            synthesized.append(new_sample)

    final = kept + synthesized

    print(f"\nFilter summary")
    print(f"  minority-chosen kept       : {len(minority_pairs)}")
    print(f"  majority-chosen anchors    : {len(anchor_pairs)} "
          f"(fraction={args.anchor_majority_fraction}, "
          f"of {len(majority_pairs)} original majority pairs)")
    print(f"  synthesized anti-majority  : {len(synthesized)} "
          f"(targets={anti_targets}, dup-skipped={skipped_dup})")
    print(f"  total output pairs         : {len(final)}")
    print(f"  final chosen dist   : {Counter(s['chosen'] for s in final).most_common()}")
    print(f"  final rejected dist : {Counter(s['rejected'] for s in final).most_common()}")
    print(f"  final selection_reason dist : "
          f"{Counter(s.get('selection_reason') for s in final).most_common()}")

    rebalanced = dict(data)
    rebalanced["samples"] = final
    rebalanced["sample_count"] = len(final)
    rebalanced["sample_indices"] = [s["sample_index"] for s in final]
    rebalanced["rebalance_info"] = {
        "source": os.path.abspath(args.input),
        "minority_classes": sorted(minority),
        "anti_majority_targets": anti_targets,
        "anchor_majority_fraction": args.anchor_majority_fraction,
        "anchor_seed": args.anchor_seed,
        "minority_pairs_kept": len(minority_pairs),
        "anchor_pairs_kept": len(anchor_pairs),
        "synthesized_pairs": len(synthesized),
        "synthesized_dup_skipped": skipped_dup,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rebalanced, f, indent=2)
    print(f"\nWrote {len(final)} pairs to {args.output}")


if __name__ == "__main__":
    main()
