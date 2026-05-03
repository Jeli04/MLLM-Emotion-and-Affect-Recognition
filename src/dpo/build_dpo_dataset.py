import argparse
import json
import logging
import math
import os
import random
import warnings
from functools import partial

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, set_seed
from peft import PeftModel
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import (
    CORRUPTION_PRESET_NAMES as MELD_CORRUPTION_PRESET_NAMES,
    CorruptedMELDDataset,
    collate_fn as meld_collate_fn,
    EMOTION2ID as MELD_EMOTION2ID,
    SYSTEM_PROMPT as MELD_SYSTEM_PROMPT,
)
from src.iemocap_dataset import (
    CORRUPTION_PRESET_NAMES as IEMOCAP_CORRUPTION_PRESET_NAMES,
    CorruptedIEMOCAPDataset,
    collate_fn as iemocap_collate_fn,
    compute_iemocap_eval_holdout_indices,
    EMOTION2ID as IEMOCAP_EMOTION2ID,
    SYSTEM_PROMPT as IEMOCAP_SYSTEM_PROMPT,
)

MELD_CONFUSION_PAIRS = {
    "neutral": ["sadness", "joy"],
    "sadness": ["neutral", "fear"],
    "joy": ["neutral", "surprise"],
    "anger": ["disgust"],
    "disgust": ["anger"],
    "surprise": ["joy", "fear"],
    "fear": ["surprise", "sadness"],
}

# Plausible confusions for IEMOCAP 10-class labels (same semantics as MELD graph where applicable).
IEMOCAP_CONFUSION_PAIRS = {
    "neutral": ["sad", "happy", "frustrated"],
    "sad": ["neutral", "fearful", "frustrated"],
    "happy": ["neutral", "excited", "surprised"],
    "angry": ["disgusted", "frustrated", "neutral"],
    "disgusted": ["angry", "frustrated"],
    "fearful": ["surprised", "sad", "neutral"],
    "surprised": ["happy", "fearful", "excited"],
    "frustrated": ["angry", "neutral", "sad"],
    "excited": ["happy", "surprised"],
    "other": ["neutral", "happy", "sad"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a DPO preference dataset from MELD or IEMOCAP using confusion-pair heuristics",
    )
    parser.add_argument(
        "--dataset",
        default="meld",
        choices=["meld", "iemocap"],
        help="Source corpus (default: meld)",
    )
    parser.add_argument("--modalities", nargs="+", default=["text"],
                        choices=["text", "audio", "video"],
                        help="Which modalities to include in the input (default: text)")
    parser.add_argument(
        "--split",
        default="train",
        help="MELD: train|dev|test. IEMOCAP: value for manifest 'split' column if present; "
             "otherwise ignored (see RawIEMOCAPDataset warning).",
    )
    parser.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw",
                        help="Path to MELD.Raw directory (MELD only)")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to IEMOCAP utterance manifest CSV (required when --dataset iemocap)",
    )
    parser.add_argument(
        "--iemocap_sessions",
        nargs="+",
        default=None,
        help="Optional IEMOCAP session filter, e.g. Session1 Session2",
    )
    parser.add_argument(
        "--iemocap_manifest_split",
        default=None,
        help="Override --split for IEMOCAP manifest filtering (same as CorruptedIEMOCAPDataset split=)",
    )
    parser.add_argument(
        "--iemocap_eval_holdout_n",
        type=int,
        default=500,
        help="IEMOCAP only: exclude this many random manifest indices from DPO mining (default 500). "
             "Uses --iemocap_eval_holdout_seed; same draw as SFT holdout / typical 500-sample eval. "
             "Set to 0 to use every row after session/split filters.",
    )
    parser.add_argument(
        "--iemocap_eval_holdout_seed",
        type=int,
        default=42,
        help="RNG seed for IEMOCAP eval holdout exclusion (default 42)",
    )
    parser.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the model")
    parser.add_argument("--adapter_path", default=None,
                        help="Path to a LoRA adapter checkpoint to load on top of the base model")
    parser.add_argument("--corrupt", dest="corrupt", action="store_true", default=True,
                        help="Apply noise/corruption to inputs (default: True)")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false",
                        help="Disable input corruption")
    parser.add_argument("--corruption_preset", default="medium",
                        help="Corruption preset when --corrupt is enabled (mild|medium|strong)")
    parser.add_argument("--output_dir", default=os.path.join("results", "dpo"),
                        help="Directory where evaluation and DPO files are saved")
    parser.add_argument("--correct_sample_ratio", type=float, default=0.15,
                        help="Target fraction of final DPO samples drawn from correct model predictions")
    parser.add_argument("--correct_sample_seed", type=int, default=42,
                        help="Random seed for selecting correct-prediction DPO samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model evaluation and input corruption")
    args = parser.parse_args()
    preset_names = (
        IEMOCAP_CORRUPTION_PRESET_NAMES
        if args.dataset == "iemocap"
        else MELD_CORRUPTION_PRESET_NAMES
    )
    if args.corruption_preset not in preset_names:
        raise SystemExit(
            f"--corruption_preset must be one of {preset_names}, got {args.corruption_preset!r}",
        )
    if args.dataset == "iemocap" and not args.manifest:
        raise SystemExit("--manifest is required when --dataset iemocap")
    return args


def build_prompt_messages(raw_sample, modalities, *, system_prompt: str, dataset: str):
    """Build the prompt side of a preference example in chat-message format."""
    user_content = []
    has_video = "video" in modalities

    for mod in modalities:
        if mod == "text":
            user_content.append({"type": "text", "text": raw_sample["text"]})
        elif mod == "video":
            user_content.append({"type": "video", "video": raw_sample["video_path"]})
        elif mod == "audio":
            if dataset == "iemocap":
                audio_ref = raw_sample.get("audio_path") or raw_sample.get("video_path") or ""
            else:
                audio_ref = raw_sample["video_path"]
            user_content.append({"type": "audio", "audio": audio_ref})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def build_dpo_sample(
    sample_index,
    raw_sample,
    gt_emotion,
    pred,
    raw_output,
    modalities,
    corrupt,
    corruption_preset,
    *,
    system_prompt: str,
    dataset: str,
    rejected_emotion=None,
    selection_reason="confusion_pair_error",
):
    """Create a preference record for DPO: ground truth is chosen, confused prediction is rejected."""
    prompt_messages = build_prompt_messages(
        raw_sample, modalities, system_prompt=system_prompt, dataset=dataset,
    )
    rejected_emotion = rejected_emotion or pred
    chosen_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": gt_emotion}],
    }
    rejected_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": rejected_emotion}],
    }

    row = {
        "sample_index": sample_index,
        "dialogue_id": raw_sample["dialogue_id"],
        "utterance_id": raw_sample["utterance_id"],
        "speaker": raw_sample.get("speaker"),
        "text": raw_sample["text"],
        "video_path": raw_sample["video_path"],
        "modalities": list(modalities),
        "corrupt": corrupt,
        "corruption_preset": corruption_preset,
        "ground_truth": gt_emotion,
        "prediction": pred,
        "raw_model_output": raw_output,
        "selection_reason": selection_reason,
        "model_was_correct": pred == gt_emotion,
        "confusion_pair": {
            "ground_truth": gt_emotion,
            "rejected": rejected_emotion,
        },
        "prompt": raw_sample["text"],
        "chosen": gt_emotion,
        "rejected": rejected_emotion,
        "prompt_messages": prompt_messages,
        "chosen_messages": prompt_messages + [chosen_message],
        "rejected_messages": prompt_messages + [rejected_message],
    }
    if dataset == "iemocap":
        row["audio_path"] = raw_sample.get("audio_path") or ""
        row["session"] = raw_sample.get("session", "")
    return row


def is_confusion_pair(pred, gt_emotion, confusion_pairs):
    return gt_emotion in confusion_pairs.get(pred, [])


def get_rejected_emotion_for_correct_sample(gt_emotion, rng, confusion_pairs):
    alts = confusion_pairs.get(gt_emotion)
    if not alts:
        raise KeyError(
            f"No confusion_pairs entry for ground-truth emotion {gt_emotion!r}; "
            "extend the confusion graph for this label.",
        )
    return rng.choice(alts)


def sample_correct_predictions(correct_candidates, confusion_sample_count, target_ratio, rng):
    if target_ratio <= 0 or not correct_candidates:
        return []
    if target_ratio >= 1:
        return list(correct_candidates)

    target_correct_count = math.ceil(
        (target_ratio * confusion_sample_count) / (1 - target_ratio)
    )
    target_correct_count = min(target_correct_count, len(correct_candidates))
    if target_correct_count <= 0:
        return []

    return rng.sample(correct_candidates, target_correct_count)


def make_eval_collate(collate_fn_impl, pad_token_id):
    """Collate wrapper that extracts non-tensor metadata before calling collate_fn."""

    def eval_collate(batch, pad_token_id_inner):
        emotions = [b["emotion"] for b in batch]
        tensor_batch = [{k: v for k, v in b.items() if k not in ("emotion", "label")} for b in batch]
        collated = collate_fn_impl(
            tensor_batch, pad_token_id=pad_token_id_inner, padding_side="left",
        )
        collated["emotions"] = emotions
        return collated

    return partial(eval_collate, pad_token_id_inner=pad_token_id)


def main():
    args = parse_args()
    set_seed(args.seed)
    iemocap_split = args.iemocap_manifest_split if args.dataset == "iemocap" else None
    split_label = iemocap_split if iemocap_split is not None else args.split

    print(
        f"Building DPO candidates dataset={args.dataset} split={split_label!r} with "
        f"modalities={args.modalities}, corrupt={args.corrupt}, "
        f"corruption_preset={args.corruption_preset}, seed={args.seed}"
    )
    correct_sample_rng = random.Random(args.correct_sample_seed)

    if args.dataset == "meld":
        confusion_pairs = MELD_CONFUSION_PAIRS
        valid_emotions = set(MELD_EMOTION2ID.keys())
        system_prompt = MELD_SYSTEM_PROMPT
        collate_fn_impl = meld_collate_fn
    else:
        confusion_pairs = IEMOCAP_CONFUSION_PAIRS
        valid_emotions = set(IEMOCAP_EMOTION2ID.keys())
        system_prompt = IEMOCAP_SYSTEM_PROMPT
        collate_fn_impl = iemocap_collate_fn

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        enable_audio_output=False,
    )
    if args.adapter_path is not None:
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model.thinker = PeftModel.from_pretrained(model.thinker, args.adapter_path)

    model.eval()

    first_device = next(model.parameters()).device

    iemocap_full_ds = None
    iemocap_index_map = None
    holdout_indices_for_json = []

    common_ds = dict(
        processor=processor,
        modalities=tuple(args.modalities),
        corrupt=args.corrupt,
        corruption_preset=args.corruption_preset,
        for_training=False,
    )
    if args.dataset == "meld":
        dataset = CorruptedMELDDataset(
            args.data_root,
            split=args.split,
            **common_ds,
        )
        use_audio_in_video = False
    else:
        split_kw = iemocap_split if iemocap_split is not None else args.split
        iemocap_full_ds = CorruptedIEMOCAPDataset(
            args.manifest,
            split=split_kw,
            sessions=args.iemocap_sessions,
            max_samples=None,
            **common_ds,
        )
        if args.iemocap_eval_holdout_n > 0:
            holdout_indices_for_json, remaining = compute_iemocap_eval_holdout_indices(
                args.manifest,
                holdout_n=args.iemocap_eval_holdout_n,
                holdout_seed=args.iemocap_eval_holdout_seed,
                sessions=args.iemocap_sessions,
                split=split_kw,
                drop_no_agreement=True,
            )
            if not remaining:
                raise SystemExit(
                    "IEMOCAP eval holdout leaves no rows for DPO mining; reduce "
                    "--iemocap_eval_holdout_n or relax session/split filters.",
                )
            dataset = Subset(iemocap_full_ds, remaining)
            iemocap_index_map = remaining
            print(
                f"IEMOCAP eval holdout: excluding {len(holdout_indices_for_json)} manifest indices "
                f"(seed={args.iemocap_eval_holdout_seed}); DPO mining on {len(remaining)}/"
                f"{len(iemocap_full_ds)} rows.",
            )
        else:
            dataset = iemocap_full_ds
            print("IEMOCAP eval holdout: disabled (--iemocap_eval_holdout_n 0); mining all filtered rows.")
        use_audio_in_video = "video" in args.modalities and "audio" in args.modalities

    eval_collate = make_eval_collate(collate_fn_impl, processor.tokenizer.pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=0,
    )

    all_preds = []
    all_labels = []
    invalid_predictions = []
    skipped_samples = []
    per_sample_results = []

    # DPO samples are examples where the model prediction is a known confusing
    # emotion for the ground-truth label. A small sampled slice of correct
    # predictions is added below with a plausible confusing emotion as rejected.
    dpo_sample_indices = []
    dpo_samples = []
    correct_dpo_candidates = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        gt_emotion = batch.pop("emotions")[0]
        if args.dataset == "meld":
            raw_idx = i
            raw_sample = dataset.raw_dataset[raw_idx]
        else:
            raw_idx = iemocap_index_map[i] if iemocap_index_map is not None else i
            raw_sample = iemocap_full_ds.raw_dataset[raw_idx]

        inputs = {
            k: v.to(first_device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        try:
            with torch.no_grad():
                text_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    return_audio=False,
                    use_audio_in_video=use_audio_in_video,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], text_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            skipped_samples.append((raw_idx, str(e), gt_emotion))
            per_sample_results.append({
                "sample_index": raw_idx,
                "dialogue_id": raw_sample["dialogue_id"],
                "utterance_id": raw_sample["utterance_id"],
                "text": raw_sample["text"],
                "ground_truth": gt_emotion,
                "prediction": None,
                "raw_model_output": None,
                "valid": False,
                "skipped": True,
                "error": str(e),
            })
            torch.cuda.empty_cache()
            tqdm.write(f"  Skipped sample {raw_idx} (OOM/error): {str(e)[:100]}")
            continue

        pred = output_text[0].strip().lower()
        raw_output = output_text[0].strip()
        is_valid = pred in valid_emotions

        row_meta = {
            "sample_index": raw_idx,
            "dialogue_id": raw_sample["dialogue_id"],
            "utterance_id": raw_sample["utterance_id"],
            "text": raw_sample["text"],
            "ground_truth": gt_emotion,
            "prediction": pred,
            "raw_model_output": raw_output,
            "valid": is_valid,
            "skipped": False,
        }
        if args.dataset == "iemocap":
            row_meta["session"] = raw_sample.get("session", "")
        per_sample_results.append(row_meta)

        if not is_valid:
            invalid_predictions.append((raw_idx, raw_output, gt_emotion))
        else:
            all_preds.append(pred)
            all_labels.append(gt_emotion)

            if is_confusion_pair(pred, gt_emotion, confusion_pairs):
                dpo_sample_indices.append(raw_idx)
                dpo_samples.append(
                    build_dpo_sample(
                        sample_index=raw_idx,
                        raw_sample=raw_sample,
                        gt_emotion=gt_emotion,
                        pred=pred,
                        raw_output=raw_output,
                        modalities=args.modalities,
                        corrupt=args.corrupt,
                        corruption_preset=args.corruption_preset,
                        system_prompt=system_prompt,
                        dataset=args.dataset,
                    )
                )
            elif pred == gt_emotion:
                correct_dpo_candidates.append(
                    build_dpo_sample(
                        sample_index=raw_idx,
                        raw_sample=raw_sample,
                        gt_emotion=gt_emotion,
                        pred=pred,
                        raw_output=raw_output,
                        modalities=args.modalities,
                        corrupt=args.corrupt,
                        corruption_preset=args.corruption_preset,
                        system_prompt=system_prompt,
                        dataset=args.dataset,
                        rejected_emotion=get_rejected_emotion_for_correct_sample(
                            gt_emotion,
                            correct_sample_rng,
                            confusion_pairs,
                        ),
                        selection_reason="correct_prediction",
                    )
                )

    confusion_pair_sample_count = len(dpo_samples)
    correct_dpo_samples = sample_correct_predictions(
        correct_dpo_candidates,
        confusion_pair_sample_count,
        args.correct_sample_ratio,
        correct_sample_rng,
    )
    dpo_samples.extend(correct_dpo_samples)
    dpo_sample_indices.extend(sample["sample_index"] for sample in correct_dpo_samples)

    # --- Report metrics ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"Dataset: {args.dataset} | Split: {split_label!r} | Modalities: {args.modalities} | "
        f"Corrupt: {args.corrupt} | Preset: {args.corruption_preset}"
    )
    print(f"Total samples: {len(dataset)}")
    print(f"Valid predictions: {len(all_preds)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Skipped (OOM/error): {len(skipped_samples)}")
    print(f"DPO confusion-pair samples: {confusion_pair_sample_count}")
    print(f"DPO correct-prediction samples: {len(correct_dpo_samples)}")
    print(f"DPO total samples: {len(dpo_samples)}")

    if invalid_predictions:
        print(f"\n--- Invalid Predictions ({len(invalid_predictions)}) ---")
        for idx, model_out, gt in invalid_predictions:
            print(f"  Sample {idx}: model='{model_out}' | gt='{gt}'")

    label_names = sorted(valid_emotions)
    if all_preds:
        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, labels=label_names, zero_division=0))
        acc = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {acc:.4f}")

        y_true_bin = label_binarize(all_labels, classes=label_names)
        y_pred_bin = label_binarize(all_preds, classes=label_names)
        auroc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
        auprc = average_precision_score(y_true_bin, y_pred_bin, average="macro")
        mcc = matthews_corrcoef(all_labels, all_preds)
        print(f"Macro AUROC (OVR):            {auroc:.4f}")
        print(f"Macro Avg Precision (AUPRC):  {auprc:.4f}")
        print(f"MCC:                          {mcc:.4f}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nPeak VRAM usage: {peak_vram:.2f} GB")

    modalities_str = "+".join(sorted(args.modalities))
    corrupt_str = f"corrupt_{args.corruption_preset}" if args.corrupt else "clean"
    model_str = "finetuned" if args.adapter_path else "base"
    # Keep MELD output filenames unchanged; prefix IEMOCAP runs for clarity.
    ds_prefix = f"{args.dataset}_" if args.dataset == "iemocap" else ""
    output_filename = f"results_{ds_prefix}{split_label}_{modalities_str}_{corrupt_str}_{model_str}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    dpo_output_filename = f"dpo_samples_{ds_prefix}{split_label}_{modalities_str}_{corrupt_str}_{model_str}.json"
    dpo_output_path = os.path.join(args.output_dir, dpo_output_filename)

    results_common = {
        "dataset": args.dataset,
        "split": split_label,
        "modalities": args.modalities,
        "corrupt": args.corrupt,
        "corruption_preset": args.corruption_preset,
        "adapter_path": args.adapter_path,
        "seed": args.seed,
        "total_samples": len(dataset),
        "valid_predictions": len(all_preds),
        "invalid_predictions": len(invalid_predictions),
        "skipped_samples": len(skipped_samples),
        "accuracy": accuracy_score(all_labels, all_preds) if all_preds else None,
        "auroc_macro_ovr": roc_auc_score(
            label_binarize(all_labels, classes=label_names),
            label_binarize(all_preds, classes=label_names),
            average="macro",
        ) if all_preds else None,
        "auprc_macro_ovr": average_precision_score(
            label_binarize(all_labels, classes=label_names),
            label_binarize(all_preds, classes=label_names),
            average="macro",
        ) if all_preds else None,
        "mcc": matthews_corrcoef(all_labels, all_preds) if all_preds else None,
        "confusion_pairs": confusion_pairs,
        "dpo_sample_count": len(dpo_samples),
        "dpo_confusion_pair_sample_count": confusion_pair_sample_count,
        "dpo_correct_prediction_sample_count": len(correct_dpo_samples),
        "dpo_correct_prediction_target_ratio": args.correct_sample_ratio,
        "dpo_correct_prediction_actual_ratio": (
            len(correct_dpo_samples) / len(dpo_samples) if dpo_samples else 0.0
        ),
        "dpo_sample_indices": dpo_sample_indices,
        "predictions": per_sample_results,
    }
    if args.dataset == "iemocap":
        results_common["manifest"] = os.path.abspath(args.manifest)
        results_common["iemocap_sessions"] = args.iemocap_sessions
        results_common["iemocap_rows_after_filters"] = len(iemocap_full_ds)
        results_common["iemocap_eval_holdout_n"] = args.iemocap_eval_holdout_n
        results_common["iemocap_eval_holdout_seed"] = args.iemocap_eval_holdout_seed
        results_common["iemocap_eval_holdout_indices"] = (
            holdout_indices_for_json if holdout_indices_for_json else None
        )
    else:
        results_common["data_root"] = args.data_root

    dpo_json = {
        "dataset": args.dataset,
        "split": split_label,
        "modalities": args.modalities,
        "corrupt": args.corrupt,
        "corruption_preset": args.corruption_preset,
        "adapter_path": args.adapter_path,
        "model_path": args.model_path,
        "seed": args.seed,
        "confusion_pairs": confusion_pairs,
        "sample_count": len(dpo_samples),
        "confusion_pair_sample_count": confusion_pair_sample_count,
        "correct_prediction_sample_count": len(correct_dpo_samples),
        "correct_prediction_target_ratio": args.correct_sample_ratio,
        "correct_prediction_actual_ratio": (
            len(correct_dpo_samples) / len(dpo_samples) if dpo_samples else 0.0
        ),
        "correct_prediction_seed": args.correct_sample_seed,
        "sample_indices": dpo_sample_indices,
        "samples": dpo_samples,
    }
    if args.dataset == "iemocap":
        dpo_json["manifest"] = os.path.abspath(args.manifest)
        dpo_json["iemocap_sessions"] = args.iemocap_sessions
        dpo_json["iemocap_rows_after_filters"] = len(iemocap_full_ds)
        dpo_json["iemocap_eval_holdout_n"] = args.iemocap_eval_holdout_n
        dpo_json["iemocap_eval_holdout_seed"] = args.iemocap_eval_holdout_seed
        dpo_json["iemocap_eval_holdout_indices"] = (
            holdout_indices_for_json if holdout_indices_for_json else None
        )
    else:
        dpo_json["data_root"] = args.data_root

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_common, f, indent=2)
    print(f"\nResults saved to {output_path}")

    with open(dpo_output_path, "w") as f:
        json.dump(dpo_json, f, indent=2)
    print(f"DPO samples saved to {dpo_output_path}")


if __name__ == "__main__":
    main()
