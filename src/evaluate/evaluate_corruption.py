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
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import (
    CORRUPTION_PRESET_NAMES as MELD_CORRUPTION_PRESET_NAMES,
    CorruptedMELDDataset,
    collate_fn,
    EMOTION2ID as MELD_EMOTION2ID,
)
from src.iemocap_dataset import (
    CORRUPTION_PRESET_NAMES as IEMOCAP_CORRUPTION_PRESET_NAMES,
    CorruptedIEMOCAPDataset,
    collate_fn as iemocap_collate_fn,
    compute_iemocap_train_val_holdout_split,
    EMOTION2ID as IEMOCAP_EMOTION2ID,
)

MELD_VALID_EMOTIONS = set(MELD_EMOTION2ID.keys())
IEMOCAP_VALID_EMOTIONS = set(IEMOCAP_EMOTION2ID.keys())
CORRUPTION_PRESET_NAMES = tuple(
    sorted(set(MELD_CORRUPTION_PRESET_NAMES) | set(IEMOCAP_CORRUPTION_PRESET_NAMES))
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni on MELD/IEMOCAP emotion recognition with corruption"
    )
    parser.add_argument("--dataset", default="meld", choices=["meld", "iemocap"],
                        help="Dataset backend (default: meld)")
    parser.add_argument("--modalities", nargs="+", default=["text"],
                        choices=["text", "audio", "video"],
                        help="Which modalities to include in the input (default: text)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw",
                        help="Path to MELD.Raw directory")
    parser.add_argument("--manifest", default=None,
                        help="Path to IEMOCAP utterance manifest CSV (required for --dataset iemocap)")
    parser.add_argument("--iemocap_sessions", nargs="+", default=None,
                        help="Optional IEMOCAP session filter, e.g. Session1 Session2")
    parser.add_argument("--iemocap_manifest_split", default=None,
                        help="If the IEMOCAP manifest has a split column, keep only rows matching this value")
    parser.add_argument("--iemocap_holdout_n", type=int, default=500,
                        help="Random eval holdout size for IEMOCAP test split; 0 disables")
    parser.add_argument("--iemocap_holdout_seed", type=int, default=42,
                        help="RNG seed for the IEMOCAP holdout subset")
    parser.add_argument("--iemocap_val_ratio", type=float, default=0.1,
                        help="Fraction of non-holdout IEMOCAP samples used for dev split")
    parser.add_argument("--iemocap_split_seed", type=int, default=43,
                        help="RNG seed for shuffling non-holdout IEMOCAP rows before train/dev split")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on IEMOCAP rows before train/dev/test indexing")
    parser.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the model")
    parser.add_argument("--adapter_path", default=None,
                        help="Path to a LoRA adapter checkpoint to load on top of the base model")
    parser.add_argument("--run_label", default=None,
                        help="Tag for the saved-results filename (e.g. 'finetune', 'student_teacher'). "
                             "Overrides the default 'base'/'finetuned' tag so different adapters don't collide.")
    parser.add_argument("--corrupt", dest="corrupt", action="store_true", default=True,
                        help="Apply noise/corruption to inputs (default: True)")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false",
                        help="Disable input corruption")
    parser.add_argument("--corruption_preset", default="medium",
                        choices=CORRUPTION_PRESET_NAMES,
                        help="Corruption preset to use when corruption is enabled")
    parser.add_argument("--output_dir", default=None,
                        help="Directory where corruption results are saved. "
                             "Defaults to results/meld or results/iemocap.")
    parser.add_argument("--data_subset_percent", type=float, default=100.0,
                        help="Percentage of the split to evaluate, sampled deterministically "
                             "with --seed (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for python/numpy/torch (controls corruption RNG)")
    args = parser.parse_args()
    if not 0 < args.data_subset_percent <= 100:
        parser.error("--data_subset_percent must be greater than 0 and at most 100")
    preset_names = (
        IEMOCAP_CORRUPTION_PRESET_NAMES
        if args.dataset == "iemocap"
        else MELD_CORRUPTION_PRESET_NAMES
    )
    if args.corruption_preset not in preset_names:
        parser.error(
            f"--corruption_preset must be one of {preset_names}, "
            f"got {args.corruption_preset!r}"
        )
    if args.dataset == "iemocap":
        if not args.manifest:
            parser.error("--manifest is required when --dataset iemocap")
        if not 0.0 <= args.iemocap_val_ratio < 1.0:
            parser.error("--iemocap_val_ratio must be in [0, 1)")
    return args


def build_eval_indices(total_samples, subset_percent, seed):
    if subset_percent >= 100 or total_samples == 0:
        return list(range(total_samples))

    subset_size = min(
        total_samples,
        max(1, math.ceil(total_samples * subset_percent / 100.0)),
    )
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_samples), subset_size))


def eval_collate(batch, pad_token_id, collate_fn_impl):
    """Collate wrapper that extracts non-tensor metadata before calling collate_fn."""
    emotions = [b["emotion"] for b in batch]
    tensor_batch = [{k: v for k, v in b.items() if k not in ("emotion", "label")} for b in batch]
    collated = collate_fn_impl(tensor_batch, pad_token_id=pad_token_id, padding_side="left")
    collated["emotions"] = emotions
    return collated


def build_iemocap_eval_indices(args):
    train_idx, val_idx, holdout_idx = compute_iemocap_train_val_holdout_split(
        args.manifest,
        holdout_n=args.iemocap_holdout_n,
        holdout_seed=args.iemocap_holdout_seed,
        val_ratio=args.iemocap_val_ratio,
        split_seed=args.iemocap_split_seed,
        sessions=args.iemocap_sessions,
        split=args.iemocap_manifest_split,
    )
    if args.split == "train":
        return train_idx
    if args.split == "dev":
        return val_idx
    return holdout_idx


def get_cuda_eval_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but the GPTQ eval model must run on GPU. "
            "Check the Slurm GPU allocation/CUDA_VISIBLE_DEVICES for this job."
        )
    return f"cuda:{torch.cuda.current_device()}"


def main():
    args = parse_args()
    set_seed(args.seed)
    eval_device = get_cuda_eval_device()
    print(
        f"Evaluating dataset='{args.dataset}' on split='{args.split}' with modalities={args.modalities}, "
        f"corrupt={args.corrupt}, corruption_preset={args.corruption_preset}, "
        f"data_subset_percent={args.data_subset_percent:g}, seed={args.seed}, "
        f"device={eval_device}"
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map={"": eval_device},
        enable_audio_output=False,
    )
    if args.adapter_path is not None:
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model.thinker = PeftModel.from_pretrained(
            model.thinker,
            args.adapter_path,
            torch_device=eval_device,
        )

    model.eval()

    first_device = torch.device(eval_device)

    if args.dataset == "iemocap":
        dataset = CorruptedIEMOCAPDataset(
            args.manifest,
            processor=processor,
            split=args.iemocap_manifest_split,
            sessions=args.iemocap_sessions,
            modalities=tuple(args.modalities),
            corrupt=args.corrupt,
            corruption_preset=args.corruption_preset,
            for_training=False,
            max_samples=args.max_samples,
        )
        valid_emotions = IEMOCAP_VALID_EMOTIONS
        collate_fn_impl = iemocap_collate_fn
        eval_indices = build_iemocap_eval_indices(args)
        if args.max_samples is not None:
            eval_indices = [i for i in eval_indices if i < len(dataset)]
    else:
        dataset = CorruptedMELDDataset(
            args.data_root,
            processor=processor,
            split=args.split,
            modalities=tuple(args.modalities),
            corrupt=args.corrupt,
            corruption_preset=args.corruption_preset,
            for_training=False,
        )
        valid_emotions = MELD_VALID_EMOTIONS
        collate_fn_impl = collate_fn
        eval_indices = list(range(len(dataset)))

    full_dataset_samples = len(dataset)
    if args.data_subset_percent < 100:
        subset_positions = build_eval_indices(
            len(eval_indices),
            args.data_subset_percent,
            args.seed,
        )
        eval_indices = [eval_indices[i] for i in subset_positions]
    using_subset = len(eval_indices) != full_dataset_samples
    eval_dataset = Subset(dataset, eval_indices) if using_subset else dataset
    if using_subset:
        print(
            f"Using {len(eval_indices)}/{full_dataset_samples} samples "
            f"({args.data_subset_percent:g}% requested)"
        )

    loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(
            eval_collate,
            pad_token_id=processor.tokenizer.pad_token_id,
            collate_fn_impl=collate_fn_impl,
        ),
        num_workers=0,
    )

    all_preds = []
    all_labels = []
    invalid_predictions = []
    skipped_samples = []
    per_sample_results = []

    for subset_pos, batch in enumerate(tqdm(loader, desc="Evaluating")):
        raw_idx = eval_indices[subset_pos]
        gt_emotion = batch.pop("emotions")[0]
        raw_sample = dataset.raw_dataset[raw_idx]

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
                    use_audio_in_video=False,
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
            sample_result = {
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
            }
            if using_subset:
                sample_result["subset_position"] = subset_pos
            per_sample_results.append(sample_result)
            torch.cuda.empty_cache()
            tqdm.write(f"  Skipped sample {raw_idx} (OOM/error): {str(e)[:100]}")
            continue

        pred = output_text[0].strip().lower()
        raw_output = output_text[0].strip()
        is_valid = pred in valid_emotions

        sample_result = {
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
        if using_subset:
            sample_result["subset_position"] = subset_pos
        per_sample_results.append(sample_result)

        if not is_valid:
            invalid_predictions.append((raw_idx, raw_output, gt_emotion))
        else:
            all_preds.append(pred)
            all_labels.append(gt_emotion)

    # --- Report metrics ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"Split: {args.split} | Modalities: {args.modalities} | "
        f"Corrupt: {args.corrupt} | Preset: {args.corruption_preset}"
    )
    print(f"Total samples: {len(eval_dataset)}")
    if using_subset:
        print(f"Full dataset samples: {full_dataset_samples}")
        print(f"Data subset percent: {args.data_subset_percent:g}")
    print(f"Valid predictions: {len(all_preds)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Skipped (OOM/error): {len(skipped_samples)}")

    if invalid_predictions:
        print(f"\n--- Invalid Predictions ({len(invalid_predictions)}) ---")
        for idx, model_out, gt in invalid_predictions:
            print(f"  Sample {idx}: model='{model_out}' | gt='{gt}'")

    label_names = sorted(valid_emotions)

    acc = None
    macro_f1 = None
    weighted_f1 = None
    auroc = None
    auprc = None
    mcc = None

    if all_preds:
        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, labels=label_names, zero_division=0))

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(
            all_labels,
            all_preds,
            labels=label_names,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            all_labels,
            all_preds,
            labels=label_names,
            average="weighted",
            zero_division=0,
        )

        print(f"Accuracy:                     {acc:.4f}")
        print(f"Macro F1:                     {macro_f1:.4f}")
        print(f"Weighted F1:                  {weighted_f1:.4f}")

        y_true_bin = label_binarize(all_labels, classes=label_names)
        y_pred_bin = label_binarize(all_preds, classes=label_names)
        try:
            auroc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
        except ValueError as e:
            print(f"Macro AUROC (OVR):            unavailable ({e})")

        try:
            auprc = average_precision_score(y_true_bin, y_pred_bin, average="macro")
        except ValueError as e:
            print(f"Macro Avg Precision (AUPRC):  unavailable ({e})")

        mcc = matthews_corrcoef(all_labels, all_preds)

        if auroc is not None:
            print(f"Macro AUROC (OVR):            {auroc:.4f}")
        if auprc is not None:
            print(f"Macro Avg Precision (AUPRC):  {auprc:.4f}")
        print(f"MCC:                          {mcc:.4f}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nPeak VRAM usage: {peak_vram:.2f} GB")

    modalities_str = "+".join(sorted(args.modalities))
    corrupt_str = f"corrupt_{args.corruption_preset}" if args.corrupt else "clean"
    model_str = args.run_label or ("finetuned" if args.adapter_path else "base")
    subset_str = ""
    if using_subset:
        subset_pct = f"{args.data_subset_percent:g}".replace(".", "p")
        subset_str = f"_subset{subset_pct}pct"
    ds_prefix = "results_iemocap" if args.dataset == "iemocap" else "results"
    output_filename = f"{ds_prefix}_{args.split}_{modalities_str}_{corrupt_str}_{model_str}{subset_str}.json"
    output_dir = args.output_dir or os.path.join("results", args.dataset)
    output_path = os.path.join(output_dir, output_filename)

    results_json = {
        "dataset": args.dataset,
        "split": args.split,
        "modalities": args.modalities,
        "corrupt": args.corrupt,
        "corruption_preset": args.corruption_preset,
        "adapter_path": args.adapter_path,
        "manifest": os.path.abspath(args.manifest) if args.manifest else None,
        "iemocap_sessions": args.iemocap_sessions,
        "iemocap_manifest_split": args.iemocap_manifest_split,
        "iemocap_holdout_n": args.iemocap_holdout_n if args.dataset == "iemocap" else None,
        "iemocap_holdout_seed": args.iemocap_holdout_seed if args.dataset == "iemocap" else None,
        "iemocap_val_ratio": args.iemocap_val_ratio if args.dataset == "iemocap" else None,
        "iemocap_split_seed": args.iemocap_split_seed if args.dataset == "iemocap" else None,
        "data_subset_percent": args.data_subset_percent,
        "total_samples": len(eval_dataset),
        "full_dataset_samples": full_dataset_samples,
        "sample_indices": eval_indices if (using_subset or args.dataset == "iemocap") else None,
        "valid_predictions": len(all_preds),
        "invalid_predictions": len(invalid_predictions),
        "skipped_samples": len(skipped_samples),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auroc_macro_ovr": auroc,
        "auprc_macro_ovr": auprc,
        "mcc": mcc,
        "predictions": per_sample_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
