import argparse
import json
import logging
import os
import warnings
from functools import partial

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    set_seed,
)

import optimum.gptq.constants

from src.meld_dataset import (
    CORRUPTION_PRESET_NAMES,
    CorruptedMELDDataset,
    EMOTION2ID,
    collate_fn,
)

# Patch optimum to recognize Qwen2.5-Omni's layer structure.
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

VALID_EMOTIONS = set(EMOTION2ID.keys())
ALL_MODALITY_VARIATIONS = [
    ("text",),
    ("audio",),
    ("video",),
    ("text", "audio"),
    ("text", "video"),
    ("audio", "video"),
    ("text", "audio", "video"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni/AffectGPT-style checkpoints on corrupted MELD inputs."
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["text"],
        choices=["text", "audio", "video"],
        help="Which modalities to include in the input.",
    )
    parser.add_argument(
        "--run_all_modalities",
        action="store_true",
        help="Run all 7 modality variations sequentially and save one result file per combination.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--data_root",
        default="/project2/robinjia_875/lijc/data/MELD.Raw",
        help="Path to MELD.Raw directory.",
    )
    parser.add_argument(
        "--model_path",
        default="./Qwen2.5-Omni-7B-GPTQ-Int4",
        help="Path to the Qwen base model.",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Optional LoRA adapter checkpoint to load on top of the base model.",
    )
    parser.add_argument(
        "--run_label",
        default=None,
        help="Optional tag appended to saved result filenames.",
    )
    parser.add_argument(
        "--corrupt",
        dest="corrupt",
        action="store_true",
        default=True,
        help="Apply corruption to inputs.",
    )
    parser.add_argument(
        "--no_corrupt",
        dest="corrupt",
        action="store_false",
        help="Disable corruption.",
    )
    parser.add_argument(
        "--corruption_preset",
        default="medium",
        choices=CORRUPTION_PRESET_NAMES,
        help="Corruption preset to use when corruption is enabled.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("results", "meld"),
        help="Directory where per-run JSON outputs are saved.",
    )
    parser.add_argument(
        "--summary_filename",
        default="summary.json",
        help="Filename to use for the multi-run summary when --run_all_modalities is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for python/numpy/torch.",
    )
    return parser.parse_args()


def eval_collate(batch, pad_token_id):
    emotions = [b["emotion"] for b in batch]
    tensor_batch = [{k: v for k, v in b.items() if k not in ("emotion", "label")} for b in batch]
    collated = collate_fn(tensor_batch, pad_token_id=pad_token_id, padding_side="left")
    collated["emotions"] = emotions
    return collated


def canonicalize_modalities(modalities):
    ordered = tuple(mod for mod in ("text", "audio", "video") if mod in set(modalities))
    if not ordered:
        raise ValueError("At least one modality must be selected.")
    return ordered


def modality_tag(modalities):
    return "+".join(modalities)


def safe_multiclass_metric(metric_name, all_labels, all_preds, label_names):
    if not all_preds:
        return None
    try:
        y_true_bin = label_binarize(all_labels, classes=label_names)
        y_pred_bin = label_binarize(all_preds, classes=label_names)
        if metric_name == "auroc":
            return roc_auc_score(y_true_bin, y_pred_bin, average="macro")
        if metric_name == "auprc":
            return average_precision_score(y_true_bin, y_pred_bin, average="macro")
        raise ValueError(f"Unknown metric: {metric_name}")
    except ValueError:
        return None


def build_output_path(output_dir, split, modalities, corrupt, corruption_preset, run_label, adapter_path):
    modalities_str = modality_tag(modalities)
    corrupt_str = f"corrupt_{corruption_preset}" if corrupt else "clean"
    model_str = run_label or ("finetuned" if adapter_path else "base")
    filename = f"results_{split}_{modalities_str}_{corrupt_str}_{model_str}.json"
    return os.path.join(output_dir, filename)


def evaluate_modalities(args, processor, model, first_device, modalities):
    print(
        f"Evaluating split='{args.split}' | modalities={list(modalities)} | "
        f"corrupt={args.corrupt} | preset={args.corruption_preset} | seed={args.seed}"
    )

    dataset = CorruptedMELDDataset(
        args.data_root,
        processor=processor,
        split=args.split,
        modalities=modalities,
        corrupt=args.corrupt,
        corruption_preset=args.corruption_preset,
        for_training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(eval_collate, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=0,
    )

    all_preds = []
    all_labels = []
    invalid_predictions = []
    skipped_samples = []
    per_sample_results = []

    for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {modality_tag(modalities)}")):
        gt_emotion = batch.pop("emotions")[0]
        raw_sample = dataset.raw_dataset[i]
        inputs = {
            key: value.to(first_device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
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
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            skipped_samples.append((i, str(exc), gt_emotion))
            per_sample_results.append(
                {
                    "sample_index": i,
                    "dialogue_id": raw_sample["dialogue_id"],
                    "utterance_id": raw_sample["utterance_id"],
                    "text": raw_sample["text"],
                    "ground_truth": gt_emotion,
                    "prediction": None,
                    "raw_model_output": None,
                    "valid": False,
                    "skipped": True,
                    "error": str(exc),
                }
            )
            torch.cuda.empty_cache()
            tqdm.write(f"Skipped sample {i} ({modality_tag(modalities)}): {str(exc)[:120]}")
            continue

        pred = output_text[0].strip().lower()
        raw_output = output_text[0].strip()
        is_valid = pred in VALID_EMOTIONS
        per_sample_results.append(
            {
                "sample_index": i,
                "dialogue_id": raw_sample["dialogue_id"],
                "utterance_id": raw_sample["utterance_id"],
                "text": raw_sample["text"],
                "ground_truth": gt_emotion,
                "prediction": pred,
                "raw_model_output": raw_output,
                "valid": is_valid,
                "skipped": False,
            }
        )

        if is_valid:
            all_preds.append(pred)
            all_labels.append(gt_emotion)
        else:
            invalid_predictions.append((i, raw_output, gt_emotion))

    label_names = sorted(VALID_EMOTIONS)
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else None
    auroc = safe_multiclass_metric("auroc", all_labels, all_preds, label_names)
    auprc = safe_multiclass_metric("auprc", all_labels, all_preds, label_names)
    mcc = matthews_corrcoef(all_labels, all_preds) if all_preds else None

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"Split: {args.split} | Modalities: {list(modalities)} | "
        f"Corrupt: {args.corrupt} | Preset: {args.corruption_preset}"
    )
    print(f"Total samples: {len(dataset)}")
    print(f"Valid predictions: {len(all_preds)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Skipped (OOM/error): {len(skipped_samples)}")

    if invalid_predictions:
        print(f"\n--- Invalid Predictions ({len(invalid_predictions)}) ---")
        for idx, model_out, gt in invalid_predictions:
            print(f"  Sample {idx}: model='{model_out}' | gt='{gt}'")

    if all_preds:
        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, labels=label_names, zero_division=0))
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro AUROC (OVR):            {auroc if auroc is not None else 'n/a'}")
        print(f"Macro Avg Precision (AUPRC):  {auprc if auprc is not None else 'n/a'}")
        print(f"MCC:                          {mcc:.4f}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nPeak VRAM usage: {peak_vram:.2f} GB")

    result = {
        "split": args.split,
        "modalities": list(modalities),
        "corrupt": args.corrupt,
        "corruption_preset": args.corruption_preset,
        "adapter_path": args.adapter_path,
        "model_path": args.model_path,
        "total_samples": len(dataset),
        "valid_predictions": len(all_preds),
        "invalid_predictions": len(invalid_predictions),
        "skipped_samples": len(skipped_samples),
        "accuracy": accuracy,
        "auroc_macro_ovr": auroc,
        "auprc_macro_ovr": auprc,
        "mcc": mcc,
        "peak_vram_gb": peak_vram,
        "predictions": per_sample_results,
    }
    return result


def save_run_result(args, modalities, result):
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(
        args.output_dir,
        args.split,
        modalities,
        args.corrupt,
        args.corruption_preset,
        args.run_label,
        args.adapter_path,
    )
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Results saved to {output_path}")
    return output_path


def main():
    args = parse_args()
    set_seed(args.seed)

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
    modality_runs = (
        ALL_MODALITY_VARIATIONS
        if args.run_all_modalities
        else [canonicalize_modalities(args.modalities)]
    )

    run_summaries = []
    for modalities in modality_runs:
        result = evaluate_modalities(args, processor, model, first_device, modalities)
        output_path = save_run_result(args, modalities, result)
        run_summaries.append(
            {
                "modalities": list(modalities),
                "modalities_tag": modality_tag(modalities),
                "output_path": output_path,
                "accuracy": result["accuracy"],
                "auroc_macro_ovr": result["auroc_macro_ovr"],
                "auprc_macro_ovr": result["auprc_macro_ovr"],
                "mcc": result["mcc"],
                "valid_predictions": result["valid_predictions"],
                "invalid_predictions": result["invalid_predictions"],
                "skipped_samples": result["skipped_samples"],
            }
        )

    if args.run_all_modalities:
        summary = {
            "split": args.split,
            "corrupt": args.corrupt,
            "corruption_preset": args.corruption_preset,
            "adapter_path": args.adapter_path,
            "model_path": args.model_path,
            "run_label": args.run_label,
            "runs": run_summaries,
        }
        summary_path = os.path.join(args.output_dir, args.summary_filename)
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nSaved modality sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
