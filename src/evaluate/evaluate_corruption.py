import argparse
import json
import logging
import os
import warnings
from functools import partial

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import CorruptedMELDDataset, collate_fn, EMOTION2ID

ID2EMOTION = {v: k for k, v in EMOTION2ID.items()}
VALID_EMOTIONS = set(EMOTION2ID.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on MELD emotion recognition")
    parser.add_argument("--modalities", nargs="+", default=["text"],
                        choices=["text", "audio", "video"],
                        help="Which modalities to include in the input (default: text)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw",
                        help="Path to MELD.Raw directory")
    parser.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the model")
    parser.add_argument("--adapter_path", default=None,
                        help="Path to a LoRA adapter checkpoint to load on top of the base model")
    parser.add_argument("--corrupt", dest="corrupt", action="store_true", default=True,
                        help="Apply noise/corruption to inputs (default: True)")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false",
                        help="Disable input corruption")
    return parser.parse_args()


def eval_collate(batch, pad_token_id):
    """Collate wrapper that extracts non-tensor metadata before calling collate_fn."""
    emotions = [b["emotion"] for b in batch]
    tensor_batch = [{k: v for k, v in b.items() if k not in ("emotion", "label")} for b in batch]
    collated = collate_fn(tensor_batch, pad_token_id=pad_token_id, padding_side="left")
    collated["emotions"] = emotions
    return collated


def main():
    args = parse_args()
    print(f"Evaluating on split='{args.split}' with modalities={args.modalities}, corrupt={args.corrupt}")

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

    dataset = CorruptedMELDDataset(
        args.data_root,
        processor=processor,
        split=args.split,
        modalities=tuple(args.modalities),
        corrupt=args.corrupt,
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

    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        gt_emotion = batch.pop("emotions")[0]
        raw_sample = dataset.raw_dataset[i]

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
                    use_audio_in_video=("video" in args.modalities),
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
            skipped_samples.append((i, str(e), gt_emotion))
            per_sample_results.append({
                "sample_index": i,
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
            tqdm.write(f"  Skipped sample {i} (OOM/error): {str(e)[:100]}")
            continue

        pred = output_text[0].strip().lower()
        raw_output = output_text[0].strip()
        is_valid = pred in VALID_EMOTIONS

        per_sample_results.append({
            "sample_index": i,
            "dialogue_id": raw_sample["dialogue_id"],
            "utterance_id": raw_sample["utterance_id"],
            "text": raw_sample["text"],
            "ground_truth": gt_emotion,
            "prediction": pred,
            "raw_model_output": raw_output,
            "valid": is_valid,
            "skipped": False,
        })

        if not is_valid:
            invalid_predictions.append((i, raw_output, gt_emotion))
        else:
            all_preds.append(pred)
            all_labels.append(gt_emotion)

    # --- Report metrics ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Split: {args.split} | Modalities: {args.modalities} | Corrupt: {args.corrupt}")
    print(f"Total samples: {len(dataset)}")
    print(f"Valid predictions: {len(all_preds)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Skipped (OOM/error): {len(skipped_samples)}")

    if invalid_predictions:
        print(f"\n--- Invalid Predictions ({len(invalid_predictions)}) ---")
        for idx, model_out, gt in invalid_predictions:
            print(f"  Sample {idx}: model='{model_out}' | gt='{gt}'")

    if all_preds:
        label_names = sorted(VALID_EMOTIONS)
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
    corrupt_str = "corrupt" if args.corrupt else "clean"
    model_str = "finetuned" if args.adapter_path else "base"
    output_filename = f"results_{args.split}_{modalities_str}_{corrupt_str}_{model_str}.json"
    output_path = os.path.join("results", output_filename)

    results_json = {
        "split": args.split,
        "modalities": args.modalities,
        "corrupt": args.corrupt,
        "adapter_path": args.adapter_path,
        "total_samples": len(dataset),
        "valid_predictions": len(all_preds),
        "invalid_predictions": len(invalid_predictions),
        "skipped_samples": len(skipped_samples),
        "accuracy": accuracy_score(all_labels, all_preds) if all_preds else None,
        "auroc_macro_ovr": roc_auc_score(
            label_binarize(all_labels, classes=sorted(VALID_EMOTIONS)),
            label_binarize(all_preds, classes=sorted(VALID_EMOTIONS)),
            average="macro",
        ) if all_preds else None,
        "auprc_macro_ovr": average_precision_score(
            label_binarize(all_labels, classes=sorted(VALID_EMOTIONS)),
            label_binarize(all_preds, classes=sorted(VALID_EMOTIONS)),
            average="macro",
        ) if all_preds else None,
        "mcc": matthews_corrcoef(all_labels, all_preds) if all_preds else None,
        "predictions": per_sample_results,
    }

    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
