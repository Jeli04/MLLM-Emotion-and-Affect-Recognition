import argparse
import json
import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from datasets.meld_dataset import RawMELDDataset, EMOTION2ID

ID2EMOTION = {v: k for k, v in EMOTION2ID.items()}
VALID_EMOTIONS = set(EMOTION2ID.keys())

SYSTEM_PROMPT = (
    "Your job as a helpful assistant is to detect what emotion is being expressed "
    "from the inputs. Output only one word. Here are the options: "
    "anger, disgust, fear, joy, neutral, sadness, surprise."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on MELD emotion recognition")
    parser.add_argument("--modalities", nargs="+", default=["text"],
                        choices=["text", "audio", "video"],
                        help="Which modalities to include in the input (default: text)")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw",
                        help="Path to MELD.Raw directory")
    parser.add_argument("--model_path", default="./Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the model")
    return parser.parse_args()


def build_messages(sample, modalities):
    """Build chat messages based on enabled modalities.

    When video is enabled, audio is extracted from the video via
    use_audio_in_video=True, so we skip adding a separate audio entry.
    A standalone audio entry is only added when audio is requested without video.
    """
    user_content = []
    has_video = "video" in modalities
    for mod in modalities:
        if mod == "text":
            user_content.append({"type": "text", "text": sample["text"]})
        elif mod == "video":
            user_content.append({"type": "video", "video": sample["video_path"]})
        elif mod == "audio" and not has_video:
            user_content.append({"type": "audio", "audio": sample["video_path"]})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


def main():
    args = parse_args()
    print(f"Evaluating on split='{args.split}' with modalities={args.modalities}")

    dataset = RawMELDDataset(args.data_root, split=args.split, load_audio=False)

    # Load model
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
    )

    all_preds = []
    all_labels = []
    invalid_predictions = []  # (index, model_output, ground_truth)
    skipped_samples = []  # (index, error_message, ground_truth)
    per_sample_results = []

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        gt_emotion = sample["emotion"]
        messages = build_messages(sample, args.modalities)

        try:
            use_audio = "video" in args.modalities or "audio" in args.modalities
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                load_audio_from_video=("video" in args.modalities),
                use_audio_in_video=("video" in args.modalities),
                fps=1,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            text_ids = model.generate(**inputs, max_new_tokens=128, return_audio=False)

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
                "dialogue_id": sample["dialogue_id"],
                "utterance_id": sample["utterance_id"],
                "text": sample["text"],
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
            "dialogue_id": sample["dialogue_id"],
            "utterance_id": sample["utterance_id"],
            "text": sample["text"],
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
    print(f"Split: {args.split} | Modalities: {args.modalities}")
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

        # Binary-style metrics via one-vs-rest binarization
        y_true_bin = label_binarize(all_labels, classes=label_names)
        y_pred_bin = label_binarize(all_preds, classes=label_names)
        auroc = roc_auc_score(y_true_bin, y_pred_bin, average="macro")
        auprc = average_precision_score(y_true_bin, y_pred_bin, average="macro")
        mcc = matthews_corrcoef(all_labels, all_preds)
        print(f"Macro AUROC (OVR):            {auroc:.4f}")
        print(f"Macro Avg Precision (AUPRC):  {auprc:.4f}")
        print(f"MCC:                          {mcc:.4f}")

    # Peak VRAM
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nPeak VRAM usage: {peak_vram:.2f} GB")

    # Save results to JSON
    modalities_str = "+".join(sorted(args.modalities))
    output_filename = f"results_{args.split}_{modalities_str}.json"
    output_path = os.path.join("results", output_filename)

    results_json = {
        "split": args.split,
        "modalities": args.modalities,
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

    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
