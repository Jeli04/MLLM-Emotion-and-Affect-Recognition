import argparse
import csv
import json
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("root").setLevel(logging.ERROR)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, set_seed
from peft import PeftModel
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

from src.meld_dataset import RawMELDDataset, EMOTION2ID

MELD_VALID_EMOTIONS = set(EMOTION2ID.keys())
MELD_SYSTEM_PROMPT = (
    "Your job as a helpful assistant is to detect what emotion is being expressed "
    "from the inputs. Output only one word. Here are the options: "
    "anger, disgust, fear, joy, neutral, sadness, surprise."
)
IEMOCAP_EMOTIONS = (
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
)
IEMOCAP_VALID_EMOTIONS = set(IEMOCAP_EMOTIONS)
IEMOCAP_SYSTEM_PROMPT = (
    "Your job as a helpful assistant is to detect what emotion is being expressed "
    "from the inputs. Output only one word. Here are the options: "
    + ", ".join(IEMOCAP_EMOTIONS)
    + "."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-Omni on MELD/IEMOCAP emotion recognition"
    )
    parser.add_argument(
        "--dataset",
        choices=["meld", "iemocap"],
        default="meld",
        help="Dataset backend (default: meld)",
    )
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
    parser.add_argument("--run_label", default=None,
                        help="Tag for the saved-results filename (e.g. 'finetune', 'student_teacher'). "
                             "Overrides the default 'base'/'finetuned' tag so different adapters don't collide.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to iemocap_utterance_labels.csv (required for --dataset iemocap)",
    )
    parser.add_argument(
        "--iemocap_sessions",
        nargs="*",
        default=None,
        help="Optional Session1 Session2 ... filter for iemocap",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of IEMOCAP utterances (for smoke tests)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where results are saved. Defaults to results/meld or results/iemocap.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for python/numpy/torch")
    return parser.parse_args()


def _resolve_iemocap_media_path(raw_path: str, iemocap_root: Path) -> str:
    p = (raw_path or "").strip()
    if not p:
        return ""
    path = Path(p)
    if path.is_file():
        return str(path.resolve())
    if not path.is_absolute():
        cand = iemocap_root / path
        if cand.is_file():
            return str(cand.resolve())
    if path.is_absolute():
        parts = path.parts
        for i, part in enumerate(parts):
            if part == "IEMOCAP_full_release" and i + 1 < len(parts):
                cand = iemocap_root / Path(*parts[i + 1 :])
                if cand.is_file():
                    return str(cand.resolve())
                break
    return p


def load_iemocap_manifest(manifest_path, split=None, sessions=None, max_samples=None):
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    iemocap_root = manifest.resolve().parent.parent
    rows = []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_split_col = "split" in fieldnames
        for row in reader:
            emo = (row.get("emotion") or "").strip().lower()
            if not emo or emo == "no_agreement" or emo not in IEMOCAP_VALID_EMOTIONS:
                continue
            if sessions:
                sess = (row.get("session") or "").strip()
                if sess not in sessions:
                    continue
            if split is not None and has_split_col:
                if (row.get("split") or "").strip() != split:
                    continue
            rows.append(
                {
                    "text": (row.get("text") or "").strip(),
                    "emotion": emo,
                    "dialogue_id": (row.get("recording_id") or "").strip(),
                    "utterance_id": (row.get("utterance_id") or "").strip(),
                    "session": (row.get("session") or "").strip(),
                    "video_path": _resolve_iemocap_media_path(
                        row.get("video_path") or "", iemocap_root
                    ),
                    "audio_path": _resolve_iemocap_media_path(
                        row.get("wav_path") or "", iemocap_root
                    ),
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_messages(sample, modalities, system_prompt):
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
            audio_source = sample.get("audio_path") or sample["video_path"]
            user_content.append({"type": "audio", "audio": audio_source})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return messages


def main():
    args = parse_args()
    set_seed(args.seed)
    print(
        f"Evaluating dataset='{args.dataset}' split='{args.split}' with modalities={args.modalities}, "
        f"seed={args.seed}"
    )

    if args.dataset == "meld":
        dataset = RawMELDDataset(args.data_root, split=args.split, load_audio=False)
        valid_emotions = MELD_VALID_EMOTIONS
        system_prompt = MELD_SYSTEM_PROMPT
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required when --dataset iemocap")
        dataset = load_iemocap_manifest(
            args.manifest,
            split=args.split,
            sessions=args.iemocap_sessions,
            max_samples=args.max_samples,
        )
        valid_emotions = IEMOCAP_VALID_EMOTIONS
        system_prompt = IEMOCAP_SYSTEM_PROMPT

    # Load model
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

    all_preds = []
    all_labels = []
    invalid_predictions = []  # (index, model_output, ground_truth)
    skipped_samples = []  # (index, error_message, ground_truth)
    per_sample_results = []

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        gt_emotion = sample["emotion"]
        messages = build_messages(sample, args.modalities, system_prompt)

        try:
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
            inputs = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

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
                "dialogue_id": sample.get("dialogue_id"),
                "utterance_id": sample.get("utterance_id"),
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
        is_valid = pred in valid_emotions

        per_sample_results.append({
            "sample_index": i,
            "dialogue_id": sample.get("dialogue_id"),
            "utterance_id": sample.get("utterance_id"),
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
        label_names = sorted(valid_emotions)
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
    model_str = args.run_label or ("finetuned" if args.adapter_path else "base")
    if args.dataset == "meld":
        output_filename = f"results_{args.split}_{modalities_str}_{model_str}.json"
        output_dir = args.output_dir or os.path.join("results", "meld")
    else:
        output_filename = f"results_iemocap_{args.split}_{modalities_str}_{model_str}.json"
        output_dir = args.output_dir or os.path.join("results", "iemocap")
    output_path = os.path.join(output_dir, output_filename)

    results_json = {
        "dataset": args.dataset,
        "split": args.split,
        "modalities": args.modalities,
        "adapter_path": args.adapter_path,
        "total_samples": len(dataset),
        "valid_predictions": len(all_preds),
        "invalid_predictions": len(invalid_predictions),
        "skipped_samples": len(skipped_samples),
        "accuracy": accuracy_score(all_labels, all_preds) if all_preds else None,
        "auroc_macro_ovr": roc_auc_score(
            label_binarize(all_labels, classes=sorted(valid_emotions)),
            label_binarize(all_preds, classes=sorted(valid_emotions)),
            average="macro",
        ) if all_preds else None,
        "auprc_macro_ovr": average_precision_score(
            label_binarize(all_labels, classes=sorted(valid_emotions)),
            label_binarize(all_preds, classes=sorted(valid_emotions)),
            average="macro",
        ) if all_preds else None,
        "mcc": matthews_corrcoef(all_labels, all_preds) if all_preds else None,
        "predictions": per_sample_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
