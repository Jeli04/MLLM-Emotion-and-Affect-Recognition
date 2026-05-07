import argparse
import json
import logging
import math
import os
import random
import warnings
from collections import Counter
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

# Mapping pred -> plausible ground-truth labels. Used to keep DPO pairs where
# the model's wrong prediction is in a known confusion direction. Edges include
# both "natural" emotional confusions and observed over-prediction directions
# (e.g. anger/surprise/joy frequently produced when the gold is neutral), so
# DPO can correct the SFT's most common false positives.
MELD_CONFUSION_PAIRS = {
    "neutral":  ["sadness", "joy", "anger", "surprise"],
    "sadness":  ["neutral", "fear", "anger"],
    "joy":      ["neutral", "surprise", "anger"],
    "anger":    ["disgust", "neutral", "joy", "sadness"],
    "disgust":  ["anger", "neutral", "sadness"],
    "surprise": ["joy", "fear", "neutral"],
    "fear":     ["surprise", "sadness", "neutral"],
}

# Plausible confusions for IEMOCAP 10-class labels (same semantics as MELD graph where applicable).
IEMOCAP_CONFUSION_PAIRS = {
    "neutral":    ["sad", "happy", "frustrated", "angry", "excited"],
    "sad":        ["neutral", "fearful", "frustrated", "angry"],
    "happy":      ["neutral", "excited", "surprised", "frustrated"],
    "angry":      ["disgusted", "frustrated", "neutral", "sad"],
    "disgusted":  ["angry", "frustrated", "neutral"],
    "fearful":    ["surprised", "sad", "neutral"],
    "surprised":  ["happy", "fearful", "excited", "neutral"],
    "frustrated": ["angry", "neutral", "sad", "disgusted"],
    "excited":    ["happy", "surprised", "neutral"],
    "other":      ["neutral", "happy", "sad"],
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
                        help="Corruption preset to use when --corrupt is enabled")
    parser.add_argument("--output_dir", default=os.path.join("results", "dpo"),
                        help="Directory where evaluation and DPO files are saved")
    parser.add_argument(
        "--output_name_suffix",
        default=None,
        help="Optional suffix to append to output filenames, e.g. student_teacher",
    )
    parser.add_argument("--correct_sample_ratio", type=float, default=0.0,
                        help="Target fraction of final DPO samples drawn from correct model predictions. "
                             "Correct predictions are eligible only when the highest-scoring "
                             "non-ground-truth emotion is a known confusion-pair partner. "
                             "Use 0 to disable correct-prediction samples.")
    parser.add_argument("--correct_sample_seed", type=int, default=42,
                        help="Random seed for selecting correct-prediction DPO samples")
    parser.add_argument("--max_chosen_per_class", type=int, default=None,
                        help="Optional cap on the number of DPO pairs per `chosen` label. "
                             "Applied after confusion-pair and correct-prediction selection. "
                             "Use to prevent majority `chosen` (e.g. neutral, joy) from "
                             "dominating the preference signal. None disables the cap.")
    parser.add_argument("--max_chosen_per_class_seed", type=int, default=42,
                        help="Random seed used when subsampling pairs to enforce "
                             "--max_chosen_per_class")
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
    emotion_score_info=None,
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

    result = {
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
    if emotion_score_info is not None:
        result["emotion_score_info"] = emotion_score_info

    if dataset == "iemocap":
        result["audio_path"] = raw_sample.get("audio_path") or ""
        result["session"] = raw_sample.get("session", "")
    return result


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


def cap_samples_per_chosen(samples, max_per_class, seed):
    """Downsample DPO pairs so no `chosen` label exceeds ``max_per_class``.

    Returns ``(kept, dropped_per_class)`` where ``dropped_per_class`` maps the
    `chosen` label to the number of pairs dropped for that label. When
    ``max_per_class`` is ``None`` or no class exceeds the cap, samples are
    returned unchanged.
    """
    if max_per_class is None or max_per_class <= 0:
        return list(samples), {}

    by_chosen = {}
    for idx, sample in enumerate(samples):
        by_chosen.setdefault(sample["chosen"], []).append(idx)

    rng = random.Random(seed)
    keep_indices = set()
    dropped = {}
    for chosen, indices in by_chosen.items():
        if len(indices) <= max_per_class:
            keep_indices.update(indices)
            continue
        kept = rng.sample(indices, max_per_class)
        keep_indices.update(kept)
        dropped[chosen] = len(indices) - max_per_class

    kept_samples = [s for i, s in enumerate(samples) if i in keep_indices]
    return kept_samples, dropped


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


def score_candidate_emotions(model, tokenizer, inputs, candidate_emotions):
    """Score every emotion as the assistant response for the current prompt.

    The DPO builder uses generation for the argmax prediction, but correct
    predictions need a runner-up label. This helper teacher-forces each valid
    emotion after the exact processed prompt and records length-normalized
    response logprobs, avoiding randomly invented rejected labels.
    """
    prompt_ids = inputs["input_ids"]
    prompt_attention = inputs["attention_mask"]
    prompt_len = prompt_ids.shape[-1]
    scores = {}
    token_logprobs = {}

    for emotion in candidate_emotions:
        response_ids = tokenizer(
            emotion,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(prompt_ids.device)
        if response_ids.numel() == 0:
            continue

        full_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask")
        }
        full_inputs["input_ids"] = torch.cat([prompt_ids, response_ids], dim=-1)
        full_inputs["attention_mask"] = torch.cat(
            [
                prompt_attention,
                prompt_attention.new_ones(response_ids.shape),
            ],
            dim=-1,
        )

        outputs = model.thinker(**full_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        emotion_token_logprobs = []
        for offset, token_id in enumerate(response_ids[0]):
            logit_index = prompt_len + offset - 1
            next_token_logprobs = logits[0, logit_index].float().log_softmax(dim=-1)
            emotion_token_logprobs.append(float(next_token_logprobs[token_id].item()))

        mean_logprob = sum(emotion_token_logprobs) / len(emotion_token_logprobs)
        scores[emotion] = mean_logprob
        token_logprobs[emotion] = emotion_token_logprobs

    ranking = sorted(scores, key=scores.get, reverse=True)
    return {
        "emotion_mean_logprobs": scores,
        "emotion_token_logprobs": token_logprobs,
        "emotion_ranking": ranking,
    }


def add_runner_up_info(score_info, gt_emotion, confusion_pairs):
    ranking = score_info["emotion_ranking"]
    runner_up = next((emotion for emotion in ranking if emotion != gt_emotion), None)
    score_info["top_emotion"] = ranking[0] if ranking else None
    score_info["runner_up_emotion"] = runner_up
    score_info["runner_up_is_confusion_pair"] = (
        runner_up in confusion_pairs.get(gt_emotion, [])
    )
    return score_info


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
    # emotion for the ground-truth label. Correct predictions are considered
    # only if their forced-choice runner-up emotion is a confusion-pair partner.
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

        if args.dataset == "iemocap":
            sample_result["session"] = raw_sample.get("session", "")
        per_sample_results.append(sample_result)

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
            elif pred == gt_emotion and args.correct_sample_ratio > 0:
                try:
                    with torch.no_grad():
                        score_info = score_candidate_emotions(
                            model,
                            processor.tokenizer,
                            inputs,
                            sorted(valid_emotions),
                        )
                    score_info = add_runner_up_info(score_info, gt_emotion, confusion_pairs)
                    sample_result["emotion_score_info"] = score_info
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    sample_result["emotion_score_error"] = str(e)
                    torch.cuda.empty_cache()
                    tqdm.write(
                        f"  Could not score correct sample {i}: {str(e)[:100]}"
                    )
                    continue

                runner_up = score_info["runner_up_emotion"]
                if score_info["runner_up_is_confusion_pair"]:
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
                            rejected_emotion=runner_up,
                            selection_reason="correct_prediction_runner_up_confusion_pair",
                            emotion_score_info=score_info,
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

    pre_cap_count = len(dpo_samples)
    pre_cap_chosen_counts = dict(Counter(s["chosen"] for s in dpo_samples))
    dpo_samples, dropped_per_class = cap_samples_per_chosen(
        dpo_samples,
        args.max_chosen_per_class,
        args.max_chosen_per_class_seed,
    )
    dpo_sample_indices = [s["sample_index"] for s in dpo_samples]

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
    print(f"DPO eligible correct-prediction candidates: {len(correct_dpo_candidates)}")
    print(f"DPO sampled correct-prediction samples: {len(correct_dpo_samples)}")
    print(f"DPO total samples (pre-cap): {pre_cap_count}")
    print(f"DPO chosen distribution (pre-cap): {pre_cap_chosen_counts}")
    if args.max_chosen_per_class is not None:
        print(
            f"DPO --max_chosen_per_class={args.max_chosen_per_class} "
            f"(seed={args.max_chosen_per_class_seed}); dropped per class: {dropped_per_class}"
        )
    post_cap_chosen_counts = dict(Counter(s["chosen"] for s in dpo_samples))
    print(f"DPO total samples (post-cap): {len(dpo_samples)}")
    print(f"DPO chosen distribution (post-cap): {post_cap_chosen_counts}")

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
    if args.output_name_suffix:
        model_str = args.output_name_suffix.strip().replace(" ", "_")
    else:
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
        "dpo_correct_prediction_eligible_candidate_count": len(correct_dpo_candidates),
        "dpo_correct_prediction_sample_count": len(correct_dpo_samples),
        "dpo_correct_prediction_target_ratio": args.correct_sample_ratio,
        "dpo_correct_prediction_actual_ratio": (
            len(correct_dpo_samples) / len(dpo_samples) if dpo_samples else 0.0
        ),
        "dpo_correct_prediction_selection": "runner_up_confusion_pair",
        "dpo_max_chosen_per_class": args.max_chosen_per_class,
        "dpo_max_chosen_per_class_seed": args.max_chosen_per_class_seed,
        "dpo_pre_cap_sample_count": pre_cap_count,
        "dpo_pre_cap_chosen_counts": pre_cap_chosen_counts,
        "dpo_post_cap_chosen_counts": post_cap_chosen_counts,
        "dpo_dropped_per_class": dropped_per_class,
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
        "correct_prediction_eligible_candidate_count": len(correct_dpo_candidates),
        "correct_prediction_sample_count": len(correct_dpo_samples),
        "correct_prediction_target_ratio": args.correct_sample_ratio,
        "correct_prediction_actual_ratio": (
            len(correct_dpo_samples) / len(dpo_samples) if dpo_samples else 0.0
        ),
        "correct_prediction_selection": "runner_up_confusion_pair",
        "correct_prediction_seed": args.correct_sample_seed,
        "max_chosen_per_class": args.max_chosen_per_class,
        "max_chosen_per_class_seed": args.max_chosen_per_class_seed,
        "pre_cap_sample_count": pre_cap_count,
        "pre_cap_chosen_counts": pre_cap_chosen_counts,
        "post_cap_chosen_counts": post_cap_chosen_counts,
        "dropped_per_class": dropped_per_class,
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
