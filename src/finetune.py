import argparse
import faulthandler
import logging
import os
from functools import partial

import numpy as np
import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType

faulthandler.enable(all_threads=True)

# prevents warning message from being displayed
logging.getLogger().addFilter(
    lambda r: "System prompt modified" not in r.getMessage()
)

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import (
    CORRUPTION_PRESET_NAMES,
    CorruptedMELDDataset,
    collate_fn,
    EMOTION2ID,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen2.5-Omni on MELD emotion recognition")
    parser.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the pretrained model")
    parser.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw",
                        help="Path to MELD.Raw directory")
    parser.add_argument("--modalities", nargs="+", default=["text"],
                        choices=["text", "audio", "video"],
                        help="Which modalities to include in the input")
    parser.add_argument("--output_dir", default="./ckpts/finetuned",
                        help="Directory to save finetuned model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir to resume from. Pass "
                             "'true', 'latest', or 'auto' to use the latest "
                             "checkpoint in output_dir.")
    parser.add_argument("--corrupt", action="store_true", default=True,
                        help="Apply noise/corruption to raw inputs")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false")
    parser.add_argument("--corruption_preset", default="medium",
                        choices=CORRUPTION_PRESET_NAMES,
                        help="Corruption preset to use when --corrupt is enabled")
    parser.add_argument("--predict_corruption", action="store_true",
                        help="Train the assistant to output emotion plus corrupted input modalities")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for python, numpy, torch, and HF Trainer")

    # W&B args
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qwen25-omni-meld",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional W&B run name")

    return parser.parse_args()


def resolve_resume_from_checkpoint(resume_from_checkpoint, output_dir):
    if resume_from_checkpoint is None:
        return None

    value = resume_from_checkpoint.strip()
    lower = value.lower()
    if lower in {"false", "0", "no", "none"}:
        return None

    if lower in {"true", "1", "yes", "latest", "auto"}:
        checkpoint = get_last_checkpoint(output_dir)
        if checkpoint is None:
            raise ValueError(
                "--resume_from_checkpoint requested auto-resume, but no "
                f"checkpoint-* directory was found in {output_dir!r}."
            )
        print(f"Resuming from latest checkpoint: {checkpoint}")
        return checkpoint

    if not os.path.isdir(value):
        raise ValueError(
            f"--resume_from_checkpoint points to a missing directory: {value}"
        )
    print(f"Resuming from checkpoint: {value}")
    return value


def _unwrap_logits(logits, labels):
    """Extract the token logits tensor from model outputs passed by Trainer."""
    if isinstance(logits, torch.Tensor):
        return logits
    if hasattr(logits, "logits"):
        return logits.logits
    if isinstance(logits, (tuple, list)):
        for item in logits:
            if isinstance(item, torch.Tensor) and item.ndim >= 3:
                if labels is None or item.shape[:2] == labels.shape[:2]:
                    return item
        for item in logits:
            if isinstance(item, torch.Tensor):
                return item
        for item in logits:
            try:
                return _unwrap_logits(item, labels)
            except TypeError:
                continue
    raise TypeError(f"Could not find logits tensor in output type {type(logits)}")


def preprocess_logits_for_metrics(logits, labels):
    # Reduce [batch, seq_len, vocab_size] → [batch, seq_len] before Trainer
    # stores them, otherwise the full logit tensor OOMs on large sequences.
    logits = _unwrap_logits(logits, labels)
    return logits.argmax(dim=-1)


def make_compute_metrics(emotion_first_token_ids):
    """Return a compute_metrics closure over the per-emotion first-token IDs.

    emotion_first_token_ids: dict mapping emotion name → token ID of its first
    subword (e.g. {"neutral": 19282, ...}), built from the processor tokenizer.
    """
    id2emotion = {v: k for k, v in emotion_first_token_ids.items()}

    def compute_metrics(eval_pred):
        pred_tokens, label_ids = eval_pred
        # pred_tokens: [n, seq_len] — argmax over vocab at each position
        # label_ids:   [n, seq_len] — -100 for prompt/padding, real token elsewhere

        per_class_correct = {e: 0 for e in emotion_first_token_ids}
        per_class_total   = {e: 0 for e in emotion_first_token_ids}

        for pred_seq, label_seq in zip(pred_tokens, label_ids):
            resp = np.where(label_seq != -100)[0]
            if len(resp) == 0 or resp[0] == 0:
                continue
            first_pos = int(resp[0])
            true_tok  = int(label_seq[first_pos])
            # logits[j] predicts the token at position j+1, so the prediction
            # for the first response token lives at position first_pos - 1.
            pred_tok  = int(pred_seq[first_pos - 1])

            true_emotion = id2emotion.get(true_tok)
            if true_emotion is None:
                continue
            per_class_total[true_emotion] += 1
            if pred_tok == true_tok:
                per_class_correct[true_emotion] += 1

        total   = sum(per_class_total.values())
        correct = sum(per_class_correct.values())
        metrics = {"accuracy": correct / total if total > 0 else 0.0}
        for emotion in emotion_first_token_ids:
            n = per_class_total[emotion]
            metrics[f"acc_{emotion}"] = (
                per_class_correct[emotion] / n if n > 0 else 0.0
            )
        return metrics

    return compute_metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    print(
        f"Finetuning with modalities={args.modalities}, corrupt={args.corrupt}, "
        f"corruption_preset={args.corruption_preset}, "
        f"predict_corruption={args.predict_corruption}, wandb={args.wandb}, "
        f"seed={args.seed}"
    )

    # Set W&B env vars before Trainer is created
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    # Build emotion -> first-subword-token-ID map for compute_metrics.
    emotion_first_token_ids = {
        emotion: processor.tokenizer(
            emotion, add_special_tokens=False
        )["input_ids"][0]
        for emotion in EMOTION2ID
    }
    print("Emotion first token IDs:", emotion_first_token_ids)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],#["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    thinker = get_peft_model(model.thinker, lora_config)

    # remove unused speech-generation side
    if hasattr(model, "talker"):
        del model.talker
    if hasattr(model, "token2wav"):
        del model.token2wav
    if hasattr(model, "audio_tokenizer"):
        del model.audio_tokenizer
    del model
    torch.cuda.empty_cache()

    thinker.gradient_checkpointing_enable()
    thinker.print_trainable_parameters()

    common = dict(
        processor=processor,
        modalities=tuple(args.modalities),
        corrupt=args.corrupt,
        corruption_preset=args.corruption_preset,
        predict_corruption=args.predict_corruption,
        for_training=True,
    )
    train_dataset = CorruptedMELDDataset(args.data_root, split="train", **common)
    val_dataset = CorruptedMELDDataset(args.data_root, split="dev", **common)

    data_collator = partial(
        collate_fn,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    run_name = args.wandb_run_name or os.path.basename(os.path.abspath(args.output_dir))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        fp16=True,
        report_to="wandb" if args.wandb else "none",
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=thinker,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(emotion_first_token_ids),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    resume_from_checkpoint = resolve_resume_from_checkpoint(
        args.resume_from_checkpoint,
        args.output_dir,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    thinker.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Model saved to {adapter_dir}")


if __name__ == "__main__":
    main()
