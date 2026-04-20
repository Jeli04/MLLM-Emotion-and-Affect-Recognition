import argparse
import os
from functools import partial

import torch
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import CorruptedMELDDataset, collate_fn


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
    parser.add_argument("--corrupt", action="store_true", default=True,
                        help="Apply noise/corruption to raw inputs")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false")

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


def main():
    args = parse_args()
    print(f"Finetuning with modalities={args.modalities}, corrupt={args.corrupt}, wandb={args.wandb}")

    # Set W&B env vars before Trainer is created
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
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
    )

    trainer = Trainer(
        model=thinker,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    thinker.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Model saved to {adapter_dir}")


if __name__ == "__main__":
    main()