import argparse
import logging
import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.getLogger().addFilter(
    lambda r: "System prompt modified" not in r.getMessage()
)

# Patch optimum to recognize Qwen2.5-Omni's layer structure
import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import CorruptedMELDDataset, collate_fn


def set_adapter_trainability(model, adapter_name, trainable):
    marker = f".{adapter_name}."
    for name, param in model.named_parameters():
        if marker in name:
            param.requires_grad = trainable


class StudentTeacherTrainer(Trainer):
    """Trainer for paired full/masked inputs.

    Distill mode optimizes weighted CE on both full and masked student inputs,
    plus lambda*KL(sg(p_teacher_full) || p_student_mask). It falls back to the
    default CE path for unpaired inputs, so non-distill runs are unaffected.
    """

    def __init__(
        self,
        *args,
        lambda_kl=0.1,
        distill_temperature=2.0,
        full_ce_weight=0.5,
        mask_ce_weight=0.5,
        student_adapter_name="default",
        teacher_adapter_name=None,
        base_teacher=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_kl = lambda_kl
        if distill_temperature <= 0:
            raise ValueError("distill_temperature must be > 0")
        if full_ce_weight < 0 or mask_ce_weight < 0:
            raise ValueError("CE weights must be non-negative")
        self.distill_temperature = distill_temperature
        self.full_ce_weight = full_ce_weight
        self.mask_ce_weight = mask_ce_weight
        self.student_adapter_name = student_adapter_name
        self.teacher_adapter_name = teacher_adapter_name
        self.base_teacher = base_teacher

    def _set_adapter(self, model, adapter_name, trainable=None):
        if adapter_name is not None and hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
            if trainable is not None:
                set_adapter_trainability(model, adapter_name, trainable)

    def _teacher_forward(self, model, full_inputs_no_labels, full_student_out):
        if self.teacher_adapter_name is None and not self.base_teacher:
            return full_student_out

        was_training = model.training
        teacher_context = nullcontext()
        if self.teacher_adapter_name is not None:
            self._set_adapter(model, self.teacher_adapter_name, trainable=False)
        else:
            teacher_context = model.disable_adapter()

        model.eval()
        # Bypass accelerate's ConvertOutputsToFp32 wrapper on model.forward —
        # it would cast the full [B, T, V] logits to fp32, costing ~8 GB
        # transient memory we don't need (KL/CE only cast the response slice).
        inner_forward = getattr(model.forward, "model_forward", model.forward)
        with torch.no_grad(), teacher_context:
            teacher_out = inner_forward(**full_inputs_no_labels)
        if was_training:
            model.train()
        self._set_adapter(model, self.student_adapter_name, trainable=True)
        return teacher_out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_cached_distill = "mask" in inputs and "teacher_response_logits" in inputs
        if ("full" not in inputs or "mask" not in inputs) and not is_cached_distill:
            return super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        full_inputs = inputs.get("full")
        full_inputs_no_labels = (
            {k: v for k, v in full_inputs.items() if k != "labels"}
            if full_inputs is not None
            else None
        )
        mask_inputs = inputs["mask"]
        mask_inputs_no_labels = {k: v for k, v in mask_inputs.items() if k != "labels"}

        self._set_adapter(model, self.student_adapter_name, trainable=True)
        use_cached_teacher = "teacher_response_logits" in inputs
        needs_full_student = (
            full_inputs is not None
            and (
                self.full_ce_weight > 0
                or (
                    not use_cached_teacher
                    and self.teacher_adapter_name is None
                    and not self.base_teacher
                )
            )
        )
        if full_inputs is None and (
            self.full_ce_weight > 0
            or (
                not use_cached_teacher
                and self.teacher_adapter_name is None
                and not self.base_teacher
            )
        ):
            raise ValueError("Distillation needs full inputs, but the batch only has mask inputs")

        def response_logits_and_labels(logits, labels):
            # Only gather supervised response row to reduce memory
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            keep = shift_labels != -100
            return shift_logits[keep], shift_labels[keep]

        full_student_out = model(**full_inputs_no_labels) if needs_full_student else None
        mask_out = model(**mask_inputs_no_labels)
        student_resp, student_labels = response_logits_and_labels(
            mask_out.logits, mask_inputs["labels"],
        )

        if student_resp.numel() == 0:
            mask_ce_loss = torch.zeros((), device=mask_out.logits.device, dtype=mask_out.logits.dtype)
        else:
            mask_ce_loss = F.cross_entropy(student_resp.float(), student_labels)

        if full_student_out is not None:
            full_resp, full_labels = response_logits_and_labels(
                full_student_out.logits, full_inputs["labels"],
            )
            full_ce_loss = (
                F.cross_entropy(full_resp.float(), full_labels)
                if full_resp.numel() > 0
                else torch.zeros((), device=mask_ce_loss.device, dtype=mask_ce_loss.dtype)
            )
        else:
            full_ce_loss = torch.zeros((), device=mask_ce_loss.device, dtype=mask_ce_loss.dtype)
        ce_loss = self.full_ce_weight * full_ce_loss + self.mask_ce_weight * mask_ce_loss

        if use_cached_teacher:
            device = mask_out.logits.device
            teacher_resp = torch.cat(
                [t.to(device) for t in inputs["teacher_response_logits"]], dim=0
            ).detach()
            teacher_labels = torch.cat(
                [t.to(device) for t in inputs["teacher_response_labels"]], dim=0
            ).detach()
        else:
            teacher_out = self._teacher_forward(
                model, full_inputs_no_labels, full_student_out,
            )
            teacher_logits = teacher_out.logits.detach()
            teacher_resp, teacher_labels = response_logits_and_labels(
                teacher_logits, full_inputs["labels"],
            )
            teacher_labels = teacher_labels.detach()

        # checks for cases where teacher output is completely empty
        if teacher_resp.numel() == 0:
            teacher_nll = torch.zeros((), device=ce_loss.device)
            teacher_token_acc = torch.zeros((), device=ce_loss.device)
        else:
            teacher_nll = F.cross_entropy(teacher_resp.float(), teacher_labels)
            teacher_token_acc = (
                teacher_resp.argmax(dim=-1).eq(teacher_labels).float().mean()
            )

        # checks for cases where the student output is nothing if all the modalilites are masked out
        if student_resp.numel() == 0 or student_resp.shape != teacher_resp.shape:
            loss = ce_loss
            kl_val = torch.zeros((), device=ce_loss.device)
            kl_raw_val = kl_val
            kl_weighted_val = kl_val
        else:
            # KL in fp32 for numerical stability under fp16 training.
            t = self.distill_temperature
            log_p_mask = F.log_softmax(student_resp.float() / t, dim=-1)
            p_full = F.softmax(teacher_resp.float() / t, dim=-1)
            kl_raw = F.kl_div(log_p_mask, p_full, reduction="batchmean")
            kl = kl_raw * (t ** 2)
            kl_weighted = self.lambda_kl * kl
            loss = ce_loss + kl_weighted
            kl_val = kl.detach()
            kl_raw_val = kl_raw.detach()
            kl_weighted_val = kl_weighted.detach()

        self._last_ce = ce_loss.detach()
        self._last_full_ce = full_ce_loss.detach()
        self._last_mask_ce = mask_ce_loss.detach()
        self._last_kl = kl_val
        self._last_kl_raw = kl_raw_val
        self._last_kl_weighted = kl_weighted_val
        self._last_teacher_nll = teacher_nll.detach()
        self._last_teacher_token_acc = teacher_token_acc.detach()

        return (loss, mask_out) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_ce"):
            logs = {
                **logs,
                "loss/ce": float(self._last_ce),
                "loss/full_ce": float(self._last_full_ce),
                "loss/mask_ce": float(self._last_mask_ce),
                "loss/kl": float(self._last_kl),
                "loss/kl_raw": float(self._last_kl_raw),
                "loss/kl_weighted": float(self._last_kl_weighted),
                "teacher/nll": float(self._last_teacher_nll),
                "teacher/token_acc": float(self._last_teacher_token_acc),
            }
        return super().log(logs, *args, **kwargs)


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
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (parallel video/audio decoding)")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir to resume from (e.g. "
                             "./ckpts/finetuned_distill/checkpoint-1800). "
                             "Pass 'True' to auto-pick the latest in output_dir.")
    parser.add_argument("--corrupt", action="store_true", default=True,
                        help="Apply noise/corruption to raw inputs")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false")
    parser.add_argument("--modality_mask", action="store_true", default=True,
                        help="Randomly drop modalities from the student branch in distill mode")
    parser.add_argument("--no_modality_mask", dest="modality_mask", action="store_false",
                        help="Disable modality dropout; corruption can still be applied")
    parser.add_argument("--clean_teacher", action="store_true", default=False,
                        help="In distill mode, keep the full/teacher branch uncorrupted "
                             "while the mask/student branch follows --corrupt")

    parser.add_argument("--distill", action="store_true", default=False,
                        help="Enable student-teacher distillation: two forward passes "
                             "per sample (full modalities vs. random masked subset), "
                             "loss = weighted CE(full/mask student) + "
                             "lambda_kl * KL(sg(p_teacher_full) || p_student_mask).")
    parser.add_argument("--lambda_kl", type=float, default=0.1,
                        help="Weight on the KL consistency term when --distill is set")
    parser.add_argument("--distill_temperature", type=float, default=2.0,
                        help="Temperature for distillation soft targets. The KL term "
                             "is multiplied by temperature^2.")
    parser.add_argument("--full_ce_weight", type=float, default=0.5,
                        help="Weight for full-modality student CE in distill mode")
    parser.add_argument("--mask_ce_weight", type=float, default=0.5,
                        help="Weight for masked-modality student CE in distill mode")
    parser.add_argument("--teacher_adapter_path", type=str, default=None,
                        help="Optional LoRA adapter path for a frozen full-modality "
                             "teacher. If omitted, the full-modality student pass is "
                             "used as an online adapter-enabled teacher.")
    parser.add_argument("--base_teacher", action="store_true", default=False,
                        help="Use the frozen base model with adapters disabled as the "
                             "teacher. Ignored when --teacher_adapter_path is set.")
    parser.add_argument("--teacher_logits_dir", type=str, default=None,
                        help="Directory of precomputed teacher response-slice logits "
                             "(produced by scripts/precompute_teacher_logits.py). When "
                             "set, the live teacher forward is skipped and cached "
                             "logits are used for KL.")

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
    using_cached_teacher = args.distill and args.teacher_logits_dir is not None
    if using_cached_teacher and args.teacher_adapter_path is not None:
        print("Using cached teacher logits; teacher_adapter_path will not be loaded at train time.")

    print(
        f"Finetuning with modalities={args.modalities}, corrupt={args.corrupt}, "
        f"modality_mask={args.modality_mask}, clean_teacher={args.clean_teacher}, "
        f"distill={args.distill}, lambda_kl={args.lambda_kl}, "
        f"distill_temperature={args.distill_temperature}, "
        f"full_ce_weight={args.full_ce_weight}, mask_ce_weight={args.mask_ce_weight}, "
        f"teacher_adapter_path={args.teacher_adapter_path}, base_teacher={args.base_teacher}, "
        f"wandb={args.wandb}"
    )

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],#["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    thinker = get_peft_model(model.thinker, lora_config)
    teacher_adapter_name = None
    if args.teacher_adapter_path is not None and not using_cached_teacher:
        teacher_adapter_name = "teacher"
        thinker.load_adapter(
            args.teacher_adapter_path,
            adapter_name=teacher_adapter_name,
            is_trainable=False,
        )
        set_adapter_trainability(thinker, teacher_adapter_name, False)
        thinker.set_adapter("default")

    # remove unused speech-generation side
    if hasattr(model, "talker"):
        del model.talker
    if hasattr(model, "token2wav"):
        del model.token2wav
    if hasattr(model, "audio_tokenizer"):
        del model.audio_tokenizer
    del model
    torch.cuda.empty_cache()

    # Required when grad checkpointing with a frozen base: forces the embedding
    # output to require_grad so the autograd graph reaches LoRA adapters.
    if hasattr(thinker, "enable_input_require_grads"):
        thinker.enable_input_require_grads()
    thinker.gradient_checkpointing_enable()
    thinker.print_trainable_parameters()

    common = dict(
        processor=processor,
        modalities=tuple(args.modalities),
        corrupt=args.corrupt,
        for_training=True,
        distill=args.distill,
        modality_mask=args.modality_mask,
        clean_teacher=args.clean_teacher,
        include_full_branch=not (
            args.distill and args.teacher_logits_dir is not None and args.full_ce_weight == 0
        ),
    )
    train_logits_dir = (
        os.path.join(args.teacher_logits_dir, "train")
        if args.teacher_logits_dir else None
    )
    val_logits_dir = (
        os.path.join(args.teacher_logits_dir, "dev")
        if args.teacher_logits_dir else None
    )
    train_dataset = CorruptedMELDDataset(
        args.data_root, split="train", teacher_logits_dir=train_logits_dir, **common,
    )
    val_dataset = CorruptedMELDDataset(
        args.data_root, split="dev", teacher_logits_dir=val_logits_dir, **common,
    )

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
        dataloader_num_workers=args.num_workers,
    )

    trainer = StudentTeacherTrainer(
        model=thinker,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        lambda_kl=args.lambda_kl,
        distill_temperature=args.distill_temperature,
        full_ce_weight=args.full_ce_weight,
        mask_ce_weight=args.mask_ce_weight,
        student_adapter_name="default",
        teacher_adapter_name=teacher_adapter_name,
        base_teacher=args.base_teacher and teacher_adapter_name is None,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    thinker.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Model saved to {adapter_dir}")


if __name__ == "__main__":
    main()
