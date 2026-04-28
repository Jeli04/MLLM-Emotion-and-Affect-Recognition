import argparse
import faulthandler
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, random_split
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

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
    SYSTEM_PROMPT,
    apply_audio_corruptions,
    apply_video_corruptions,
    collate_fn,
    corrupt_text,
    get_corruption_config,
    load_audio_from_video,
    load_video_frames,
)


BATCH_DIM_KEYS = {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}


def parse_args():
    parser = argparse.ArgumentParser(description="DPO-tune Qwen2.5-Omni on collected MELD preference data")
    parser.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the pretrained model")
    parser.add_argument("--dpo_data_path",
                        default="results/dpo/dpo_samples_train_audio+text+video_corrupt_finetuned.json",
                        help="Path to the JSON file produced by src.dpo.build_dpo_dataset")
    parser.add_argument("--modalities", nargs="+", default=["text"],
                        choices=["text", "audio", "video"],
                        help="Which modalities to include in the input")
    parser.add_argument("--output_dir", default="./ckpts/dpo_finetuned",
                        help="Directory to save DPO-tuned LoRA adapter")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO inverse-temperature parameter")
    parser.add_argument("--eval_ratio", type=float, default=0.0,
                        help="Optional fraction of DPO data held out for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--reference_free", action="store_true",
                        help="Use reference-free preference optimization instead of disabling the LoRA adapter for reference logprobs")
    parser.add_argument("--corrupt", action="store_true", default=True,
                        help="Apply the same random input corruption style used by supervised finetuning")
    parser.add_argument("--no_corrupt", dest="corrupt", action="store_false")
    parser.add_argument("--corruption_preset", default=None,
                        choices=CORRUPTION_PRESET_NAMES,
                        help="Corruption preset to use when --corrupt is enabled. "
                             "Defaults to the preset stored in the DPO JSON, or 'medium'.")
    parser.add_argument("--max_audio_seconds", type=float, default=20.0,
                        help="Cap each DPO audio clip to this many seconds to avoid "
                             "rare long-sample OOMs. Use <=0 to disable.")
    parser.add_argument("--max_video_frames", type=int, default=16,
                        help="Uniformly downsample each DPO video clip to at most this "
                             "many frames. Use <=0 to disable.")
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1,
                        help="Ask Trainer to release unused CUDA cache before backward "
                             "every N optimizer steps. Use <=0 to disable.")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Resume Trainer state from a checkpoint path. Use 'auto' "
                             "to resume from the latest checkpoint in output_dir.")

    # W&B args
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="qwen25-omni-meld-dpo",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional W&B run name")
    return parser.parse_args()


class MELDDPODataset(Dataset):
    """Preference dataset built from dpo_samples_*.json.

    Each item returns two model-ready examples:
      - chosen: prompt + ground-truth emotion
      - rejected: prompt + confused model prediction
    """

    def __init__(
        self,
        dpo_data_path,
        processor,
        modalities,
        corrupt=True,
        audio_sr=16000,
        fps=1,
        corruption_preset=None,
        text_char_swap_prob=None,
        text_word_drop_prob=None,
        audio_noise_level=None,
        video_noise_level=None,
        max_audio_seconds=None,
        max_video_frames=None,
    ):
        self.dpo_data_path = Path(dpo_data_path)
        self.processor = processor
        self.modalities = list(modalities)
        self.corrupt = corrupt
        self.audio_sr = audio_sr
        self.fps = fps
        self.max_audio_seconds = (
            float(max_audio_seconds) if max_audio_seconds and max_audio_seconds > 0 else None
        )
        self.max_video_frames = (
            int(max_video_frames) if max_video_frames and max_video_frames > 0 else None
        )

        with open(self.dpo_data_path) as f:
            data = json.load(f)

        self.samples = data["samples"] if isinstance(data, dict) else data
        if not self.samples:
            raise ValueError(f"No DPO samples found in {self.dpo_data_path}")

        data_preset = data.get("corruption_preset") if isinstance(data, dict) else None
        sample_preset = next(
            (
                sample.get("corruption_preset")
                for sample in self.samples
                if isinstance(sample, dict) and sample.get("corruption_preset")
            ),
            None,
        )
        self.corruption_preset = corruption_preset or data_preset or sample_preset or "medium"
        self.corruption_config = get_corruption_config(
            self.corruption_preset,
            text_char_swap_prob=text_char_swap_prob,
            text_word_drop_prob=text_word_drop_prob,
            audio_noise_level=audio_noise_level,
            video_noise_level=video_noise_level,
        )
        self.text_char_swap_prob = self.corruption_config["text_char_swap_prob"]
        self.text_word_drop_prob = self.corruption_config["text_word_drop_prob"]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _sample_video_frames(frames, max_frames):
        if max_frames is None or len(frames) <= max_frames:
            return frames

        # Qwen2.5-Omni expects an even frame count for temporal patching.
        max_frames = max(2, int(max_frames))
        if max_frames % 2:
            max_frames -= 1
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return frames[indices]

    def _cap_audio(self, waveform):
        if self.max_audio_seconds is None:
            return waveform

        max_samples = max(1, int(round(self.max_audio_seconds * self.audio_sr)))
        if len(waveform) <= max_samples:
            return waveform
        return waveform[:max_samples].copy()

    def _build_prompt_messages(self, sample, text):
        user_content = []
        for mod in self.modalities:
            if mod == "text":
                user_content.append({"type": "text", "text": text})
            elif mod == "video":
                user_content.append({"type": "video", "video": sample["video_path"]})
            elif mod == "audio":
                user_content.append({"type": "audio", "audio": sample["video_path"]})

        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    def _load_media(self, sample):
        videos = None
        audio = None
        has_video = "video" in self.modalities
        has_audio = "audio" in self.modalities

        if has_video:
            try:
                frames = load_video_frames(sample["video_path"], fps=self.fps)
            except Exception:
                frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            frames = self._sample_video_frames(frames, self.max_video_frames)
            if self.corrupt:
                frames = apply_video_corruptions(frames, self.corruption_config)
            videos = [frames]

        if has_audio:
            try:
                waveform, _ = load_audio_from_video(sample["video_path"], target_sr=self.audio_sr)
            except Exception:
                waveform = np.zeros(self.audio_sr, dtype=np.float32)
            waveform = self._cap_audio(waveform)
            if self.corrupt:
                waveform = apply_audio_corruptions(
                    waveform,
                    sample_rate=self.audio_sr,
                    config=self.corruption_config,
                )
            audio = [waveform]

        return dict(
            videos=videos,
            audio=audio,
            use_audio_in_video=False,
            fps=self.fps,
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )

    def _encode_response(self, prompt_messages, response_text, processor_kwargs, prompt_len):
        messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
        ]
        rendered_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(text=rendered_text, **processor_kwargs)

        result = {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k in BATCH_DIM_KEYS else v)
            for k, v in inputs.items()
        }
        result["prompt_len"] = prompt_len
        return result

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        if self.corrupt and "text" in self.modalities:
            text = corrupt_text(
                text,
                char_swap_prob=self.text_char_swap_prob,
                word_drop_prob=self.text_word_drop_prob,
            )

        prompt_messages = self._build_prompt_messages(sample, text)
        prompt_rendered = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        processor_kwargs = self._load_media(sample)
        prompt_inputs = self.processor(text=prompt_rendered, **processor_kwargs)
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        return {
            "chosen": self._encode_response(
                prompt_messages, sample["chosen"], processor_kwargs, prompt_len,
            ),
            "rejected": self._encode_response(
                prompt_messages, sample["rejected"], processor_kwargs, prompt_len,
            ),
        }


class DPODataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        chosen = collate_fn(
            [feature["chosen"] for feature in features],
            pad_token_id=self.pad_token_id,
            padding_side="right",
        )
        rejected = collate_fn(
            [feature["rejected"] for feature in features],
            pad_token_id=self.pad_token_id,
            padding_side="right",
        )

        batch = {}
        for key, value in chosen.items():
            batch[f"chosen_{key}"] = value
        for key, value in rejected.items():
            batch[f"rejected_{key}"] = value
        return batch


class PreferenceTrainer(Trainer):
    def __init__(self, *args, beta=0.1, reference_free=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reference_free = reference_free

    @staticmethod
    def _split_batch(inputs, prefix):
        prefix = f"{prefix}_"
        return {
            key[len(prefix):]: value
            for key, value in inputs.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _get_batch_logps(logits, labels):
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:].clone()
        loss_mask = shifted_labels != -100

        if not loss_mask.any():
            return logits.new_zeros(labels.size(0))

        selected_logits = shifted_logits[loss_mask].float()
        selected_labels = shifted_labels[loss_mask]
        token_logps = (
            selected_logits.gather(1, selected_labels.unsqueeze(1)).squeeze(1)
            - selected_logits.logsumexp(dim=-1)
        )

        batch_indices = loss_mask.nonzero(as_tuple=True)[0]
        batch_logps = token_logps.new_zeros(labels.size(0))
        batch_logps.index_add_(0, batch_indices, token_logps)
        return batch_logps

    def _forward_logps(self, model, batch):
        labels = batch["labels"]
        model_inputs = {key: value for key, value in batch.items() if key != "labels"}
        # Bypass Accelerate's ConvertOutputsToFp32 wrapper. DPO only needs
        # response-token logprobs, so casting the full [B, T, V] logits tensor
        # to fp32 can OOM before we get a chance to slice it down.
        inner_forward = getattr(model.forward, "model_forward", model.forward)
        outputs = inner_forward(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return self._get_batch_logps(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_batch = self._split_batch(inputs, "chosen")
        rejected_batch = self._split_batch(inputs, "rejected")

        if not self.reference_free:
            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen_logps = self._forward_logps(model, chosen_batch)
                    ref_rejected_logps = self._forward_logps(model, rejected_batch)
            if torch.cuda.is_available() and self.args.torch_empty_cache_steps is not None:
                torch.cuda.empty_cache()

        policy_chosen_logps = self._forward_logps(model, chosen_batch)
        policy_rejected_logps = self._forward_logps(model, rejected_batch)

        if self.reference_free:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        rewards = policy_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * rewards)
        loss = losses.mean()

        if return_outputs:
            return loss, {
                "rewards": rewards.detach(),
                "policy_chosen_logps": policy_chosen_logps.detach(),
                "policy_rejected_logps": policy_rejected_logps.detach(),
            }
        return loss


def maybe_split_dataset(dataset, eval_ratio, seed):
    if eval_ratio <= 0 or len(dataset) < 2:
        return dataset, None

    eval_size = max(1, int(len(dataset) * eval_ratio))
    train_size = len(dataset) - eval_size
    if train_size == 0:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, eval_size], generator=generator)


def main():
    args = parse_args()
    set_seed(args.seed)
    print(
        f"DPO training with data={args.dpo_data_path}, modalities={args.modalities}, "
        f"corrupt={args.corrupt}, corruption_preset={args.corruption_preset or 'from_data_or_medium'}, "
        f"beta={args.beta}, wandb={args.wandb}, seed={args.seed}, "
        f"max_audio_seconds={args.max_audio_seconds}, max_video_frames={args.max_video_frames}, "
        f"torch_empty_cache_steps={args.torch_empty_cache_steps}"
    )

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        enable_audio_output=False,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    thinker = get_peft_model(model.thinker, lora_config)

    if hasattr(model, "talker"):
        del model.talker
    if hasattr(model, "token2wav"):
        del model.token2wav
    if hasattr(model, "audio_tokenizer"):
        del model.audio_tokenizer
    del model
    torch.cuda.empty_cache()

    if hasattr(thinker.config, "use_cache"):
        thinker.config.use_cache = False
    thinker.gradient_checkpointing_enable()
    thinker.print_trainable_parameters()

    dataset = MELDDPODataset(
        args.dpo_data_path,
        processor=processor,
        modalities=tuple(args.modalities),
        corrupt=args.corrupt,
        corruption_preset=args.corruption_preset,
        max_audio_seconds=args.max_audio_seconds,
        max_video_frames=args.max_video_frames,
    )
    print(f"Resolved DPO corruption_preset={dataset.corruption_preset}")
    train_dataset, eval_dataset = maybe_split_dataset(dataset, args.eval_ratio, args.seed)

    run_name = args.wandb_run_name or os.path.basename(os.path.abspath(args.output_dir))
    eval_strategy = "steps" if eval_dataset is not None else "no"

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
        eval_strategy=eval_strategy,
        eval_steps=args.save_steps if eval_dataset is not None else None,
        fp16=True,
        report_to="wandb" if args.wandb else "none",
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        torch_empty_cache_steps=(
            args.torch_empty_cache_steps if args.torch_empty_cache_steps > 0 else None
        ),
    )

    trainer = PreferenceTrainer(
        model=thinker,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DPODataCollator(processor.tokenizer.pad_token_id),
        beta=args.beta,
        reference_free=args.reference_free,
    )

    resume_from_checkpoint = (
        True if args.resume_from_checkpoint == "auto" else args.resume_from_checkpoint
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    thinker.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"DPO adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
