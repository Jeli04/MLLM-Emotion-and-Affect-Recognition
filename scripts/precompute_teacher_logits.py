"""Precompute teacher response-slice logits for distillation.

For each sample in the requested split(s), runs the teacher model on the
clean full-modality inputs and saves only the logits at response-token
positions (i.e. where labels != -100 after the standard +1 shift). Storing
the response slice instead of the full [B, T, V] tensor keeps the cache
small (~MBs/sample instead of GBs).

Usage:
    PYTHONPATH=. python scripts/precompute_teacher_logits.py \
        --modalities text audio video \
        --splits train dev \
        --teacher_adapter_path ./ckpts/finetuned_base/lora_adapter \
        --output_dir ./cache/teacher_logits/finetuned_base_t+a+v

Then point training at it:
    python src/finetune.py ... \
        --teacher_logits_dir ./cache/teacher_logits/finetuned_base_t+a+v
"""

import argparse
import random
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, set_seed
from peft import PeftModel

import optimum.gptq.constants
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import CORRUPTION_PRESET_NAMES, CorruptedMELDDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4")
    p.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw")
    p.add_argument("--modalities", nargs="+", default=["text", "audio", "video"],
                   choices=["text", "audio", "video"])
    p.add_argument("--splits", nargs="+", default=["train", "dev"],
                   choices=["train", "dev", "test"])
    p.add_argument("--teacher_adapter_path", default=None,
                   help="LoRA adapter to load on top of the base thinker")
    p.add_argument("--base_teacher", action="store_true", default=False,
                   help="Use the base model with no adapter as teacher")
    p.add_argument("--corrupt", action="store_true", default=False,
                   help="Apply input corruption before teacher forward")
    p.add_argument("--no_corrupt", dest="corrupt", action="store_false",
                   help="Disable input corruption before teacher forward")
    p.add_argument("--corruption_preset", default="medium",
                   choices=CORRUPTION_PRESET_NAMES,
                   help="Corruption preset to use when --corrupt is enabled")
    p.add_argument("--output_dir", required=True,
                   help="Where to write per-sample logits "
                        "(structured as {output_dir}/{split}/sample_{idx:06d}.pt)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for python, numpy, torch, and dataloader workers")
    p.add_argument("--force", action="store_true", default=False,
                   help="Recompute even if a cache file already exists")
    return p.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.teacher_adapter_path is None and not args.base_teacher:
        raise SystemExit("Specify either --teacher_adapter_path or --base_teacher")

    print(
        f"Precomputing teacher logits | modalities={args.modalities} "
        f"splits={args.splits} adapter={args.teacher_adapter_path} "
        f"base_teacher={args.base_teacher} corrupt={args.corrupt} "
        f"corruption_preset={args.corruption_preset} seed={args.seed}"
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", torch_dtype=torch.float16,
    )

    for attr in ("talker", "token2wav", "audio_tokenizer"):
        if hasattr(model, attr):
            delattr(model, attr)

    teacher = model.thinker
    if args.teacher_adapter_path is not None:
        teacher = PeftModel.from_pretrained(teacher, args.teacher_adapter_path)
    teacher.eval()
    first_device = next(teacher.parameters()).device

    for split in args.splits:
        out_dir = Path(args.output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = CorruptedMELDDataset(
            args.data_root,
            processor=processor,
            split=split,
            modalities=tuple(args.modalities),
            corrupt=args.corrupt,
            corruption_preset=args.corruption_preset,
            for_training=True,
            distill=False,
        )
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=partial(
                collate_fn,
                pad_token_id=processor.tokenizer.pad_token_id,
                padding_side="right",
            ),
        )

        for i, batch in enumerate(tqdm(loader, desc=split)):
            cache_file = out_dir / f"sample_{i:06d}.pt"
            if cache_file.exists() and not args.force:
                continue

            labels = batch.pop("labels", None)
            if labels is None:
                continue

            inputs = {
                k: v.to(first_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels_dev = labels.to(first_device)

            with torch.no_grad():
                out = teacher(**inputs)

            logits = out.logits  # [1, T, V] fp16
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_dev[..., 1:].contiguous()
            keep = shift_labels != -100
            response_logits = shift_logits[keep].to(torch.float16).cpu()
            response_labels = shift_labels[keep].cpu()

            torch.save(
                {
                    "response_logits": response_logits,
                    "response_labels": response_labels,
                },
                cache_file,
            )


if __name__ == "__main__":
    main()
