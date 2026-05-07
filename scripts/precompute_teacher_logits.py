"""Precompute teacher response-slice logits for distillation.

For each sample in the requested split(s), runs the teacher model on the
requested teacher inputs and saves only the logits at response-token
positions (i.e. where labels != -100 after the standard +1 shift). Storing
the response slice instead of the full [B, T, V] tensor keeps the cache
small (~MBs/sample instead of GBs).

MELD caches are written as {output_dir}/{split}/sample_{idx:06d}.pt.
IEMOCAP caches are written flat as {output_dir}/sample_{manifest_idx:06d}.pt,
matching CorruptedIEMOCAPDataset lookups through train/val Subset indices.

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
import json
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

from src.meld_dataset import (
    CORRUPTION_PRESET_NAMES as MELD_CORRUPTION_PRESET_NAMES,
    CorruptedMELDDataset,
    collate_fn as meld_collate_fn,
)
from src.iemocap_dataset import (
    CORRUPTION_PRESET_NAMES as IEMOCAP_CORRUPTION_PRESET_NAMES,
    CorruptedIEMOCAPDataset,
    RawIEMOCAPDataset,
    collate_fn as iemocap_collate_fn,
    compute_iemocap_train_val_holdout_split,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="meld", choices=["meld", "iemocap"],
                   help="Dataset backend (default: meld)")
    p.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4")
    p.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw")
    p.add_argument("--manifest", default=None,
                   help="Path to IEMOCAP utterance manifest CSV (required for --dataset iemocap)")
    p.add_argument("--iemocap_sessions", nargs="+", default=None,
                   help="Optional IEMOCAP session filter, e.g. Session1 Session2")
    p.add_argument("--iemocap_manifest_split", default=None,
                   help="If the manifest has a 'split' column, keep only rows matching this value")
    p.add_argument("--iemocap_holdout_n", type=int, default=500,
                   help="Random eval holdout size excluded from IEMOCAP train/val; 0 disables")
    p.add_argument("--iemocap_holdout_seed", type=int, default=42,
                   help="RNG seed for the IEMOCAP holdout subset")
    p.add_argument("--iemocap_val_ratio", type=float, default=0.1,
                   help="Fraction of non-holdout IEMOCAP samples used for validation")
    p.add_argument("--iemocap_split_seed", type=int, default=43,
                   help="RNG seed for shuffling non-holdout IEMOCAP rows before train/val split")
    p.add_argument("--modalities", nargs="+", default=["text", "audio", "video"],
                   choices=["text", "audio", "video"])
    p.add_argument("--splits", nargs="+", default=["train", "dev"],
                   help="MELD: train/dev/test. IEMOCAP: train/dev|val/test|holdout/all.")
    p.add_argument("--teacher_adapter_path", default=None,
                   help="LoRA adapter to load on top of the base thinker")
    p.add_argument("--base_teacher", action="store_true", default=False,
                   help="Use the base model with no adapter as teacher")
    p.add_argument("--corrupt", action="store_true", default=False,
                   help="Apply input corruption before teacher forward")
    p.add_argument("--no_corrupt", dest="corrupt", action="store_false",
                   help="Disable input corruption before teacher forward")
    p.add_argument("--corruption_preset", default="medium",
                   help="Corruption preset to use when --corrupt is enabled")
    p.add_argument("--output_dir", required=True,
                   help="Where to write per-sample logits "
                        "(MELD: {output_dir}/{split}/sample_*.pt; "
                        "IEMOCAP: {output_dir}/sample_*.pt)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for python, numpy, torch, and dataloader workers")
    p.add_argument("--force", action="store_true", default=False,
                   help="Recompute even if a cache file already exists")
    args = p.parse_args()

    preset_names = (
        IEMOCAP_CORRUPTION_PRESET_NAMES
        if args.dataset == "iemocap"
        else MELD_CORRUPTION_PRESET_NAMES
    )
    if args.corruption_preset not in preset_names:
        p.error(
            f"--corruption_preset must be one of {preset_names}, got {args.corruption_preset!r}",
        )
    if args.dataset == "meld":
        valid = {"train", "dev", "test"}
        bad = sorted(set(args.splits) - valid)
        if bad:
            p.error(f"MELD --splits must be drawn from {sorted(valid)}, got {bad}")
    else:
        if not args.manifest:
            p.error("--manifest is required when --dataset iemocap")
        if not 0.0 <= args.iemocap_val_ratio < 1.0:
            p.error("--iemocap_val_ratio must be in [0, 1)")
        valid = {"train", "dev", "val", "test", "holdout", "all"}
        bad = sorted(set(args.splits) - valid)
        if bad:
            p.error(f"IEMOCAP --splits must be drawn from {sorted(valid)}, got {bad}")
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_response_logits(teacher, batch, cache_file, first_device, force):
    if cache_file.exists() and not force:
        return False

    labels = batch.pop("labels", None)
    if labels is None:
        return False

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

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "response_logits": response_logits,
            "response_labels": response_labels,
        },
        cache_file,
    )
    return True


def build_iemocap_split_indices(args):
    train_idx, val_idx, holdout_idx = compute_iemocap_train_val_holdout_split(
        args.manifest,
        holdout_n=args.iemocap_holdout_n,
        holdout_seed=args.iemocap_holdout_seed,
        val_ratio=args.iemocap_val_ratio,
        split_seed=args.iemocap_split_seed,
        sessions=args.iemocap_sessions,
        split=args.iemocap_manifest_split,
        drop_no_agreement=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_record = {
        "manifest": str(Path(args.manifest).resolve()),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "holdout_indices": holdout_idx,
        "holdout_n": args.iemocap_holdout_n,
        "holdout_seed": args.iemocap_holdout_seed,
        "val_ratio": args.iemocap_val_ratio,
        "split_seed": args.iemocap_split_seed,
        "sessions": args.iemocap_sessions,
        "manifest_split_filter": args.iemocap_manifest_split,
    }
    with (out_dir / "iemocap_train_val_holdout_indices.json").open("w", encoding="utf-8") as f:
        json.dump(split_record, f, indent=2)

    full = RawIEMOCAPDataset(
        args.manifest,
        split=args.iemocap_manifest_split,
        sessions=args.iemocap_sessions,
        drop_no_agreement=True,
        load_audio=False,
        max_samples=None,
    )
    n = len(full)
    requested = {}
    for split in args.splits:
        if split == "train":
            requested["train"] = train_idx
        elif split in {"dev", "val"}:
            requested["dev"] = val_idx
        elif split in {"test", "holdout"}:
            requested["holdout"] = holdout_idx
        elif split == "all":
            requested["all"] = list(range(n))
    return requested, split_record


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.teacher_adapter_path is None and not args.base_teacher:
        raise SystemExit("Specify either --teacher_adapter_path or --base_teacher")

    print(
        f"Precomputing teacher logits | dataset={args.dataset} modalities={args.modalities} "
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

    if args.dataset == "iemocap":
        split_indices, split_record = build_iemocap_split_indices(args)
        print(
            "IEMOCAP split: "
            f"train={len(split_record['train_indices'])} "
            f"val={len(split_record['val_indices'])} "
            f"holdout={len(split_record['holdout_indices'])}"
        )
        dataset = CorruptedIEMOCAPDataset(
            args.manifest,
            processor=processor,
            split=args.iemocap_manifest_split,
            sessions=args.iemocap_sessions,
            modalities=tuple(args.modalities),
            corrupt=args.corrupt,
            corruption_preset=args.corruption_preset,
            for_training=True,
            distill=False,
            drop_no_agreement=True,
            max_samples=None,
        )
        collate_fn = iemocap_collate_fn

        seen = set()
        for split, indices in split_indices.items():
            new_indices = [idx for idx in indices if idx not in seen]
            seen.update(new_indices)
            if not new_indices:
                print(f"Skipping empty IEMOCAP split {split!r}")
                continue

            subset = torch.utils.data.Subset(dataset, new_indices)
            generator = torch.Generator().manual_seed(args.seed)
            loader = DataLoader(
                subset,
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

            for j, batch in enumerate(tqdm(loader, desc=split)):
                raw_idx = int(new_indices[j])
                cache_file = Path(args.output_dir) / f"sample_{raw_idx:06d}.pt"
                save_response_logits(
                    teacher, batch, cache_file, first_device, args.force,
                )
        return

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
                meld_collate_fn,
                pad_token_id=processor.tokenizer.pad_token_id,
                padding_side="right",
            ),
        )

        for i, batch in enumerate(tqdm(loader, desc=split)):
            cache_file = out_dir / f"sample_{i:06d}.pt"
            save_response_logits(teacher, batch, cache_file, first_device, args.force)


if __name__ == "__main__":
    main()
