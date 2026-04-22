import argparse

import optimum.gptq.constants

import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Patch optimum to recognize Qwen2.5-Omni's layer structure (same as finetune.py)
optimum.gptq.constants.BLOCK_PATTERNS.insert(0, "thinker.model.layers")

from src.meld_dataset import SYSTEM_PROMPT, RawMELDDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with finetuned Qwen2.5-Omni on MELD")
    parser.add_argument("--model_path", default="./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4",
                        help="Path to the base pretrained model")
    parser.add_argument("--adapter_path", default="./ckpts/finetuned/checkpoint-3747",
                        help="Path to the LoRA adapter checkpoint")
    parser.add_argument("--data_root", default="/project2/robinjia_875/lijc/data/MELD.Raw",
                        help="Path to MELD.Raw directory")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    print("Loading base model...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model.thinker = PeftModel.from_pretrained(model.thinker, args.adapter_path)
    model.thinker.eval()
    model.eval()

    # Load the first sample (text only for basic inference)
    dataset = RawMELDDataset(args.data_root, split=args.split, load_audio=False)
    sample = dataset[args.sample_idx]

    print(f"\n--- Sample {args.sample_idx} ({args.split} split) ---")
    print(f"  Speaker:     {sample['speaker']}")
    print(f"  Text:        {sample['text']}")
    print(f"  Ground truth: {sample['emotion']}")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")

    # Move inputs to the model's first device
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("\nRunning inference...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_audio_in_video=False,
            return_audio=False,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    prediction = processor.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"\n--- Result ---")
    print(f"  Prediction:   {prediction}")
    print(f"  Ground truth: {sample['emotion']}")
    print(f"  Correct:      {prediction.lower() == sample['emotion'].lower()}")


if __name__ == "__main__":
    main()
