import math
import os
import random
import string
from pathlib import Path

import av
import decord
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EMOTION2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

SPLIT_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}

SYSTEM_PROMPT = (
    "Your job as a helpful assistant is to detect what emotion is being expressed "
    "from the inputs. Output only one word. Here are the options: "
    "anger, disgust, fear, joy, neutral, sadness, surprise."
)

ID2EMOTION = {v: k for k, v in EMOTION2ID.items()}


class RawMELDDataset(Dataset):
    def __init__(
        self,
        meld_root,
        split="train",
        label_type="emotion",
        load_audio=True,
        audio_sr=16000,
    ):
        """
        Args:
            meld_root: path to extracted MELD folder
            split: 'train', 'dev', or 'test'
            label_type: currently supports 'emotion'
            load_audio: whether to load raw audio waveform from mp4
            audio_sr: target audio sample rate
        """
        self.meld_root = Path(meld_root)
        self.split = split
        self.label_type = label_type.lower()
        self.load_audio = load_audio
        self.audio_sr = audio_sr

        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be one of: train, dev, test")

        if self.label_type != "emotion":
            raise ValueError("Only 'emotion' is supported in this version")

        csv_name = {
            "train": "train_sent_emo.csv",
            "dev": "dev_sent_emo.csv",
            "test": "test_sent_emo.csv",
        }[self.split]

        self.csv_path = self.meld_root / csv_name
        self.video_dir = self.meld_root / SPLIT_DIRS[self.split]

        self.df = pd.read_csv(self.csv_path)

        # Basic cleanup
        self.df["Utterance"] = self.df["Utterance"].astype(str).str.strip()
        self.df["Emotion"] = self.df["Emotion"].astype(str).str.strip().str.lower()
        self.df["Speaker"] = self.df["Speaker"].astype(str).str.strip()

        # Preserve dialogue order
        self.df = self.df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _get_video_path(self, dialogue_id, utterance_id):
        filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        return self.video_dir / filename

    def _load_audio(self, video_path):
        """
        Load mono audio waveform from an mp4 file.
        Returns:
            waveform: np.ndarray of shape [num_samples]
            sample_rate: int
        """
        return load_audio_from_video(video_path, target_sr=self.audio_sr)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        dialogue_id = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])
        utterance = row["Utterance"]
        speaker = row["Speaker"]
        emotion = row["Emotion"]
        label = EMOTION2ID[emotion]

        video_path = self._get_video_path(dialogue_id, utterance_id)

        sample = {
            "text": utterance,
            "label": label,
            "emotion": emotion,
            "speaker": speaker,
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "video_path": str(video_path),
        }

        # if self.load_audio:
        #     try:
        #         audio, sr = self._load_audio(video_path)
        #     except Exception as e:
        #         audio, sr = None, None
        #         sample["audio_error"] = str(e)

        #     sample["audio"] = audio
        #     sample["audio_sr"] = sr

        if self.load_audio:
            print("Loading audio from:", video_path)
            try:
                audio, sr = self._load_audio(video_path)
                print("Loaded audio:", type(audio), None if audio is None else audio.shape, sr)
            except Exception as e:
                print("Audio exception:", e)
                audio, sr = None, None
                sample["audio_error"] = str(e)

            sample["audio"] = audio
            sample["audio_sr"] = sr

        return sample


def load_video_frames(video_path, fps=1, temporal_patch_size=2, max_frames=16):
    """Decode a video file with decord and sample frames at target fps.

    Rounds the frame count to a multiple of temporal_patch_size (2 for Qwen2.5-Omni).
    Caps total frames at `max_frames` so a single long clip can't blow up VRAM
    via thousands of vision tokens. Set max_frames=None to disable the cap.
    Returns a [N, H, W, C] uint8 numpy array.
    """
    vr = decord.VideoReader(str(video_path), num_threads=1)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    num_frames = max(1, math.floor(total_frames / video_fps * fps))
    num_frames = max(temporal_patch_size, round(num_frames / temporal_patch_size) * temporal_patch_size)
    num_frames = min(num_frames, total_frames)
    if max_frames is not None:
        # Round cap down to a multiple of temporal_patch_size
        cap = (max_frames // temporal_patch_size) * temporal_patch_size
        num_frames = min(num_frames, cap)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return vr.get_batch(indices).asnumpy()


def load_audio_from_video(path, target_sr=16000):
    """Decode mono audio from an mp4 using PyAV, resampled to target_sr."""
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"No audio stream in {path}")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32), target_sr
    return np.concatenate(chunks).astype(np.float32), target_sr


def corrupt_text(text, char_swap_prob=0.1, word_drop_prob=0.1):
    """Corrupt text by randomly swapping characters and dropping words."""
    # Word-level dropout
    words = text.split()
    if len(words) > 1:
        words = [w for w in words if random.random() > word_drop_prob]
        if not words:  # keep at least one word
            words = [text.split()[0]]

    # Character-level swaps
    corrupted = []
    for word in words:
        chars = list(word)
        for i in range(len(chars)):
            if random.random() < char_swap_prob:
                chars[i] = random.choice(string.ascii_lowercase)
        corrupted.append("".join(chars))

    return " ".join(corrupted)


def corrupt_audio(waveform, noise_level=0.05):
    """Add Gaussian noise to an audio waveform (numpy array)."""
    noise = np.random.randn(*waveform.shape).astype(waveform.dtype) * noise_level
    return waveform + noise


def corrupt_video_frames(video_path, noise_level=0.05):
    """Return the video path as-is; frame-level noise is applied post-processor
    on pixel_values_videos in the collate function. This is a placeholder so the
    dataset can flag that corruption is enabled."""
    return video_path

def build_messages(sample, modalities):
    """Build chat messages for a single sample (same format as evaluate.py)."""
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
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": sample["emotion"]}]},
    ]
    return messages


class CorruptedMELDDataset(Dataset):
    """MELD dataset with optional corruption applied to raw modality data.

    Loads raw video frames, audio waveforms, and text, then runs the
    processor to produce model-ready inputs.

    When `distill=True`, __getitem__ returns a paired {"full": ..., "mask": ...}
    dict where:
      - "full"  = teacher input: CLEAN, uncorrupted media, all modalities
      - "mask"  = student input: CORRUPTED media (via `corrupt`), all modalities
    The modalities are the same in both passes; the only difference is that
    the student sees noisy media while the teacher sees clean media.

    When `distill=False`, a single processed dict is returned with corrupted
    media (if `corrupt=True`) for standard noisy-input training.
    """

    def __init__(
        self,
        meld_root,
        processor,
        split="train",
        modalities=("text",),
        audio_sr=16000,
        fps=1,
        corrupt=True,
        text_char_swap_prob=0.1,
        text_word_drop_prob=0.1,
        audio_noise_level=0.05,
        video_noise_level=0.05,
        for_training=False,
        distill=False,
    ):
        self.raw_dataset = RawMELDDataset(
            meld_root, split=split, load_audio=False, audio_sr=audio_sr,
        )
        self.processor = processor
        self.modalities = list(modalities)
        self.audio_sr = audio_sr
        self.fps = fps
        self.corrupt = corrupt
        self.text_char_swap_prob = text_char_swap_prob
        self.text_word_drop_prob = text_word_drop_prob
        self.audio_noise_level = audio_noise_level
        self.video_noise_level = video_noise_level
        self.for_training = for_training
        self.distill = distill

    def __len__(self):
        return len(self.raw_dataset)

    def _load_video_frames(self, video_path, fps):
        return load_video_frames(video_path, fps=fps)

    def _process(self, sample, text, frames, waveform, modalities):
        """Render chat + run processor for one modality configuration.

        `frames` / `waveform` are pre-loaded (and pre-corrupted) raw media or
        None; this method only includes them if `modalities` asks for them.
        """
        has_video = "video" in modalities
        has_audio = "audio" in modalities or has_video

        messages = build_messages({**sample, "text": text}, modalities)
        if self.for_training:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_rendered = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
        else:
            # messages[-1] is the assistant turn carrying the ground-truth
            # label. Drop it before rendering so the model generates from the
            # prompt without seeing the answer. add_generation_prompt=True
            # appends the marker where generation begins.
            rendered_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
            prompt_rendered = None

        videos = [frames] if (has_video and frames is not None) else None
        audio = [waveform] if (has_audio and waveform is not None) else None

        processor_kwargs = dict(
            videos=videos,
            audio=audio,
            use_audio_in_video=has_video and has_audio,
            fps=self.fps,
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        inputs = self.processor(text=rendered_text, **processor_kwargs)

        prompt_len = None
        if self.for_training:
            prompt_inputs = self.processor(text=prompt_rendered, **processor_kwargs)
            prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        # Only squeeze keys that carry a real batch dim from the processor.
        # Video/audio "count" dims (video_grid_thw, video_second_per_grid, pixel_values_videos)
        # are semantic, not batch — squeezing them breaks collation when N=1.
        BATCH_DIM_KEYS = {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
        result = {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k in BATCH_DIM_KEYS else v)
            for k, v in inputs.items()
        }
        if prompt_len is not None:
            result["prompt_len"] = prompt_len
        return result

    def _sample_kept_modalities(self):
        mods = list(self.modalities)
        num_to_mask = random.randint(1, len(mods))
        dropped = set(random.sample(mods, num_to_mask))
        return [m for m in mods if m not in dropped]

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]

        # Load clean raw media once. If file is missing, fall back to zero
        # tensors so the sample still contributes rather than crashing.
        clean_text = sample["text"]
        clean_frames = None
        clean_waveform = None
        if "video" in self.modalities:
            try:
                clean_frames = self._load_video_frames(sample["video_path"], self.fps)
            except Exception:
                clean_frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        if "audio" in self.modalities or "video" in self.modalities:
            try:
                clean_waveform, _ = load_audio_from_video(sample["video_path"], target_sr=self.audio_sr)
            except Exception:
                clean_waveform = np.zeros(self.audio_sr, dtype=np.float32)

        # Build corrupted copies (only used where noise applies).
        corrupt_text_str = clean_text
        corrupt_frames = clean_frames
        corrupt_waveform = clean_waveform
        if self.corrupt:
            if "text" in self.modalities:
                corrupt_text_str = corrupt_text(
                    clean_text,
                    char_swap_prob=self.text_char_swap_prob,
                    word_drop_prob=self.text_word_drop_prob,
                )
            if clean_frames is not None:
                noise = np.random.randn(*clean_frames.shape).astype(np.float32) * self.video_noise_level * 255
                corrupt_frames = np.clip(clean_frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            if clean_waveform is not None:
                corrupt_waveform = corrupt_audio(clean_waveform, noise_level=self.audio_noise_level)

        if self.distill:
            # Teacher: clean inputs, all modalities (no masking).
            full_item = self._process(sample, clean_text, clean_frames, clean_waveform, self.modalities)
            full_item["emotion"] = sample["emotion"]
            full_item["label"] = sample["label"]

            # Student: corrupted inputs, all modalities (no masking).
            mask_item = self._process(sample, corrupt_text_str, corrupt_frames, corrupt_waveform, self.modalities)
            mask_item["emotion"] = sample["emotion"]
            mask_item["label"] = sample["label"]

            return {
                "full": full_item,
                "mask": mask_item,
                "emotion": sample["emotion"],
                "label": sample["label"],
            }

        # Non-distill: single pass with corrupted inputs (if corrupt=True).
        result = self._process(sample, corrupt_text_str, corrupt_frames, corrupt_waveform, self.modalities)
        result["emotion"] = sample["emotion"]
        result["label"] = sample["label"]
        return result


def _collate_single(batch, pad_token_id, padding_side, label_pad_id):
    """Collate a flat list of per-sample dicts (no 'full'/'mask' nesting)."""
    def pad_1d(seqs, pad_value):
        max_len = max(s.size(0) for s in seqs)
        out = []
        for s in seqs:
            pad_len = max_len - s.size(0)
            if pad_len == 0:
                out.append(s)
                continue
            pad = torch.full((pad_len,), pad_value, dtype=s.dtype)
            out.append(torch.cat([pad, s] if padding_side == "left" else [s, pad], dim=0))
        return torch.stack(out, dim=0)

    def pad_last(seqs, pad_value):
        max_len = max(s.shape[-1] for s in seqs)
        out = []
        for s in seqs:
            pad_len = max_len - s.shape[-1]
            if pad_len > 0:
                s = torch.nn.functional.pad(s, (0, pad_len), value=pad_value)
            out.append(s)
        return torch.stack(out, dim=0)

    out = {
        "input_ids": pad_1d([b["input_ids"] for b in batch], pad_token_id),
        "attention_mask": pad_1d([b["attention_mask"] for b in batch], 0),
    }

    # Guard against samples in a batch with different modality presence: in
    # distill mode the mask variant may omit video/audio for some samples.
    video_samples = [b for b in batch if "pixel_values_videos" in b]
    if video_samples:
        out["pixel_values_videos"] = torch.cat([b["pixel_values_videos"] for b in video_samples], dim=0)
        out["video_grid_thw"] = torch.cat([b["video_grid_thw"] for b in video_samples], dim=0)
        if "video_second_per_grid" in video_samples[0]:
            vals = [b["video_second_per_grid"] for b in video_samples]
            if isinstance(vals[0], torch.Tensor):
                out["video_second_per_grid"] = torch.cat(vals, dim=0)
            else:
                out["video_second_per_grid"] = [v for sub in vals for v in sub]

    audio_samples = [b for b in batch if "input_features" in b]
    if audio_samples:
        out["input_features"] = pad_last([b["input_features"] for b in audio_samples], 0.0)
        if "feature_attention_mask" in audio_samples[0]:
            out["feature_attention_mask"] = pad_last(
                [b["feature_attention_mask"] for b in audio_samples], 0,
            )

    if "prompt_len" in batch[0]:
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = label_pad_id
        for i, b in enumerate(batch):
            pl = int(b["prompt_len"])
            if padding_side == "right":
                labels[i, :pl] = label_pad_id
            else:
                pad_count = int((attention_mask[i] == 0).sum().item())
                labels[i, pad_count:pad_count + pl] = label_pad_id
        out["labels"] = labels

    return out


def collate_fn(batch, pad_token_id, padding_side="left", label_pad_id=-100):
    """Collate per-sample dicts from CorruptedMELDDataset into a padded batch.

    Text is padded to max length; video patches and grid_thw are concatenated
    along dim 0 (token-packed, not stackable); audio features are right-padded
    along the frames dim.

    If samples carry `prompt_len`, builds an HF-Trainer-ready `labels` tensor
    where prompt and padding positions are masked to `label_pad_id`.

    If samples are paired {"full": ..., "mask": ...} (distill mode), returns
    {"full": <collated>, "mask": <collated>, "labels": <mask labels>}. The
    top-level "labels" key exists so HF Trainer's num_items_in_batch
    accounting can find a labels tensor.
    """
    if "full" in batch[0]:
        full = _collate_single([b["full"] for b in batch], pad_token_id, padding_side, label_pad_id)
        mask = _collate_single([b["mask"] for b in batch], pad_token_id, padding_side, label_pad_id)
        out = {"full": full, "mask": mask}
        if "labels" in mask:
            out["labels"] = mask["labels"]
        return out

    return _collate_single(batch, pad_token_id, padding_side, label_pad_id)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = RawMELDDataset("/project2/robinjia_875/lijc/data/MELD.Raw", split="test", load_audio=True)

    sample = dataset[0]
    print(sample["text"])
    print(sample["audio"].shape if sample["audio"] is not None else None)
    print(sample["audio_sr"])
    print(sample["video_path"])
    print(sample.get("audio_error"))

    print(len(dataset))
    print(type(sample["audio"]))