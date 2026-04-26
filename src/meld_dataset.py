import math
import os
import random
import string
import io
from pathlib import Path

import av
import decord
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import signal
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

CORRUPTION_PRESETS = {
    "mild": {
        "text_char_swap_prob": 0.02,
        "text_word_drop_prob": 0.01,
        "audio_noise_level": 0.01,
        "audio_corruptions": ["snr_noise", "lowpass"],
        "audio_snr_db": 20.0,
        "audio_dropout_ratio": 0.02,
        "audio_dropout_chunks": 1,
        "audio_clip_gain": 1.2,
        "audio_clip_level": 0.9,
        "audio_lowpass_hz": 6000.0,
        "video_noise_level": 0.01,
        "video_corruptions": ["noise", "blur", "brightness_contrast"],
        "video_occlusion_area_ratio": 0.04,
        "video_blur_radius": 0.75,
        "video_pixelate_factor": 2,
        "video_drop_frame_ratio": 0.05,
        "video_brightness": 0.9,
        "video_contrast": 1.1,
        "video_crop_scale": 0.95,
        "video_jpeg_quality": 55,
    },
    "medium": {
        "text_char_swap_prob": 0.04,
        "text_word_drop_prob": 0.03,
        "audio_noise_level": 0.02,
        "audio_corruptions": ["snr_noise", "dropout", "lowpass"],
        "audio_snr_db": 12.0,
        "audio_dropout_ratio": 0.06,
        "audio_dropout_chunks": 2,
        "audio_clip_gain": 1.4,
        "audio_clip_level": 0.8,
        "audio_lowpass_hz": 4500.0,
        "video_noise_level": 0.025,
        "video_corruptions": [
            "noise", "occlusion", "blur", "pixelate",
            "brightness_contrast",
        ],
        "video_occlusion_area_ratio": 0.08,
        "video_blur_radius": 1.5,
        "video_pixelate_factor": 4,
        "video_drop_frame_ratio": 0.10,
        "video_brightness": 0.80,
        "video_contrast": 1.25,
        "video_crop_scale": 0.90,
        "video_jpeg_quality": 35,
    },
    "strong": {
        "text_char_swap_prob": 0.06,
        "text_word_drop_prob": 0.05,
        "audio_noise_level": 0.05,
        "audio_corruptions": ["snr_noise", "dropout", "clip", "lowpass"],
        "audio_snr_db": 5.0,
        "audio_dropout_ratio": 0.15,
        "audio_dropout_chunks": 3,
        "audio_clip_gain": 2.5,
        "audio_clip_level": 0.5,
        "audio_lowpass_hz": 3000.0,
        "video_noise_level": 0.05,
        "video_corruptions": [
            "noise", "occlusion", "blur", "pixelate", "drop_frames",
            "brightness_contrast",
        ],
        "video_occlusion_area_ratio": 0.20,
        "video_blur_radius": 4.0,
        "video_pixelate_factor": 8,
        "video_drop_frame_ratio": 0.25,
        "video_brightness": 0.55,
        "video_contrast": 1.8,
        "video_crop_scale": 0.75,
        "video_jpeg_quality": 12,
    },
}

CORRUPTION_PRESET_NAMES = tuple(CORRUPTION_PRESETS.keys())


def get_corruption_config(preset="medium", **overrides):
    if preset not in CORRUPTION_PRESETS:
        raise ValueError(
            f"corruption_preset must be one of {CORRUPTION_PRESET_NAMES}, got {preset!r}"
        )

    config = {
        k: (list(v) if isinstance(v, list) else v)
        for k, v in CORRUPTION_PRESETS[preset].items()
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


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


def load_video_frames(video_path, fps=1, temporal_patch_size=2):
    """Decode a video file with decord and sample frames at target fps.

    Rounds the frame count to a multiple of temporal_patch_size (2 for Qwen2.5-Omni).
    Returns a [N, H, W, C] uint8 numpy array.
    """
    vr = decord.VideoReader(str(video_path), num_threads=1)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    num_frames = max(1, math.floor(total_frames / video_fps * fps))
    num_frames = max(temporal_patch_size, round(num_frames / temporal_patch_size) * temporal_patch_size)
    num_frames = min(num_frames, total_frames)
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


def add_snr_noise(waveform, snr_db):
    """Add Gaussian noise scaled to a target signal-to-noise ratio."""
    signal_power = float(np.mean(np.square(waveform)))
    if signal_power <= 0:
        noise_std = 0.01
    else:
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = math.sqrt(noise_power)
    noise = np.random.randn(*waveform.shape).astype(np.float32) * noise_std
    return waveform.astype(np.float32) + noise


def apply_audio_dropout(waveform, dropout_ratio, chunks):
    """Silence random contiguous chunks of audio."""
    corrupted = waveform.astype(np.float32).copy()
    if len(corrupted) == 0 or dropout_ratio <= 0 or chunks <= 0:
        return corrupted

    total_drop = max(1, int(len(corrupted) * dropout_ratio))
    chunk_len = max(1, total_drop // chunks)
    for _ in range(chunks):
        if chunk_len >= len(corrupted):
            corrupted[:] = 0.0
            break
        start = random.randint(0, len(corrupted) - chunk_len)
        corrupted[start:start + chunk_len] = 0.0
    return corrupted


def apply_lowpass(waveform, sample_rate, cutoff_hz):
    """Muffle speech with a low-pass filter."""
    if len(waveform) < 16 or cutoff_hz <= 0:
        return waveform
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        return waveform
    sos = signal.butter(6, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, waveform).astype(np.float32)


def apply_audio_corruptions(waveform, sample_rate, config):
    """Apply configured audio corruptions in order."""
    corrupted = waveform.astype(np.float32).copy()
    for corruption in config["audio_corruptions"]:
        if corruption == "noise":
            corrupted = corrupt_audio(corrupted, noise_level=config["audio_noise_level"])
        elif corruption == "snr_noise":
            corrupted = add_snr_noise(corrupted, snr_db=config["audio_snr_db"])
        elif corruption == "dropout":
            corrupted = apply_audio_dropout(
                corrupted,
                dropout_ratio=config["audio_dropout_ratio"],
                chunks=config["audio_dropout_chunks"],
            )
        elif corruption == "clip":
            corrupted = np.clip(
                corrupted * config["audio_clip_gain"],
                -config["audio_clip_level"],
                config["audio_clip_level"],
            ).astype(np.float32)
        elif corruption == "lowpass":
            corrupted = apply_lowpass(
                corrupted,
                sample_rate=sample_rate,
                cutoff_hz=config["audio_lowpass_hz"],
            )
        else:
            raise ValueError(f"Unknown audio corruption: {corruption}")
    return corrupted.astype(np.float32)


def add_video_noise(frames, noise_level):
    noise = np.random.randn(*frames.shape).astype(np.float32) * noise_level * 255.0
    return np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_video_occlusion(frames, area_ratio):
    corrupted = frames.copy()
    _, height, width, _ = corrupted.shape
    box_scale = math.sqrt(max(0.0, min(area_ratio, 1.0)))
    box_h = max(1, int(height * box_scale))
    box_w = max(1, int(width * box_scale))
    y = random.randint(0, max(0, height - box_h))
    x = random.randint(0, max(0, width - box_w))
    corrupted[:, y:y + box_h, x:x + box_w, :] = 0
    return corrupted


def apply_video_blur(frames, radius):
    return np.stack([
        np.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=radius)))
        for frame in frames
    ]).astype(np.uint8)


def apply_video_pixelation(frames, factor):
    factor = max(2, int(factor))
    output = []
    for frame in frames:
        image = Image.fromarray(frame)
        width, height = image.size
        small = image.resize(
            (max(1, width // factor), max(1, height // factor)),
            Image.Resampling.BILINEAR,
        )
        output.append(np.asarray(small.resize((width, height), Image.Resampling.NEAREST)))
    return np.stack(output).astype(np.uint8)


def apply_video_frame_dropout(frames, drop_ratio):
    corrupted = frames.copy()
    frame_count = len(corrupted)
    if frame_count == 0 or drop_ratio <= 0:
        return corrupted
    drop_count = max(1, int(round(frame_count * drop_ratio)))
    drop_count = min(drop_count, frame_count)
    drop_indices = random.sample(range(frame_count), drop_count)
    corrupted[drop_indices] = 0
    return corrupted


def apply_video_freeze(frames):
    if len(frames) == 0:
        return frames
    frozen = frames[random.randrange(len(frames))].copy()
    return np.repeat(frozen[None, ...], len(frames), axis=0).astype(np.uint8)


def apply_video_brightness_contrast(frames, brightness, contrast):
    output = []
    for frame in frames:
        image = Image.fromarray(frame)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        output.append(np.asarray(image))
    return np.stack(output).astype(np.uint8)


def apply_video_crop_resize(frames, crop_scale):
    crop_scale = max(0.1, min(1.0, crop_scale))
    _, height, width, _ = frames.shape
    crop_h = max(1, int(height * crop_scale))
    crop_w = max(1, int(width * crop_scale))
    y = random.randint(0, max(0, height - crop_h))
    x = random.randint(0, max(0, width - crop_w))

    output = []
    for frame in frames:
        image = Image.fromarray(frame)
        cropped = image.crop((x, y, x + crop_w, y + crop_h))
        output.append(np.asarray(cropped.resize((width, height), Image.Resampling.BILINEAR)))
    return np.stack(output).astype(np.uint8)


def apply_video_jpeg(frames, quality):
    output = []
    quality = max(1, min(95, int(quality)))
    for frame in frames:
        image = Image.fromarray(frame)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        output.append(np.asarray(Image.open(buffer).convert("RGB")))
    return np.stack(output).astype(np.uint8)


def apply_video_corruptions(frames, config):
    """Apply configured video corruptions in order."""
    corrupted = frames.astype(np.uint8).copy()
    for corruption in config["video_corruptions"]:
        if corruption == "noise":
            corrupted = add_video_noise(corrupted, config["video_noise_level"])
        elif corruption == "occlusion":
            corrupted = apply_video_occlusion(corrupted, config["video_occlusion_area_ratio"])
        elif corruption == "blur":
            corrupted = apply_video_blur(corrupted, config["video_blur_radius"])
        elif corruption == "pixelate":
            corrupted = apply_video_pixelation(corrupted, config["video_pixelate_factor"])
        elif corruption == "drop_frames":
            corrupted = apply_video_frame_dropout(corrupted, config["video_drop_frame_ratio"])
        elif corruption == "freeze":
            corrupted = apply_video_freeze(corrupted)
        elif corruption == "brightness_contrast":
            corrupted = apply_video_brightness_contrast(
                corrupted,
                config["video_brightness"],
                config["video_contrast"],
            )
        elif corruption == "crop_resize":
            corrupted = apply_video_crop_resize(corrupted, config["video_crop_scale"])
        elif corruption == "jpeg":
            corrupted = apply_video_jpeg(corrupted, config["video_jpeg_quality"])
        else:
            raise ValueError(f"Unknown video corruption: {corruption}")
    return corrupted.astype(np.uint8)


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

    Loads raw video frames, audio waveforms, and text, applies corruption,
    then runs the processor to produce model-ready inputs.
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
        corruption_preset="medium",
        text_char_swap_prob=None,
        text_word_drop_prob=None,
        audio_noise_level=None,
        video_noise_level=None,
        audio_corruptions=None,
        audio_snr_db=None,
        audio_dropout_ratio=None,
        audio_dropout_chunks=None,
        audio_clip_gain=None,
        audio_clip_level=None,
        audio_lowpass_hz=None,
        video_corruptions=None,
        video_occlusion_area_ratio=None,
        video_blur_radius=None,
        video_pixelate_factor=None,
        video_drop_frame_ratio=None,
        video_brightness=None,
        video_contrast=None,
        video_crop_scale=None,
        video_jpeg_quality=None,
        for_training=False,
    ):
        corruption_config = get_corruption_config(
            corruption_preset,
            text_char_swap_prob=text_char_swap_prob,
            text_word_drop_prob=text_word_drop_prob,
            audio_noise_level=audio_noise_level,
            video_noise_level=video_noise_level,
            audio_corruptions=audio_corruptions,
            audio_snr_db=audio_snr_db,
            audio_dropout_ratio=audio_dropout_ratio,
            audio_dropout_chunks=audio_dropout_chunks,
            audio_clip_gain=audio_clip_gain,
            audio_clip_level=audio_clip_level,
            audio_lowpass_hz=audio_lowpass_hz,
            video_corruptions=video_corruptions,
            video_occlusion_area_ratio=video_occlusion_area_ratio,
            video_blur_radius=video_blur_radius,
            video_pixelate_factor=video_pixelate_factor,
            video_drop_frame_ratio=video_drop_frame_ratio,
            video_brightness=video_brightness,
            video_contrast=video_contrast,
            video_crop_scale=video_crop_scale,
            video_jpeg_quality=video_jpeg_quality,
        )
        self.raw_dataset = RawMELDDataset(
            meld_root, split=split, load_audio=False, audio_sr=audio_sr,
        )
        self.processor = processor
        self.modalities = list(modalities)
        self.audio_sr = audio_sr
        self.fps = fps
        self.corrupt = corrupt
        self.corruption_preset = corruption_preset
        self.corruption_config = corruption_config
        self.text_char_swap_prob = corruption_config["text_char_swap_prob"]
        self.text_word_drop_prob = corruption_config["text_word_drop_prob"]
        self.audio_noise_level = corruption_config["audio_noise_level"]
        self.video_noise_level = corruption_config["video_noise_level"]
        self.for_training = for_training

    def __len__(self):
        return len(self.raw_dataset)

    def _load_video_frames(self, video_path, fps):
        return load_video_frames(video_path, fps=fps)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]

        # Text corruption
        text = sample["text"]
        if self.corrupt and "text" in self.modalities:
            text = corrupt_text(
                text,
                char_swap_prob=self.text_char_swap_prob,
                word_drop_prob=self.text_word_drop_prob,
            )

        # Build messages and render chat template (text only, no media loading).
        # Training needs the assistant response IN the sequence + a way to mask
        # prompt tokens from the loss; inference needs the generation prompt.
        messages = build_messages({**sample, "text": text}, self.modalities)
        if self.for_training:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_rendered = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
        else:
            rendered_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
            prompt_rendered = None

        # Load and optionally corrupt raw media.
        # If a file is missing or unreadable, fall back to a zero-filled tensor
        # so the sample still contributes (with padding) rather than crashing.
        videos = None
        audio = None
        has_video = "video" in self.modalities
        has_audio = "audio" in self.modalities or has_video

        if has_video:
            try:
                frames = self._load_video_frames(sample["video_path"], self.fps)
            except Exception:
                # 2 black frames (minimum for temporal_patch_size=2), 224×224 RGB
                frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            if self.corrupt:
                frames = apply_video_corruptions(frames, self.corruption_config)
            videos = [frames]

        if has_audio:
            try:
                waveform, sr = load_audio_from_video(sample["video_path"], target_sr=self.audio_sr)
            except Exception:
                # 1 second of silence at the target sample rate
                waveform = np.zeros(self.audio_sr, dtype=np.float32)
                sr = self.audio_sr
            if self.corrupt:
                waveform = apply_audio_corruptions(waveform, sr, self.corruption_config)
            audio = [waveform]

        # Run processor on corrupted raw data
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
            # Second pass over the prompt-only text with identical media gives the
            # exact token count that precedes the assistant response in `inputs`.
            prompt_inputs = self.processor(text=prompt_rendered, **processor_kwargs)
            prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        # Only squeeze keys that carry a real batch dim from the processor.
        # Video/audio "count" dims (video_grid_thw, video_second_per_grid, pixel_values_videos)
        BATCH_DIM_KEYS = {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
        result = {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k in BATCH_DIM_KEYS else v)
            for k, v in inputs.items()
        }
        result["emotion"] = sample["emotion"]
        result["label"] = sample["label"]
        if prompt_len is not None:
            result["prompt_len"] = prompt_len
        return result


def collate_fn(batch, pad_token_id, padding_side="left", label_pad_id=-100):
    """Collate per-sample dicts from CorruptedMELDDataset into a padded batch.

    Text is padded to max length; video patches and grid_thw are concatenated
    along dim 0 (token-packed, not stackable); audio features are right-padded
    along the frames dim.

    If samples carry `prompt_len`, builds an HF-Trainer-ready `labels` tensor
    where prompt and padding positions are masked to `label_pad_id`.
    """
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

    if "pixel_values_videos" in batch[0]:
        out["pixel_values_videos"] = torch.cat([b["pixel_values_videos"] for b in batch], dim=0)
        out["video_grid_thw"] = torch.cat([b["video_grid_thw"] for b in batch], dim=0)
        if "video_second_per_grid" in batch[0]:
            vals = [b["video_second_per_grid"] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out["video_second_per_grid"] = torch.cat(vals, dim=0)
            else:
                out["video_second_per_grid"] = [v for sub in vals for v in sub]

    if "input_features" in batch[0]:
        out["input_features"] = pad_last([b["input_features"] for b in batch], 0.0)
        if "feature_attention_mask" in batch[0]:
            out["feature_attention_mask"] = pad_last(
                [b["feature_attention_mask"] for b in batch], 0,
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
