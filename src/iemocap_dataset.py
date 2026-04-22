"""
IEMOCAP dataset utilities.

Expects CSV from scripts/build_iemocap_emotion_manifest.py with columns including
utterance_id, text, emotion, wav_path, video_path (and optionally session, split).
"""

import math
import random
import string
import warnings
from pathlib import Path

import av
import librosa

try:
    import decord
except ImportError:
    decord = None
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EMOTION2ID = {
    "angry": 0,
    "disgusted": 1,
    "excited": 2,
    "fearful": 3,
    "frustrated": 4,
    "happy": 5,
    "neutral": 6,
    "other": 7,
    "sad": 8,
    "surprised": 9,
}

ID2EMOTION = {v: k for k, v in EMOTION2ID.items()}
IEMOCAP_EMOTIONS = tuple(EMOTION2ID.keys())

SYSTEM_PROMPT = (
    "Your job as a helpful assistant is to detect what emotion is being expressed "
    "from the inputs. Output only one word. Here are the options: "
    + ", ".join(IEMOCAP_EMOTIONS)
    + "."
)
IEMOCAP_SYSTEM_PROMPT = SYSTEM_PROMPT


def _default_iemocap_root_from_manifest(manifest_path: Path) -> Path:
    """.../IEMOCAP_full_release/manifests/*.csv -> .../IEMOCAP_full_release."""
    return manifest_path.resolve().parent.parent


def _resolve_media_path(raw: str, iemocap_root: Path) -> str:
    """Resolve manifest wav/video entries after folder moves or relative rows."""
    p = (raw or "").strip()
    if not p:
        return ""
    path = Path(p)
    if path.is_file():
        return str(path.resolve())
    rel = Path(p)
    if not rel.is_absolute():
        cand = iemocap_root / rel
        if cand.is_file():
            return str(cand.resolve())
    if path.is_absolute():
        parts = path.parts
        for i, part in enumerate(parts):
            if part == "IEMOCAP_full_release" and i + 1 < len(parts):
                tail = Path(*parts[i + 1 :])
                cand = iemocap_root / tail
                if cand.is_file():
                    return str(cand.resolve())
                break
    return p


class RawIEMOCAPDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split=None,
        sessions=None,
        label_type="emotion",
        drop_no_agreement=True,
        load_audio=True,
        audio_sr=16000,
        max_samples=None,
    ):
        """
        Args:
            manifest_path: path to iemocap utterance manifest CSV
            split: optional split filter if a split column exists
            sessions: optional list of session names (e.g. Session1)
            label_type: currently supports 'emotion'
            drop_no_agreement: whether to drop rows with no_agreement labels
            load_audio: whether to load raw audio waveform
            audio_sr: target audio sample rate
            max_samples: optional cap for quick smoke tests
        """
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.sessions = sessions
        self.label_type = str(label_type).lower()
        self.drop_no_agreement = bool(drop_no_agreement)
        self.load_audio = bool(load_audio)
        self.audio_sr = int(audio_sr)
        self.max_samples = max_samples

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if self.label_type != "emotion":
            raise ValueError("Only 'emotion' is supported in this version")

        self.iemocap_root = _default_iemocap_root_from_manifest(self.manifest_path)
        self.df = pd.read_csv(self.manifest_path)

        # Basic cleanup
        for c in ["utterance_id", "recording_id", "session", "text", "emotion", "wav_path", "video_path"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()
        if "emotion" in self.df.columns:
            self.df["emotion"] = self.df["emotion"].str.lower()

        if "emotion" not in self.df.columns:
            raise ValueError("Manifest is missing required 'emotion' column")

        if self.drop_no_agreement:
            self.df = self.df[self.df["emotion"] != "no_agreement"]
        self.df = self.df[self.df["emotion"].isin(EMOTION2ID.keys())]

        if self.sessions:
            if "session" not in self.df.columns:
                raise ValueError("sessions filter requested but manifest has no 'session' column")
            self.df = self.df[self.df["session"].isin(self.sessions)]

        if self.split is not None:
            if "split" in self.df.columns:
                self.df = self.df[self.df["split"].astype(str).str.strip() == str(self.split)]
            else:
                warnings.warn(
                    "Manifest has no 'split' column; ignoring split filter. "
                    "Use sessions or max_samples for subsetting.",
                    stacklevel=2,
                )

        sort_cols = [c for c in ["session", "recording_id", "utterance_id"] if c in self.df.columns]
        if sort_cols:
            self.df = self.df.sort_values(sort_cols).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        if self.max_samples is not None:
            self.df = self.df.iloc[: int(self.max_samples)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, audio_path, video_path):
        if audio_path and Path(audio_path).is_file():
            waveform, sr = librosa.load(audio_path, sr=self.audio_sr, mono=True)
            return waveform.astype(np.float32), sr
        if video_path and Path(video_path).is_file():
            return load_audio_from_video(video_path, target_sr=self.audio_sr)
        raise RuntimeError("No valid audio source for sample")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        utt = str(row.get("utterance_id", "")).strip()
        rec = str(row.get("recording_id", "")).strip()
        text = str(row.get("text", "")).strip()
        emotion = str(row.get("emotion", "")).strip().lower()

        wav = _resolve_media_path(str(row.get("wav_path", "")), self.iemocap_root)
        vid = _resolve_media_path(str(row.get("video_path", "")), self.iemocap_root)

        label = EMOTION2ID[emotion]
        sample = {
            "text": text,
            "label": label,
            "emotion": emotion,
            "session": str(row.get("session", "")).strip(),
            "dialogue_id": rec,
            "utterance_id": utt,
            "video_path": vid,
            "audio_path": wav,
        }

        if self.load_audio:
            try:
                audio, sr = self._load_audio(wav, vid)
            except Exception as e:
                audio, sr = None, None
                sample["audio_error"] = str(e)
            sample["audio"] = audio
            sample["audio_sr"] = sr

        return sample


def _load_video_frames_torchvision(video_path, fps, temporal_patch_size):
    import torchvision.io as tvio

    video, _, info = tvio.read_video(str(video_path), pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"No frames in {video_path}")
    total_frames = int(video.shape[0])
    video_fps = float(info.get("video_fps") or 25.0)
    num_frames = max(1, math.floor(total_frames / video_fps * fps))
    num_frames = max(
        temporal_patch_size,
        round(num_frames / temporal_patch_size) * temporal_patch_size,
    )
    num_frames = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return video[indices].numpy()


def load_video_frames(video_path, fps=1, temporal_patch_size=2):
    """Decode a video file and sample frames at target fps."""
    path = str(video_path)
    if decord is not None:
        vr = decord.VideoReader(path, num_threads=1)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        num_frames = max(1, math.floor(total_frames / video_fps * fps))
        num_frames = max(
            temporal_patch_size,
            round(num_frames / temporal_patch_size) * temporal_patch_size,
        )
        num_frames = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()
    return _load_video_frames_torchvision(path, fps, temporal_patch_size)


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
    words = text.split()
    if len(words) > 1:
        words = [w for w in words if random.random() > word_drop_prob]
        if not words:
            words = [text.split()[0]]

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
    """Return the video path as-is; frame-level noise is applied after decode."""
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
            user_content.append({"type": "audio", "audio": sample["audio_path"] or sample["video_path"]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": sample["emotion"]}]},
    ]
    return messages


class CorruptedIEMOCAPDataset(Dataset):
    """IEMOCAP dataset with optional corruption applied to raw modality data."""

    def __init__(
        self,
        manifest_path,
        processor,
        split=None,
        sessions=None,
        modalities=("text",),
        audio_sr=16000,
        fps=1,
        corrupt=True,
        text_char_swap_prob=0.1,
        text_word_drop_prob=0.1,
        audio_noise_level=0.05,
        video_noise_level=0.05,
        for_training=False,
        drop_no_agreement=True,
        max_samples=None,
    ):
        self.raw_dataset = RawIEMOCAPDataset(
            manifest_path,
            split=split,
            sessions=sessions,
            drop_no_agreement=drop_no_agreement,
            load_audio=False,
            audio_sr=audio_sr,
            max_samples=max_samples,
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

    def __len__(self):
        return len(self.raw_dataset)

    def _load_video_frames(self, video_path, fps):
        return load_video_frames(video_path, fps=fps)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]

        text = sample["text"]
        if self.corrupt and "text" in self.modalities:
            text = corrupt_text(
                text,
                char_swap_prob=self.text_char_swap_prob,
                word_drop_prob=self.text_word_drop_prob,
            )

        messages = build_messages({**sample, "text": text}, self.modalities)
        if self.for_training:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_rendered = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
        else:
            # Evaluation should not include the assistant label turn; otherwise
            # the ground-truth emotion leaks into the model input prompt.
            rendered_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
            prompt_rendered = None

        videos = None
        audio = None
        has_video = "video" in self.modalities
        # Keep video-only runs visual-only unless audio is explicitly requested.
        has_audio = "audio" in self.modalities

        if has_video:
            frames = self._load_video_frames(sample["video_path"], self.fps)
            if self.corrupt:
                noise = np.random.randn(*frames.shape).astype(np.float32) * self.video_noise_level * 255
                frames = np.clip(frames.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            videos = [frames]

        if has_audio:
            audio_path = sample.get("audio_path") or ""
            if audio_path and Path(audio_path).is_file():
                waveform, _ = librosa.load(audio_path, sr=self.audio_sr, mono=True)
                waveform = waveform.astype(np.float32)
            else:
                waveform, _ = load_audio_from_video(sample["video_path"], target_sr=self.audio_sr)
            if self.corrupt:
                waveform = corrupt_audio(waveform, noise_level=self.audio_noise_level)
            audio = [waveform]

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

        batch_dim_keys = {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
        result = {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k in batch_dim_keys else v)
            for k, v in inputs.items()
        }
        result["emotion"] = sample["emotion"]
        result["label"] = sample["label"]
        if prompt_len is not None:
            result["prompt_len"] = prompt_len
        return result


def collate_fn(batch, pad_token_id, padding_side="left", label_pad_id=-100):
    """Collate per-sample dicts from CorruptedIEMOCAPDataset into a padded batch."""

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


# Backward compatibility with prior name used in this repo.
ManifestIEMOCAPDataset = RawIEMOCAPDataset
