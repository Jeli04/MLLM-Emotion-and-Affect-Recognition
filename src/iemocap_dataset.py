# iemocap data loader same format as meld data loader
import math
import random
import string
import io
import warnings
from collections import Counter
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
from PIL import Image, ImageEnhance, ImageFilter
from scipy import signal
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

CORRUPTION_AWARE_SYSTEM_PROMPT = (
    "Your job as a helpful assistant is to detect what emotion is being expressed "
    "from the inputs and identify which input modalities are corrupted. "
    "Output exactly two lines. The first line must be one emotion word from: "
    + ", ".join(IEMOCAP_EMOTIONS)
    + ". The second line must be formatted exactly as: "
    "corrupted_modalities: <comma-separated modalities or none>. "
    "Valid modalities are text, audio, video."
)

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
            "noise",
            "occlusion",
            "blur",
            "pixelate",
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
            "noise",
            "occlusion",
            "blur",
            "pixelate",
            "drop_frames",
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


def _default_iemocap_root_from_manifest(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent.parent


def _resolve_media_path(raw: str, iemocap_root: Path) -> str:
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
    noise = np.random.randn(*waveform.shape).astype(waveform.dtype) * noise_level
    return waveform + noise


def add_snr_noise(waveform, snr_db):
    signal_power = float(np.mean(np.square(waveform)))
    
    if signal_power <= 0:
        noise_std = 0.01
        
    else:
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = math.sqrt(noise_power)
        
    noise = np.random.randn(*waveform.shape).astype(np.float32) * noise_std
    return waveform.astype(np.float32) + noise


def apply_audio_dropout(waveform, dropout_ratio, chunks):
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
        corrupted[start : start + chunk_len] = 0.0
        
    return corrupted


def apply_lowpass(waveform, sample_rate, cutoff_hz):
    if len(waveform) < 16 or cutoff_hz <= 0:
        return waveform
        
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        return waveform
        
    sos = signal.butter(6, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return signal.sosfiltfilt(sos, waveform).astype(np.float32)


def apply_audio_corruptions(waveform, sample_rate, config):
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
    corrupted[:, y : y + box_h, x : x + box_w, :] = 0
    return corrupted


def apply_video_blur(frames, radius):
    return np.stack(
        [
            np.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=radius)))
            for frame in frames
        ]
    ).astype(np.uint8)


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
            raise ValueError(f"invalid video corruption name {corruption}")
    return corrupted.astype(np.uint8)


def corrupt_video_frames(video_path, noise_level=0.05):
    return video_path


def get_corrupted_modalities(modalities, corrupt):
    if not corrupt:
        return []
    return [mod for mod in ("text", "audio", "video") if mod in modalities]


def format_assistant_response(sample, modalities, corrupt, predict_corruption=False):
    if not predict_corruption:
        return sample["emotion"]
        
    corrupted_modalities = get_corrupted_modalities(modalities, corrupt)
    corrupted_text = ",".join(corrupted_modalities) if corrupted_modalities else "none"
    return f"{sample['emotion']}\ncorrupted_modalities: {corrupted_text}"


def build_messages(sample, modalities, corrupt=False, predict_corruption=False):
    user_content = []
    has_video = "video" in modalities
    for mod in modalities:
        if mod == "text":
            user_content.append({"type": "text", "text": sample["text"]})
        elif mod == "video":
            user_content.append({"type": "video", "video": sample["video_path"]})
        elif mod == "audio" and not has_video:
            user_content.append({"type": "audio", "audio": sample["audio_path"] or sample["video_path"]})

    assistant_response = format_assistant_response(
        sample, modalities, corrupt=corrupt, predict_corruption=predict_corruption,
    )
    system_prompt = CORRUPTION_AWARE_SYSTEM_PROMPT if predict_corruption else SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]},
    ]
    return messages


def compute_iemocap_eval_holdout_indices(manifest_path,*,holdout_n=500,holdout_seed=42,sessions=None,split=None,drop_no_agreement=True,):
    base = RawIEMOCAPDataset(
        manifest_path,
        split=split,
        sessions=sessions,
        drop_no_agreement=drop_no_agreement,
        load_audio=False,
        max_samples=None,
    )
    n = len(base)
    if n == 0:
        raise ValueError("no iemocap samples available after filtering")

    holdout_n = int(holdout_n) if holdout_n is not None else 0
    if holdout_n > 0:
        k = min(holdout_n, n)
        rng_h = random.Random(int(holdout_seed))
        holdout_indices = sorted(rng_h.sample(range(n), k))
    else:
        holdout_indices = []

    holdout_set = set(holdout_indices)
    remaining_indices = [i for i in range(n) if i not in holdout_set]
    return holdout_indices, remaining_indices


def compute_iemocap_train_val_holdout_split(manifest_path,*,holdout_n=500,holdout_seed=42,val_ratio=0.1,split_seed=43,sessions=None,split=None,drop_no_agreement=True,):
    holdout_indices, remaining = compute_iemocap_eval_holdout_indices(
        manifest_path,
        holdout_n=holdout_n,
        holdout_seed=holdout_seed,
        sessions=sessions,
        split=split,
        drop_no_agreement=drop_no_agreement,
    )
    if not remaining:
        raise ValueError("hold out too high no samples left")

    val_ratio = float(val_ratio)
    rng_s = random.Random(int(split_seed))
    order = remaining[:]
    rng_s.shuffle(order)

    if val_ratio <= 0 or len(order) == 1:
        train_indices = sorted(order)
        val_indices = []
        
    else:
        val_n = min(
            len(order) - 1,
            max(1, int(round(len(order) * val_ratio))),
        )
        val_indices = sorted(order[:val_n])
        train_indices = sorted(order[val_n:])

    return train_indices, val_indices, holdout_indices


def iemocap_train_subset_sample_weights(train_indices,raw_df: pd.DataFrame,*,power: float = 0.5,max_ratio_to_majority: float = 40.0,) -> torch.Tensor:
    
    train_indices = list(train_indices)
    counts: Counter[str] = Counter()
    for i in train_indices:
        emo = str(raw_df.iloc[int(i)]["emotion"]).strip().lower()
        
        if emo not in EMOTION2ID:
            continue
            
        counts[emo] += 1
        
    if not counts:
        raise ValueError("iemocap_train_subset_sample_weights: empty or unlabeled train_indices")
        
    n_max = max(counts.values())
    class_weight = {}
    
    for emo, nc in counts.items():
        raw_ratio = (n_max / float(nc)) ** float(power)
        class_weight[emo] = float(min(raw_ratio, max_ratio_to_majority))

    weights = []
    for i in train_indices:
        emo = str(raw_df.iloc[int(i)]["emotion"]).strip().lower()
        weights.append(class_weight.get(emo, 1.0))
        
    return torch.tensor(weights, dtype=torch.double)


class CorruptedIEMOCAPDataset(Dataset):
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
        predict_corruption=False,
        distill=False,
        modality_mask=True,
        clean_teacher=False,
        teacher_logits_dir=None,
        include_full_branch=True,
        drop_no_agreement=True,
        max_samples=None,
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
        self.corruption_preset = corruption_preset
        self.corruption_config = corruption_config
        self.text_char_swap_prob = corruption_config["text_char_swap_prob"]
        self.text_word_drop_prob = corruption_config["text_word_drop_prob"]
        self.audio_noise_level = corruption_config["audio_noise_level"]
        self.video_noise_level = corruption_config["video_noise_level"]
        self.for_training = for_training
        self.predict_corruption = predict_corruption
        self.distill = distill
        self.modality_mask = modality_mask
        self.clean_teacher = clean_teacher
        self.teacher_logits_dir = teacher_logits_dir
        self.include_full_branch = include_full_branch

    def __len__(self):
        return len(self.raw_dataset)

    def _load_video_frames(self, video_path, fps):
        return load_video_frames(video_path, fps=fps)

    def _load_audio_waveform(self, sample):
        audio_path = sample.get("audio_path") or ""
        if audio_path and Path(audio_path).is_file():
            waveform, sr = librosa.load(audio_path, sr=self.audio_sr, mono=True)
            return waveform.astype(np.float32), sr
        return load_audio_from_video(sample["video_path"], target_sr=self.audio_sr)

    def _process(self, sample, text, frames, waveform, modalities):
        """Render chat + run processor for one modality configuration."""
        messages = build_messages(
            {**sample, "text": text},
            modalities,
            corrupt=self.corrupt,
            predict_corruption=self.predict_corruption,
        )
        if self.for_training:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_rendered = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
        else:
            # fix ground truth label leaking into prompt message 
            rendered_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
            )
            prompt_rendered = None

        has_video = "video" in modalities
        has_audio = "audio" in modalities
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

        batch_dim_keys = {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
        result = {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k in batch_dim_keys else v)
            for k, v in inputs.items()
        }
        if prompt_len is not None:
            result["prompt_len"] = prompt_len
        return result

    def _sample_kept_modalities(self):
        mods = list(self.modalities)
        if len(mods) <= 1:
            return mods
        keep_count = random.randint(1, len(mods) - 1)
        kept = set(random.sample(mods, keep_count))
        return [m for m in mods if m in kept]

    def _corrupt_media(self, text, frames, waveform):
        if "text" in self.modalities:
            text = corrupt_text(
                text,
                char_swap_prob=self.text_char_swap_prob,
                word_drop_prob=self.text_word_drop_prob,
            )
        if frames is not None:
            frames = apply_video_corruptions(frames, self.corruption_config)
        if waveform is not None:
            waveform = apply_audio_corruptions(
                waveform,
                sample_rate=self.audio_sr,
                config=self.corruption_config,
            )
        return text, frames, waveform

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]

        text = sample["text"]
        frames = None
        waveform = None
        if "video" in self.modalities:
            try:
                frames = self._load_video_frames(sample["video_path"], self.fps)
            except Exception:
            
                frames = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        if "audio" in self.modalities:
            try:
                waveform, _ = self._load_audio_waveform(sample)
            except Exception:
                waveform = np.zeros(self.audio_sr, dtype=np.float32)

        if self.distill:
            full_text, full_frames, full_waveform = text, frames, waveform
            mask_text, mask_frames, mask_waveform = text, frames, waveform
            if self.corrupt:
                if self.clean_teacher:
                    mask_text, mask_frames, mask_waveform = self._corrupt_media(
                        text, frames, waveform,
                    )
                else:
                    full_text, full_frames, full_waveform = self._corrupt_media(
                        text, frames, waveform,
                    )
                    mask_text, mask_frames, mask_waveform = (
                        full_text, full_frames, full_waveform,
                    )

            kept = self._sample_kept_modalities() if self.modality_mask else self.modalities
            mask_item = self._process(sample, mask_text, mask_frames, mask_waveform, kept)
            mask_item["emotion"] = sample["emotion"]
            mask_item["label"] = sample["label"]

            result = {
                "mask": mask_item,
                "emotion": sample["emotion"],
                "label": sample["label"],
            }
            if self.include_full_branch:
                full_item = self._process(
                    sample, full_text, full_frames, full_waveform, self.modalities,
                )
                full_item["emotion"] = sample["emotion"]
                full_item["label"] = sample["label"]
                result["full"] = full_item
            if self.teacher_logits_dir is not None:
                cache_path = Path(self.teacher_logits_dir) / f"sample_{idx:06d}.pt"
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                result["teacher_response_logits"] = cached["response_logits"]
                result["teacher_response_labels"] = cached["response_labels"]
            return result

        if self.corrupt:
            text, frames, waveform = self._corrupt_media(text, frames, waveform)
        result = self._process(sample, text, frames, waveform, self.modalities)
        result["emotion"] = sample["emotion"]
        result["label"] = sample["label"]
        return result


def _collate_single(batch, pad_token_id, padding_side, label_pad_id):
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
    if "full" in batch[0] or "mask" in batch[0]:
        mask = _collate_single([b["mask"] for b in batch], pad_token_id, padding_side, label_pad_id)
        out = {"mask": mask}
        if "full" in batch[0]:
            out["full"] = _collate_single(
                [b["full"] for b in batch],
                pad_token_id,
                padding_side,
                label_pad_id,
            )
        if "labels" in mask:
            out["labels"] = mask["labels"]
        if "teacher_response_logits" in batch[0]:
            out["teacher_response_logits"] = [b["teacher_response_logits"] for b in batch]
            out["teacher_response_labels"] = [b["teacher_response_labels"] for b in batch]
        return out

    return _collate_single(batch, pad_token_id, padding_side, label_pad_id)

ManifestIEMOCAPDataset = RawIEMOCAPDataset
