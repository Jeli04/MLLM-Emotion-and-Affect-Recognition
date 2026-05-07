import io
import math
import random
import string

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import signal


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
        "video_motion_blur_kernel": 3,
        "video_defocus_radius": 1.0,
        "video_pixelate_factor": 2,
        "video_drop_frame_ratio": 0.05,
        "video_brightness": 0.9,
        "video_contrast": 1.1,
        "video_crop_scale": 0.95,
        "video_jpeg_quality": 55,
    },
    "medium": {
        "text_char_swap_prob": 0.1,
        "text_word_drop_prob": 0.1,
        "audio_noise_level": 0.05,
        "audio_corruptions": ["snr_noise", "noise", "dropout", "lowpass"],
        "audio_snr_db": 12.0,
        "audio_dropout_ratio": 0.06,
        "audio_dropout_chunks": 2,
        "audio_clip_gain": 1.4,
        "audio_clip_level": 0.8,
        "audio_lowpass_hz": 4500.0,
        "video_noise_level": 0.05,
        "video_corruptions": [
            "noise", "occlusion", "blur", "motion_blur", "defocus_blur",
            "pixelate", "brightness_contrast",
        ],
        "video_occlusion_area_ratio": 0.08,
        "video_blur_radius": 1.5,
        "video_motion_blur_kernel": 7,
        "video_defocus_radius": 2.5,
        "video_pixelate_factor": 4,
        "video_drop_frame_ratio": 0.10,
        "video_brightness": 0.80,
        "video_contrast": 1.25,
        "video_crop_scale": 0.90,
        "video_jpeg_quality": 35,
    },
    "strong": {
        "text_char_swap_prob": 0.2,
        "text_word_drop_prob": 0.2,
        "audio_noise_level": 0.12,
        "audio_corruptions": ["snr_noise", "noise", "dropout", "clip", "lowpass"],
        "audio_snr_db": 5.0,
        "audio_dropout_ratio": 0.15,
        "audio_dropout_chunks": 3,
        "audio_clip_gain": 2.5,
        "audio_clip_level": 0.5,
        "audio_lowpass_hz": 3000.0,
        "video_noise_level": 0.10,
        "video_corruptions": [
            "noise", "occlusion", "blur", "motion_blur", "defocus_blur",
            "pixelate", "drop_frames", "brightness_contrast",
        ],
        "video_occlusion_area_ratio": 0.20,
        "video_blur_radius": 4.0,
        "video_motion_blur_kernel": 15,
        "video_defocus_radius": 5.0,
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
        corrupted[start:start + chunk_len] = 0.0
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
    corrupted[:, y:y + box_h, x:x + box_w, :] = 0
    return corrupted


def apply_video_blur(frames, radius):
    return np.stack([
        np.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=radius)))
        for frame in frames
    ]).astype(np.uint8)


def _convolve_frame_per_channel(frame, kernel):
    from scipy.ndimage import convolve as nd_convolve
    out = np.empty_like(frame, dtype=np.float32)
    for c in range(frame.shape[-1]):
        out[..., c] = nd_convolve(frame[..., c].astype(np.float32), kernel, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_video_motion_blur(frames, kernel_size, angle_deg=None):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return frames
    if angle_deg is None:
        angle_deg = random.uniform(0.0, 180.0)
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cx = cy = (kernel_size - 1) / 2.0
    for t in np.linspace(-cx, cx, kernel_size * 2):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[int(cy), int(cx)] = 1.0
    kernel /= kernel.sum()
    return np.stack([_convolve_frame_per_channel(f, kernel) for f in frames])


def apply_video_defocus_blur(frames, radius):
    radius = float(radius)
    if radius <= 0.5:
        return frames
    size = 2 * int(math.ceil(radius)) + 1
    cx = cy = size // 2
    yy, xx = np.ogrid[:size, :size]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    kernel = mask.astype(np.float32)
    if kernel.sum() == 0:
        kernel[cy, cx] = 1.0
    kernel /= kernel.sum()
    return np.stack([_convolve_frame_per_channel(f, kernel) for f in frames])


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
        elif corruption == "motion_blur":
            corrupted = apply_video_motion_blur(corrupted, config["video_motion_blur_kernel"])
        elif corruption == "defocus_blur":
            corrupted = apply_video_defocus_blur(corrupted, config["video_defocus_radius"])
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
