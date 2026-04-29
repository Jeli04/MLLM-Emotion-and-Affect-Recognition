"""
Pre-corrupt a MELD-process directory using main's corruption pipeline.

Produces a parallel directory with corrupted media for the chosen modalities,
suitable for AffectGPT eval. Modalities not being corrupted are symlinked
from the source so we don't duplicate disk usage.

Outputs at <dst>:
    label.npz                              -> symlink (labels never change)
    transcription-engchi-polish.csv        -> corrupted CSV or symlink
    subaudio/<name>.wav                    -> corrupted .wav files or symlinked dir
    subvideo/<name>.mp4                    -> corrupted .mp4 files or symlinked dir
    openface_face/<name>.npy               -> re-extracted from corrupted videos
                                              if 'video' is being corrupted, else
                                              symlinked dir

Usage:
    python preprocess_corruption.py \
        --src ../dataset/meld-process \
        --dst ../dataset/meld-process-textaudiovideo-medium \
        --corrupt_modalities text audio video \
        --preset medium \
        --seed 42
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Self-contained corruption code copied from main's src/meld_dataset.py
from corruption_lib import (
    CORRUPTION_PRESET_NAMES,
    apply_audio_corruptions,
    apply_video_corruptions,
    corrupt_text,
    get_corruption_config,
)


def corrupt_text_csv(src_csv, dst_csv, char_swap_prob, word_drop_prob):
    df = pd.read_csv(src_csv)
    if "english" not in df.columns:
        raise RuntimeError(f"{src_csv} missing 'english' column; got {df.columns.tolist()}")
    df["english"] = (
        df["english"].fillna("").astype(str)
        .apply(lambda s: corrupt_text(s, char_swap_prob, word_drop_prob) if s else s)
    )
    df.to_csv(dst_csv, index=False)


def corrupt_audio_file(src_wav, dst_wav, sample_rate, config):
    import scipy.io.wavfile as wavfile
    sr, wav = wavfile.read(str(src_wav))
    if wav.dtype == np.int16:
        wav_f = wav.astype(np.float32) / 32768.0
        was_int = True
    elif wav.dtype == np.int32:
        wav_f = wav.astype(np.float32) / 2147483648.0
        was_int = True
    else:
        wav_f = wav.astype(np.float32)
        was_int = False

    # Mono only (AffectGPT pre-processing already converts to mono 16k)
    if wav_f.ndim > 1:
        wav_f = wav_f.mean(axis=1)

    wav_corrupted = apply_audio_corruptions(wav_f, sr, config)

    if was_int:
        out = np.clip(wav_corrupted * 32768.0, -32768, 32767).astype(np.int16)
    else:
        out = wav_corrupted.astype(np.float32)
    wavfile.write(str(dst_wav), sr, out)


def corrupt_video_file(src_mp4, dst_mp4, config):
    """Decode mp4 → corrupt frames → re-encode mp4."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(src_mp4), "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 25)
    frames = []
    for frame in reader:
        frames.append(np.asarray(frame))
    reader.close()
    if len(frames) == 0:
        # No frames; copy as-is to avoid producing empty output
        shutil.copy(src_mp4, dst_mp4)
        return
    frames_arr = np.stack(frames, axis=0).astype(np.uint8)  # [N, H, W, 3]

    corrupted = apply_video_corruptions(frames_arr, config)

    writer = imageio.get_writer(
        str(dst_mp4),
        fps=fps,
        codec="libx264",
        quality=8,           # libx264 quality (1-10), 8 ~= medium-high
        macro_block_size=1,  # don't pad odd dims
    )
    for f in corrupted:
        writer.append_data(f)
    writer.close()


def parse_args():
    p = argparse.ArgumentParser(description="Pre-corrupt MELD-process for AffectGPT eval")
    p.add_argument("--src", required=True, help="Source meld-process dir (clean)")
    p.add_argument("--dst", required=True, help="Destination corrupted dir")
    p.add_argument("--corrupt_modalities", nargs="*", default=[],
                   choices=["text", "audio", "video"],
                   help="Which modalities to corrupt. Empty = symlink everything.")
    p.add_argument("--preset", default="medium", choices=list(CORRUPTION_PRESET_NAMES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio_sr", type=int, default=16000)
    p.add_argument("--extract_faces_script", default="extract_faces.py",
                   help="Path to extract_faces.py for re-extraction on corrupted videos.")
    p.add_argument("--face_workers", type=int, default=4,
                   help="Parallel workers for face re-extraction on corrupted videos.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only corrupt this many files per modality (debug).")
    p.add_argument("--name_prefix", default="test_",
                   help="Only corrupt files whose basename starts with this. "
                        "Default 'test_' = only the MELD test split. Use '' to "
                        "process every file.")
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if dst.exists():
        print(f"Destination already exists, leaving alone: {dst}", flush=True)
        return

    config = get_corruption_config(args.preset)
    corrupt_set = set(args.corrupt_modalities)

    print(f"Preset:             {args.preset}")
    print(f"Corrupt modalities: {sorted(corrupt_set) if corrupt_set else '(none — full symlink)'}")
    print(f"Source:             {src}")
    print(f"Dest:               {dst}")
    print(f"Seed:               {args.seed}")

    dst.mkdir(parents=True, exist_ok=True)

    # ---- label.npz: never changes
    src_label = src / "label.npz"
    if src_label.exists():
        os.symlink(src_label, dst / "label.npz")

    # ---- transcription
    src_csv = src / "transcription-engchi-polish.csv"
    dst_csv = dst / "transcription-engchi-polish.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing transcription file: {src_csv}")
    if "text" in corrupt_set:
        print("Corrupting text...")
        corrupt_text_csv(
            src_csv, dst_csv,
            config["text_char_swap_prob"],
            config["text_word_drop_prob"],
        )
    else:
        os.symlink(src_csv, dst_csv)

    # ---- audio
    if "audio" in corrupt_set:
        src_aud = src / "subaudio"
        dst_aud = dst / "subaudio"
        dst_aud.mkdir()
        wavs = sorted(p for p in src_aud.glob("*.wav") if p.name.startswith(args.name_prefix))
        if args.limit:
            wavs = wavs[:args.limit]
        print(f"Corrupting {len(wavs)} audio files matching prefix='{args.name_prefix}'...")
        for i, wav in enumerate(wavs):
            if i % 200 == 0:
                print(f"  audio {i}/{len(wavs)}", flush=True)
            try:
                corrupt_audio_file(wav, dst_aud / wav.name, args.audio_sr, config)
            except Exception as e:
                print(f"  Skip {wav.name}: {e}", flush=True)
                shutil.copy(wav, dst_aud / wav.name)
        # Symlink any other-split files so AffectGPT's loader doesn't break if
        # it ever asks for them (it shouldn't if it only reads test names).
        for wav in src_aud.glob("*.wav"):
            if wav.name.startswith(args.name_prefix):
                continue
            link = dst_aud / wav.name
            if not link.exists():
                os.symlink(wav, link)
    else:
        os.symlink(src / "subaudio", dst / "subaudio")

    # ---- video (and re-extracted faces if video corrupted)
    if "video" in corrupt_set:
        src_vid = src / "subvideo"
        dst_vid = dst / "subvideo"
        dst_vid.mkdir()
        mp4s = sorted(p for p in src_vid.glob("*.mp4") if p.name.startswith(args.name_prefix))
        if args.limit:
            mp4s = mp4s[:args.limit]
        print(f"Corrupting {len(mp4s)} video files matching prefix='{args.name_prefix}'...")
        for i, mp4 in enumerate(mp4s):
            if i % 200 == 0:
                print(f"  video {i}/{len(mp4s)}", flush=True)
            try:
                corrupt_video_file(mp4, dst_vid / mp4.name, config)
            except Exception as e:
                print(f"  Skip {mp4.name}: {e}", flush=True)
                shutil.copy(mp4, dst_vid / mp4.name)
        # Symlink other-split videos for completeness
        for mp4 in src_vid.glob("*.mp4"):
            if mp4.name.startswith(args.name_prefix):
                continue
            link = dst_vid / mp4.name
            if not link.exists():
                os.symlink(mp4, link)

        # Re-extract faces ONLY for the corrupted (test) videos; symlink the
        # rest from the clean openface_face dir.
        dst_face = dst / "openface_face"
        dst_face.mkdir()
        face_script = Path(args.extract_faces_script).resolve()
        if not face_script.exists():
            raise FileNotFoundError(f"extract_faces.py not found at {face_script}")
        # Stage a tmp dir of just the corrupted (prefix-matching) videos for
        # extract_faces.py — it processes everything in --video_dir.
        tmp_vid_dir = dst / "_face_extract_tmp"
        tmp_vid_dir.mkdir(exist_ok=True)
        for mp4 in mp4s:
            tlink = tmp_vid_dir / mp4.name
            if not tlink.exists():
                os.symlink((dst_vid / mp4.name).resolve(), tlink)
        print(f"Re-extracting faces from {len(mp4s)} corrupted videos with {args.face_workers} workers...")
        subprocess.check_call([
            sys.executable, str(face_script),
            "--video_dir", str(tmp_vid_dir),
            "--output_dir", str(dst_face),
            "--workers", str(args.face_workers),
        ])
        shutil.rmtree(tmp_vid_dir)

        # Symlink other-split face npys from clean
        clean_face_dir = src / "openface_face"
        if clean_face_dir.exists():
            for npy in clean_face_dir.glob("*.npy"):
                if npy.name.startswith(args.name_prefix):
                    continue
                link = dst_face / npy.name
                if not link.exists():
                    os.symlink(npy, link)
    else:
        os.symlink(src / "subvideo", dst / "subvideo")
        os.symlink(src / "openface_face", dst / "openface_face")

    print(f"Done. Corrupted dataset at {dst}")


if __name__ == "__main__":
    main()
