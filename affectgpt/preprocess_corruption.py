import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

    if wav_f.ndim > 1:
        wav_f = wav_f.mean(axis=1)

    wav_corrupted = apply_audio_corruptions(wav_f, sr, config)

    if was_int:
        out = np.clip(wav_corrupted * 32768.0, -32768, 32767).astype(np.int16)
    else:
        out = wav_corrupted.astype(np.float32)
    wavfile.write(str(dst_wav), sr, out)


def corrupt_video_file(src_mp4, dst_mp4, config):
    import imageio.v2 as imageio
    reader = imageio.get_reader(str(src_mp4), "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 25)
    frames = []
    for frame in reader:
        frames.append(np.asarray(frame))
    reader.close()
    if len(frames) == 0:
        shutil.copy(src_mp4, dst_mp4)
        return
    frames_arr = np.stack(frames, axis=0).astype(np.uint8)

    corrupted = apply_video_corruptions(frames_arr, config)

    writer = imageio.get_writer(
        str(dst_mp4),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for f in corrupted:
        writer.append_data(f)
    writer.close()


def _audio_worker(task):
    src_path, dst_path, audio_sr, config, seed = task
    if dst_path.exists():
        return ("skip", src_path.name)
    try:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        corrupt_audio_file(src_path, dst_path, audio_sr, config)
        return ("ok", src_path.name)
    except Exception as e:
        try:
            shutil.copy(src_path, dst_path)
        except Exception:
            pass
        return ("err", f"{src_path.name}: {e}")


def _video_worker(task):
    src_path, dst_path, config, seed = task
    if dst_path.exists():
        return ("skip", src_path.name)
    try:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        corrupt_video_file(src_path, dst_path, config)
        return ("ok", src_path.name)
    except Exception as e:
        try:
            shutil.copy(src_path, dst_path)
        except Exception:
            pass
        return ("err", f"{src_path.name}: {e}")


def _run_parallel(tasks, worker_fn, n_workers, label):
    if n_workers <= 1:
        for i, t in enumerate(tasks):
            if i % 100 == 0:
                print(f"  {label} {i}/{len(tasks)}", flush=True)
            status, info = worker_fn(t)
            if status == "err":
                print(f"  Skip {info}", flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(worker_fn, t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  {label} {done}/{len(tasks)}", flush=True)
            status, info = f.result()
            if status == "err":
                print(f"  Skip {info}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--corrupt_modalities", nargs="*", default=[],
                   choices=["text", "audio", "video"])
    p.add_argument("--preset", default="medium", choices=list(CORRUPTION_PRESET_NAMES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio_sr", type=int, default=16000)
    p.add_argument("--extract_faces_script", default="extract_faces.py")
    p.add_argument("--face_workers", type=int, default=4)
    p.add_argument("--video_workers", type=int, default=8)
    p.add_argument("--audio_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--name_prefix", default="test_")
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

    dst.mkdir(parents=True, exist_ok=True)

    src_label = src / "label.npz"
    if src_label.exists():
        os.symlink(src_label, dst / "label.npz")

    src_csv = src / "transcription-engchi-polish.csv"
    dst_csv = dst / "transcription-engchi-polish.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing transcription file: {src_csv}")
    if "text" in corrupt_set:
        corrupt_text_csv(
            src_csv, dst_csv,
            config["text_char_swap_prob"],
            config["text_word_drop_prob"],
        )
    else:
        os.symlink(src_csv, dst_csv)

    if "audio" in corrupt_set:
        src_aud = src / "subaudio"
        dst_aud = dst / "subaudio"
        dst_aud.mkdir(exist_ok=True)
        wavs = sorted(p for p in src_aud.glob("*.wav") if p.name.startswith(args.name_prefix))
        if args.limit:
            wavs = wavs[:args.limit]
        tasks = [(wav, dst_aud / wav.name, args.audio_sr, config, args.seed + i)
                 for i, wav in enumerate(wavs)]
        _run_parallel(tasks, _audio_worker, args.audio_workers, "audio")
        for wav in src_aud.glob("*.wav"):
            if wav.name.startswith(args.name_prefix):
                continue
            link = dst_aud / wav.name
            if not link.exists():
                os.symlink(wav, link)
    else:
        os.symlink(src / "subaudio", dst / "subaudio")

    if "video" in corrupt_set:
        src_vid = src / "subvideo"
        dst_vid = dst / "subvideo"
        dst_vid.mkdir(exist_ok=True)
        mp4s = sorted(p for p in src_vid.glob("*.mp4") if p.name.startswith(args.name_prefix))
        if args.limit:
            mp4s = mp4s[:args.limit]
        tasks = [(mp4, dst_vid / mp4.name, config, args.seed + i)
                 for i, mp4 in enumerate(mp4s)]
        _run_parallel(tasks, _video_worker, args.video_workers, "video")
        for mp4 in src_vid.glob("*.mp4"):
            if mp4.name.startswith(args.name_prefix):
                continue
            link = dst_vid / mp4.name
            if not link.exists():
                os.symlink(mp4, link)

        dst_face = dst / "openface_face"
        dst_face.mkdir()
        face_script = Path(args.extract_faces_script).resolve()
        if not face_script.exists():
            raise FileNotFoundError(f"extract_faces.py not found at {face_script}")
        tmp_vid_dir = dst / "_face_extract_tmp"
        tmp_vid_dir.mkdir(exist_ok=True)
        for mp4 in mp4s:
            tlink = tmp_vid_dir / mp4.name
            if not tlink.exists():
                os.symlink((dst_vid / mp4.name).resolve(), tlink)
        subprocess.check_call([
            sys.executable, str(face_script),
            "--video_dir", str(tmp_vid_dir),
            "--output_dir", str(dst_face),
            "--workers", str(args.face_workers),
        ])
        shutil.rmtree(tmp_vid_dir)

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


if __name__ == "__main__":
    main()
