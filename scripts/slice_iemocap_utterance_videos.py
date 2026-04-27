#!/usr/bin/env python3
"""
Slice IEMOCAP dialog-level AVIs into per-utterance video clips using timestamps from
SessionX/dialog/transcriptions/*.txt (same timing as utterance-level WAV under sentences/wav).

Output layout (mirrors sentences/wav):
  <iemocap_root>/SessionK/sentences/avi/<recording_id>/<utterance_id>.mp4

Requires ffmpeg in PATH.

Example:
  python scripts/slice_iemocap_utterance_videos.py \\
    --iemocap_root "/path/to/IEMOCAP_full_release"

If ``--iemocap_root`` is omitted, uses ``<repo>/IEMOCAP_full_release``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


# Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
TRANSCRIPTION_LINE_RE = re.compile(
    r"^(?P<utt_id>\S+)\s+\[(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\]\s*:\s*(?P<text>.*)$"
)

# Utterance id ends with _F### or _M### (three digits); recording id is the prefix.
UTT_TO_RECORDING_RE = re.compile(r"^(?P<rec>.+)_[FM]\d{3}$")


def _default_iemocap_root() -> Path:
    return Path(__file__).resolve().parent.parent / "IEMOCAP_full_release"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create per-utterance video clips for IEMOCAP (dialog AVI + transcription times).",
    )
    p.add_argument(
        "--iemocap_root",
        type=Path,
        default=None,
        help=f"Path to IEMOCAP_full_release (default: {_default_iemocap_root()})",
    )
    p.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional session folder names, e.g. Session1 Session2. Default: all Session* under root.",
    )
    p.add_argument(
        "--output_subdir",
        default="avi",
        help="Under sentences/, clips go to sentences/<output_subdir>/ (default: avi).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode clips even if the output file already exists.",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Use stream copy (-c copy) instead of re-encoding to H.264/AAC. Faster but cuts may be less accurate.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned ffmpeg commands without running them.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many utterances (for smoke tests).",
    )
    return p.parse_args()


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not found in PATH. Install ffmpeg and retry.")
    return exe


def session_dirs(root: Path, names: list[str] | None) -> list[Path]:
    if names:
        out = []
        for n in names:
            d = root / n
            if not d.is_dir():
                sys.exit(f"Not a directory: {d}")
            out.append(d)
        return sorted(out, key=lambda p: p.name)
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("Session")])


def recording_id_from_utterance(utt_id: str) -> str | None:
    m = UTT_TO_RECORDING_RE.match(utt_id)
    return m.group("rec") if m else None


def find_dialog_avi(session_dir: Path, recording_id: str) -> Path | None:
    """
    Official layout uses SessionX/dialog/avi/<name>.avi; some releases nest under dialog/avi/DivX/.
    """
    avi_root = session_dir / "dialog" / "avi"
    candidates = [
        avi_root / f"{recording_id}.avi",
        avi_root / "DivX" / f"{recording_id}.avi",
    ]
    for c in candidates:
        if c.is_file():
            return c
    if avi_root.is_dir():
        for p in avi_root.rglob(f"{recording_id}.avi"):
            if p.is_file():
                return p
    return None


def run_ffmpeg(
    ffmpeg: str,
    src: Path,
    start_sec: float,
    end_sec: float,
    dst: Path,
    *,
    use_copy: bool,
    dry_run: bool,
) -> bool:
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        print(f"  skip bad span {start_sec}-{end_sec} -> {dst.name}", file=sys.stderr)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    # Accurate seek: place -ss after -i for frame-accurate cuts (slower) — for dataset prep we prefer accuracy.
    cmd += ["-ss", f"{start_sec:.6f}", "-i", str(src), "-t", f"{duration:.6f}"]
    if use_copy:
        cmd += ["-c", "copy", str(dst)]
    else:
        # Re-encode for broad player/model compatibility; keep audio (Qwen may use audio from video).
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(dst),
        ]

    if dry_run:
        print(" ".join(cmd))
        return True

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed for {dst.name}: {r.stderr or r.stdout}", file=sys.stderr)
        return False
    return True


def iter_transcription_lines(trans_dir: Path) -> list[tuple[Path, str]]:
    if not trans_dir.is_dir():
        return []
    rows: list[tuple[Path, str]] = []
    for p in sorted(trans_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in text:
            line = line.strip()
            if not line:
                continue
            rows.append((p, line))
    return rows


def main() -> None:
    args = parse_args()
    ffmpeg_exe = find_ffmpeg()
    root = (args.iemocap_root or _default_iemocap_root()).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    sessions = session_dirs(root, args.sessions)
    if not sessions:
        sys.exit(f"No Session* folders under {root}")

    done = 0
    skipped_exists = 0
    missing_video = 0
    parse_fail = 0
    failed = 0

    for session in sessions:
        trans_dir = session / "dialog" / "transcriptions"
        out_base = session / "sentences" / args.output_subdir
        for trans_path, raw in iter_transcription_lines(trans_dir):
            if args.limit is not None and done >= args.limit:
                print("Reached --limit, stopping.")
                return

            m = TRANSCRIPTION_LINE_RE.match(raw)
            if not m:
                parse_fail += 1
                continue

            utt_id = m.group("utt_id")
            start = float(m.group("start"))
            end = float(m.group("end"))
            rec_id = recording_id_from_utterance(utt_id)
            if not rec_id:
                parse_fail += 1
                continue

            src = find_dialog_avi(session, rec_id)
            if src is None:
                missing_video += 1
                print(f"  missing dialog AVI for {rec_id} (session {session.name})", file=sys.stderr)
                continue

            dst = out_base / rec_id / f"{utt_id}.mp4"
            if dst.exists() and not args.overwrite:
                skipped_exists += 1
                continue

            ok = run_ffmpeg(
                ffmpeg_exe,
                src,
                start,
                end,
                dst,
                use_copy=args.copy,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                done += 1
                continue
            if ok:
                done += 1
            else:
                failed += 1

    print(
        f"Finished. encoded={done} skipped_existing={skipped_exists} "
        f"missing_source_avi={missing_video} parse_skips={parse_fail} failed={failed}"
    )


if __name__ == "__main__":
    main()
