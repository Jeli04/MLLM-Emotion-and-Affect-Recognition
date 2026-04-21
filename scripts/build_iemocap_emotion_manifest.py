#!/usr/bin/env python3
"""
Build a per-utterance manifest with IEMOCAP categorical emotion labels from
SessionX/dialog/EmoEvaluation/Ses*.txt (first summary line per utterance block).

Optionally merge transcript text from SessionX/dialog/transcriptions/*.txt and
fill expected paths for sentences/wav and sentences/avi utterance clips.

Example:
  python scripts/build_iemocap_emotion_manifest.py \\
    --iemocap_root "/path/to/IEMOCAP_full_release" \\
    --with_text --with_paths

If ``--iemocap_root`` is omitted, uses ``<repo>/IEMOCAP_full_release`` (parent of ``scripts/``).
With ``--relative_paths`` (default), wav/video columns are stored relative to the corpus root so
moving the data folder only requires keeping ``manifests/`` under ``IEMOCAP_full_release/``.

Outputs (default): <iemocap_root>/manifests/iemocap_utterance_labels.csv
                  and iemocap_utterance_labels.jsonl
With --limit N: .../iemocap_utterance_labels_limitN.csv (same stem .jsonl)

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


# First line of each utterance block in EmoEvaluation/<recording>.txt
# [start - end]  Ses01F_impro01_F000  neu  [2.5, 2.5, 2.5]
SUMMARY_LINE_RE = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+"
    r"(?P<utt_id>\S+)\s+(?P<emo_raw>\S+)\s+"
    r"\[(?P<vad>[^\]]*)\]\s*$"
)

# Utterance id -> recording prefix (Ses01F_impro01 from Ses01F_impro01_F000)
UTT_TO_RECORDING_RE = re.compile(r"^(?P<rec>.+)_[FM]\d{3}$")

# Transcription: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
TRANSCRIPTION_LINE_RE = re.compile(
    r"^(?P<utt_id>\S+)\s+\[(?P<start>[\d.]+)-(?P<end>[\d.]+)\]\s*:\s*(?P<text>.*)$"
)

# Map 3-letter corpus tags to single-token English labels (IEMOCAP README set).
EMOTION_CANONICAL = {
    "neu": "neutral",
    "ang": "angry",
    "hap": "happy",
    "sad": "sad",
    "sur": "surprised",
    "fea": "fearful",
    "dis": "disgusted",
    "fru": "frustrated",
    "exc": "excited",
    "oth": "other",
    "xxx": "no_agreement",
}


def _default_iemocap_root() -> Path:
    return Path(__file__).resolve().parent.parent / "IEMOCAP_full_release"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build IEMOCAP utterance-level emotion manifest.")
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
        help="Session folder names (e.g. Session1). Default: all Session*.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write CSV/JSONL. Default: <iemocap_root>/manifests",
    )
    p.add_argument(
        "--with_text",
        action="store_true",
        help="Merge transcript text from dialog/transcriptions/*.txt",
    )
    p.add_argument(
        "--with_paths",
        action="store_true",
        help="Add expected wav_path and video_path (utterance clips under sentences/).",
    )
    p.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Write absolute wav/video paths. Default is relative to --iemocap_root (portable).",
    )
    p.add_argument(
        "--video_ext",
        default="mp4",
        help="Filename extension for sliced video clips (default: mp4).",
    )
    p.add_argument(
        "--drop_no_agreement",
        action="store_true",
        help="Omit rows where corpus label is xxx (no majority).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many utterances (processing order: sorted EmoEvaluation files, top to bottom).",
    )
    return p.parse_args()


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


def load_transcriptions(session_dir: Path) -> dict[str, str]:
    """utt_id -> text"""
    trans_dir = session_dir / "dialog" / "transcriptions"
    out: dict[str, str] = {}
    if not trans_dir.is_dir():
        return out
    for p in sorted(trans_dir.glob("*.txt")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            m = TRANSCRIPTION_LINE_RE.match(line)
            if m:
                out[m.group("utt_id")] = m.group("text").strip()
    return out


def iter_emoeval_summaries(emo_dir: Path) -> list[dict]:
    """Parse all Ses*.txt at top level of EmoEvaluation."""
    rows: list[dict] = []
    for p in sorted(emo_dir.glob("Ses*.txt")):
        if p.is_dir():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("["):
                continue
            m = SUMMARY_LINE_RE.match(line)
            if not m:
                continue
            rows.append(
                {
                    "source_file": str(p.name),
                    "start_sec": float(m.group("start")),
                    "end_sec": float(m.group("end")),
                    "utterance_id": m.group("utt_id"),
                    "emotion_raw": m.group("emo_raw"),
                    "vad": m.group("vad").strip(),
                }
            )
    return rows


def canonical_emotion(emo_raw: str) -> str:
    key = emo_raw.lower().strip()
    if key in EMOTION_CANONICAL:
        return EMOTION_CANONICAL[key]
    return key


def main() -> None:
    args = parse_args()
    root = (args.iemocap_root or _default_iemocap_root()).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    out_dir = args.output_dir or (root / "manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit is not None:
        base = f"iemocap_utterance_labels_limit{args.limit}"
    else:
        base = "iemocap_utterance_labels"
    csv_path = out_dir / f"{base}.csv"
    jsonl_path = out_dir / f"{base}.jsonl"

    sessions = session_dirs(root, args.sessions)
    if not sessions:
        sys.exit(f"No Session* under {root}")

    all_rows: list[dict] = []
    unknown_raw: set[str] = set()
    stop = False

    for session in sessions:
        emo_dir = session / "dialog" / "EmoEvaluation"
        if not emo_dir.is_dir():
            print(f"Warning: missing {emo_dir}", file=sys.stderr)
            continue

        trans_map = load_transcriptions(session) if args.with_text else {}
        summaries = iter_emoeval_summaries(emo_dir)

        for s in summaries:
            if args.limit is not None and len(all_rows) >= args.limit:
                stop = True
                break
            utt = s["utterance_id"]
            emo_raw = s["emotion_raw"]
            if args.drop_no_agreement and emo_raw.lower() == "xxx":
                continue

            rec = recording_id_from_utterance(utt)
            if not rec:
                print(f"Warning: bad utterance id {utt}", file=sys.stderr)
                continue

            emo = canonical_emotion(emo_raw)
            if emo_raw.lower() not in EMOTION_CANONICAL:
                unknown_raw.add(emo_raw)

            row = {
                "session": session.name,
                "utterance_id": utt,
                "recording_id": rec,
                "start_sec": s["start_sec"],
                "end_sec": s["end_sec"],
                "emotion_raw": emo_raw,
                "emotion": emo,
                "has_agreement": emo_raw.lower() != "xxx",
                "vad": s["vad"],
                "emoeval_source": s["source_file"],
            }
            if args.with_text:
                row["text"] = trans_map.get(utt, "")
            if args.with_paths:
                wav = session / "sentences" / "wav" / rec / f"{utt}.wav"
                vid = session / "sentences" / "avi" / rec / f"{utt}.{args.video_ext}"
                if args.absolute_paths:
                    row["wav_path"] = str(wav)
                    row["video_path"] = str(vid)
                else:
                    row["wav_path"] = str(wav.relative_to(root))
                    row["video_path"] = str(vid.relative_to(root))

            all_rows.append(row)

        if stop:
            break

    # Full runs: stable sort. With --limit, keep encounter order (first N lines in file order).
    if args.limit is None:
        all_rows.sort(key=lambda r: (r["session"], r["utterance_id"]))

    if not all_rows:
        sys.exit("No utterance summary lines found. Check --iemocap_root and Session*/dialog/EmoEvaluation.")

    if unknown_raw:
        print(
            "Note: non-standard emotion_raw tokens (passed through as emotion): "
            + ", ".join(sorted(unknown_raw)),
            file=sys.stderr,
        )

    keys_order = [
        "session",
        "utterance_id",
        "recording_id",
        "start_sec",
        "end_sec",
        "emotion_raw",
        "emotion",
        "has_agreement",
        "vad",
        "emoeval_source",
    ]
    if args.with_text:
        keys_order.append("text")
    if args.with_paths:
        keys_order.extend(["wav_path", "video_path"])
    seen = set(keys_order)
    for r in all_rows:
        for k in r:
            if k not in seen:
                keys_order.append(k)
                seen.add(k)
    fieldnames = [k for k in keys_order if any(k in r for r in all_rows)]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_rows)} rows to {csv_path}")
    print(f"Wrote {len(all_rows)} lines to {jsonl_path}")


if __name__ == "__main__":
    main()
