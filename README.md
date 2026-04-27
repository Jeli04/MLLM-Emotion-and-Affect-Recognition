# MLLM Emotion and Affect Recognition

## Environment Setup

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management. Make sure `uv` is installed before proceeding.

ffmpeg must also be loaded as a system module:

```bash
module load ffmpeg
```

### Install dependencies

A `uv.lock` file is included in the repository. To create a virtual environment and install all pinned dependencies:

```bash
uv sync
```

This will create a `.venv` directory in the project root and install all packages from the lock file.

### Activate the environment

```bash
source .venv/bin/activate
```

Or run commands directly without activating:

```bash
uv run python main.py
```

### Full setup sequence

```bash
module load ffmpeg
uv sync
source .venv/bin/activate
```

## Model Download

This project uses [Qwen2.5-Omni-7B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4), a 4-bit quantized version of the Qwen2.5-Omni 7B model. Download it into the project root:

```bash
huggingface-cli download Qwen/Qwen2.5-Omni-7B-GPTQ-Int4 --local-dir ./Qwen2.5-Omni-7B-GPTQ-Int4
```

The model will be saved to `./Qwen2.5-Omni-7B-GPTQ-Int4`, which is the default path used by `evaluate.py`. To use a different location, pass `--model_path` when running evaluation.

## IEMOCAP

Place the IEMOCAP release tree under the project root as `IEMOCAP_full_release/` so scripts and evaluation can resolve media paths consistently.

### Corpus layout and the manifest

The official release is organized by **session** (`Session1` … `Session5`). Under each session, the pieces this repo cares about are roughly:


| Location                          | Contents                                                                                                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Session*/dialog/avi/`            | Full **dialogue** video files (`.avi`)—one file per recording/scene. Clarification: NOT utterance level.                                                                         |
| `Session*/dialog/transcriptions/` | Transcripts with **utterance-level** time spans and text.                                                                                                                        |
| `Session*/dialog/EmoEvaluation/`  | **Emotion labels** per utterance (short corpus codes, e.g. `neu`, `ang`).                                                                                                        |
| `Session*/sentences/wav/`         | Official **per-utterance** WAV files (one clip per utterance id).                                                                                                                |
| `Session*/sentences/avi/`         | **Per-utterance** video clips—often **not** in the raw release as MP4s; you can generate them with `scripts/slice_iemocap_utterance_videos.py` so paths align with the manifest. |


The **manifest** is a single **utterance-level table** (CSV plus optional JSONL) derived from the corpus: one row per utterance, with canonical English emotion labels, transcript text, timing/metadata, and paths to that utterance’s WAV and video files. Evaluation reads this file instead of walking `Session*/` at runtime. Default output location:

- `IEMOCAP_full_release/manifests/iemocap_utterance_labels.csv`
- `IEMOCAP_full_release/manifests/iemocap_utterance_labels.jsonl`

Typical columns include `session`, `utterance_id`, `recording_id`, `text`, `emotion`, `wav_path`, `video_path`, and timing fields. By default, `wav_path` / `video_path` are stored **relative to** `IEMOCAP_full_release` (portable if you move the whole tree). In `evaluate.py`, rows with `no_agreement` or labels outside the 10-class evaluation set are skipped.

A reusable PyTorch `Dataset` for the same CSV lives in `src/iemocap_dataset.py`; `evaluate.py` parses the manifest **directly** when `--dataset iemocap`.

### Data setup

You may need to create per-utterance video clips and/or (re)build the manifest:

- **Per-utterance video clips** (from full dialogue AVIs): `scripts/slice_iemocap_utterance_videos.py` — `--iemocap_root` defaults to `./IEMOCAP_full_release` when omitted. Requires `ffmpeg` on your `PATH`.
  Example commands (run from the project root):
  ```bash
  # Default data root (./IEMOCAP_full_release), all sessions
  uv run python scripts/slice_iemocap_utterance_videos.py
  ```
  ```bash
  # Explicit root (if IEMOCAP lives elsewhere)
  uv run python scripts/slice_iemocap_utterance_videos.py \
    --iemocap_root /path/to/IEMOCAP_full_release
  ```
  ```bash
  # Only Session1 and Session2
  uv run python scripts/slice_iemocap_utterance_videos.py --sessions Session1 Session2
  ```
  ```bash
  # First 20 utterances only (prints real ffmpeg work for a subset)
  uv run python scripts/slice_iemocap_utterance_videos.py --limit 20
  ```
- **Manifest CSV / JSONL**: `scripts/build_iemocap_emotion_manifest.py` reads `EmoEvaluation` (and optionally transcriptions + expected clip paths). `--iemocap_root` defaults to `./IEMOCAP_full_release`. Use `--with_text` and `--with_paths` for a full table suitable for multimodal evaluation. Paths default to **relative** to the corpus root; pass `--absolute_paths` if you want absolute paths on disk.
  Example commands:
  ```bash
  # Full manifest: labels + transcript text + wav/video paths (relative paths, default output dir)
  uv run python scripts/build_iemocap_emotion_manifest.py --with_text --with_paths
  ```


### Evaluation

Run `evaluate.py` with `--dataset iemocap` and `--manifest` pointing at your CSV. Use `--model_path` if the model is not at the default location.

```bash
uv run python evaluate.py \
  --dataset iemocap \
  --manifest IEMOCAP_full_release/manifests/iemocap_utterance_labels.csv \
  --split test \
  --modalities text video \
  --model_path ./Qwen2.5-Omni-7B-GPTQ-Int4
```

- `**--split**`: If the manifest has a `split` column, only rows matching that value are used; if there is no `split` column, this filter is ignored (use `--iemocap_sessions` or `--max_samples` to limit scope).
- `**--iemocap_sessions**`: Optional list, e.g. `Session1 Session2`, to restrict by session.
- `**--max_samples**`: Optional cap for quick smoke tests.

**Modalities:** `text`, `audio`, and/or `video`. When `video` is included, the processor loads **audio from the same video file** as the visuals (same behavior as the MELD path). Standalone `audio` without `video` uses the resolved WAV path from the manifest when present.

Results are written under `results/` as `results_iemocap_{split}_{modalities}.json` (for example `results_iemocap_test_text+video.json`).
