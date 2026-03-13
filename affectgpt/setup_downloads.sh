#!/bin/bash
# ==============================================================
# AffectGPT Setup — Download models + prepare MELD dataset
# ==============================================================
#
# Downloads from HuggingFace:
#   1. clip-vit-large-patch14        (Visual Encoder, ~1.7 GB)
#   2. chinese-hubert-large          (Audio Encoder,  ~1.2 GB)
#   3. Qwen2.5-7B-Instruct          (LLM backbone,  ~15 GB)
#   4. bert-base-uncased             (Q-Former init, ~0.4 GB)
#   5. AffectGPT fine-tuned ckpts    (7 epochs,      ~4.7 GB)
#
# Preprocesses the raw MELD.Raw.tar.gz already in the project
# into the format AffectGPT expects at ../dataset/meld-process/
#
# ==============================================================

set -eo pipefail

ROOT_DIR="$(pwd)"
DATASET_DIR="${ROOT_DIR}/../dataset"
MELD_RAW="${ROOT_DIR}/MELD.Raw"
MELD_TAR="${ROOT_DIR}/MELD.Raw.tar.gz"

# ---- Check prerequisites ----
# huggingface_hub v1.x renamed the CLI from "huggingface-cli" to "hf"
if command -v hf &>/dev/null; then
    HF_DL="hf download"
elif command -v huggingface-cli &>/dev/null; then
    HF_DL="huggingface-cli download"
else
    echo "ERROR: hf/huggingface-cli not found. Install: pip install huggingface_hub"
    exit 1
fi

HAS_FFMPEG=0
if command -v ffmpeg &>/dev/null; then
    HAS_FFMPEG=1
fi

echo "=============================================="
echo " AffectGPT Setup"
echo "=============================================="

# ==============================================================
# 1. Download pretrained models → ./models/
# ==============================================================
MODELS_DIR="${ROOT_DIR}/models"
mkdir -p "${MODELS_DIR}"

download_hf_model() {
    local repo="$1"
    local target="$2"
    local label="$3"

    echo ""
    echo "Downloading ${label}..."
    if [ -d "${target}" ] && [ -f "${target}/config.json" ]; then
        echo "  Already exists, skipping."
    else
        ${HF_DL} "${repo}" \
            --local-dir "${target}"
        echo "  Done."
    fi
}

download_hf_model "openai/clip-vit-large-patch14" \
    "${MODELS_DIR}/clip-vit-large-patch14" \
    "[1/5] clip-vit-large-patch14 (Visual Encoder)"

download_hf_model "TencentGameMate/chinese-hubert-large" \
    "${MODELS_DIR}/chinese-hubert-large" \
    "[2/5] chinese-hubert-large (Audio Encoder)"

download_hf_model "Qwen/Qwen2.5-7B-Instruct" \
    "${MODELS_DIR}/Qwen2.5-7B-Instruct" \
    "[3/5] Qwen2.5-7B-Instruct (LLM)"

download_hf_model "google-bert/bert-base-uncased" \
    "${MODELS_DIR}/bert-base-uncased" \
    "[4/5] bert-base-uncased (Q-Former init)"

# ==============================================================
# 2. Download fine-tuned AffectGPT checkpoint → ./output/
# ==============================================================
CKPT_NAME="emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz"
CKPT_SUBDIR="${CKPT_NAME}_20250110100"
CKPT_PARENT="${ROOT_DIR}/output/${CKPT_NAME}"

echo ""
echo "[5/5] Downloading AffectGPT fine-tuned checkpoints (~4.7 GB)..."
if [ -d "${CKPT_PARENT}/${CKPT_SUBDIR}" ] && ls "${CKPT_PARENT}/${CKPT_SUBDIR}"/checkpoint_*.pth &>/dev/null 2>&1; then
    echo "  Already exists, skipping."
else
    mkdir -p "${CKPT_PARENT}"
    ${HF_DL} MERChallenge/AffectGPT \
        --include "${CKPT_SUBDIR}/*" \
        --local-dir "${CKPT_PARENT}"
    echo "  Done."
fi

# ==============================================================
# 3. Preprocess MELD raw data → ../dataset/meld-process/
# ==============================================================
MELD_PROC="${DATASET_DIR}/meld-process"

echo ""
echo "=============================================="
echo " Preparing MELD dataset"
echo "=============================================="

# 3a. Extract raw MELD if needed
if [ ! -d "${MELD_RAW}" ]; then
    if [ -f "${MELD_TAR}" ]; then
        echo "Extracting MELD.Raw.tar.gz..."
        tar xzf "${MELD_TAR}" -C "$(dirname "${MELD_TAR}")"
        echo "  Done."
    else
        echo "ERROR: Cannot find MELD.Raw.tar.gz at ${MELD_TAR}"
        exit 1
    fi
fi

# 3b. Extract inner tar.gz files (videos + any CSVs packed inside)
# MELD tars extract to non-standard names: train_splits, dev_splits_complete, output_repeated_splits_test
for split_info in train:train_splits dev:dev_splits_complete test:output_repeated_splits_test; do
    split="${split_info%%:*}"
    split_dir="${split_info##*:}"
    INNER_TAR="${MELD_RAW}/${split}.tar.gz"
    INNER_DIR="${MELD_RAW}/${split_dir}"
    # Extract CSV labels if missing (train_sent_emo.csv is inside train.tar.gz)
    SPLIT_CSV="${MELD_RAW}/${split}_sent_emo.csv"
    if [ -f "${INNER_TAR}" ] && [ ! -f "${SPLIT_CSV}" ]; then
        echo "Extracting ${split}_sent_emo.csv from ${split}.tar.gz..."
        tar xzf "${INNER_TAR}" -C "${MELD_RAW}/" "${split}_sent_emo.csv" 2>/dev/null || true
    fi
    # Extract video directory
    if [ -f "${INNER_TAR}" ] && [ ! -d "${INNER_DIR}" ]; then
        echo "Extracting ${split}.tar.gz..."
        tar xzf "${INNER_TAR}" -C "${MELD_RAW}/"
        echo "  Done."
    fi
done

# 3c. Build meld-process/ directory
if [ -d "${MELD_PROC}" ] && [ -f "${MELD_PROC}/label.npz" ]; then
    echo "meld-process/ already exists, skipping preprocessing."
else
    echo "Building meld-process/..."
    mkdir -p "${MELD_PROC}/subvideo"
    mkdir -p "${MELD_PROC}/subaudio"
    mkdir -p "${MELD_PROC}/openface_face"

    # Run the Python preprocessing
    python -c "
import os, sys, shutil
import numpy as np
import pandas as pd

meld_raw = '${MELD_RAW}'
save_root = '${MELD_PROC}'

emos = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
emo2idx = {emo: i for i, emo in enumerate(emos)}

def read_labels(label_path):
    df = pd.read_csv(label_path)
    names, labels, engs = [], [], []
    for _, row in df.iterrows():
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID']
        names.append(f'dia{dia_id}_utt{utt_id}')
        labels.append(emo2idx[row['Emotion']])
        utt = row['Utterance']
        engs.append('' if pd.isna(utt) else str(utt))
    return names, labels, engs

# Read all splits
train_names, train_labels, train_engs = read_labels(os.path.join(meld_raw, 'train_sent_emo.csv'))
val_names,   val_labels,   val_engs   = read_labels(os.path.join(meld_raw, 'dev_sent_emo.csv'))
test_names,  test_labels,  test_engs  = read_labels(os.path.join(meld_raw, 'test_sent_emo.csv'))
print(f'train: {len(train_names)}, val: {len(val_names)}, test: {len(test_names)}')

# Copy videos + build labels
save_video = os.path.join(save_root, 'subvideo')
name2eng = {}
whole_corpus = {}
for datatype, names, labels, engs, video_dir in [
    ('train', train_names, train_labels, train_engs, 'train_splits'),
    ('val',   val_names,   val_labels,   val_engs,   'dev_splits_complete'),
    ('test',  test_names,  test_labels,  test_engs,  'output_repeated_splits_test'),
]:
    whole_corpus[datatype] = {}
    video_root = os.path.join(meld_raw, video_dir)
    for ii, name in enumerate(names):
        newname = f'{datatype}_{name}'
        whole_corpus[datatype][newname] = {'emo': labels[ii], 'val': -10}
        name2eng[newname] = engs[ii]

        src = os.path.join(video_root, name + '.mp4')
        dst = os.path.join(save_video, newname + '.mp4')
        if os.path.exists(dst):
            continue
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f'WARNING: missing video {src}')

# Save label.npz
np.savez_compressed(
    os.path.join(save_root, 'label.npz'),
    train_corpus=whole_corpus['train'],
    val_corpus=whole_corpus['val'],
    test_corpus=whole_corpus['test'],
)
print('Saved label.npz')

# Save transcription CSV (transcription-engchi-polish.csv)
# AffectGPT expects columns: name, english
trans_path = os.path.join(save_root, 'transcription-engchi-polish.csv')
rows = []
for name in name2eng:
    rows.append({'name': name, 'english': name2eng[name]})
df = pd.DataFrame(rows)
df.to_csv(trans_path, index=False)
print(f'Saved {trans_path} ({len(rows)} rows)')
"
    echo "  Preprocessing done."

    # 3d. Extract audio from videos using ffmpeg
    if [ "${HAS_FFMPEG}" -eq 1 ]; then
        echo "Extracting audio from videos..."
        for mp4 in "${MELD_PROC}"/subvideo/*.mp4; do
            fname="$(basename "${mp4}" .mp4)"
            wav="${MELD_PROC}/subaudio/${fname}.wav"
            if [ ! -f "${wav}" ]; then
                ffmpeg -nostats -loglevel error -i "${mp4}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${wav}" -y 2>/dev/null || true
            fi
        done
        echo "  Audio extraction done."
    else
        echo "  Skipped audio extraction (ffmpeg not found)."
    fi

    # 3e. Extract face crops from videos using MediaPipe
    echo "Extracting face crops from videos..."
    python extract_faces.py \
        --video_dir "${MELD_PROC}/subvideo" \
        --output_dir "${MELD_PROC}/openface_face"
    echo "  Face extraction done."
fi

# ==============================================================
# 4. Fill in missing audio/faces if previous run was incomplete
# ==============================================================
# Audio
if [ "${HAS_FFMPEG}" -eq 1 ]; then
    AUDIO_COUNT=$(ls "${MELD_PROC}/subaudio/" 2>/dev/null | wc -l)
    VIDEO_COUNT=$(ls "${MELD_PROC}/subvideo/" 2>/dev/null | wc -l)
    if [ "${AUDIO_COUNT}" -lt "${VIDEO_COUNT}" ]; then
        echo "Resuming audio extraction (${AUDIO_COUNT}/${VIDEO_COUNT})..."
        for mp4 in "${MELD_PROC}"/subvideo/*.mp4; do
            fname="$(basename "${mp4}" .mp4)"
            wav="${MELD_PROC}/subaudio/${fname}.wav"
            if [ ! -f "${wav}" ]; then
                ffmpeg -nostats -loglevel error -i "${mp4}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${wav}" -y 2>/dev/null || true
            fi
        done
        echo "  Audio extraction done."
    fi
fi

# Faces
FACE_COUNT=$(ls "${MELD_PROC}/openface_face/" 2>/dev/null | wc -l)
VIDEO_COUNT=$(ls "${MELD_PROC}/subvideo/" 2>/dev/null | wc -l)
if [ "${FACE_COUNT}" -lt "${VIDEO_COUNT}" ]; then
    echo "Resuming face extraction (${FACE_COUNT}/${VIDEO_COUNT})..."
    python extract_faces.py \
        --video_dir "${MELD_PROC}/subvideo" \
        --output_dir "${MELD_PROC}/openface_face"
    echo "  Face extraction done."
fi

# ==============================================================
# Summary
# ==============================================================
echo ""
echo "=============================================="
echo " SETUP COMPLETE"
echo "=============================================="
echo ""
echo " Models:       ${MODELS_DIR}/"
ls -1d "${MODELS_DIR}"/*/ 2>/dev/null | sed 's/^/   /'
echo ""
echo " Checkpoints:  ${CKPT_PARENT}/${CKPT_SUBDIR}/"
ls "${CKPT_PARENT}/${CKPT_SUBDIR}"/checkpoint_*.pth 2>/dev/null | xargs -I{} basename {} | sed 's/^/   /' || echo "   (none found)"
echo ""
echo " MELD dataset: ${MELD_PROC}/"
echo "   subvideo/:   $(ls "${MELD_PROC}/subvideo/" 2>/dev/null | wc -l | tr -d ' ') files"
echo "   subaudio/:   $(ls "${MELD_PROC}/subaudio/" 2>/dev/null | wc -l | tr -d ' ') files"
echo "   openface/:   $(ls "${MELD_PROC}/openface_face/" 2>/dev/null | wc -l | tr -d ' ') files"
echo "   label.npz:   $([ -f "${MELD_PROC}/label.npz" ] && echo 'YES' || echo 'NO')"
echo ""
echo " Next step:"
echo "   ./run_meld_missing_modality.sh"
echo ""
