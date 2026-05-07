#!/bin/bash
set -eo pipefail

ROOT_DIR="$(pwd)"
DATASET_DIR="${ROOT_DIR}/../dataset"
MELD_RAW="${ROOT_DIR}/MELD.Raw"
MELD_TAR="${ROOT_DIR}/MELD.Raw.tar.gz"

TEST_ONLY="${TEST_ONLY:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
FACE_WORKERS="${FACE_WORKERS:-4}"

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

MODELS_DIR="${ROOT_DIR}/models"
mkdir -p "${MODELS_DIR}"

download_hf_model() {
    local repo="$1"
    local target="$2"
    local label="$3"
    echo "Downloading ${label}..."
    if [ -d "${target}" ] && [ -f "${target}/config.json" ]; then
        echo "  Already exists, skipping."
    else
        ${HF_DL} "${repo}" --local-dir "${target}"
    fi
}

download_hf_model "openai/clip-vit-large-patch14" \
    "${MODELS_DIR}/clip-vit-large-patch14" \
    "clip-vit-large-patch14"

download_hf_model "TencentGameMate/chinese-hubert-large" \
    "${MODELS_DIR}/chinese-hubert-large" \
    "chinese-hubert-large"

download_hf_model "Qwen/Qwen2.5-7B-Instruct" \
    "${MODELS_DIR}/Qwen2.5-7B-Instruct" \
    "Qwen2.5-7B-Instruct"

download_hf_model "google-bert/bert-base-uncased" \
    "${MODELS_DIR}/bert-base-uncased" \
    "bert-base-uncased"

CKPT_NAME="emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz"
CKPT_SUBDIR="${CKPT_NAME}_20250110100"
CKPT_PARENT="${ROOT_DIR}/output/${CKPT_NAME}"

echo "Downloading AffectGPT checkpoints..."
if [ -d "${CKPT_PARENT}/${CKPT_SUBDIR}" ] && ls "${CKPT_PARENT}/${CKPT_SUBDIR}"/checkpoint_*.pth &>/dev/null 2>&1; then
    echo "  Already exists, skipping."
else
    mkdir -p "${CKPT_PARENT}"
    ${HF_DL} MERChallenge/AffectGPT \
        --include "${CKPT_SUBDIR}/*" \
        --local-dir "${CKPT_PARENT}"
fi

MELD_PROC="${DATASET_DIR}/meld-process"

if [ ! -d "${MELD_RAW}" ]; then
    if [ -f "${MELD_TAR}" ]; then
        echo "Extracting MELD.Raw.tar.gz..."
        tar xzf "${MELD_TAR}" -C "$(dirname "${MELD_TAR}")"
    else
        echo "ERROR: Cannot find MELD.Raw.tar.gz at ${MELD_TAR}"
        exit 1
    fi
fi

# MELD inner tars extract to non-standard names
for split_info in train:train_splits dev:dev_splits_complete test:output_repeated_splits_test; do
    split="${split_info%%:*}"
    split_dir="${split_info##*:}"
    INNER_TAR="${MELD_RAW}/${split}.tar.gz"
    INNER_DIR="${MELD_RAW}/${split_dir}"
    SPLIT_CSV="${MELD_RAW}/${split}_sent_emo.csv"
    if [ -f "${INNER_TAR}" ] && [ ! -f "${SPLIT_CSV}" ]; then
        tar xzf "${INNER_TAR}" -C "${MELD_RAW}/" "${split}_sent_emo.csv" 2>/dev/null || true
    fi
    if [ -f "${INNER_TAR}" ] && [ ! -d "${INNER_DIR}" ]; then
        echo "Extracting ${split}.tar.gz..."
        tar xzf "${INNER_TAR}" -C "${MELD_RAW}/"
    fi
done

if [ -d "${MELD_PROC}" ] && [ -f "${MELD_PROC}/label.npz" ]; then
    echo "meld-process/ already exists, skipping preprocessing."
else
    echo "Building meld-process/..."
    mkdir -p "${MELD_PROC}/subvideo"
    mkdir -p "${MELD_PROC}/subaudio"
    mkdir -p "${MELD_PROC}/openface_face"

    python -c "
import os, sys, shutil
import numpy as np
import pandas as pd

meld_raw = '${MELD_RAW}'
save_root = '${MELD_PROC}'
test_only = ${TEST_ONLY} == 1
max_test_samples = ${MAX_TEST_SAMPLES}

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

train_names, train_labels, train_engs = read_labels(os.path.join(meld_raw, 'train_sent_emo.csv'))
val_names,   val_labels,   val_engs   = read_labels(os.path.join(meld_raw, 'dev_sent_emo.csv'))
test_names,  test_labels,  test_engs  = read_labels(os.path.join(meld_raw, 'test_sent_emo.csv'))
print(f'train: {len(train_names)}, val: {len(val_names)}, test: {len(test_names)}')

save_video = os.path.join(save_root, 'subvideo')
name2eng = {}
whole_corpus = {}
splits_to_process = [
    ('train', train_names, train_labels, train_engs, 'train_splits'),
    ('val',   val_names,   val_labels,   val_engs,   'dev_splits_complete'),
    ('test',  test_names,  test_labels,  test_engs,  'output_repeated_splits_test'),
]
if test_only:
    splits_to_process = [s for s in splits_to_process if s[0] == 'test']
if max_test_samples > 0:
    capped = []
    for s in splits_to_process:
        dt, names, labels, engs, vd = s
        if dt == 'test':
            names = names[:max_test_samples]
            labels = labels[:max_test_samples]
            engs = engs[:max_test_samples]
        capped.append((dt, names, labels, engs, vd))
    splits_to_process = capped
for datatype, names, labels, engs, video_dir in splits_to_process:
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

np.savez_compressed(
    os.path.join(save_root, 'label.npz'),
    train_corpus=whole_corpus.get('train', {}),
    val_corpus=whole_corpus.get('val', {}),
    test_corpus=whole_corpus.get('test', {}),
)

trans_path = os.path.join(save_root, 'transcription-engchi-polish.csv')
rows = []
for name in name2eng:
    rows.append({'name': name, 'english': name2eng[name]})
df = pd.DataFrame(rows)
df.to_csv(trans_path, index=False)
"

    if [ "${HAS_FFMPEG}" -eq 1 ]; then
        echo "Extracting audio from videos..."
        for mp4 in "${MELD_PROC}"/subvideo/*.mp4; do
            fname="$(basename "${mp4}" .mp4)"
            wav="${MELD_PROC}/subaudio/${fname}.wav"
            if [ ! -f "${wav}" ]; then
                ffmpeg -nostats -loglevel error -i "${mp4}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "${wav}" -y 2>/dev/null || true
            fi
        done
    fi

    echo "Extracting face crops with ${FACE_WORKERS} workers..."
    python extract_faces.py \
        --video_dir "${MELD_PROC}/subvideo" \
        --output_dir "${MELD_PROC}/openface_face" \
        --workers "${FACE_WORKERS}"
fi

# Resume audio if previous run was incomplete
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
    fi
fi

# Resume face extraction if incomplete
FACE_COUNT=$(ls "${MELD_PROC}/openface_face/" 2>/dev/null | wc -l)
VIDEO_COUNT=$(ls "${MELD_PROC}/subvideo/" 2>/dev/null | wc -l)
if [ "${FACE_COUNT}" -lt "${VIDEO_COUNT}" ]; then
    echo "Resuming face extraction (${FACE_COUNT}/${VIDEO_COUNT}) with ${FACE_WORKERS} workers..."
    python extract_faces.py \
        --video_dir "${MELD_PROC}/subvideo" \
        --output_dir "${MELD_PROC}/openface_face" \
        --workers "${FACE_WORKERS}"
fi
