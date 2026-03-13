"""
Missing Modality Evaluation for AffectGPT
==========================================
Evaluates AffectGPT under missing modality regimes by randomly masking
audio, video/face, or text inputs (setting them to zero tensors / empty strings).

Supports MELD (7-class discrete emotion) and CMU-MOSEI (valence regression / 3-class sentiment).

Follows the same checkpoint discovery and model loading pattern as inference_hybird.py.

Usage:
    python missing_modality_eval.py \
        --cfg-path train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml \
        --datasets meld \
        --masking_conditions none mask_audio mask_video mask_text mask_audio_video mask_audio_text mask_video_text \
        --save_dir output/missing_modality_results \
        --options "inference.test_epoch=60"
"""

import os
import copy
import glob
import json
import random
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import *

import config
from toolkit.utils.read_files import *

# ============================================================
# Checkpoint discovery (mirrors inference_hybird.py)
# ============================================================

def search_for_ckpt_root(root_candidates):
    """Find the checkpoint directory with the most checkpoint files."""
    if len(root_candidates) == 0:
        return ''
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([p for p in os.listdir(root) if p.startswith('checkpoint_')])
        print(root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print('================================================')
    print(f'Targetroot: epoch range: 0-{maxcount-1}')
    if maxcount > 0:
        last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
        file_stat = Path(last_file).stat()
        print("Last ckpt creation time:", datetime.fromtimestamp(file_stat.st_ctime))
    print('================================================')
    return targetroot


def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    """Get checkpoint .pth paths based on inference config (mirrors inference_hybird.py)."""
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" % (ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1, f'Error: epoch {cur_epoch} not found or ambiguous in {ckpt3_root}'
        return [ckpts[0]]
    elif inference_cfg.test_epochs == 'xxx-xxx':
        last_ckpt = sorted(glob.glob("%s/*.pth" % ckpt3_root))[-1]
        return [last_ckpt]
    else:
        start_epoch, end_epoch = inference_cfg.test_epochs.split('-')
        skip_epoch = int(inference_cfg.skip_epoch)
        whole_ckpts = []
        for cur_epoch in range(int(start_epoch), int(end_epoch) + 1):
            if cur_epoch % skip_epoch == 0:
                ckpts = glob.glob("%s/*%06d*.pth" % (ckpt3_root, int(cur_epoch)))
                assert len(ckpts) == 1, f'Error: epoch {cur_epoch} not found or ambiguous'
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    """Determine face_or_frame from config or override (mirrors inference_hybird.py)."""
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1, 'must have unified face_or_frame type'
    return list(set(face_or_frame_candidates))[0]


def get_name2cls(dataset):
    """Get dataset class by name (mirrors inference_hybird.py)."""
    mapping = {
        'MELD': MELD_Dataset,
        'CMUMOSEI': CMUMOSEI_Dataset,
        'MER2023': MER2023_Dataset,
        'MER2024': MER2024_Dataset,
        'IEMOCAPFour': IEMOCAPFour_Dataset,
        'CMUMOSI': CMUMOSI_Dataset,
        'SIMS': SIMS_Dataset,
        'SIMSv2': SIMSv2_Dataset,
    }
    if dataset in mapping:
        return mapping[dataset]()
    print(f'dataset cls not provided for {dataset}!')
    return None


# ============================================================
# Dataset label helpers
# ============================================================

MELD_EMOS = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
MELD_EMO2IDX = {emo: i for i, emo in enumerate(MELD_EMOS)}
MELD_IDX2EMO = {i: emo for i, emo in enumerate(MELD_EMOS)}


# ============================================================
# Masking logic
# ============================================================

MASKING_CONDITIONS = {
    'none':             {'audio': False, 'video': False, 'text': False},
    'mask_audio':       {'audio': True,  'video': False, 'text': False},
    'mask_video':       {'audio': False, 'video': True,  'text': False},
    'mask_text':        {'audio': False, 'video': False, 'text': True},
    'mask_audio_video': {'audio': True,  'video': True,  'text': False},
    'mask_audio_text':  {'audio': True,  'video': False, 'text': True},
    'mask_video_text':  {'audio': False, 'video': True,  'text': True},
}


def zero_out_tensor(tensor):
    """Replace tensor values with zeros, preserving shape and device."""
    if tensor is None:
        return None
    return torch.zeros_like(tensor)


def apply_masking(sample_data, subtitle, mask_config):
    """
    Apply modality masking to sample_data and subtitle.

    Audio masking:  zero out audio and raw_audio tensors.
    Video masking:  zero out frame, raw_frame, face, raw_face tensors.
    Text masking:   replace subtitle with empty string.

    Returns: (masked_sample_data, masked_subtitle)
    """
    masked = {}
    for key, val in sample_data.items():
        masked[key] = val

    if mask_config['audio']:
        masked['audio'] = zero_out_tensor(sample_data['audio'])
        masked['raw_audio'] = zero_out_tensor(sample_data['raw_audio'])

    if mask_config['video']:
        masked['frame'] = zero_out_tensor(sample_data['frame'])
        masked['raw_frame'] = zero_out_tensor(sample_data['raw_frame'])
        masked['face'] = zero_out_tensor(sample_data['face'])
        masked['raw_face'] = zero_out_tensor(sample_data['raw_face'])

    masked_subtitle = "" if mask_config['text'] else subtitle

    return masked, masked_subtitle


# ============================================================
# Inference helpers (mirrors inference_hybird.py per-sample loop)
# ============================================================

def run_inference_single(chat, dataset_cls, face_or_frame, sample_data, subtitle, user_message):
    """Run inference for a single sample, returning the model's text response."""
    audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
    frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
    face_hiddens, face_llms = chat.postprocess_face(sample_data)
    _, image_llms = chat.postprocess_image(sample_data)
    multi_llms = None

    if face_or_frame.startswith('multiface'):
        if face_hiddens is not None and audio_hiddens is not None:
            _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
    elif face_or_frame.startswith('multiframe'):
        if frame_hiddens is not None and audio_hiddens is not None:
            _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

    img_list = {
        'audio': audio_llms,
        'frame': frame_llms,
        'face': face_llms,
        'image': image_llms,
        'multi': multi_llms,
    }

    prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)

    response = chat.answer_sample(
        prompt=prompt,
        img_list=img_list,
        num_beams=1,
        temperature=1,
        do_sample=True,
        top_p=0.9,
        max_new_tokens=1200,
        max_length=2000,
    )
    return response


# ============================================================
# Metric computation
# ============================================================

def parse_discrete_prediction(response, candidate_labels):
    """Extract predicted emotion from model response for discrete datasets."""
    response_lower = response.lower().strip()
    for label in candidate_labels:
        if label.lower() in response_lower:
            return label
    return None


def parse_sentiment_prediction(response):
    """Extract predicted sentiment from model response."""
    response_lower = response.lower().strip()
    for sent in ['positive', 'negative', 'neutral']:
        if sent in response_lower:
            return sent
    return None


def compute_weighted_f1(predictions, ground_truths, labels):
    """Compute weighted F1 score."""
    from sklearn.metrics import f1_score as sklearn_f1
    label2idx = {l: i for i, l in enumerate(labels)}
    invalid_idx = len(labels)
    y_true = [label2idx.get(gt, invalid_idx) for gt in ground_truths]
    y_pred = [label2idx.get(p, invalid_idx) for p in predictions]
    try:
        return sklearn_f1(y_true, y_pred, average='weighted', zero_division=0)
    except Exception:
        return 0.0


def compute_meld_metrics(predictions, ground_truths):
    """Compute accuracy and per-class accuracy for MELD."""
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for pred, gt in zip(predictions, ground_truths):
        per_class_total[gt] += 1
        total += 1
        if pred == gt:
            correct += 1
            per_class_correct[gt] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {}
    for emo in MELD_EMOS:
        if per_class_total[emo] > 0:
            per_class_acc[emo] = per_class_correct[emo] / per_class_total[emo]
        else:
            per_class_acc[emo] = 0.0

    weighted_f1 = compute_weighted_f1(predictions, ground_truths, MELD_EMOS)

    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'per_class_accuracy': per_class_acc,
        'total_samples': total,
        'correct': correct,
        'unparseable': sum(1 for p in predictions if p is None),
    }


def compute_cmumosei_metrics(pred_sentiments, gt_sentiments):
    """Compute sentiment accuracy/F1 for CMU-MOSEI."""
    from sklearn.metrics import f1_score as sklearn_f1

    non_zero_indices = [i for i, gt in enumerate(gt_sentiments) if gt != 'neutral']

    sent_correct = sum(1 for i in non_zero_indices if pred_sentiments[i] == gt_sentiments[i])
    sent_total = len(non_zero_indices)
    sent_accuracy = sent_correct / sent_total if sent_total > 0 else 0.0

    binary_correct = sum(
        1 for i in non_zero_indices
        if (pred_sentiments[i] == 'positive') == (gt_sentiments[i] == 'positive')
    )
    binary_acc = binary_correct / sent_total if sent_total > 0 else 0.0

    labels = ['positive', 'negative', 'neutral']
    label2idx = {l: i for i, l in enumerate(labels)}
    y_true = [label2idx.get(gt_sentiments[i], 3) for i in non_zero_indices]
    y_pred = [label2idx.get(pred_sentiments[i], 3) for i in non_zero_indices]
    try:
        weighted_f1 = sklearn_f1(y_true, y_pred, average='weighted', zero_division=0)
    except Exception:
        weighted_f1 = 0.0

    return {
        'binary_accuracy': binary_acc,
        'sentiment_accuracy': sent_accuracy,
        'weighted_f1': weighted_f1,
        'total_samples': len(gt_sentiments),
        'non_zero_samples': sent_total,
        'unparseable': sum(1 for p in pred_sentiments if p is None),
    }


# ============================================================
# Main evaluation loop
# ============================================================

def evaluate_dataset(chat, dataset_cls, face_or_frame, dataset_name, test_names,
                     name2subtitle, masking_conditions, save_dir, max_samples,
                     name2gt):
    """Run evaluation for one dataset across all masking conditions."""

    results = {}

    # Build user_message matching the dataset's QA format
    if dataset_name == 'MELD':
        candidate_labels = ",".join(MELD_EMOS)
        user_message = (
            f"Please select the label that can best describe the person's "
            f"emotional state from the provided candidate labels: {candidate_labels}."
        )
    elif dataset_name == 'CMUMOSEI':
        user_message = (
            "Please select the most likely sentiment label that can best describe "
            "the person's emotional state: positive, negative, neutral."
        )

    samples_to_run = test_names
    if max_samples > 0:
        samples_to_run = test_names[:max_samples]

    print(f'\nRunning on {len(samples_to_run)} test samples for {dataset_name}')

    for cond_name in masking_conditions:
        mask_config = MASKING_CONDITIONS[cond_name]
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name} | Condition: {cond_name}')
        print(f'Mask audio={mask_config["audio"]}, video={mask_config["video"]}, text={mask_config["text"]}')
        print(f'{"="*60}')

        name2reason = {}
        predictions = []
        gt_labels = []

        for ii, name in enumerate(samples_to_run):
            subtitle = name2subtitle.get(name, "")
            if ii % 50 == 0:
                print(f'  Processing {ii}/{len(samples_to_run)}: {name}')

            try:
                # Build paths (same as inference_hybird.py)
                sample = {'name': name}
                video_path, audio_path, face_npy, image_path = None, None, None, None
                if hasattr(dataset_cls, '_get_video_path'):
                    video_path = dataset_cls._get_video_path(sample)
                if hasattr(dataset_cls, '_get_audio_path'):
                    audio_path = dataset_cls._get_audio_path(sample)
                if hasattr(dataset_cls, '_get_face_path'):
                    face_npy = dataset_cls._get_face_path(sample)

                # Load multimodal data
                sample_data = dataset_cls.read_frame_face_audio_text(
                    video_path, face_npy, audio_path, image_path
                )

                # Apply masking
                masked_data, masked_subtitle = apply_masking(sample_data, subtitle, mask_config)

                # Run inference
                with torch.no_grad():
                    response = run_inference_single(
                        chat, dataset_cls, face_or_frame,
                        masked_data, masked_subtitle, user_message
                    )
            except Exception as e:
                print(f'  Error on sample {name}: {e}')
                response = ""

            name2reason[name] = response

            # Parse prediction and collect ground truth
            if dataset_name == 'MELD':
                gt_emo_idx = name2gt[name]
                gt_emo = MELD_IDX2EMO[gt_emo_idx]
                pred = parse_discrete_prediction(response, MELD_EMOS)
                predictions.append(pred)
                gt_labels.append(gt_emo)
            elif dataset_name == 'CMUMOSEI':
                gt_val = name2gt[name]
                gt_sent = 'positive' if gt_val > 0 else ('negative' if gt_val < 0 else 'neutral')
                pred_sent = parse_sentiment_prediction(response)
                predictions.append(pred_sent)
                gt_labels.append(gt_sent)

        # Compute metrics
        if dataset_name == 'MELD':
            metrics = compute_meld_metrics(predictions, gt_labels)
        elif dataset_name == 'CMUMOSEI':
            metrics = compute_cmumosei_metrics(predictions, gt_labels)

        results[cond_name] = metrics
        print(f'\n  Results for {cond_name}:')
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f'    {k}:')
                for kk, vv in v.items():
                    print(f'      {kk}: {vv:.4f}')
            elif isinstance(v, float):
                print(f'    {k}: {v:.4f}')
            else:
                print(f'    {k}: {v}')

        # Save per-condition raw responses
        cond_save_dir = os.path.join(save_dir, dataset_name.lower())
        os.makedirs(cond_save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(cond_save_dir, f'responses_{cond_name}.npz'),
            name2reason=name2reason
        )

    # Save aggregated metrics
    results_path = os.path.join(save_dir, dataset_name.lower(), 'metrics_summary.json')
    serializable = {}
    for cond, metrics in results.items():
        serializable[cond] = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serializable[cond][k] = {
                    kk: float(vv) if isinstance(vv, (float, np.floating, np.integer)) else vv
                    for kk, vv in v.items()
                }
            elif isinstance(v, (float, np.floating, np.integer)):
                serializable[cond][k] = float(v)
            else:
                serializable[cond][k] = v
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    return results


def print_summary_table(all_results):
    """Print a formatted summary table."""
    print('\n' + '=' * 80)
    print('MISSING MODALITY EVALUATION SUMMARY')
    print('=' * 80)

    for dataset_name, results in all_results.items():
        print(f'\n--- {dataset_name} ---')
        if dataset_name == 'MELD':
            header = f'{"Condition":<22} {"Accuracy":>10} {"W-F1":>10} {"Unparsed":>10} {"Total":>8}'
            print(header)
            print('-' * len(header))
            for cond, m in results.items():
                print(f'{cond:<22} {m["accuracy"]:>10.4f} {m["weighted_f1"]:>10.4f} '
                      f'{m["unparseable"]:>10} {m["total_samples"]:>8}')
        elif dataset_name == 'CMUMOSEI':
            header = f'{"Condition":<22} {"BinAcc":>10} {"SentAcc":>10} {"W-F1":>10} {"Unparsed":>10}'
            print(header)
            print('-' * len(header))
            for cond, m in results.items():
                print(f'{cond:<22} {m["binary_accuracy"]:>10.4f} '
                      f'{m["sentiment_accuracy"]:>10.4f} {m["weighted_f1"]:>10.4f} '
                      f'{m["unparseable"]:>10}')

    print('\n' + '=' * 80)


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AffectGPT Missing Modality Evaluation")
    parser.add_argument("--cfg-path", required=True, help="Path to training config YAML.")
    parser.add_argument("--options", nargs="+",
                        help="Override config, format: --options xx=xx yy=yy")
    parser.add_argument("--datasets", nargs="+", default=['MELD'],
                        help="Datasets to evaluate (e.g. MELD CMUMOSEI).")
    parser.add_argument("--masking_conditions", nargs="+",
                        default=list(MASKING_CONDITIONS.keys()),
                        choices=list(MASKING_CONDITIONS.keys()),
                        help="Which masking conditions to run.")
    parser.add_argument("--outside_face_or_frame", default=None,
                        help="Override face_or_frame (e.g. multiface_audio_face_text)")
    parser.add_argument("--save_dir", type=str,
                        default="output/missing_modality_results",
                        help="Directory to save results.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit test samples per dataset (0=all).")
    args = parser.parse_args()

    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Step 1: Discover checkpoint (same as inference_hybird.py) ----
    print('======== Step1: cfg pre-analysis ========')
    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
    else:
        print('Searching for suitable ckpt_root...')
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name + '*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)

    print(f'ckpt3 root: {ckpt3_root}')
    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    for item in whole_ckpt3s:
        print(f'  {os.path.basename(item)}')

    # Determine face_or_frame
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print(f'face_or_frame: {face_or_frame}')
    print('=======================================')

    # ---- Use the best (last) checkpoint ----
    ckpt_3 = whole_ckpt3s[-1]

    # ---- Step 2: Load model (same as inference_hybird.py) ----
    print(f'======== Step2: Loading model with ckpt_3: {os.path.basename(ckpt_3)} ========')
    model_cfg.ckpt_3 = ckpt_3
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    chat = Chat(model, model_cfg, device=device)

    # ---- Step 3: Run missing modality evaluation ----
    print('======== Step3: Missing Modality Evaluation ========')
    all_results = {}

    for dataset_name in args.datasets:
        dataset_name = dataset_name.upper()
        if dataset_name == 'MELD':
            dataset_name = 'MELD'
        elif dataset_name == 'CMUMOSEI':
            dataset_name = 'CMUMOSEI'

        print(f'\nPreparing dataset: {dataset_name}')
        dataset_cls = get_name2cls(dataset_name)
        if dataset_cls is None:
            continue

        # Configure dataset_cls for inference (same as inference_hybird.py)
        dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
        dataset_cls.vis_processor = BaseProcessor()
        dataset_cls.img_processor = BaseProcessor()
        vis_processor_cfg = inference_cfg.get("vis_processor")
        img_processor_cfg = inference_cfg.get("img_processor")
        if vis_processor_cfg is not None:
            dataset_cls.vis_processor = registry.get_processor_class(
                vis_processor_cfg.train.name
            ).from_config(vis_processor_cfg.train)
        if img_processor_cfg is not None:
            dataset_cls.img_processor = registry.get_processor_class(
                img_processor_cfg.train.name
            ).from_config(img_processor_cfg.train)
        dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms

        # Read test data (uses dataset class methods)
        test_names = dataset_cls.read_test_names()
        name2subtitle = dataset_cls.name2subtitle
        name2gt = dataset_cls.get_test_name2gt()

        results = evaluate_dataset(
            chat, dataset_cls, face_or_frame, dataset_name,
            test_names, name2subtitle, args.masking_conditions,
            args.save_dir, args.max_samples, name2gt
        )
        all_results[dataset_name] = results

    # Print final summary
    print_summary_table(all_results)

    # Save final summary
    summary_path = os.path.join(args.save_dir, 'full_summary.json')
    summary = {}
    for ds, results in all_results.items():
        summary[ds] = {}
        for cond, metrics in results.items():
            summary[ds][cond] = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    summary[ds][cond][k] = {
                        kk: float(vv) if isinstance(vv, (float, np.floating, np.integer)) else vv
                        for kk, vv in v.items()
                    }
                elif isinstance(v, (float, np.floating, np.integer)):
                    summary[ds][cond][k] = float(v)
                else:
                    summary[ds][cond][k] = v
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFull summary saved to {summary_path}')


if __name__ == '__main__':
    main()
