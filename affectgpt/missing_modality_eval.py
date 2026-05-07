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


def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([p for p in os.listdir(root) if p.startswith('checkpoint_')])
        if count > maxcount:
            maxcount = count
            targetroot = root
    if maxcount > 0:
        last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
        file_stat = Path(last_file).stat()
        print("Last ckpt creation time:", datetime.fromtimestamp(file_stat.st_ctime))
    return targetroot


def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" % (ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1
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
                assert len(ckpts) == 1
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1
    return list(set(face_or_frame_candidates))[0]


def get_name2cls(dataset):
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
    return None


MELD_EMOS = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
MELD_EMO2IDX = {emo: i for i, emo in enumerate(MELD_EMOS)}
MELD_IDX2EMO = {i: emo for i, emo in enumerate(MELD_EMOS)}


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
    if tensor is None:
        return None
    return torch.zeros_like(tensor)


def apply_masking(sample_data, subtitle, mask_config):
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


def run_inference_single(chat, dataset_cls, face_or_frame, sample_data, subtitle, user_message):
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


def parse_discrete_prediction(response, candidate_labels):
    response_lower = response.lower().strip()
    for label in candidate_labels:
        if label.lower() in response_lower:
            return label
    return None


def parse_sentiment_prediction(response):
    response_lower = response.lower().strip()
    for sent in ['positive', 'negative', 'neutral']:
        if sent in response_lower:
            return sent
    return None


def compute_weighted_f1(predictions, ground_truths, labels):
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


def evaluate_dataset(chat, dataset_cls, face_or_frame, dataset_name, test_names,
                     name2subtitle, masking_conditions, save_dir, max_samples,
                     name2gt):
    results = {}

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

    for cond_name in masking_conditions:
        mask_config = MASKING_CONDITIONS[cond_name]
        print(f'\n[{dataset_name}] {cond_name}: '
              f'audio={mask_config["audio"]}, video={mask_config["video"]}, text={mask_config["text"]}')

        name2reason = {}
        predictions = []
        gt_labels = []

        for ii, name in enumerate(samples_to_run):
            subtitle = name2subtitle.get(name, "")
            if ii % 50 == 0:
                print(f'  {ii}/{len(samples_to_run)}: {name}')

            try:
                sample = {'name': name}
                video_path, audio_path, face_npy, image_path = None, None, None, None
                if hasattr(dataset_cls, '_get_video_path'):
                    video_path = dataset_cls._get_video_path(sample)
                if hasattr(dataset_cls, '_get_audio_path'):
                    audio_path = dataset_cls._get_audio_path(sample)
                if hasattr(dataset_cls, '_get_face_path'):
                    face_npy = dataset_cls._get_face_path(sample)

                sample_data = dataset_cls.read_frame_face_audio_text(
                    video_path, face_npy, audio_path, image_path
                )

                masked_data, masked_subtitle = apply_masking(sample_data, subtitle, mask_config)

                with torch.no_grad():
                    response = run_inference_single(
                        chat, dataset_cls, face_or_frame,
                        masked_data, masked_subtitle, user_message
                    )
            except Exception as e:
                print(f'  Error on sample {name}: {e}')
                response = ""

            name2reason[name] = response

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

        if dataset_name == 'MELD':
            metrics = compute_meld_metrics(predictions, gt_labels)
        elif dataset_name == 'CMUMOSEI':
            metrics = compute_cmumosei_metrics(predictions, gt_labels)

        results[cond_name] = metrics
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f'    {k}:')
                for kk, vv in v.items():
                    print(f'      {kk}: {vv:.4f}')
            elif isinstance(v, float):
                print(f'    {k}: {v:.4f}')
            else:
                print(f'    {k}: {v}')

        cond_save_dir = os.path.join(save_dir, dataset_name.lower())
        os.makedirs(cond_save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(cond_save_dir, f'responses_{cond_name}.npz'),
            name2reason=name2reason
        )

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
    print('\n' + '=' * 80)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--options", nargs="+")
    parser.add_argument("--datasets", nargs="+", default=['MELD'])
    parser.add_argument("--masking_conditions", nargs="+",
                        default=list(MASKING_CONDITIONS.keys()),
                        choices=list(MASKING_CONDITIONS.keys()))
    parser.add_argument("--outside_face_or_frame", default=None)
    parser.add_argument("--save_dir", type=str,
                        default="output/missing_modality_results")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)

    os.makedirs(args.save_dir, exist_ok=True)

    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
    else:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name + '*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)

    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)

    ckpt_3 = whole_ckpt3s[-1]
    model_cfg.ckpt_3 = ckpt_3
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    chat = Chat(model, model_cfg, device=device)

    all_results = {}

    for dataset_name in args.datasets:
        dataset_name = dataset_name.upper()
        dataset_cls = get_name2cls(dataset_name)
        if dataset_cls is None:
            continue

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

        test_names = dataset_cls.read_test_names()
        name2subtitle = dataset_cls.name2subtitle
        name2gt = dataset_cls.get_test_name2gt()

        results = evaluate_dataset(
            chat, dataset_cls, face_or_frame, dataset_name,
            test_names, name2subtitle, args.masking_conditions,
            args.save_dir, args.max_samples, name2gt
        )
        all_results[dataset_name] = results

    print_summary_table(all_results)

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


if __name__ == '__main__':
    main()
