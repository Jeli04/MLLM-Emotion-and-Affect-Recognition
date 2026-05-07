import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms.functional as _tvf
    sys.modules["torchvision.transforms.functional_tensor"] = _tvf

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge("torch")

import cv2
import mediapipe as mp

from corruption_lib import (
    get_corruption_config,
    corrupt_text,
    apply_audio_corruptions,
    apply_video_corruptions,
)


_MP_FACE_SINGLETON = None


def _get_mp_face():
    global _MP_FACE_SINGLETON
    if _MP_FACE_SINGLETON is None:
        _MP_FACE_SINGLETON = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5,
        )
    return _MP_FACE_SINGLETON


def _extract_faces_from_frames(frames, target_size=224, padding_ratio=0.3):
    mp_face = _get_mp_face()
    output = []
    last_good = None
    prev_bbox = None

    for frame in frames:
        results = mp_face.process(frame)
        chosen = None
        if results.detections:
            cands = []
            h, w = frame.shape[:2]
            for d in results.detections:
                rb = d.location_data.relative_bounding_box
                x1 = max(0, int(rb.xmin * w))
                y1 = max(0, int(rb.ymin * h))
                x2 = min(w, int((rb.xmin + rb.width) * w))
                y2 = min(h, int((rb.ymin + rb.height) * h))
                score = d.score[0] if d.score else 0.0
                cands.append((x1, y1, x2, y2, score))
            if prev_bbox is None:
                chosen = max(cands, key=lambda b: b[4])
            else:
                pcx = (prev_bbox[0] + prev_bbox[2]) * 0.5
                pcy = (prev_bbox[1] + prev_bbox[3]) * 0.5
                chosen = min(cands, key=lambda b: (
                    ((b[0] + b[2]) * 0.5 - pcx) ** 2
                    + ((b[1] + b[3]) * 0.5 - pcy) ** 2
                ))

        face_crop = None
        if chosen is not None:
            x1, y1, x2, y2, _ = chosen
            bw_ = x2 - x1
            bh_ = y2 - y1
            pad_w = int(bw_ * padding_ratio)
            pad_h = int(bh_ * padding_ratio)
            xa = max(0, x1 - pad_w)
            ya = max(0, y1 - pad_h)
            xb = min(frame.shape[1], x2 + pad_w)
            yb = min(frame.shape[0], y2 + pad_h)
            face_crop = frame[ya:yb, xa:xb]
            prev_bbox = chosen
        elif prev_bbox is not None:
            x1, y1, x2, y2, _ = prev_bbox
            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1], x2))
            y2 = max(0, min(frame.shape[0], y2))
            face_crop = frame[y1:y2, x1:x2]

        if face_crop is not None and face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (target_size, target_size))
            last_good = face_crop
        elif last_good is not None:
            face_crop = last_good
        else:
            mn = min(frame.shape[:2])
            cy_, cx_ = frame.shape[0] // 2, frame.shape[1] // 2
            half = mn // 2
            face_crop = frame[cy_ - half:cy_ + half, cx_ - half:cx_ + half]
            face_crop = cv2.resize(face_crop, (target_size, target_size))

        output.append(face_crop)

    return np.stack(output).astype(np.uint8)


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


MELD_EMOS = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
MELD_IDX2EMO = {i: emo for i, emo in enumerate(MELD_EMOS)}
CMUMOSEI_SENT = ['positive', 'negative', 'neutral']


class GenerateScoreCapture:
    def __init__(self):
        self._original = None
        self._target = None
        self._all_logits = None
        self._sequences = None

    def install(self, candidate_models):
        for cand in candidate_models:
            if cand is None:
                continue
            if hasattr(cand, "generate"):
                self._target = cand
                self._original = cand.generate
                cap = self
                def wrapper(*args, **kwargs):
                    # Force greedy + drop top_p/top_k so captured scores are raw
                    # logits comparable across the 7 emotion classes.
                    kwargs["do_sample"] = False
                    kwargs["temperature"] = 1.0
                    kwargs.pop("top_p", None)
                    kwargs.pop("top_k", None)
                    kwargs.pop("typical_p", None)
                    kwargs["max_new_tokens"] = min(int(kwargs.get("max_new_tokens", 48)), 48)
                    kwargs.setdefault("output_logits", True)
                    kwargs.setdefault("output_scores", True)
                    kwargs.setdefault("return_dict_in_generate", True)
                    try:
                        out = cap._original(*args, **kwargs)
                    except TypeError:
                        kwargs.pop("output_logits", None)
                        out = cap._original(*args, **kwargs)
                    raw = getattr(out, "logits", None)
                    if raw is None or not raw:
                        raw = getattr(out, "scores", None)
                    cap._all_logits = (
                        tuple(t.detach().cpu() for t in raw) if raw else None
                    )
                    cap._sequences = (out.sequences.detach().cpu()
                                      if hasattr(out, "sequences") else None)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(out, "sequences"):
                        return out.sequences
                    return out
                cand.generate = wrapper
                return True
        return False

    def reset(self):
        self._all_logits = None
        self._sequences = None

    def all_logits(self):
        return self._all_logits

    def sequences(self):
        return self._sequences


def find_emotion_step_logits(all_logits, sequences, emo_token_ids):
    if all_logits is None or sequences is None:
        return None
    total_len = sequences.shape[-1]
    n_generated = len(all_logits)
    gen_start = total_len - n_generated
    if gen_start < 0:
        return None
    emo_id_set = set(emo_token_ids.values())
    for i in range(n_generated):
        gen_token = int(sequences[0, gen_start + i].item())
        if gen_token in emo_id_set:
            return all_logits[i][0]
    return all_logits[-1][0]


def get_emotion_first_token_ids(tokenizer, emotions):
    out = {}
    for emo in emotions:
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + emo, add_special_tokens=False)
            if ids:
                out[emo] = int(ids[0])
                break
    return out


def find_tokenizer(chat):
    candidates = [
        getattr(chat, "tokenizer", None),
        getattr(getattr(chat, "model", None), "llama_tokenizer", None),
        getattr(getattr(chat, "model", None), "tokenizer", None),
    ]
    for c in candidates:
        if c is not None and hasattr(c, "encode"):
            return c
    return None


def find_generate_targets(chat):
    cands = []
    m = getattr(chat, "model", None)
    if m is None:
        return cands
    for name in ("llama_model", "language_model", "lm", "llm", "model"):
        cands.append(getattr(m, name, None))
    cands.append(m)
    return cands


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


def get_face_or_frame(datasets_cfg, override):
    if override is not None:
        return override
    cands = []
    if 'mercaptionplus' in datasets_cfg:
        cands.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        cands.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(cands)) == 1
    return list(set(cands))[0]


def get_name2cls(name):
    mapping = {
        'MELD': MELD_Dataset, 'CMUMOSEI': CMUMOSEI_Dataset,
        'MER2023': MER2023_Dataset, 'MER2024': MER2024_Dataset,
        'IEMOCAPFour': IEMOCAPFour_Dataset, 'CMUMOSI': CMUMOSI_Dataset,
        'SIMS': SIMS_Dataset, 'SIMSv2': SIMSv2_Dataset,
    }
    if name in mapping:
        return mapping[name]()
    return None


def _to_numpy(t):
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return t


def _put_back(orig, arr):
    if orig is None:
        return None
    if hasattr(orig, "detach"):
        import torch as _torch
        return _torch.from_numpy(arr.copy()).to(orig.device, dtype=orig.dtype)
    return arr.astype(orig.dtype if hasattr(orig, "dtype") else arr.dtype)


def apply_corruption_to_sample(sample_data, subtitle, corrupt_modalities, config,
                                audio_sr=16000):
    new_sub = subtitle
    if "text" in corrupt_modalities:
        new_sub = corrupt_text(
            subtitle or "",
            char_swap_prob=config["text_char_swap_prob"],
            word_drop_prob=config["text_word_drop_prob"],
        )

    if "audio" in corrupt_modalities:
        for key in ("raw_audio",):
            if key in sample_data and sample_data[key] is not None:
                try:
                    arr = _to_numpy(sample_data[key]).astype(np.float32)
                    shape = arr.shape
                    wav = arr.reshape(-1)
                    wav = apply_audio_corruptions(wav, audio_sr, config)
                    arr = wav.reshape(shape).astype(np.float32)
                    sample_data[key] = _put_back(sample_data[key], arr)
                except Exception as e:
                    print(f"  [warn] audio corrupt skipped on key={key}: {e}", flush=True)

    if "video" in corrupt_modalities:
        if "raw_frame" in sample_data and sample_data["raw_frame"] is not None:
            try:
                arr = _to_numpy(sample_data["raw_frame"])
                if arr.ndim == 4 and arr.shape[-1] in (1, 3):
                    was_float = arr.dtype != np.uint8
                    if was_float:
                        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                    corrupted_frames = apply_video_corruptions(arr, config)
                    out = corrupted_frames
                    if was_float:
                        out = out.astype(np.float32) / 255.0
                    sample_data["raw_frame"] = _put_back(sample_data["raw_frame"], out)

                    if "raw_face" in sample_data and sample_data["raw_face"] is not None:
                        try:
                            target = sample_data["raw_face"].shape[-2] \
                                if hasattr(sample_data["raw_face"], "shape") else 224
                            target = 224 if target not in (96, 112, 128, 160, 224, 256) else target
                            face_crops = _extract_faces_from_frames(
                                corrupted_frames, target_size=int(target),
                            )
                            sample_data["raw_face"] = _put_back(
                                sample_data["raw_face"], face_crops,
                            )
                        except Exception as e:
                            print(f"  [warn] face re-extract failed: {e}", flush=True)
            except Exception as e:
                print(f"  [warn] video corrupt skipped: {e}", flush=True)

    return sample_data, new_sub


def run_inference_single(chat, dataset_cls, face_or_frame, sample_data,
                         subtitle, user_message):
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
        'audio': audio_llms, 'frame': frame_llms, 'face': face_llms,
        'image': image_llms, 'multi': multi_llms,
    }
    prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
    return chat.answer_sample(
        prompt=prompt, img_list=img_list,
        num_beams=1, temperature=1, do_sample=True, top_p=0.9,
        max_new_tokens=1200, max_length=2000,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--options", nargs="+")
    p.add_argument("--datasets", nargs="+", default=['MELD'])
    p.add_argument("--outside_face_or_frame", default=None)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--corrupt_modalities", nargs="*", default=[],
                   choices=["text", "audio", "video"])
    p.add_argument("--corruption_preset", default="strong",
                   choices=["mild", "medium", "strong"])
    return p.parse_args()


def cache_dataset(chat, dataset_cls, face_or_frame, dataset_name,
                  test_names, name2subtitle, name2gt, max_samples,
                  out_jsonl, resume, score_capture, emo_token_ids,
                  corrupt_modalities=None, corruption_config=None):
    if dataset_name == 'MELD':
        candidates = ",".join(MELD_EMOS)
        user_message = (
            f"Please select the label that can best describe the person's "
            f"emotional state from the provided candidate labels: {candidates}."
        )
        emo_list = MELD_EMOS
    elif dataset_name == 'CMUMOSEI':
        user_message = (
            "Please select the most likely sentiment label that can best describe "
            "the person's emotional state: positive, negative, neutral."
        )
        emo_list = CMUMOSEI_SENT
    else:
        user_message = ""
        emo_list = []

    samples = test_names if max_samples == 0 else test_names[:max_samples]

    seen = set()
    if resume and Path(out_jsonl).exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["name"])
                except Exception:
                    pass

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    fout = open(out_jsonl, "a")
    try:
        for i, name in enumerate(samples):
            if name in seen:
                continue
            subtitle = name2subtitle.get(name, "")
            if i % 50 == 0:
                print(f"  {dataset_name} {i}/{len(samples)} ({name})", flush=True)

            response, error, emotion_logits = "", None, None
            try:
                sample = {"name": name}
                video_path = audio_path = face_npy = image_path = None
                if hasattr(dataset_cls, "_get_video_path"):
                    video_path = dataset_cls._get_video_path(sample)
                if hasattr(dataset_cls, "_get_audio_path"):
                    audio_path = dataset_cls._get_audio_path(sample)
                if hasattr(dataset_cls, "_get_face_path"):
                    face_npy = dataset_cls._get_face_path(sample)
                sample_data = dataset_cls.read_frame_face_audio_text(
                    video_path, face_npy, audio_path, image_path
                )
                if corrupt_modalities and corruption_config is not None:
                    sample_data, subtitle = apply_corruption_to_sample(
                        sample_data, subtitle, corrupt_modalities, corruption_config,
                    )
                if score_capture is not None:
                    score_capture.reset()
                with torch.no_grad():
                    response = run_inference_single(
                        chat, dataset_cls, face_or_frame,
                        sample_data, subtitle, user_message,
                    )
                if score_capture is not None and emo_token_ids:
                    all_logits = score_capture.all_logits()
                    seqs = score_capture.sequences()
                    row = find_emotion_step_logits(all_logits, seqs, emo_token_ids)
                    if row is not None:
                        row = row.float().cpu()
                        emotion_logits = {
                            emo: float(row[emo_token_ids[emo]].item())
                            for emo in emo_list if emo in emo_token_ids
                        }
            except Exception as e:
                error = str(e)

            record = {"name": name, "subtitle": subtitle, "response": response}
            if emotion_logits is not None:
                record["emotion_logits"] = emotion_logits
            if error is not None:
                record["error"] = error

            if dataset_name == "MELD":
                gt_idx = name2gt[name]
                record["ground_truth_idx"] = int(gt_idx)
                record["ground_truth_label"] = MELD_IDX2EMO[gt_idx]
            elif dataset_name == "CMUMOSEI":
                gt_val = name2gt[name]
                record["ground_truth_value"] = float(gt_val)
                record["ground_truth_label"] = (
                    "positive" if gt_val > 0 else ("negative" if gt_val < 0 else "neutral")
                )

            fout.write(json.dumps(record) + "\n")
            fout.flush()
    finally:
        fout.close()


def main():
    args = parse_args()
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = f"cuda:{inference_cfg.gpu}"

    os.makedirs(args.save_dir, exist_ok=True)

    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
    else:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        roots = glob.glob(os.path.join('output', cfg_name, cfg_name + '*'))
        ckpt3_root = search_for_ckpt_root(roots)
    whole_ckpts = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)

    ckpt_3 = whole_ckpts[-1]
    model_cfg.ckpt_3 = ckpt_3
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    chat = Chat(model, model_cfg, device=device)

    score_capture = GenerateScoreCapture()
    score_capture.install(find_generate_targets(chat))

    tokenizer = find_tokenizer(chat)
    if tokenizer is None:
        emo_token_ids = {}
    else:
        emo_token_ids = get_emotion_first_token_ids(tokenizer, MELD_EMOS + CMUMOSEI_SENT)

    for dataset_name in args.datasets:
        dataset_name = dataset_name.upper()
        dataset_cls = get_name2cls(dataset_name)
        if dataset_cls is None:
            continue

        dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
        dataset_cls.vis_processor = BaseProcessor()
        dataset_cls.img_processor = BaseProcessor()
        if inference_cfg.get("vis_processor") is not None:
            dataset_cls.vis_processor = registry.get_processor_class(
                inference_cfg.vis_processor.train.name
            ).from_config(inference_cfg.vis_processor.train)
        if inference_cfg.get("img_processor") is not None:
            dataset_cls.img_processor = registry.get_processor_class(
                inference_cfg.img_processor.train.name
            ).from_config(inference_cfg.img_processor.train)
        dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms

        test_names = dataset_cls.read_test_names()
        name2subtitle = dataset_cls.name2subtitle
        name2gt = dataset_cls.get_test_name2gt()

        out_jsonl = os.path.join(args.save_dir, f"{dataset_name.lower()}.jsonl")
        corruption_config = None
        if args.corrupt_modalities:
            corruption_config = get_corruption_config(args.corruption_preset)
        cache_dataset(
            chat, dataset_cls, face_or_frame, dataset_name,
            test_names, name2subtitle, name2gt, args.max_samples,
            out_jsonl, args.resume, score_capture, emo_token_ids,
            corrupt_modalities=args.corrupt_modalities,
            corruption_config=corruption_config,
        )


if __name__ == "__main__":
    main()
