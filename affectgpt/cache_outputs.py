"""
Run AffectGPT inference and cache (a) raw text response and (b) per-candidate
emotion logits at the first answer position to disk, one JSON record per sample.

Per-sample record:
    {
      "name": ...,
      "subtitle": ...,
      "ground_truth_idx": ...,
      "ground_truth_label": ...,
      "response": "<generated text>",
      "emotion_logits": {"anger": -3.2, "joy": -1.1, ...}   # raw model logits
    }

Logits capture works by wrapping `chat.model.<llm>.generate` so the score tensor
from the first generated step is stashed during each call. Sampling-based
text generation is unchanged (we still call chat.answer_sample as before); the
wrapper only adds output_scores=True to grab the deterministic distribution
that precedes the first sampled token.

If the wrapper can't find a target generate method (AffectGPT internals differ
from expectation), we fall back to caching only the text response and the
metric script falls back to text parsing.

Usage:
    python cache_outputs.py \
        --cfg-path train_configs/<the yaml> \
        --datasets MELD \
        --save_dir output/cache/<cond>-<preset>/ \
        --options "inference.test_epoch=60"
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Compat shim: pytorchvideo imports torchvision.transforms.functional_tensor,
# which torchvision removed in 0.17+. Alias to the merged module.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms.functional as _tvf
    sys.modules["torchvision.transforms.functional_tensor"] = _tvf

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge("torch")

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
# Constants
# ============================================================

MELD_EMOS = ['anger', 'joy', 'sadness', 'neutral', 'disgust', 'fear', 'surprise']
MELD_IDX2EMO = {i: emo for i, emo in enumerate(MELD_EMOS)}

# CMUMOSEI sentiment is not handled below, but listed for parity with the
# original eval. Add candidates here if you extend.
CMUMOSEI_SENT = ['positive', 'negative', 'neutral']


# ============================================================
# Score-capture infrastructure
# ============================================================

class GenerateScoreCapture:
    """Monkey-patches a model's `generate` method to capture per-step RAW
    logits (pre-sampling-filter). Captures ALL steps so we can later find
    the position where an emotion token was actually generated.

    Uses transformers' `output_logits=True` (raw, pre-processor logits).
    Falls back to `output_scores=True` if not supported, but those will be
    -inf for tokens filtered out by top_p/top_k.
    """
    def __init__(self):
        self._original = None
        self._target = None
        self._all_logits = None     # tuple of [batch, vocab] tensors per step
        self._sequences = None      # generated token IDs

    def install(self, candidate_models):
        for cand in candidate_models:
            if cand is None:
                continue
            if hasattr(cand, "generate"):
                self._target = cand
                self._original = cand.generate
                cap = self
                def wrapper(*args, **kwargs):
                    # Force greedy decoding & disable top-p/top-k so the captured
                    # `scores` are RAW logits (no -inf filtering). This makes
                    # candidate-emotion logits comparable across the 7 classes.
                    # Trade-off: text response is now deterministic / greedy.
                    kwargs["do_sample"] = False
                    kwargs["temperature"] = 1.0
                    kwargs.pop("top_p", None)
                    kwargs.pop("top_k", None)
                    kwargs.pop("typical_p", None)
                    # Cap max_new_tokens — chat.answer_sample passes 1200 but
                    # the response is ~10 tokens. Generating + storing 1200
                    # logit tensors per sample blows GPU memory. Hard cap to 48.
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
                    # Move to CPU immediately to free GPU memory
                    cap._all_logits = (
                        tuple(t.detach().cpu() for t in raw) if raw else None
                    )
                    cap._sequences = (out.sequences.detach().cpu()
                                      if hasattr(out, "sequences") else None)
                    # Free GPU cache between samples
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Diagnostic: print once on first call so we can verify
                    # which logit source we got
                    if not getattr(cap, "_diag_printed", False):
                        kind = "logits (raw)" if getattr(out, "logits", None) else "scores (filtered, may be -inf)"
                        n_steps = len(cap._all_logits) if cap._all_logits else 0
                        print(f"[score-capture] using {kind}, captured {n_steps} steps", flush=True)
                        cap._diag_printed = True
                    if hasattr(out, "sequences"):
                        return out.sequences
                    return out
                cand.generate = wrapper
                print(f"[score-capture] patched generate on {type(cand).__name__}")
                return True
        print("[score-capture] WARNING: no generate() target found; logits unavailable")
        return False

    def reset(self):
        self._all_logits = None
        self._sequences = None

    def all_logits(self):
        """Return tuple of [batch, vocab] logit tensors per generation step."""
        return self._all_logits

    def sequences(self):
        return self._sequences


def find_emotion_step_logits(all_logits, sequences, emo_token_ids, prompt_len=None):
    """Pick the step whose generated token matches an emotion first-token.

    Returns the [vocab] logit tensor at that step (over the candidate set's
    underlying vocab), or the last step's logits as a fallback.
    """
    if all_logits is None or sequences is None:
        return None
    # sequences shape: [batch=1, total_len]. The first prompt_len tokens are
    # input; the remaining are generated. all_logits[i] is the logits used to
    # predict generated token i (so its argmax → sequences[batch, prompt_len + i]).
    # We don't have prompt_len, but we can match by length: number of generated
    # tokens equals len(all_logits).
    total_len = sequences.shape[-1]
    n_generated = len(all_logits)
    gen_start = total_len - n_generated
    if gen_start < 0:
        return None
    emo_id_set = set(emo_token_ids.values())
    for i in range(n_generated):
        gen_token = int(sequences[0, gen_start + i].item())
        if gen_token in emo_id_set:
            return all_logits[i][0]  # [vocab]
    # No emotion token found in generation; return last step's logits as best-effort
    return all_logits[-1][0]


def get_emotion_first_token_ids(tokenizer, emotions):
    """Return {emotion: first_token_id} using a leading-space variant first.

    BPE tokenizers commonly produce different token boundaries depending on
    whether the candidate is preceded by a space. We try " emotion" first
    (chat-context typical) then fall back to "emotion".
    """
    out = {}
    for emo in emotions:
        for prefix in (" ", ""):
            ids = tokenizer.encode(prefix + emo, add_special_tokens=False)
            if ids:
                out[emo] = int(ids[0])
                break
    return out


def find_tokenizer(chat):
    """Best-effort discovery of the LLM tokenizer used by AffectGPT."""
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
    """Return a list of objects whose .generate we should consider patching."""
    cands = []
    m = getattr(chat, "model", None)
    if m is None:
        return cands
    for name in ("llama_model", "language_model", "lm", "llm", "model"):
        cands.append(getattr(m, name, None))
    cands.append(m)
    return cands


# ============================================================
# Borrowed verbatim (config / model / dataset boilerplate)
# ============================================================

def search_for_ckpt_root(root_candidates):
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
    if maxcount > 0:
        last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
        file_stat = Path(last_file).stat()
        print("Last ckpt creation time:", datetime.fromtimestamp(file_stat.st_ctime))
    return targetroot


def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" % (ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1, f'epoch {cur_epoch} not found / ambiguous'
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


# ============================================================
# Main loop
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="AffectGPT inference + response/logits cache")
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--options", nargs="+",
                   help="Override config (e.g. inference.test_epoch=60)")
    p.add_argument("--datasets", nargs="+", default=['MELD'])
    p.add_argument("--outside_face_or_frame", default=None)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def cache_dataset(chat, dataset_cls, face_or_frame, dataset_name,
                  test_names, name2subtitle, name2gt, max_samples,
                  out_jsonl, resume, score_capture, emo_token_ids):
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
        print(f"  resume: {len(seen)} already cached.")

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

    print("======== Step1: cfg pre-analysis ========")
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
    print(f"======== Step2: Loading model with ckpt_3: {os.path.basename(ckpt_3)} ========")
    model_cfg.ckpt_3 = ckpt_3
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device).eval()
    chat = Chat(model, model_cfg, device=device)

    # ---- Install score capture
    score_capture = GenerateScoreCapture()
    score_capture.install(find_generate_targets(chat))

    tokenizer = find_tokenizer(chat)
    if tokenizer is None:
        print("WARNING: tokenizer not found; emotion_logits will not be cached.")
        emo_token_ids = {}
    else:
        emo_token_ids = get_emotion_first_token_ids(tokenizer, MELD_EMOS + CMUMOSEI_SENT)
        print(f"emo first-token IDs: {emo_token_ids}")

    print("======== Step3: Cache outputs ========")
    for dataset_name in args.datasets:
        dataset_name = dataset_name.upper()
        print(f"\nDataset: {dataset_name}")
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
        cache_dataset(
            chat, dataset_cls, face_or_frame, dataset_name,
            test_names, name2subtitle, name2gt, args.max_samples,
            out_jsonl, args.resume, score_capture, emo_token_ids,
        )
        print(f"  cached -> {out_jsonl}")

    print("\nDone.")


if __name__ == "__main__":
    main()
