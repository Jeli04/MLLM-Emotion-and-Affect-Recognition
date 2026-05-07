import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
from tqdm import tqdm


def _det_to_bbox(det, w, h):
    rb = det.location_data.relative_bounding_box
    x1 = max(0, int(rb.xmin * w))
    y1 = max(0, int(rb.ymin * h))
    x2 = min(w, int((rb.xmin + rb.width) * w))
    y2 = min(h, int((rb.ymin + rb.height) * h))
    score = det.score[0] if det.score else 0.0
    return x1, y1, x2, y2, float(score)


def _bbox_center(b):
    x1, y1, x2, y2, _ = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _pick_tracked_detection(detections, prev_bbox, w, h):
    if not detections:
        return None
    bboxes = [_det_to_bbox(d, w, h) for d in detections]
    if prev_bbox is None:
        return max(bboxes, key=lambda b: b[4])
    pcx, pcy = _bbox_center(prev_bbox)
    return min(bboxes, key=lambda b: ((b[0] + b[2]) * 0.5 - pcx) ** 2
                                     + ((b[1] + b[3]) * 0.5 - pcy) ** 2)


def extract_faces_from_video(video_path, target_size=224, padding_ratio=0.3,
                              mp_face=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    owns_mp_face = False
    if mp_face is None:
        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        owns_mp_face = True

    faces = []
    prev_bbox = None
    last_good_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        chosen = _pick_tracked_detection(
            results.detections if results.detections else [],
            prev_bbox, w, h,
        )

        face_crop = None
        if chosen is not None:
            x1, y1, x2, y2, _ = chosen
            bw = x2 - x1
            bh = y2 - y1
            pad_w = int(bw * padding_ratio)
            pad_h = int(bh * padding_ratio)
            x1p = max(0, x1 - pad_w)
            y1p = max(0, y1 - pad_h)
            x2p = min(w, x2 + pad_w)
            y2p = min(h, y2 + pad_h)
            face_crop = frame[y1p:y2p, x1p:x2p]
            prev_bbox = chosen
        elif prev_bbox is not None:
            x1, y1, x2, y2, _ = prev_bbox
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            face_crop = frame[y1:y2, x1:x2]

        if face_crop is not None and face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (target_size, target_size))
            last_good_face = face_crop
        elif last_good_face is not None:
            face_crop = last_good_face
        else:
            min_dim = min(h, w)
            cy, cx = h // 2, w // 2
            half = min_dim // 2
            face_crop = frame[cy - half:cy + half, cx - half:cx + half]
            face_crop = cv2.resize(face_crop, (target_size, target_size))

        faces.append(face_crop)

    cap.release()
    if owns_mp_face:
        mp_face.close()

    if len(faces) == 0:
        return None

    return np.array(faces)


_WORKER_MP_FACE = None


def _get_worker_mp_face():
    global _WORKER_MP_FACE
    if _WORKER_MP_FACE is None:
        _WORKER_MP_FACE = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5,
        )
    return _WORKER_MP_FACE


def _process_one(vpath, output_dir, target_size):
    name = os.path.splitext(os.path.basename(vpath))[0]
    out_path = os.path.join(output_dir, name + ".npy")
    if os.path.exists(out_path):
        return ("skipped", name)
    try:
        faces = extract_faces_from_video(
            vpath, target_size=target_size, mp_face=_get_worker_mp_face(),
        )
        if faces is not None:
            np.save(out_path, faces)
            return ("done", name)
        return ("failed", f"{name}: no frames extracted")
    except Exception as e:
        return ("failed", f"{name}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos = sorted(glob(os.path.join(args.video_dir, "*.mp4")))
    print(f"Found {len(videos)} videos in {args.video_dir}")

    done = 0
    failed = 0
    skipped = 0

    if args.workers <= 1:
        mp_face = _get_worker_mp_face()
        for vpath in tqdm(videos, desc="Extracting faces"):
            name = os.path.splitext(os.path.basename(vpath))[0]
            out_path = os.path.join(args.output_dir, name + ".npy")
            if os.path.exists(out_path):
                skipped += 1
                continue
            try:
                faces = extract_faces_from_video(
                    vpath, target_size=args.target_size, mp_face=mp_face,
                )
                if faces is not None:
                    np.save(out_path, faces)
                    done += 1
                else:
                    failed += 1
                    print(f"WARNING: no frames extracted from {name}")
            except Exception as e:
                failed += 1
                print(f"ERROR: {name}: {e}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        worker = partial(_process_one,
                         output_dir=args.output_dir,
                         target_size=args.target_size)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, v) for v in videos]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Extracting faces ({args.workers} workers)"):
                status, info = f.result()
                if status == "done":
                    done += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"FAIL: {info}")

    print(f"\nDone: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total .npy files: {len(glob(os.path.join(args.output_dir, '*.npy')))}")


if __name__ == "__main__":
    main()
