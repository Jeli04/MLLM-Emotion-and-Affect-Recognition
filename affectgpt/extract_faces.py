"""
Extract face crops from videos using MediaPipe and save as .npy files.
Produces the same format as OpenFace output expected by AffectGPT:
  - .npy file per video with shape [num_frames, H, W, 3] (BGR uint8)
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
from tqdm import tqdm


def extract_faces_from_video(video_path, target_size=224, padding_ratio=0.3):
    """Extract face crops from all frames of a video.

    Args:
        video_path: Path to .mp4 file
        target_size: Output face crop size (square)
        padding_ratio: Extra padding around detected face box

    Returns:
        numpy array of shape [num_frames, target_size, target_size, 3] or None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # full-range model (better for varied distances)
        min_detection_confidence=0.5,
    )

    faces = []
    last_good_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        face_crop = None
        if results.detections:
            det = results.detections[0]  # use first (most confident) face
            bbox = det.location_data.relative_bounding_box

            # Convert relative coords to absolute
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Add padding
            pad_w = int(bw * padding_ratio)
            pad_h = int(bh * padding_ratio)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x1 + bw + 2 * pad_w)
            y2 = min(h, y1 + bh + 2 * pad_h)

            face_crop = frame[y1:y2, x1:x2]

        if face_crop is not None and face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (target_size, target_size))
            last_good_face = face_crop
        elif last_good_face is not None:
            # Use last detected face if detection fails on this frame
            face_crop = last_good_face
        else:
            # No face detected yet, use center crop of frame
            min_dim = min(h, w)
            cy, cx = h // 2, w // 2
            half = min_dim // 2
            face_crop = frame[cy - half:cy + half, cx - half:cx + half]
            face_crop = cv2.resize(face_crop, (target_size, target_size))

        faces.append(face_crop)

    cap.release()
    mp_face.close()

    if len(faces) == 0:
        return None

    return np.array(faces)  # [num_frames, H, W, 3]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory with .mp4 files")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npy files")
    parser.add_argument("--target_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos = sorted(glob(os.path.join(args.video_dir, "*.mp4")))
    print(f"Found {len(videos)} videos in {args.video_dir}")

    done = 0
    failed = 0
    skipped = 0

    for vpath in tqdm(videos, desc="Extracting faces"):
        name = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(args.output_dir, name + ".npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            faces = extract_faces_from_video(vpath, target_size=args.target_size)
            if faces is not None:
                np.save(out_path, faces)
                done += 1
            else:
                failed += 1
                print(f"WARNING: no frames extracted from {name}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {name}: {e}")

    print(f"\nDone: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total .npy files: {len(glob(os.path.join(args.output_dir, '*.npy')))}")


if __name__ == "__main__":
    main()
