import os
from pathlib import Path

import librosa
import pandas as pd
from torch.utils.data import Dataset


EMOTION2ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

SPLIT_DIRS = {
    "train": "train_splits",
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}


class RawMELDDataset(Dataset):
    def __init__(
        self,
        meld_root,
        split="train",
        label_type="emotion",
        load_audio=True,
        audio_sr=16000,
    ):
        """
        Args:
            meld_root: path to extracted MELD folder
            split: 'train', 'dev', or 'test'
            label_type: currently supports 'emotion'
            load_audio: whether to load raw audio waveform from mp4
            audio_sr: target audio sample rate
        """
        self.meld_root = Path(meld_root)
        self.split = split
        self.label_type = label_type.lower()
        self.load_audio = load_audio
        self.audio_sr = audio_sr

        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be one of: train, dev, test")

        if self.label_type != "emotion":
            raise ValueError("Only 'emotion' is supported in this version")

        csv_name = {
            "train": "train_sent_emo.csv",
            "dev": "dev_sent_emo.csv",
            "test": "test_sent_emo.csv",
        }[self.split]

        self.csv_path = self.meld_root / csv_name
        self.video_dir = self.meld_root / SPLIT_DIRS[self.split]

        self.df = pd.read_csv(self.csv_path)

        # Basic cleanup
        self.df["Utterance"] = self.df["Utterance"].astype(str).str.strip()
        self.df["Emotion"] = self.df["Emotion"].astype(str).str.strip().str.lower()
        self.df["Speaker"] = self.df["Speaker"].astype(str).str.strip()

        # Preserve dialogue order
        self.df = self.df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _get_video_path(self, dialogue_id, utterance_id):
        filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        return self.video_dir / filename

    def _load_audio(self, video_path):
        """
        Load mono audio waveform from an mp4 file.
        Returns:
            waveform: np.ndarray of shape [num_samples]
            sample_rate: int
        """
        waveform, sr = librosa.load(str(video_path), sr=self.audio_sr, mono=True)
        return waveform, sr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        dialogue_id = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])
        utterance = row["Utterance"]
        speaker = row["Speaker"]
        emotion = row["Emotion"]
        label = EMOTION2ID[emotion]

        video_path = self._get_video_path(dialogue_id, utterance_id)

        sample = {
            "text": utterance,
            "label": label,
            "emotion": emotion,
            "speaker": speaker,
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "video_path": str(video_path),
        }

        # if self.load_audio:
        #     try:
        #         audio, sr = self._load_audio(video_path)
        #     except Exception as e:
        #         audio, sr = None, None
        #         sample["audio_error"] = str(e)

        #     sample["audio"] = audio
        #     sample["audio_sr"] = sr

        if self.load_audio:
            print("Loading audio from:", video_path)
            try:
                audio, sr = self._load_audio(video_path)
                print("Loaded audio:", type(audio), None if audio is None else audio.shape, sr)
            except Exception as e:
                print("Audio exception:", e)
                audio, sr = None, None
                sample["audio_error"] = str(e)

            sample["audio"] = audio
            sample["audio_sr"] = sr

        return sample
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = RawMELDDataset("/project2/robinjia_875/lijc/data/MELD.Raw", split="test", load_audio=True)

    sample = dataset[0]
    print(sample["text"])
    print(sample["audio"].shape if sample["audio"] is not None else None)
    print(sample["audio_sr"])
    print(sample["video_path"])
    print(sample.get("audio_error"))

    print(len(dataset))
    print(type(sample["audio"]))