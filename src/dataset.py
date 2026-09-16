import json
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from decord import VideoReader, cpu


class UCFCrimeTrainingDataset(Dataset):
    def __init__(
            self,
            split_json_path: Path,
            processor,
            num_frames: int = 10,
            window_size_sec: float = 10.0,
            prompt_template: str = None
    ):
        """
        Unified DataLoader exclusively for Training and Validation splits.
        """
        super().__init__()
        self.processor = processor
        self.num_frames = num_frames
        self.window_size_sec = window_size_sec

        if prompt_template is None:
            self.prompt_template = (
                "You are an AI surveillance assistant. Analyze this video for any suspicious or anomalous behavior. "
                "You MUST output your response in this exact format: '[Probability: X.XX] Description'. "
                "X.XX must be a float between 0.00 (completely normal) and 1.00 (definite anomaly). "
                "Follow this with a detailed description of the scene."
            )
        else:
            self.prompt_template = prompt_template

        with open(split_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        project_root = Path(__file__).resolve().parents[1]
        raw_dir = project_root / "data" / "raw"

        print("Indexing video files...")
        self.video_index = {}
        for p in raw_dir.rglob("*.mp4"):
            self.video_index[p.name] = p

        print(f"Dataset Initialized: {len(self.data)} records mapped.")

    def __len__(self):
        return len(self.data)

    def _extract_window(self, vr, start_sec, fps, total_frames):
        """Extracts exactly `num_frames` from a specific `window_size_sec`."""
        start_frame = int(start_sec * fps)
        end_frame = int((start_sec + self.window_size_sec) * fps)

        # Prevent out-of-bounds errors
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        if start_frame == end_frame:
            return None

        indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item["label"]

        video_filename = Path(item["video_path"]).name
        if video_filename not in self.video_index:
            return self.__getitem__((idx + 1) % len(self))

        video_full_path = self.video_index[video_filename]

        try:
            vr = VideoReader(str(video_full_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_sec = len(vr) / fps
            total_frames = len(vr)
        except Exception as e:
            print(f"\nError reading video {video_full_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Dynamic Temporal Cropping
        if label == "Normal":
            max_start = max(0.0, total_sec - self.window_size_sec)
            start_sec = random.uniform(0, max_start)
            # Force the model to learn that Normal = 0.00
            target_text = "[Probability: 0.00] This is a normal surveillance video with no anomalies."
        else:
            # Force the 10-second window to overlap with the UCA ground truth anomaly
            sentences = item.get("uca_data", {}).get("sentences", [])
            if not sentences:
                print(f"Skipping {video_filename}: No UCA text description found.")
                return self.__getitem__((idx + 1) % len(self))
            base_desc = sentences[0]

            # Force the model to learn that Anomaly = 1.00
            target_text = f"[Probability: 1.00] {base_desc}"

            timestamps = item.get("uca_data", {}).get("timestamps", [])
            if timestamps and len(timestamps[0]) == 2:
                ano_start, ano_end = timestamps[0]
                min_start = max(0.0, ano_start - (self.window_size_sec / 2))
                max_start = min(ano_end, total_sec - self.window_size_sec)
                start_sec = random.uniform(min_start, max_start) if max_start > min_start else ano_start
            else:
                start_sec = 0.0

        frames = self._extract_window(vr, start_sec, fps, total_frames)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))

        return {
            "frames": frames,
            "prompt": self.prompt_template,
            "target": target_text,
            "label": label
        }