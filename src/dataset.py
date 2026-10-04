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
        Consumes LLM-labeled UCA dense captions to enable robust Intra-Video Negative Sampling.
        """
        super().__init__()
        self.processor = processor
        self.num_frames = num_frames
        self.window_size_sec = window_size_sec

        if prompt_template is None:
            self.prompt_template = (
                "Is there any suspicious or anomalous behavior (such as fighting, violence, abuse, stealing, or accidents) happening in this video clip? "
                "Answer with exactly 'Yes.' or 'No.' as your very first word. "
                "Then, on a new line, provide a detailed, objective description of exactly what is happening in the scene."
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

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))

        if start_frame >= end_frame:
            return None

        indices = np.linspace(start_frame, end_frame, self.num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item["label"]

        sentences = item.get("uca_data", {}).get("sentences", [])
        timestamps = item.get("uca_data", {}).get("timestamps", [])

        if not sentences or not timestamps or len(sentences) != len(timestamps):
            return self.__getitem__((idx + 1) % len(self))

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

        # --------------------------------------------------------
        # 1. PURE NORMAL VIDEOS
        # --------------------------------------------------------
        if label == "Normal":
            idx_choice = random.randint(0, len(sentences) - 1)
            ano_start, ano_end = timestamps[idx_choice]

            # Safely unpack the dictionary structure created by the LLM labeler
            sentence_obj = sentences[idx_choice]
            specific_action = sentence_obj["text"] if isinstance(sentence_obj, dict) else sentence_obj

            target_text = f"No.\n{specific_action}. The scene appears completely peaceful and there is no suspicious or anomalous behavior occurring."

            min_start = max(0.0, ano_start - (self.window_size_sec / 2))
            max_start = min(ano_end, total_sec - self.window_size_sec)
            start_sec = random.uniform(min_start, max_start) if max_start > min_start else ano_start

        # --------------------------------------------------------
        # 2. ANOMALY VIDEOS (Intra-Video Negative Sampling via LLM Labels)
        # --------------------------------------------------------
        else:
            positive_pool = []
            negative_pool = []

            # Segregate annotations using Llama 3's pre-calculated boolean flags
            for i, (t_start, t_end) in enumerate(timestamps):
                sentence_obj = sentences[i]

                # Check if the sentence has been processed into a dictionary by our script
                if isinstance(sentence_obj, dict):
                    desc = sentence_obj["text"]
                    is_ano = sentence_obj.get("is_anomaly", False)
                else:
                    # Fallback if raw un-labeled JSON is accidentally passed
                    desc = sentence_obj
                    is_ano = False

                if is_ano:
                    positive_pool.append((t_start, t_end, desc))
                else:
                    negative_pool.append((t_start, t_end, desc))

            # 50% chance for an Anomaly (Yes), 50% chance for Background (No)
            if random.random() > 0.5 and positive_pool:
                # POSITIVE SAMPLE
                chosen_start, chosen_end, desc = random.choice(positive_pool)
                target_text = f"Yes.\n{desc}"

                min_start = max(0.0, chosen_start - (self.window_size_sec / 2))
                max_start = min(chosen_end, total_sec - self.window_size_sec)
                start_sec = random.uniform(min_start, max_start) if max_start > min_start else chosen_start

            elif negative_pool:
                # NEGATIVE SAMPLE (Peaceful background interval from the same anomaly video)
                chosen_start, chosen_end, desc = random.choice(negative_pool)
                target_text = f"No.\n{desc}. The scene appears peaceful with no suspicious or anomalous behavior."

                min_start = max(0.0, chosen_start - (self.window_size_sec / 2))
                max_start = min(chosen_end, total_sec - self.window_size_sec)
                start_sec = random.uniform(min_start, max_start) if max_start > min_start else chosen_start

            else:
                # Fallback if pools are empty
                return self.__getitem__((idx + 1) % len(self))

        frames = self._extract_window(vr, start_sec, fps, total_frames)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))

        return {
            "frames": frames,
            "prompt": self.prompt_template,
            "target": target_text,
            "label": label
        }