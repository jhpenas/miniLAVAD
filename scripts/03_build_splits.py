import json
import random
import re
from pathlib import Path


def get_label_from_stem(stem: str) -> str:
    """Extracts the anomaly class label from the video stem (e.g., 'Abuse037_x264' -> 'Abuse')."""
    if "Normal" in stem or "normal" in stem.lower():
        return "Normal"
    # Extract alphabetical characters before any numbers
    match = re.match(r"([a-zA-Z]+)", stem)
    if match:
        return match.group(1)
    return "Unknown"


def parse_original_test_annotations(txt_path: Path):
    test_annotations = {}
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        print(f"ERROR: {txt_path.name} is missing or empty!")
        return test_annotations

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            video_name = parts[0]
            anomaly_class = parts[1]
            segments = []
            fps = 30.0 #Used on the original paper

            start1, end1 = int(parts[2]), int(parts[3])
            if start1 != -1 and end1 != -1:
                segments.append([round(start1 / fps, 1), round(end1 / fps, 1)])

            if len(parts) >= 6:
                start2, end2 = int(parts[4]), int(parts[5])
                if start2 != -1 and end2 != -1:
                    segments.append([round(start2 / fps, 1), round(end2 / fps, 1)])

            test_annotations[Path(video_name).stem] = {
                "video_path": video_name,
                "label": anomaly_class,
                "temporal_segments_sec": segments
            }
    return test_annotations


def load_uca_annotations(uca_dir: Path):
    uca_lookup = {}
    json_dir = uca_dir / "json"

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for path, ann in data.items():
                        uca_lookup[Path(path).stem] = ann
                elif isinstance(data, list):
                    for item in data:
                        key = item.get('video_name') or item.get('video_path')
                        if key:
                            uca_lookup[Path(key).stem] = item
        except json.JSONDecodeError:
            pass  # Safely ignore any empty/corrupted JSON files

    return uca_lookup


def build_hybrid_dataset(raw_dir: Path, processed_dir: Path, val_split_ratio=0.15):
    splits_dir = raw_dir / "UCF_Crimes-Train-Test-Split" / "Anomaly_Detection_splits"
    test_txt = splits_dir / "Anomaly_Test.txt"
    uca_dir = raw_dir / "UCF_Annotation"

    # 1. Load the Strict Original Test Set
    with open(test_txt, 'r', encoding='utf-8') as f:
        test_list = [line.strip() for line in f if line.strip()]

    test_stems = {Path(p).stem for p in test_list}
    print(f"Loaded Original Test Set: {len(test_stems)} videos.")

    # 2. Load UCA master lookup pool (The Universe of Annotated Videos)
    print("Loading UCA master lookup pool...")
    uca_lookup = load_uca_annotations(uca_dir)

    # 3. DEDUCE THE TRAIN SET (Universe - Test)
    train_val_universe = []
    for stem in uca_lookup.keys():
        if stem not in test_stems:
            label = get_label_from_stem(stem)
            vid_path = f"{label}/{stem}.mp4"
            train_val_universe.append((stem, vid_path, label))

    # 4. Create Validation Split
    random.seed(42)
    random.shuffle(train_val_universe)
    split_idx = int(len(train_val_universe) * (1 - val_split_ratio))

    train_videos = train_val_universe[:split_idx]
    val_videos = train_val_universe[split_idx:]

    print(f"Deduced Train Set split into: {len(train_videos)} Train, {len(val_videos)} Val")

    # 5. Map and Export
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Original Test annotations (Frames -> Seconds)...")
    test_annotations = parse_original_test_annotations(raw_dir / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")

    # Export Train and Val with full language prompts
    for split_name, video_list in [("train", train_videos), ("val", val_videos)]:
        split_data = []

        for stem, video_path, label in video_list:
            if label == "Normal":
                split_data.append({"video_path": video_path, "label": "Normal", "temporal_segments_sec": []})
            else:
                split_data.append({
                    "video_path": video_path,
                    "label": label,
                    "uca_data": uca_lookup[stem]  # Injects the timestamps and sentences!
                })

        with open(processed_dir / f"{split_name}_split.json", 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=4)
        print(f"Saved {split_name}_split.json ({len(split_data)} videos).")

    # Export Test strictly matching original baseline
    test_data = []
    for video_path in test_list:
        stem = Path(video_path).stem
        if stem in test_annotations:
            test_data.append(test_annotations[stem])
        else:
            test_data.append({"video_path": video_path, "label": "Normal", "temporal_segments_sec": []})

    with open(processed_dir / "test_split.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
    print(f"Saved test_split.json ({len(test_data)} videos).")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

    build_hybrid_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)