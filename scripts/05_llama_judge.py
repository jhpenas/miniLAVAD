import json
import requests
from tqdm import tqdm
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path.cwd().parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # Change to llama3.1 if you have the newer version installed


def classify_text_with_llama(text):
    prompt = (
        "You are an objective CCTV anomaly detection judge. "
        "Read the following video description. If the description contains any anomalous, "
        "suspicious, or dangerous behavior (such as fighting, violence, abuse, stealing, or accidents), "
        "respond with the exact number 1. If it describes normal, peaceful, or everyday behavior, "
        "respond with the exact number 0. Output nothing else.\n\n"
        f"Description: {text}"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}  # Zero temperature for strict classification
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json().get("response", "").strip()
        # Parse the output to a float
        if "1" in result:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return 0.0


def process_captions_to_scores(input_json_path, output_json_path):
    print(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    raw_captions = data.get("raw_captions", {})
    if not raw_captions:
        print("No raw_captions found in this JSON.")
        return

    new_scores = {}

    print("Evaluating Qwen's text descriptions using Llama-3...")
    for video_id, chunks in tqdm(raw_captions.items(), desc="Processing Videos"):
        chunk_scores = []
        for chunk in chunks:
            description = chunk.get("raw_output", "")
            if not description:
                chunk_scores.append(0.0)
                continue

            score = classify_text_with_llama(description)
            chunk_scores.append(score)

        # We need to stretch these chunk scores to frame-level scores just like evaluate.py does
        # However, since we don't have total_frames explicitly here easily, we will
        # let calculate_auc handle the chunk-to-frame stretching, OR we assume evaluate.py
        # already stretched the original scores. For simplicity, we will save the raw chunk scores
        # and stretch them to match the original score array length.

        original_frame_count = len(data["scores"][video_id])
        if len(chunk_scores) > 0:
            stretched_scores = np.interp(
                np.arange(original_frame_count),
                np.linspace(0, original_frame_count - 1, len(chunk_scores)),
                chunk_scores
            ).tolist()
        else:
            stretched_scores = [0.0] * original_frame_count

        new_scores[video_id] = stretched_scores

    # Create the new data structure
    new_data = {
        "__metadata__": data.get("__metadata__", {}),
        "scores": new_scores
    }

    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved Llama-3 evaluated scores to {output_json_path}")


if __name__ == "__main__":
    input_file = PROJECT_ROOT / "models" / "eval_results_epoch2.json"
    output_file = PROJECT_ROOT / "models" / "eval_results_epoch2_llama3.json"

    process_captions_to_scores(input_file, output_file)