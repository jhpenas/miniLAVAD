import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configuration
INPUT_JSON = PROJECT_ROOT / "data/processed/train_split.json"
OUTPUT_JSON = PROJECT_ROOT / "data/processed/train_split_labeled.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


# Classification Engine
def classify_sentence(sentence: str) -> bool:
    """
    Asks local Llama 3 if the sentence contains an anomaly.
    Returns True if yes, False if no.
    """
    prompt = (
        "You are an expert CCTV surveillance behavioral analyst. "
        "Read the sentence describing a video scene and determine if it describes an anomaly, "
        "a crime, violence, abuse, theft, vandalism, or an accident. "
        "If it describes a normal, peaceful, or mundane action, answer NO. "
        "Answer with exactly one word: YES or NO.\n\n"

        "Example 1:\n"
        "Sentence: 'Three cars passed by on the road.'\n"
        "Answer: NO\n\n"

        "Example 2:\n"
        "Sentence: 'A man in black spray paints on a green device.'\n"
        "Answer: YES\n\n"

        "Example 3:\n"
        "Sentence: 'People walking forward on the sidewalk.'\n"
        "Answer: NO\n\n"

        "Example 4:\n"
        "Sentence: 'A man suddenly grabs the woman's bag and runs away.'\n"
        "Answer: YES\n\n"

        f"Sentence: '{sentence}'\n"
        "Answer:"
    )

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        })

        if res.status_code == 200:
            response_text = res.json().get("response", "").strip().upper()
            return "YES" in response_text
        else:
            print(f"\n[Warning] Ollama returned status code {res.status_code}")
            return False

    except Exception as e:
        print(f"\n[Error] Could not connect to local Ollama server: {e}")
        return False


# Processing Loop
def run_labeler():
    print(f"Loading dataset from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Memoization Cache to save time/API calls for repeated sentences
    cache = {}
    total_sentences = 0
    unique_sentences = 0

    print(f"Analyzing UCA temporal descriptions using local {MODEL_NAME}...")
    for item in tqdm(data, desc="Videos Processed"):
        uca_data = item.get("uca_data", {})
        sentences = uca_data.get("sentences", [])

        if not sentences:
            continue

        new_sentences_format = []
        for text in sentences:
            # Handle cases where the script might be re-run on already processed data
            if isinstance(text, dict):
                text = text.get("text", "")

            total_sentences += 1
            text = text.strip()

            # Check cache first
            if text in cache:
                is_anomaly = cache[text]
            else:
                is_anomaly = classify_sentence(text)
                cache[text] = is_anomaly
                unique_sentences += 1

            # Convert the raw string into a structured dictionary
            new_sentences_format.append({
                "text": text,
                "is_anomaly": is_anomaly
            })

        # Overwrite the old array of strings with the new array of dictionaries
        item["uca_data"]["sentences"] = new_sentences_format

    print("\n--- Processing Complete ---")
    print(f"Total Sentences Scanned: {total_sentences}")
    print(f"Unique Inferences Made: {unique_sentences}")
    print(f"Cache saved {total_sentences - unique_sentences} duplicate inferences.")

    print(f"Saving LLM-labeled dataset to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print("Done! You can now point dataset.py to this new JSON.")


if __name__ == "__main__":
    run_labeler()