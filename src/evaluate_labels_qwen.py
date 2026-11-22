import json
import torch
import numpy as np
import mlflow
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = PROJECT_ROOT / "models" / "val_eval_0ef5b4_epoch4.json" #best model


def softmax(logits, temperature=1.0):
    """Calculates the smoothed probability between Yes and No."""
    scaled = np.array(logits) / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    probs = exp_scaled / np.sum(exp_scaled)
    return probs[0]  # Returns the probability of "Yes"


def run_cascade_evaluation():
    # 1. Load the JSON from the VLM
    json_path = JSON_PATH
    print(f"Reading descriptions from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_captions = data.get("raw_captions", {})
    if not raw_captions:
        print("Error: No captions found in the file.")
        return

    print("Loading Qwen2-VL-2B-Instruct as the Classification Judge...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Para modelos VLM, torch_dtype ainda é o padrão
        device_map="auto"
    )
    model.eval()

    # Token IDs for Yes/No (using processor.tokenizer)
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    system_prompt = (
        "You are an automated CCTV security analyzer. "
        "Read the scene description and answer with exactly 'Yes.' or 'No.' as your very first word if there is "
        "any crime, severe violence, or severe accident happening."
    )

    cascade_results = {}

    with torch.no_grad():
        for video_id, frames in tqdm(raw_captions.items(), desc="Evaluating Cascade Texts"):
            frame_scores = []

            for frame in frames:
                description = frame.get("raw_output", "")

                if not description:
                    frame_scores.append({
                        "prob_yes": 0.0,
                        "raw_yes": 0.0,
                        "raw_no": 0.0
                    })
                    continue

                # Formatting without image/video tags
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Scene description: {description}"}
                ]

                # Process text purely through the VLM processor
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")

                # Extract the logits of the first generated token
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                first_token_logits = outputs.scores[0][0]
                yes_logit = first_token_logits[yes_id].item()
                no_logit = first_token_logits[no_id].item()

                # Calculate the mathematical probability
                score = softmax([yes_logit, no_logit], temperature=5.0)

                frame_scores.append({
                    "prob_yes": float(score),
                    "raw_yes": yes_logit,
                    "raw_no": no_logit
                })

            cascade_results[video_id] = frame_scores

    # Save the new JSON locally
    safe_model_name = model_id.split("/")[-1].replace(".", "_")
    output_path = Path(f"models/val_eval_cascade_{safe_model_name}.json")

    final_output = {
        "__metadata__": {
            "source_model": data["__metadata__"]["run_id"],
            "evaluator_model": model_id,
            "pipeline": "cascade_logits"
        },
        "scores": cascade_results
    }

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"\nCascade prediction matrix saved locally to: {output_path}")

    # MLFLOW TRACKING INTEGRATION
    print("Logging cascade evaluation to MLflow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Start a new run using the dynamic model name
    with mlflow.start_run(run_name=f"Cascade_{safe_model_name}_Eval"):
        mlflow.log_param("evaluator_model", model_id)
        mlflow.log_param("source_vlm_run", data["__metadata__"]["run_id"])
        mlflow.log_param("pipeline_type", "cascade")

        # Upload the generated JSON as an artifact
        mlflow.log_artifact(str(output_path), artifact_path="evaluations")

    print("Successfully logged to MLflow!")


if __name__ == "__main__":
    run_cascade_evaluation()