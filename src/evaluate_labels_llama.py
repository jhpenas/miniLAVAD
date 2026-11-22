import json
import torch
import numpy as np
import mlflow
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import transformers

PROJECT_ROOT = Path(__file__).resolve().parent.parent

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)
from transformers import AutoModelForCausalLM, AutoTokenizer

JSON_PATH = PROJECT_ROOT / "models" / "val_eval_0ef5b4_epoch4.json"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Remove warnings
transformers.logging.set_verbosity_error()

def softmax(logits, temperature=1.0):
    """Calculates the smoothed probability between Yes and No."""
    scaled = np.array(logits) / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    probs = exp_scaled / np.sum(exp_scaled)
    return probs[0]  # Returns the probability of "Yes"


def run_cascade_evaluation(json_path=JSON_PATH):
    print(f"Reading descriptions from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    raw_captions = data.get("raw_captions", {})
    if not raw_captions:
        print("Error: No captions found in the file.")
        return

    print(f"Loading Model {MODEL_ID} as the Classification Judge...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Llama 3 necessita de um pad token explícito
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Llama 3 estabiliza melhor com bfloat16
        device_map="auto"
    )
    model.eval()

    # Token IDs for Yes/No
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

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
                    #Manter a estrutura de dicionário para frames vazios
                    frame_scores.append({
                        "prob_yes": 0.0,
                        "raw_yes": 0.0,
                        "raw_no": 0.0
                    })
                    continue

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Scene description: {description}"}
                ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

                # Extrai os logits do primeiro token gerado
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    max_lenght=None,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id
                )

                first_token_logits = outputs.scores[0][0]
                yes_logit = first_token_logits[yes_id].item()
                no_logit = first_token_logits[no_id].item()

                score = softmax([yes_logit, no_logit], temperature=1.0)


                frame_scores.append({
                    "prob_yes": float(score),
                    "raw_yes": yes_logit,
                    "raw_no": no_logit
                })

            cascade_results[video_id] = frame_scores

    model_name = MODEL_ID.split("/")[-1].replace(".", "_")
    output_path = PROJECT_ROOT / "models" / f"val_eval_cascade_{model_name}.json"

    final_output = {
        "__metadata__": {
            "source_model": data["__metadata__"]["run_id"],
            "evaluator_model": MODEL_ID,
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

    with mlflow.start_run(run_name=f"Cascade_{model_name}_Eval"):
        mlflow.log_param("evaluator_model", MODEL_ID)
        mlflow.log_param("source_vlm_run", data["__metadata__"]["run_id"])
        mlflow.log_param("pipeline_type", "cascade")
        mlflow.log_artifact(str(output_path), artifact_path="evaluations")

    print("Successfully logged to MLflow!")


if __name__ == "__main__":
    run_cascade_evaluation()