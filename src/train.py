import sys
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from omegaconf import OmegaConf
import mlflow
from qwen_vl_utils import process_vision_info

# Local imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))
from dataset import UCFCrimeTrainingDataset


def collate_fn(batch, processor):
    """Formats raw frames and text into Qwen2-VL ChatML conversational tensors."""
    formatted_texts = []
    video_inputs = []

    for item in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy_path.mp4"},
                    {"type": "text", "text": item["prompt"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["target"]}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
        video_inputs.append(item["frames"])

    inputs = processor(
        text=formatted_texts,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        cap_pixels_per_frame=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # PROMPT MASKING: Only calculate loss on the assistant's response
    for i in range(labels.shape[0]):
        im_start_indices = (labels[i] == 151644).nonzero(as_tuple=True)[0]
        if len(im_start_indices) > 0:
            assistant_header_end = im_start_indices[-1] + 3
            labels[i, :assistant_header_end] = -100

    inputs["labels"] = labels
    return inputs


def main():
    # 1. Load Configurations
    config_path = project_root / "configs" / "config.yaml"
    base_config = OmegaConf.load(str(config_path))
    cli_config = OmegaConf.from_cli(sys.argv[1:])
    config = OmegaConf.merge(base_config, cli_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Configure MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(config.project_name)

    # 3. Setup 4-bit Quantization (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base Qwen2-VL-2B model in 4-bit...")
    processor = AutoProcessor.from_pretrained(config.model.name_or_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model.name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 4. Inject qLoRA Adapters
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Initialize Dataset & Dataloader
    train_dataset = UCFCrimeTrainingDataset(
        split_json_path=project_root / config.data.train_json_path,
        processor=processor,
        num_frames=config.data.num_frames,
        window_size_sec=config.data.window_size_sec
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    # Initialize Validation Dataset & Dataloader
    val_dataset = UCFCrimeTrainingDataset(
        split_json_path=project_root / config.data.val_json_path,
        processor=processor,
        num_frames=config.data.num_frames,
        window_size_sec=config.data.window_size_sec
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # No need to shuffle validation data
        collate_fn=lambda b: collate_fn(b, processor)
    )

    # 6. Optimizer & Mixed Precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.learning_rate))
    scaler = torch.cuda.amp.GradScaler()

    # 7. Native Training Loop tracked by MLflow
    run_name = config.get("run_name", "v3_lr5e-6")
    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow Run started: {run.info.run_id} ({run_name})")

        # Log parameters and original config file
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        mlflow.log_artifact(str(config_path), artifact_path="configuration")

        global_step = 0

        for epoch in range(config.training.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss / config.training.gradient_accumulation_steps

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * config.training.gradient_accumulation_steps

                if (step + 1) % config.training.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    current_loss = loss.item() * config.training.gradient_accumulation_steps
                    mlflow.log_metric("train_loss", current_loss, step=global_step)
                    print(f"Epoch [{epoch + 1}/{config.training.epochs}] Step [{step + 1}] - Loss: {current_loss:.4f}")

                del outputs, loss, batch
                torch.cuda.empty_cache()

            # Flush remaining gradients at the end of the epoch
            if len(train_loader) % config.training.gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_epoch_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("epoch_loss", avg_epoch_loss, step=epoch)
            print(f"--- Epoch {epoch + 1} Complete | Average Loss: {avg_epoch_loss:.4f} ---")

            model.eval()  # Switch to evaluation mode
            epoch_val_loss = 0.0

            with torch.no_grad():  # Do not compute gradients
                for val_batch in val_loader:
                    val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}

                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        val_outputs = model(**val_batch)
                        epoch_val_loss += val_outputs.loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(f"--- Epoch {epoch + 1} Complete | Average Val Loss: {avg_val_loss:.4f} ---")

            # Ephemeral Mid-Training Checkpointing
            print(f"Securing Epoch {epoch + 1} checkpoint to MLflow...")

            # Bundle both the model and processor so this intermediate
            # epoch is a fully functional, deployable model.
            components = {
                "model": model,
                "tokenizer": processor.tokenizer,
                "image_processor": processor.image_processor
            }

            # Log natively, using the dynamic epoch number for the path
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path=f"checkpoints_epoch_{epoch + 1}",
                task="text-generation"
            )

            print(f"Epoch {epoch + 1} checkpoint uploaded.")

        # 8. Native MLflow Model Checkpointing
        print("Securing final adapter and processor to MLflow as a Model...")

        components = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "image_processor": processor.image_processor
        }

        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="final_adapter",
            task="text-generation"  # MLflow kind of pipeline
        )

        print("Training complete! All weights and configs safely recorded in MLflow.")


if __name__ == "__main__":
    main()