import sys
from pathlib import Path
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

# Critical VRAM Protections Built Into This Loop
# set_to_none=True on Gradients: Instead of zeroing out gradient tensors, setting them to None immediately deallocates memory on the GPU.
#
# Explicit Tensor Deletion: The variables outputs, loss, and batch are explicitly unlinked and purged via torch.cuda.empty_cache() at every iteration to eliminate memory leaks across video sequences.
#
# Targeted Label Masking: Padding tokens are masked with -100, instructing PyTorch’s cross-entropy loss to bypass non-informative tokens and conserve computation.


def collate_fn(batch, processor):
    """Formats raw frames and text into Qwen2-VL ChatML conversational tensors."""
    formatted_texts = []
    video_inputs = []

    for item in batch:
        # Construct ChatML sequence: Prompt (User) -> Target Explanation (Assistant)
        messages = [
            {
                "role": "user",
                "content": [
                    # apply_chat_template only looks at the "type" to insert <|video_pad|> tokens.
                    # Using a dummy path since dataset.py already extracted the frames.
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

        # Generate text prompt with special vision tokens
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)

        # Bypass process_vision_info and directly append the frames from dataset.py
        video_inputs.append(item["frames"])

    # Tokenize and encode multimodal inputs natively
    inputs = processor(
        text=formatted_texts,
        videos=video_inputs, #the video tensors enter here, ignoring the dummy path on apply_chat_template
        padding=True,
        return_tensors="pt",
        cap_pixels_per_frame=True  # Enforces safe token counts
    )

    # Create target labels for loss computation (masking padding tokens with -100)
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
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

    # 6. Optimizer & Mixed Precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.training.learning_rate))
    scaler = torch.cuda.amp.GradScaler()

    # 7. Native Training Loop
    with mlflow.start_run() as run:
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        global_step = 0

        for epoch in range(config.training.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0

            for step, batch in enumerate(train_loader):
                # Move tensors to GPU
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss / config.training.gradient_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * config.training.gradient_accumulation_steps

                # Gradient accumulation step
                if (step + 1) % config.training.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    current_loss = loss.item() * config.training.gradient_accumulation_steps
                    mlflow.log_metric("train_loss", current_loss, step=global_step)
                    print(f"Epoch [{epoch + 1}/{config.training.epochs}] Step [{step + 1}] - Loss: {current_loss:.4f}")

                # Immediate memory cleanup to protect VRAM
                del outputs, loss, batch
                torch.cuda.empty_cache()

            avg_epoch_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("epoch_loss", avg_epoch_loss, step=epoch)
            print(f"--- Epoch {epoch + 1} Complete | Average Loss: {avg_epoch_loss:.4f} ---")

            # Mid-Training Checkpointing
            checkpoint_dir = project_root / config.training.output_dir / f"checkpoint-epoch-{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"Saving checkpoint to {checkpoint_dir}...")
            # This only saves the lightweight LoRA adapters, not the massive base model
            model.save_pretrained(str(checkpoint_dir))

            # Instantly upload to local MLflow server
            mlflow.log_artifacts(str(checkpoint_dir), artifact_path=f"checkpoints/epoch_{epoch + 1}")
            print(f"Epoch {epoch + 1} checkpoint secured!")


        # 8. Save Checkpoint Artifact
        output_dir = project_root / config.training.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir / "final_adapter"))
        mlflow.log_artifacts(str(output_dir / "final_adapter"), artifact_path="lora_adapters")
        print("Training complete! Adapter weights saved and logged to MLflow.")


if __name__ == "__main__":
    main()