# miniLAVAD

**Multimodal Anomaly Detection via Vision-Language Models**
This repository contains the training and evaluation pipeline for fine-tuning Vision-Language Models (VLMs) on the UCF-Crime dataset. To ensure mathematically rigorous comparisons against state-of-the-art baselines, this project uses a hybrid data engineering approach: it evaluates strictly on the original CVPR 2018 test splits while leveraging the rich, frame-level language annotations from the CVPR 2024 UCA dataset for training and validation.

## Set Up on a New Server

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd miniLAVAD
```

### 2. Environment Configuration
Create a .env file in the root of the project to store your access tokens for downloading the heavy video files.
```bash
touch .env
```

Add the following line to your .env file:
```bash
DROPBOX_ACCESS_TOKEN=your_token_here
```

### 3. Data Engineering Pipeline
Run the following scripts in sequential order to set up the dataset. The pipeline is designed to be idempotent (it will safely skip files that are already downloaded or extracted).

- `python scripts/00_setup_project.py`
Initializes the local project architecture by creating the necessary data/ and models/ directory structures that are ignored by .gitignore.

- `python scripts/01_download_data.py`
Handles all network ingestion. Downloads the UCF-Crime .mp4 video archives from the original authors' Dropbox and recursively pulls the UCA temporal annotations directly from GitHub. (Requires DROPBOX_ACCESS_TOKEN).

- `python scripts/02_extract_dataset.py`
Unzips the downloaded UCF-Crime video archives into the data/raw/ directory with progress tracking and corruption handling.

- `python scripts/03_build_splits.py`
Generates the VLM-ready JSON splits (train_split.json, val_split.json, test_split.json). It isolates the exact test baseline from the original UCF-Crime Anomaly_Test.txt to guarantee strict benchmarking compliance, automatically converting the original frame-level annotations into precise second-based timestamps (adhering to the original authors' specified 30 FPS). It then deduces the training and validation sets from the remaining videos, mapping them to the rich temporal sentence annotations provided by the UCA dataset.

### 4. Training & Evaluation Engine (src/)
The core machine learning pipeline is intentionally decoupled into distinct scripts. This design is highly optimized for resource-constrained environments (e.g., fine-tuning on an 8 GB RTX 4060), ensuring stability without sacrificing methodological rigor.
- `python src/dataset.py`
Implements a custom PyTorch Dataset (UCFCrimeTrainingDataset). It loads raw .mp4 files, extracts a specific number of frames over a defined temporal window, and formats the inputs into ChatML dictionaries mapped to the UCA text annotations.
Dynamically processing multimodal data prevents the system from loading entire surveillance videos into memory at once. Furthermore, it completely bypasses Hugging Face's default video fetchers, allowing us to safely feed pre-processed tensor frames directly to Qwen2-VL's processor. This guarantees that visual token counts remain strictly bounded (cap_pixels_per_frame) before they ever hit the GPU.

- `python src/train.py`
The primary optimization engine. It loads the Qwen2-VL base model using 4-bit NF4 quantization, injects qLoRA adapter matrices, and executes a native PyTorch training loop over the training split.
This script explicitly avoids high-level wrappers like Hugging Face's SFTTrainer. SFTTrainer abstracts gradient accumulation and caching in ways that cause unpredictable VRAM spikes during multimodal training. By writing a native loop, we enforce aggressive, surgical memory management—specifically utilizing torch.cuda.empty_cache() and set_to_none=True after every single batch. This guarantees the 2.2B parameter model trains stably within an 8 GB VRAM ceiling. Additionally, it forces mid-training checkpoints at the end of every epoch to prevent data loss.

- `python src/evaluate.py`
A standalone evaluation script (CCTV Simulator) that loads the saved LoRA adapter checkpoints and runs inference against the isolated CVPR 2018 test split.
Evaluation and validation are completely decoupled from the train.py script. Running an inline validation loop during training requires PyTorch to simultaneously hold the training computation graph and allocate memory for the inference KV cache. On constrained hardware, this triggers an immediate Out-Of-Memory (OOM) crash. By running evaluation post-hoc on saved checkpoints, we protect the hardware while maintaining strict, academically rigorous benchmark comparisons against state-of-the-art 2025 anomaly detection models.
### 5. Experiment Tracking (MLflow)
Because fine-tuning multimodal models involves balancing numerous hyperparameters (quantization precision, LoRA rank, visual pixel caps, learning rates), this project relies on MLflow for local experiment tracking.

To initialize the tracking server:
`mlflow ui --port 5000`
The tracking server runs locally ([http://127.0.0.1:5000](http://127.0.0.1:5000)) and is natively integrated into train.py. It automatically logs the merged config.yaml state, tracks the train_loss curve step-by-step, and securely archives the LoRA adapter weights at the end of each epoch. This provides a mathematically verifiable audit trail of the model's optimization phase, establishing the reproducibility required for thesis defense.