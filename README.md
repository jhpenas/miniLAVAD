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
### `evaluate.py` (CCTV Inference & Signal Processing Engine)

`python src/evaluate.py [ARGS]`

A standalone evaluation engine that loads the base Vision-Language Model (and optional LoRA adapters) to run mathematically rigorous inference against the isolated CVPR 2018 test split. 

Evaluation and validation are completely decoupled from the `train.py` script. Running an inline validation loop during training requires PyTorch to simultaneously hold the training computation graph and allocate memory for the inference KV cache. On constrained hardware, this triggers an immediate Out-Of-Memory (OOM) crash. By running evaluation post-hoc on saved checkpoints, we protect the hardware while maintaining strict, academically rigorous benchmark comparisons against state-of-the-art 2025 anomaly detection models.

Beyond hardware protection, this script implements a highly specialized pipeline to convert discrete autoregressive text generation into a continuous, frame-accurate mathematical signal for VAD (Video Anomaly Detection) metrics:

#### Core Methodologies

* **Mathematical Logit Extraction:** Instead of relying on brittle string parsing (which is highly vulnerable to VLM hallucination and format bias), the script bypasses the text output entirely for its mathematical scoring. It extracts the raw neural activation logits for the exact token IDs of `Yes` and `No` at the first generation step, applying a Softmax function to calculate a precise continuous probability (e.g., `0.1523`) of an anomaly.
* **Temperature Scaling for VLM Overconfidence:** Foundational VLMs are heavily polarized, frequently pushing negative logits below PyTorch's 16-bit physical memory limit (`-65,504`), resulting in `-Infinity` underflows and snapping probabilities to a hard `0.0` or `1.0`. The script implements a configurable Temperature Scaling factor (`--temperature`) to soften logit overconfidence, preserving the model's true underlying uncertainty for highly accurate ROC-AUC curves.
* **Forced Caption Generation (EOS Masking):** To prevent premature early-stopping (where the model outputs a binary answer and immediately drops an `<|im_end|>` token to save compute), the generation block enforces a `min_new_tokens=30` override. This forces the cross-attention layers to visually ground the binary decision by generating a detailed qualitative caption of the physical scene, allowing for deep forensic analysis of model hallucinations.
* **Dynamic Signal Boundary Processing:** VLM inference on raw 30 FPS video is computationally impossible. The script processes sparse 10-frame temporal chunks, then utilizes a three-step physical signal processor to rebuild a frame-accurate array that perfectly matches the UCA ground-truth annotations:
  1. **Interpolation:** Stretches the sparse chunk logits across the native frame count.
  2. **Temporal Smoothing:** Applies a rolling average to eliminate transient logit spikes.
  3. **Non-Linear Suppression:** Applies power crushing to suppress low-confidence background noise while preserving high-confidence anomaly spikes.

#### Command Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--adapter_path` | String | `None` | Path to the trained LoRA weights. If omitted, runs the zero-shot base model. |
| `--no_quantize` | Flag | `False` | Disables 4-bit NF4 quantization. Forces pure FP16 loading (Requires heavy VRAM). |
| `--max_frames` | Int | `10` | The number of frames passed to the VLM per temporal chunk. |
| `--temperature` | Float | `5.0` | Calibrates the Softmax function. `1.0` uses raw logits; higher values soften VLM overconfidence to prevent `-Infinity` underflows. |
| `--smoothing_window` | Int | `60` | Size of the rolling average window applied to the interpolated signal. `1` disables smoothing. |
| `--suppression_power` | Float | `2.0` | Exponent applied to the final signal curve to crush low-confidence noise. `1.0` disables suppression. |
| `--limit` | Int | `None` | Restricts the evaluation to *N* videos for rapid ablation testing and debugging. |
| `--test_json` | String | `data/processed/test_split.json` | Path to the evaluation dataset index. |
| `--output_file` | String | `models/eval_results.json` | Destination for the JSON log containing final ROC-AUC scores, raw logits, and forensic captions. |


### 5. Experiment Tracking (MLflow)
Because fine-tuning multimodal models involves balancing numerous hyperparameters (quantization precision, LoRA rank, visual pixel caps, learning rates), this project relies on MLflow for local experiment tracking.

To initialize the tracking server:
`mlflow ui --port 5000`
The tracking server runs locally ([http://127.0.0.1:5000](http://127.0.0.1:5000)) and is natively integrated into train.py. It automatically logs the merged config.yaml state, tracks the train_loss curve step-by-step, and securely archives the LoRA adapter weights at the end of each epoch. This provides a mathematically verifiable audit trail of the model's optimization phase, establishing the reproducibility required for thesis defense.