import os
import json
import argparse
import torch
import torchvision.io as io
import torchvision.transforms as T
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import time


def load_cctv_model(base_model_id="Qwen/Qwen2-VL-2B-Instruct", adapter_path=None, use_quantization=True):
    print(f"Initializing Evaluation. Base model: {base_model_id}")

    if use_quantization:
        print("[Config] Applying 4-bit NF4 Quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
    else:
        print("[Config] Loading in pure FP16 (No Quantization)...")
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    if adapter_path and os.path.exists(adapter_path):
        print(f"[Config] Injecting LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("[Config] No LoRA adapters provided. Running zero-shot base model.")
        model = base_model

    model.eval()
    processor = AutoProcessor.from_pretrained(base_model_id)

    return model, processor


def get_streaming_chunks(video_path, max_frames=10, fps_rate=1.0):
    """
    Simulates a live CCTV feed by breaking the video into chronological chunks.
    Extracts native resolution frames to perfectly match the training distribution.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"\n[WARNING] Could not decode video {video_path}: {e}")
        return []

    native_fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Calculate stride for 1 FPS
    stride = max(1, int(round(native_fps / fps_rate)))
    sampled_indices = list(range(0, total_frames, stride))

    chunks = []
    # Group the stream into 10-frame (10-second) windows
    for i in range(0, len(sampled_indices), max_frames):
        window_indices = sampled_indices[i: i + max_frames]

        # Skip fragments at the end of the video that are too short
        if len(window_indices) < 2:
            continue

        # Extract native frames directly (NO PIL RESIZING)
        frames = vr.get_batch(window_indices).asnumpy()

        start_sec = window_indices[0] / native_fps
        end_sec = window_indices[-1] / native_fps
        timestamp_label = f"{start_sec:.1f}s-{end_sec:.1f}s"

        chunks.append({
            "timestamp": timestamp_label,
            "frames": frames
        })

    return chunks


def run_evaluation(model, processor, args):
    print("Indexing physical video files in data/raw/...")
    video_index = {}
    for root, dirs, files in os.walk(os.path.join("data", "raw")):
        for f in files:
            if f.endswith(".mp4"):
                video_index[f] = os.path.abspath(os.path.join(root, f))
    print(f"Found {len(video_index)} videos on disk.")

    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []
    print(f"Starting inference on {len(test_data)} test records...")
    print(f"[Constraints] Max Frames: {args.max_frames}")

    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating Videos"):
            inference_prompt = item.get(
                "prompt",
                "You are an AI surveillance assistant. Analyze this video for any suspicious or anomalous behavior. "
                "You MUST output your response in this exact format: '[Probability: X.XX] Description'. "
                "X.XX must be a float between 0.00 (completely normal) and 1.00 (definite anomaly). "
                "Follow this with a detailed description of the scene."
            )

            filename = os.path.basename(item["video_path"])

            if filename not in video_index:
                print(f"\n[DEBUG ERROR] File Not Found: {filename}")
                continue

            abs_path = video_index[filename]
            video_start_time = time.time()

            # Extract chronological chunks simulating a live buffer
            video_chunks = get_streaming_chunks(abs_path, max_frames=args.max_frames, fps_rate=1.0)

            window_predictions = []

            # Run inference on each window sequentially
            for chunk in video_chunks:
                safe_video_frames = chunk["frames"]

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": safe_video_frames},
                            {"type": "text", "text": inference_prompt}
                        ]
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    videos=[safe_video_frames],
                    padding=True,
                    return_tensors="pt",
                    cap_pixels_per_frame=True
                ).to("cuda")

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    generated_ids = model.generate(**inputs, max_new_tokens=50)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                prediction = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # Store the prediction mapped directly to its temporal window
                window_predictions.append({
                    "timestamp": chunk["timestamp"],
                    "prediction": prediction.strip()
                })

                del inputs, generated_ids, generated_ids_trimmed, safe_video_frames
                torch.cuda.empty_cache()

            video_elapsed_time = time.time() - video_start_time

            # Automatically extract the real filename so we can check timestamps!
            fallback_id = os.path.splitext(os.path.basename(abs_path))[0]

            # Save the aggregated timeline of predictions for the entire video
            results.append({
                "video_id": item.get("video_id", fallback_id),
                "ground_truth": item.get("target", item.get("label", item.get("class", "unknown"))),
                "processing_time_seconds": round(video_elapsed_time, 2),
                "total_chunks_evaluated": len(video_chunks),
                "timeline": window_predictions
            })

    # Wrap the output in a metadata dictionary for perfect academic provenance
    output_data = {
        "metadata": {
            "adapter_path": args.adapter_path,
            "quantized": not args.no_quantize,
            "max_frames": args.max_frames,
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="miniLAVAD Evaluation & Ablation Script")

    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--test_json", type=str, default="data/processed/test_split.json")
    parser.add_argument("--output_file", type=str, default="models/eval_results.json")

    # Updated default to 10 to match your training set
    parser.add_argument("--max_frames", type=int, default=10,
                        help="Strict limit on number of frames to extract per video")

    args = parser.parse_args()

    cctv_model, vlm_processor = load_cctv_model(
        adapter_path=args.adapter_path,
        use_quantization=not args.no_quantize
    )

    # Pass the entire args namespace to the function
    run_evaluation(
        cctv_model,
        vlm_processor,
        args
    )