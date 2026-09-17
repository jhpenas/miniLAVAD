import os
import json
import argparse
import re
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from decord import VideoReader, cpu
import torch.nn.functional as F


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
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"\n[WARNING] Could not decode video {video_path}: {e}")
        return [], 0

    native_fps = vr.get_avg_fps()
    total_frames = len(vr)

    stride = max(1, int(round(native_fps / fps_rate)))
    sampled_indices = list(range(0, total_frames, stride))

    chunks = []
    for i in range(0, len(sampled_indices), max_frames):
        window_indices = sampled_indices[i: i + max_frames]
        if len(window_indices) < 2:
            continue

        frames = vr.get_batch(window_indices).asnumpy()
        start_sec = window_indices[0] / native_fps
        end_sec = window_indices[-1] / native_fps
        timestamp_label = f"{start_sec:.1f}s-{end_sec:.1f}s"

        chunks.append({
            "timestamp": timestamp_label,
            "frames": frames
        })

    return chunks, total_frames


def apply_dynamic_boundaries(raw_scores, total_frames, window_size=60, suppression_power=2.0):
    """
    Processes the raw chunk probabilities into a continuous, filtered physical signal.
    """
    if not raw_scores:
        return np.zeros(total_frames).tolist()

    # 1. Interpolation: Stretch the chunk scores to match the exact video frame count
    frame_level_raw = np.interp(
        np.arange(total_frames),
        np.linspace(0, total_frames - 1, len(raw_scores)),
        raw_scores
    )

    # 2. Temporal Smoothing: Apply rolling average if window is greater than 1
    if window_size > 1:
        smoothed = np.convolve(frame_level_raw, np.ones(window_size) / window_size, mode='same')
    else:
        smoothed = frame_level_raw

    # 3. Non-Linear Suppression: Apply power crushing if exponent is not exactly 1.0
    if suppression_power != 1.0:
        amplified = smoothed ** suppression_power
    else:
        amplified = smoothed

    # Normalize safely back to [0, 1] probability curve
    max_val = np.max(amplified)
    if max_val > 0:
        amplified = amplified / max_val

    return amplified.tolist()


def run_evaluation(model, processor, args):
    print("Indexing physical video files in data/raw/...")
    video_index = {}
    for root, dirs, files in os.walk(os.path.join("data", "raw")):
        for f in files:
            if f.endswith(".mp4"):
                video_index[f] = os.path.abspath(os.path.join(root, f))

    with open(args.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]

    results = {}
    raw_captions = {}
    video_timings = {}
    total_eval_start = time.time()
    print(f"Starting inference on {len(test_data)} test records...")

    print("Extracting Token IDs for logit evaluation...")
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    with torch.no_grad():
        for item in tqdm(test_data, desc="Evaluating Videos"):
            video_start_time = time.time()
            inference_prompt = item.get(
                "prompt",
                "Is there any suspicious or anomalous behavior (such as fighting, violence, abuse, stealing, or accidents) happening in this video clip? "
                "Answer with exactly 'Yes.' or 'No.' as your very first word. "
                "Then, on a new line, provide a detailed, objective description of exactly what is happening in the scene."
            )

            filename = os.path.basename(item["video_path"])
            if filename not in video_index:
                continue

            abs_path = video_index[filename]
            video_chunks, total_frames = get_streaming_chunks(abs_path, max_frames=args.max_frames, fps_rate=1.0)

            if total_frames == 0:
                continue

            raw_chunk_scores = []
            video_text_logs = []
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

                # 1. Command model to return internal neural scores
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        min_new_tokens=30, #To avoid only output Yes/No
                        output_scores=True,
                        return_dict_in_generate=True
                    )

                # 2. Decode the text specifically to save in JSON logs
                generated_ids = outputs.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                prediction = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                # Extract the logits for the very first generated token (which is 'Yes' or 'No')
                first_token_logits = outputs.scores[0][0]
                yes_logit = first_token_logits[yes_id]
                no_logit = first_token_logits[no_id]

                # Apply Temperature Scaling to soften the LLM's overconfidence
                temperature = args.temperature
                confidence_scores = F.softmax(
                    torch.tensor([yes_logit / temperature, no_logit / temperature], dtype=torch.float32),
                    dim=0
                )

                # 4. Calculate the mathematical probability of "Yes" vs "No"
                #confidence_scores = F.softmax(torch.tensor([yes_logit, no_logit], dtype=torch.float32), dim=0)
                score = confidence_scores[0].item()

                raw_chunk_scores.append(score)
                video_text_logs.append({
                    "timestamp": chunk["timestamp"],
                    "raw_output": prediction,
                    "parsed_score": round(score, 6),
                    "raw_logits": {"yes": round(yes_logit.item(), 2), "no": round(no_logit.item(), 2)}
                })


                del inputs, generated_ids, generated_ids_trimmed, safe_video_frames
                torch.cuda.empty_cache()

            video_elapsed = time.time() - video_start_time

            # Pass the new CLI arguments into the signal processor
            optimized_frame_scores = apply_dynamic_boundaries(
                raw_scores=raw_chunk_scores,
                total_frames=total_frames,
                window_size=args.smoothing_window,
                suppression_power=args.suppression_power
            )

            fallback_id = os.path.splitext(os.path.basename(abs_path))[0]
            video_id = item.get("video_id", fallback_id)

            video_timings[video_id] = round(video_elapsed, 2)
            results[video_id] = optimized_frame_scores
            raw_captions[video_id] = video_text_logs

    final_output = {
        "__metadata__": {
            "adapter_path": args.adapter_path,
            "quantized": not args.no_quantize,
            "max_frames": args.max_frames,
            "smoothing_window": args.smoothing_window,
            "suppression_power": args.suppression_power,
            "total_inference_time_seconds": round(time.time() - total_eval_start, 2),
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_timings_seconds": video_timings
        },
        "scores": results,
        "raw_captions": raw_captions
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    print(f"Evaluation complete! Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="miniLAVAD Evaluation & Ablation Script")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--test_json", type=str, default="data/processed/test_split.json")
    parser.add_argument("--output_file", type=str, default="models/eval_results.json")
    parser.add_argument("--max_frames", type=int, default=10)

    # New Signal Processing Parameters
    parser.add_argument("--smoothing_window", type=int, default=60,
                        help="Size of the rolling average window (1 = no smoothing)")
    parser.add_argument("--suppression_power", type=float, default=2.0,
                        help="Exponent for non-linear noise suppression (1.0 = no suppression)")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Temperature scaling factor for logits (1.0 = raw logits, >1.0 = softened)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of videos processed for quick testing")

    args = parser.parse_args()

    cctv_model, vlm_processor = load_cctv_model(
        adapter_path=args.adapter_path,
        use_quantization=not args.no_quantize
    )

    run_evaluation(cctv_model, vlm_processor, args)