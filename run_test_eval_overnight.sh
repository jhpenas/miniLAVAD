#!/bin/bash

# ==========================================
# 1. Evaluate Vanilla Baseline on Test Set
# ==========================================
echo "=========================================================="
echo "Running Vanilla Baseline on Test Set..."
echo "=========================================================="
python src/evaluate.py \
  --test_json "data/processed/test_split.json" \
  --smoothing_window 630 \
  --suppression_power 1.0 \
  --output_file "models/test_base_eval_vanilla.json"

# ==========================================
# 2. Evaluate the Best Checkpoint (Epoch 4)
# ==========================================
BEST_RUN="8718c5ed001f4fe5bc2e57b2fa4f0af3"

echo "=========================================================="
echo "Running Best Fine-Tuned Model (Epoch 4) on Test Set..."
echo "=========================================================="
python src/evaluate.py \
  --run_id "$BEST_RUN" \
  --epoch "checkpoints_epoch_4" \
  --test_json "data/processed/test_split.json" \
  --smoothing_window 630 \
  --suppression_power 1.0 \
  --output_file "models/test_eval_8718c5_epoch4_final.json"

echo "=========================================================="
echo "ALL OVERNIGHT EVALUATIONS COMPLETE!"
echo "=========================================================="