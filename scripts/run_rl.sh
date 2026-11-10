#!/usr/bin/env bash
set -euo pipefail
: "${UPSTREAM_DIR:?Set UPSTREAM_DIR to the MiniOneRec checkout}"
: "${DATA_DIR:?Set DATA_DIR to the prepared MIND export}"
: "${SFT_OUT:?Set SFT_OUT to the SFT checkpoint directory}"
: "${RL_OUT:?Set RL_OUT to the RL checkpoint output directory}"
NUM_GPUS="${NUM_GPUS:-4}"
cd "$UPSTREAM_DIR"
accelerate launch --config_file config/zero2_opt.yaml --num_processes "$NUM_GPUS" rl.py \
  --model_path "$SFT_OUT" \
  --train_file "$DATA_DIR/train/MIND.csv" \
  --eval_file "$DATA_DIR/valid/MIND.csv" \
  --info_file "$DATA_DIR/info/MIND.txt" \
  --sid_index_path "$DATA_DIR/MIND.index.json" \
  --item_meta_path "$DATA_DIR/MIND.item.json" \
  --category MIND \
  --output_dir "$RL_OUT" \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --reward_type ranking \
  --sample "${RL_SAMPLE:-10000}" \
  --test_during_training False \
  --seed 42
