#!/usr/bin/env bash
set -euo pipefail
: "${UPSTREAM_DIR:?Set UPSTREAM_DIR to the MiniOneRec checkout}"
: "${DATA_DIR:?Set DATA_DIR to the prepared MIND export}"
: "${BASE_MODEL:?Set BASE_MODEL to the base language model path or ID}"
: "${SFT_OUT:?Set SFT_OUT to the checkpoint output directory}"
NUM_GPUS="${NUM_GPUS:-4}"
cd "$UPSTREAM_DIR"
torchrun --nproc_per_node "$NUM_GPUS" sft.py \
  --base_model "$BASE_MODEL" \
  --train_file "$DATA_DIR/train/MIND.csv" \
  --eval_file "$DATA_DIR/valid/MIND.csv" \
  --output_dir "$SFT_OUT" \
  --sid_index_path "$DATA_DIR/MIND.index.json" \
  --item_meta_path "$DATA_DIR/MIND.item.json" \
  --category MIND \
  --batch_size "${BATCH_SIZE:-128}" \
  --micro_batch_size "${MICRO_BATCH_SIZE:-4}" \
  --num_epochs "${SFT_EPOCHS:-3}" \
  --seed 42
