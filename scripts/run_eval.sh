#!/usr/bin/env bash
set -euo pipefail
: "${UPSTREAM_DIR:?Set UPSTREAM_DIR to the MiniOneRec checkout}"
: "${DATA_DIR:?Set DATA_DIR to the prepared MIND export}"
: "${MODEL_DIR:?Set MODEL_DIR to the SFT or RL checkpoint directory}"
SPLIT="${SPLIT:-test}"
RESULT_JSON="${RESULT_JSON:-$DATA_DIR/$SPLIT/MIND.predictions.json}"
cd "$UPSTREAM_DIR"
python evaluate.py \
  --base_model "$MODEL_DIR" \
  --info_file "$DATA_DIR/info/MIND.txt" \
  --category MIND \
  --test_data_path "$DATA_DIR/$SPLIT/MIND.csv" \
  --result_json_data "$RESULT_JSON" \
  --batch_size "${EVAL_BATCH_SIZE:-4}" \
  --num_beams "${NUM_BEAMS:-50}" \
  --max_new_tokens 64
