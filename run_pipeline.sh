#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

ENCODED_DIR="data/encoded"
OUTPUT_DIR="outputs"

echo "Running end-to-end pipeline on the first 300,000 raw reviews..."
echo "Encoded outputs: $ENCODED_DIR"
echo "Model and figure outputs: $OUTPUT_DIR"

"$PYTHON_BIN" project/execution/run_pipeline.py \
  --raw-json data/IMDB_reviews.json \
  --encoded-dir "$ENCODED_DIR" \
  --input-files \
    "$ENCODED_DIR/reviews_encoded_all-MiniLM-L6-v2.csv" \
    "$ENCODED_DIR/reviews_encoded_LaBSE.csv" \
    "$ENCODED_DIR/reviews_encoded_distilbert-base-multilingual-cased.csv" \
  --names MiniLM LaBSE DistilBERT \
  --output-dir "$OUTPUT_DIR" \
  --max-reviews 300000

echo
echo "Refreshing visualization outputs from generated tables..."

"$PYTHON_BIN" project/execution/run_visualizations.py \
  --output-dir "$OUTPUT_DIR" \
  --names MiniLM LaBSE DistilBERT

echo
echo "Complete."
echo "Review results under:"
echo "  - $ENCODED_DIR"
echo "  - $OUTPUT_DIR"
