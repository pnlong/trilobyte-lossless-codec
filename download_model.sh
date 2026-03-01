#!/bin/bash
# Download the Trilobyte model checkpoint.
# Run from the project root: ./download_model.sh [OUTPUT_PATH]

set -e
cd "$(dirname "$0")"

DEFAULT_OUTPUT="model.ckpt"
OUTPUT="${1:-$DEFAULT_OUTPUT}"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "usage: $0 [OUTPUT_PATH]"
    echo ""
    echo "  Download the Trilobyte model checkpoint. Run from the project root."
    echo ""
    echo "  OUTPUT_PATH  Where to save the checkpoint (default: model.ckpt in project root)."
    echo "  -h, --help   Show this help."
    exit 0
fi

MODEL_URL="https://drive.google.com/uc?id=1TjGG7NtAb-FdxoF1Cq04t2u0mqeB9q5W"

if [ ! -d ".venv" ]; then
    echo "error: virtual environment not found. Run: bash setup.sh"
    exit 1
fi

echo "downloading checkpoint to $OUTPUT ..."
.venv/bin/gdown "$MODEL_URL" -O "$OUTPUT"
echo "done."
