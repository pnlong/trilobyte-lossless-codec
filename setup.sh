#!/bin/bash
# set up a pip venv and install dependencies for trilobyte_lossless_codec.
# run from the project root: ./setup.sh

set -e
cd "$(dirname "$0")"

# show usage and exit
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "usage: $0 [OPTIONS]"
    echo ""
    echo "  Set up a pip venv and install dependencies. Run from the project root."
    echo ""
    echo "options:"
    echo "  --dev    also install dev requirements (e.g. pytest)"
    echo "  -h, --help   show this help"
    exit 0
fi

# parse --dev flag (install dev requirements e.g. pytest)
INSTALL_DEV=false
if [ "$1" = "--dev" ]; then
    INSTALL_DEV=true
fi

# create venv if it does not exist
if [ ! -d ".venv" ]; then
    echo "creating virtual environment at .venv ..."
    python3 -m venv .venv
fi

# install dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
if [ "$INSTALL_DEV" = true ]; then
    .venv/bin/pip install -r requirements-dev.txt
fi

echo "setup complete. activate the environment with: source .venv/bin/activate"
echo "then download the model weights with: bash download_model.sh [OUTPUT_PATH]  (default: model.ckpt)"
if [ "$INSTALL_DEV" = true ]; then
    echo "then run e.g.: python -m pytest tests/ -v"
fi
