# Trilobyte Lossless Codec

The Trilobyte Lossless Codec (TLC) encodes and decodes audio losslessly to and from **Trilobyte Lossless Codec (.tlc)** files using a neural model and arithmetic coding.

Research and experiments related to Trilobyte are available at [pnlong/trilobyte-experiments](https://github.com/pnlong/trilobyte-experiments).

---

## Setup

**1. Set up the environment** — from the project root:

```bash
bash setup.sh
```

Use `bash setup.sh --help` to see options. This creates a virtual environment in `.venv` (if needed), upgrades pip, and installs dependencies from `requirements.txt`. Pass **`--dev`** to also install dev requirements (e.g. pytest) from `requirements-dev.txt`.

**2. Download the model weights** — default location is **`model.ckpt`** in the project root:

```bash
bash download_model.sh
```

To save the checkpoint elsewhere, pass the output path:

```bash
bash download_model.sh /path/to/model.ckpt
```

Use `bash download_model.sh --help` to see options. The download script requires the virtual environment from step 1 (it uses `gdown` from `.venv`). If you download the model to a path other than **`model.ckpt`** in the project root, see **Model path and bit depth** below for how to point the CLI or code at your checkpoint.

**Alternatively: set up yourself**

- **Environment** — create and activate a virtual environment, then install dependencies:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # on Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

  For dev (e.g. pytest): `pip install -r requirements-dev.txt`

- **Model weights** — download the checkpoint yourself and save it as **`model.ckpt`** (or another path and use **--modelpath**). Weights link:

  **https://drive.google.com/uc?id=1TjGG7NtAb-FdxoF1Cq04t2u0mqeB9q5W**

  From the command line you can use `gdown` (e.g. after `pip install gdown`):

  ```bash
  gdown "https://drive.google.com/uc?id=1TjGG7NtAb-FdxoF1Cq04t2u0mqeB9q5W" -O model.ckpt
  ```

After setup, activate the environment and run the CLI:

```bash
source .venv/bin/activate
python tlc.py --help
```

---

## Project layout and Python files

**`tlc.py`** is the main entry point and command-line interface. It parses arguments, loads the model, and calls the encode or decode pipeline. The rest of the logic is split across other modules for clarity and maintainability.

- **tlc.py** — Main entry point and CLI: argument parsing, logging, model loading, and dispatching to encode or decode.
- **encode.py** — Encoding pipeline: reads raw audio, converts to tokens with the model, writes a compressed file (header + arithmetic-coded blocks).
- **decode.py** — Decoding pipeline: reads a compressed file, decodes blocks with the arithmetic coder, converts tokens back to samples, writes raw audio.
- **utils/audio.py** — Audio I/O: `load_audio` and `save_audio` for reading/writing signed integer PCM.
- **utils/constants.py** — Shared constants (block size, batch size, model path, bit depth, file extension, arithmetic coder settings, etc.).
- **utils/model.py** — Model loading and token conversion: `load_model`, `convert_waveform_to_tokens`, `convert_tokens_to_waveform`.
- **utils/header.py** — .tlc file header: `encode_header` and `decode_header` for the binary header.
- **utils/arithmetic_coder.py** — Arithmetic coding: `encode_arithmetic` and `decode_arithmetic` for compressing and decompressing symbol streams.

---

## Command-line interface (`tlc.py`)

**Usage:**

```text
python tlc.py INFILE [OPTIONS]
```

**Positional argument:**

- **infile** — Path to the input file (raw audio when encoding, .tlc when decoding).

**General options:**

- **-o**, **--outfile** — Path to the output file. If omitted, the output is the input base name with extension `.tlc` in the current directory.
- **--modelpath** — Path to the Trilobyte model checkpoint. If used, **--modeldepth** must also be set.
- **--modeldepth** — Model bit depth (e.g. 24). If used, **--modelpath** must also be set.
- **--gpu** — Use GPU if available.
- **-s**, **--silent** — Do not print runtime encode/decode statistics to stderr.
- **-v**, **--verbose** — Print detailed runtime statistics to stderr.

**Mode:**

- **-d**, **--decode** — Decode a .tlc file to raw audio. Default is to encode.

**Encoding-only options (ignored when decoding):**

- **-b**, **--blocksize** — Block size (in samples) (default from constants).

**Examples:**

```bash
# Encode raw audio to recording.tlc (output path inferred)
python tlc.py recording.wav

# Encode with custom output path and GPU
python tlc.py recording.wav -o out/recording.tlc --gpu

# Decode .tlc back to raw audio
python tlc.py recording.tlc -d -o decoded.wav
```

---

## Model path and bit depth

By default, the code looks for the Trilobyte checkpoint at **`model.ckpt`** in the project directory and assumes a **native bit depth of 24**. You can change where the model is stored or use different weights in three ways:

**1. Same bit depth, different location (symlink)**  
If the model’s bit depth is unchanged, you can keep the weights anywhere and point the project at them with a symlink named `model.ckpt` in this directory:

```bash
# From the project root (directory that contains setup.sh and tlc.py):
ln -sf /path/to/your/checkpoint.ckpt model.ckpt
```

Replace `/path/to/your/checkpoint.ckpt` with the real path. The code will then load `model.ckpt` (which points to your file) and use the default bit depth from constants.

**2. Command-line arguments**  
Pass the checkpoint path and bit depth explicitly:

```bash
python tlc.py INFILE --modelpath /path/to/checkpoint.ckpt --modeldepth 24
```

You must provide both **--modelpath** and **--modeldepth** together.

**3. Defaults in code**  
Edit **`utils/constants.py`** and set:

- **`MODEL_PATH`** – default path to the checkpoint (e.g. absolute path or path relative to the project).
- **`MODEL_BIT_DEPTH`** – default model bit depth (e.g. `24`).

These defaults are used when you do not pass **--modelpath** or **--modeldepth** to `tlc.py`.
