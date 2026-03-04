# CLI compression-rate test
#
# Not pytest: run as `python tests/test_cli.py AUDIO_FILE [OPTIONS]`.
# Loads an audio file, extracts num_chunks random segments of chunk_size seconds
# (skipping silence via RMS filtering), runs tlc.py encode on each chunk sequentially,
# and reports compression rate. Uses progress bars (tqdm) and logging (stderr).

# IMPORTS
##################################################

# standard library
import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from os.path import abspath, dirname, getsize, join, splitext
import torch
from tqdm import tqdm

# ensure project root is on path when run as python tests/test_cli.py
_script_dir = dirname(abspath(__file__))
_project_root = dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.audio import (
    load_waveform,
    save_waveform,
)

logger = logging.getLogger(__name__)

##################################################


# HELPERS
##################################################


def _run_one(wav_path, project_root, tlc_script):
    """
    Encode a single chunk WAV with tlc.py via subprocess.

    Returns the compression rate (original size / compressed size) for that chunk.
    """
    tlc_path = splitext(wav_path)[0] + ".tlc"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    subprocess.run(
        [sys.executable, tlc_script, wav_path, "-o", tlc_path],
        cwd=project_root,
        check=True,
        env=env,
    )
    return getsize(wav_path) / getsize(tlc_path)

##################################################


# MAIN
##################################################

def main():

    # ARGUMENT PARSING AND VALIDATION
    ##################################################

    parser = argparse.ArgumentParser(
        description="Test TLC compression rate on random chunks of an audio file."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the input audio file (e.g. WAV).",
    )
    parser.add_argument(
        "--chunk_size",
        type=float,
        default=5.0,
        help="Chunk duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=5,
        help="Number of random chunks to extract and compress (default: 5).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write chunk WAVs and .tlc outputs (num_chunks * 2 files). If omitted, use a temp dir (discarded after).",
    )
    args = parser.parse_args()

    filepath = abspath(args.filepath)
    if not os.path.isfile(filepath):
        logger.error("Input file not found: %s", filepath)
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    project_root = _project_root
    tlc_script = join(project_root, "tlc.py")
    if not os.path.isfile(tlc_script):
        logger.error("tlc.py not found at %s", tlc_script)
        print(f"Error: tlc.py not found at {tlc_script}", file=sys.stderr)
        sys.exit(1)

    logger.info("Input: %s  chunk_size=%.1fs  num_chunks=%d", filepath, args.chunk_size, args.num_chunks)

    ##################################################


    # LOAD AUDIO AND COMPUTE CHUNK LAYOUT
    ##################################################

    logger.info("Loading audio...")
    waveform, sample_rate, bit_depth = load_waveform(filepath)
    num_samples = waveform.shape[1] if waveform.dim() > 1 else waveform.shape[0]
    num_channels = waveform.shape[0] if waveform.dim() > 1 else 1
    duration_s = num_samples / sample_rate
    samples_per_chunk = int(args.chunk_size * sample_rate)

    if samples_per_chunk <= 0:
        logger.error("chunk_size too small for sample rate %d", sample_rate)
        print("Error: chunk_size too small for this sample rate.", file=sys.stderr)
        sys.exit(1)
    max_start = num_samples - samples_per_chunk
    if max_start < 0:
        logger.error("File shorter than one chunk: duration=%.2fs, chunk=%.2fs", duration_s, args.chunk_size)
        print(
            f"Error: file duration {duration_s:.2f}s is shorter than one chunk ({args.chunk_size}s).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.num_chunks > max_start + 1:
        logger.warning("Requested %d chunks but only %d non-overlapping possible", args.num_chunks, max_start + 1)
        print(
            f"Warning: only {max_start + 1} non-overlapping chunks possible; using that many.",
            file=sys.stderr,
        )
        num_chunks = max_start + 1
    else:
        num_chunks = args.num_chunks

    logger.info(
        "Audio: %.2fs, %d channel(s), %d-bit, %d Hz → %d samples/chunk, %d candidate start(s)",
        duration_s, num_channels, bit_depth, sample_rate, samples_per_chunk, max_start + 1,
    )

    ##################################################


    # FILTER CHUNKS BY RMS SO WE SKIP SILENCE / NEAR-SILENCE
    ##################################################

    # Compute RMS for every possible chunk (vectorized): for each start index,
    # RMS = sqrt(mean(waveform[:, start:start+L]^2)). We use unfold to get all
    # windows at once, then mean and sqrt.
    squared = waveform.square()
    chunks_sq = squared.unfold(1, samples_per_chunk, 1)  # (C, num_windows, samples_per_chunk)
    rms_per_start = (chunks_sq.mean(dim=(0, 2)) + 1e-12).sqrt()  # (num_windows,)
    # Keep only start indices whose chunk RMS is at least the 25th percentile,
    # so we drop the quietest quartile and avoid selecting silent regions.
    rms_threshold = float(torch.quantile(rms_per_start, 0.25))
    pool_starts = [s for s in range(max_start + 1) if rms_per_start[s].item() >= rms_threshold]

    if len(pool_starts) < num_chunks:
        logger.warning("Only %d chunks above RMS threshold (using all)", len(pool_starts))
        print(
            f"Warning: only {len(pool_starts)} chunks above RMS threshold; using all of them.",
            file=sys.stderr,
        )
        num_chunks = len(pool_starts)
    if num_chunks == 0:
        logger.error("No chunks above RMS threshold (file may be silence)")
        print("Error: no chunks above RMS threshold (file may be silence).", file=sys.stderr)
        sys.exit(1)

    random.shuffle(pool_starts)
    starts = pool_starts[:num_chunks]
    logger.info("Selected %d random chunk start(s) above RMS threshold (%.2e)", num_chunks, rms_threshold)

    ##################################################


    # CHOOSE WORK DIRECTORY (PERSISTENT OUTPUT_DIR OR TEMP DIR)
    ##################################################

    if args.output_dir is not None:
        outdir = abspath(args.output_dir)
        os.makedirs(outdir, exist_ok=True)
        workdir = outdir
        cleanup = False
        logger.info("Writing chunk WAVs and .tlc files to %s", outdir)
    else:
        workdir = tempfile.mkdtemp(prefix="tlc_test_")
        cleanup = True
        logger.debug("Using temp workdir: %s", workdir)

    try:

        # Extract and write chunk WAVs
        wav_paths = []
        for i, start in enumerate(tqdm(starts, desc="Writing chunk WAVs", unit="chunk")):
            chunk = waveform[:, start : start + samples_per_chunk]
            wav_path = join(workdir, f"chunk_{i}.wav")
            save_waveform(wav_path, chunk, sample_rate, bit_depth)
            wav_paths.append(wav_path)
        logger.info("Wrote %d chunk WAV(s) to %s", len(wav_paths), workdir)

        # Encode each chunk with tlc.py (sequential subprocesses)
        rates = []
        for wav_path in tqdm(wav_paths, desc="Encoding chunks (tlc.py)", unit="chunk"):
            rates.append(_run_one(wav_path, project_root, tlc_script))
        logger.info("Encoded %d chunk(s); compression rates collected", len(rates))

    finally:
        if cleanup and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
            logger.debug("Removed temp dir %s", workdir)

    ##################################################


    # REPORT COMPRESSION RATE SUMMARY
    ##################################################

    mean_rate = sum(rates) / len(rates)
    min_rate = min(rates)
    max_rate = max(rates)
    logger.info("Compression rate: mean=%.2fx min=%.2fx max=%.2fx", mean_rate, min_rate, max_rate)
    print(f"Chunks: {num_chunks}")
    print(f"Compression rate: mean={mean_rate:.2f}x  min={min_rate:.2f}x  max={max_rate:.2f}x")
    if args.output_dir is not None:
        print(f"Wrote {num_chunks * 2} files to {outdir}")
    return 0

    ##################################################

##################################################


# MAIN
##################################################

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s\t%(name)s: %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    sys.exit(main())

##################################################
