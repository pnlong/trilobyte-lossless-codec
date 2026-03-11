# README
# Compress

# Implements a lossless compressor for encoding 
# audio data into Trilobyte Lossless Codec (.tlc) 
# files.

# IMPORTS
##################################################

# standard library
import logging
from math import ceil, log
from typing import BinaryIO
from tqdm import tqdm  # progress bar
import torch
import torch.nn.functional as F

# utils
from utils.arithmetic_coder import (
    Encoder,
    normalize_pdf_for_arithmetic_coding,
    write_bits_to_stream,
)
from utils.audio import (
    load_waveform,
    convert_waveform_to_unsigned_integers,
)
from utils.constants import (
    ARITHMETIC_CODER_BASE,
    ARITHMETIC_CODER_BOS,
    ARITHMETIC_CODER_PRECISION,
    BLOCK_SIZE,
    BATCH_SIZE,
    MODEL_BIT_DEPTH,
)
from utils.header import encode_header
from utils.model import (
    GPTAudioLightningModule,
    convert_waveform_to_tokens,
    get_vocab_size,
)

# logging
logger = logging.getLogger(__name__)

##################################################


# ENCODE
##################################################

def encode(
    waveform: torch.Tensor,
    stream: BinaryIO,
    model: GPTAudioLightningModule,
    sample_rate: int,
    bit_depth: int,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    silent: bool = False,
) -> None:
    """
    Encode waveform to compressed stream.

    Args:
        waveform: Waveform tensor, shape (num_channels, num_samples) or (num_samples,).
            Expected floating point sample values in range [-1, 1].
        stream: Binary output stream (e.g. file opened in "wb", or BytesIO).
        model: Trained GPTAudioLightningModule.
        sample_rate: Sample rate in Hz.
        bit_depth: Audio bit depth (e.g. 8, 16, 24).
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        block_size: Block size (in samples). Defaults to BLOCK_SIZE.
        batch_size: Batch size. Defaults to BATCH_SIZE.
        silent: If True, do not show progress bar. Defaults to False.
    """

    # deterministic PDFs for same context (no dropout)
    model.eval()

    # parse waveform shape
    num_channels = waveform.shape[0] if waveform.dim() > 1 else 1
    num_samples = waveform.shape[1] if waveform.dim() > 1 else waveform.shape[0]

    # convert signed integers to unsigned integers
    waveform = convert_waveform_to_unsigned_integers(
        waveform = waveform,
        bit_depth = bit_depth,
    ) # waveform is now in range [0, 2**bit_depth - 1]

    # convert waveform to tokens
    tokens = convert_waveform_to_tokens(
        waveform = waveform,
        bit_depth = bit_depth,
        model_bit_depth = model_bit_depth,
    )

    # encode header to stream
    encode_header(
        header = {
            "block_size": block_size,
            "num_samples": num_samples,
            "num_channels": num_channels,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
        },
        stream = stream,
    )

    # encode blocks to stream
    encode_blocks(
        tokens = tokens,
        model = model,
        stream = stream,
        bit_depth = bit_depth,
        model_bit_depth = model_bit_depth,
        block_size = block_size,
        batch_size = batch_size,
        silent = silent,
    )


def encode_blocks(
    tokens: torch.Tensor,
    model: GPTAudioLightningModule,
    stream: BinaryIO,
    bit_depth: int,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    silent: bool = False,
) -> None:
    """
    Encode token sequence in blocks using the model and arithmetic coding.

    Args:
        tokens: Flat token sequence.
        model: Trained GPTAudioLightningModule.
        stream: Binary output stream.
        bit_depth: Audio bit depth.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        block_size: Block size (in samples). Defaults to BLOCK_SIZE.
        batch_size: Batch size. Defaults to BATCH_SIZE.
        silent: If True, do not show progress bar. Defaults to False.
    """

    # get info
    num_tokens = tokens.shape[0]
    bytes_per_sample = int(ceil(model_bit_depth / 8))
    effective_block_size = block_size * bytes_per_sample
    num_tokens_in_last_block = (num_tokens % effective_block_size) if num_tokens % effective_block_size != 0 else effective_block_size

    # pad token sequence to be divisible by block size
    if num_tokens_in_last_block != effective_block_size:
        tokens = torch.cat([
            tokens,
            torch.zeros(effective_block_size - num_tokens_in_last_block, dtype = tokens.dtype),
        ], dim = 0)
    tokens = tokens.to(model.device) # move tokens to model device

    # partition token sequence into blocks
    num_blocks = int(tokens.shape[0] // effective_block_size)
    blocks = tokens.reshape(num_blocks, effective_block_size)

    # group blocks into batches
    num_blocks_in_last_batch = num_blocks % batch_size if num_blocks % batch_size != 0 else batch_size
    num_batches = int(ceil(num_blocks / batch_size))
    logger.debug(
        "encode_blocks entry: num_tokens=%s num_blocks=%s effective_block_size=%s batch_size=%s num_blocks_in_last_batch=%s num_batches=%s",
        num_tokens, num_blocks, effective_block_size, batch_size, num_blocks_in_last_batch, num_batches,
    )

    # create dummy tokens
    dummy_tokens = convert_waveform_to_tokens(
        waveform = torch.full(
            (1, 1), # shape (num_channels = 1, num_samples = 1)
            ARITHMETIC_CODER_BOS,
            dtype = tokens.dtype,
            device = tokens.device,
        ),
        bit_depth = bit_depth,
        model_bit_depth = model_bit_depth,
    ) # shape (model_bit_depth // 8,)
    num_dummy_tokens = dummy_tokens.shape[0]

    # encode blocks in batches
    batch_range = range(num_batches)
    if not silent:
        batch_range = tqdm(
            iterable = batch_range,
            desc = "Encoding batches",
        )
    for batch_idx in batch_range:

        # get current batch of blocks
        is_last_batch = (batch_idx == num_batches - 1) # determine if current batch is the last batch
        current_batch_size = num_blocks_in_last_batch if is_last_batch else batch_size
        current_batch_start_idx = batch_idx * batch_size
        current_batch_end_idx = current_batch_start_idx + current_batch_size
        batch = blocks[current_batch_start_idx:current_batch_end_idx, :] # shape (current_batch_size, effective_block_size)
        batch_with_dummy = torch.cat([
            dummy_tokens.unsqueeze(dim = 0).repeat(current_batch_size, 1), # shape (current_batch_size, num_dummy_tokens)
            batch,
        ], dim = -1) # shape (current_batch_size, num_dummy_tokens + effective_block_size)

        # iteratively pass block through model
        logits_batch = torch.zeros(
            (current_batch_size, effective_block_size, get_vocab_size(model_bit_depth = model_bit_depth)),
            dtype = torch.float32,
            device = model.device,
        )
        for i in range(effective_block_size):
            with torch.no_grad():
                logits = model(
                    batch_with_dummy[:, :i + num_dummy_tokens] # input of shape (current_batch_size, i + num_dummy_tokens)
                ).logits # logits of shape (current_batch_size, i + num_dummy_tokens, vocab_size)
            logits_batch[:, i, :] = logits[:, -1, :] # shape (current_batch_size, vocab_size)
        logger.debug(
            "encode_blocks: effective_block_size = %s logits_batch.shape = %s", effective_block_size, logits_batch.shape,
        )

        # get probability density functions
        pdfs_batch = torch.softmax(logits_batch, dim = -1) # shape (current_batch_size, effective_block_size, vocab_size)

        # use arithmetic coding to encode each block in the batch
        for batch_block_idx in range(current_batch_size):

            # get current block
            block_idx = current_batch_start_idx + batch_block_idx
            is_last_block = (block_idx == num_blocks - 1)
            current_block_size = num_tokens_in_last_block if is_last_block else effective_block_size
            block = batch[batch_block_idx, :current_block_size] # shape (current_block_size,)
            logits_block = logits_batch[batch_block_idx, :current_block_size, :] # shape (current_block_size, vocab_size)
            pdfs_block = pdfs_batch[batch_block_idx, :current_block_size, :] # shape (current_block_size, vocab_size)

            # debug: cross entropy of logits vs actual tokens -> BPB -> compression rate
            cross_entropy = F.cross_entropy(
                input = logits_block, # shape (current_block_size, vocab_size)
                target = block, # shape (current_block_size,)
            ).item()
            bpb = cross_entropy / log(2)
            expected_compression_rate = 8.0 / bpb
            logger.debug(
                "encode_blocks: block %s cross_entropy=%.4f bpb=%.4f expected_compression_rate=%.4fx",
                block_idx, cross_entropy, bpb, expected_compression_rate,
            )

            # encode current batch of blocks
            logger.debug(
                "encode_blocks: encode block %s first_10_data_tokens=%s",
                block_idx, (block[:10] if len(block) > 10 else block).cpu().tolist(),
            )
            bits = []
            encoder = Encoder(
                base = ARITHMETIC_CODER_BASE,
                precision = ARITHMETIC_CODER_PRECISION,
                output_fn = bits.append,
            )
            for pdf, token in zip(
                pdfs_block.cpu().numpy(),
                block.cpu().numpy().tolist(),
            ):
                encoder.encode(
                    pdf = normalize_pdf_for_arithmetic_coding(pdf = pdf),
                    symbol = token,
                )
            encoder.terminate()
            write_bits_to_stream(
                bits = bits, 
                stream = stream,
            )

            # debug: calculate actual compression rate
            raw_block_size = (current_block_size / bytes_per_sample) * ceil(bit_depth / 8)
            compressed_block_size = ceil(len(bits) / 8)
            actual_compression_rate = raw_block_size / compressed_block_size
            logger.debug(
                "encode_blocks: block %s actual_compression_rate=%.4fx",
                block_idx, actual_compression_rate,
            )
            logger.debug(
                "encode_blocks: block %s actual_compression_rate=%.4fx expected_compression_rate=%.4fx",
                block_idx, actual_compression_rate, expected_compression_rate,
            )

    return

##################################################


# WRAPPER FOR FILE I/O
##################################################

def encode_wrapper(
    path_input: str,
    path_output: str,
    model: GPTAudioLightningModule,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    silent: bool = False,
) -> None:
    """
    Encode the input file.

    Args:
        path_input: Path to the input file (raw or WAV audio file).
        path_output: Path to the output file (compressed .tlc file).
        model: Trained GPTAudioLightningModule.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        block_size: Block size (in samples). Defaults to BLOCK_SIZE.
        batch_size: Batch size. Defaults to BATCH_SIZE.
        silent: If True, do not show progress to stderr. Defaults to False.
    """

    # read audio file
    waveform, sample_rate, bit_depth = load_waveform(path_input)

    # open output file
    with open(path_output, "wb") as stream:

        # encode to stream
        encode(
            waveform = waveform,
            stream = stream,
            model = model,
            sample_rate = sample_rate,
            bit_depth = bit_depth,
            model_bit_depth = model_bit_depth,
            block_size = block_size,
            batch_size = batch_size,
            silent = silent,
        )

    return

##################################################
