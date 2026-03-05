# README
# Compress

# Implements a lossless compressor for encoding 
# audio data into Trilobyte Lossless Codec (.tlc) 
# files.

# IMPORTS
##################################################

# standard library
import logging
import math
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
        block_size: Block size. Defaults to BLOCK_SIZE.
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
        block_size = block_size,
        model_bit_depth = model_bit_depth,
        silent = silent,
    )


def encode_blocks(
    tokens: torch.Tensor,
    model: GPTAudioLightningModule,
    stream: BinaryIO,
    block_size: int,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    silent: bool = False,
) -> None:
    """
    Encode token sequence in blocks using the model and arithmetic coding.

    Args:
        tokens: Flat token sequence.
        model: Trained GPTAudioLightningModule.
        stream: Binary output stream.
        block_size: Block size.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        silent: If True, do not show progress bar. Defaults to False.
    """

    # get info
    num_tokens = tokens.shape[0]
    num_tokens_in_last_block = (num_tokens % block_size) if num_tokens % block_size != 0 else block_size

    # pad token sequence to be divisible by block size
    if num_tokens_in_last_block != block_size:
        tokens = torch.cat([
            tokens,
            torch.zeros(block_size - num_tokens_in_last_block, dtype = tokens.dtype),
        ], dim = 0)
    tokens = tokens.to(model.device) # move tokens to model device

    # partition token sequence into blocks
    num_blocks = int(tokens.shape[0] // block_size)
    blocks = tokens.reshape(num_blocks, block_size)
    logger.debug(
        "encode_blocks entry: num_tokens=%s num_blocks=%s block_size=%s",
        num_tokens, num_blocks, block_size,
    )

    # encode blocks in batches
    block_range = range(num_blocks)
    if not silent:
        block_range = tqdm(
            iterable = block_range,
            desc = "Encoding blocks",
        )
    for block_idx in block_range:

        # get current batch of blocks
        is_last_block = (block_idx == num_blocks - 1) # determine if current block is the last block
        current_block_size = num_tokens_in_last_block if is_last_block else block_size
        block = blocks[block_idx, :current_block_size] # shape (current_block_size,)
        dummy_token = torch.full(
            (1,),
            ARITHMETIC_CODER_BOS,
            dtype = block.dtype,
            device = model.device,
        )

        # get pdfs for current batch of blocks: iteratively pass blocks_batch[:, :i]
        vocab_size = get_vocab_size(model_bit_depth = model_bit_depth)
        pdfs = torch.zeros(
            (current_block_size, vocab_size), # shape (current_block_size, vocab_size)
            dtype = torch.float32,
            device = model.device,
        )
        logits_block = torch.zeros(
            (current_block_size, vocab_size),
            dtype = torch.float32,
            device = model.device,
        )
        for i in range(current_block_size):
            with torch.no_grad():
                logits = model(
                    torch.cat((block[:i], dummy_token), dim = 0).unsqueeze(dim = 0) # input of shape (batch_size = 1, i + 1)
                ).logits # logits of shape (batch_size = 1, i + 1, vocab_size)
            logits = logits[0, -1, :] # shape (vocab_size,)
            logits_block[i, :] = logits # shape (vocab_size,)
            pdf = torch.softmax(logits, dim = -1) # shape (vocab_size,)
            pdfs[i, :] = pdf
        logger.debug(
            "encode_blocks: block_size = %s pdfs.shape = %s", block_size, pdfs.shape,
        )

        # debug: cross entropy of logits vs actual tokens -> BPB -> compression rate
        cross_entropy = F.cross_entropy(
            input = logits_block, # shape (current_block_size, vocab_size)
            target = block, # shape (current_block_size,)
        ).item()
        logger.debug(
            "encode_blocks: block %s cross_entropy=%.4f bpb=%.4f compression_rate=%.4fx",
            block_idx, cross_entropy, cross_entropy / math.log(2), 8.0 * (math.log(2) / cross_entropy),
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
            pdfs.cpu().numpy(),
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
    silent: bool = False,
) -> None:
    """
    Encode the input file.

    Args:
        path_input: Path to the input file (raw or WAV audio file).
        path_output: Path to the output file (compressed .tlc file).
        model: Trained GPTAudioLightningModule.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        block_size: Block size. Defaults to BLOCK_SIZE.
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
            silent = silent,
        )

    return

##################################################
