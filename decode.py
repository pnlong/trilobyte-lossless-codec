# README
# Decompress

# Implements a lossless decompressor for decoding 
# audio data from Trilobyte Lossless Codec (.tlc) 
# files.

# IMPORTS
##################################################

# standard library
import logging
from typing import BinaryIO, Tuple
from tqdm import tqdm  # progress bar
import torch
from math import ceil

# utils
from utils.arithmetic_coder import (
    Decoder,
    normalize_pdf_for_arithmetic_coding,
    read_bits_from_stream,
)
from utils.audio import (
    save_waveform,
    convert_waveform_from_unsigned_integers,
)
from utils.constants import (
    ARITHMETIC_CODER_BASE,
    ARITHMETIC_CODER_BOS,
    ARITHMETIC_CODER_PRECISION,
    BLOCK_SIZE,
    MODEL_BIT_DEPTH,
)
from utils.header import decode_header
from utils.model import (
    convert_waveform_to_tokens,
    convert_tokens_to_waveform,
    GPTAudioLightningModule,
)

# logging
logger = logging.getLogger(__name__)

##################################################


# DECODE
##################################################

def decode(
    stream: BinaryIO,
    model: GPTAudioLightningModule,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    silent: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    """
    Decode compressed stream to waveform.

    Args:
        stream: Binary input stream (e.g. file opened in "rb", or BytesIO).
        model: Trained GPTAudioLightningModule.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        silent: If True, do not show progress bar. Defaults to False.

    Returns:
        Tuple of (waveform, sample_rate, bit_depth).
        waveform: torch.Tensor of shape (num_channels, num_samples), unsigned
            floating point sample values in range [-1, 1].
        sample_rate: Sample rate in Hz.
        bit_depth: Audio bit depth (e.g. 8, 16, 24).
    """

    # deterministic PDFs for same context (no dropout)
    model.eval()

    # decode header
    header = decode_header(stream = stream)

    # parse header
    block_size = header["block_size"]
    num_samples = header["num_samples"]
    num_channels = header["num_channels"]
    sample_rate = header["sample_rate"]
    bit_depth = header["bit_depth"]

    # decode blocks
    tokens = decode_blocks(
        stream = stream,
        model = model,
        num_samples = num_samples,
        num_channels = num_channels,
        bit_depth = bit_depth,
        model_bit_depth = model_bit_depth,
        block_size = block_size,
        silent = silent,
    )

    # convert tokens to waveform
    waveform = convert_tokens_to_waveform(
        tokens = tokens,
        num_channels = num_channels,
        bit_depth = bit_depth,
        model_bit_depth = model_bit_depth,
    )
    logger.debug(
        "after convert_tokens_to_waveform: waveform_shape=%s waveform_numel=%s num_samples_header=%s",
        list(waveform.shape), waveform.numel(), num_samples,
    )

    # convert unsigned integers waveform to signed integers
    waveform = convert_waveform_from_unsigned_integers(
        waveform = waveform,
        bit_depth = bit_depth,
    ) # waveform is now in range [-2**31, 2**31 - 1]

    return waveform, sample_rate, bit_depth


def decode_blocks(
    stream: BinaryIO,
    model: GPTAudioLightningModule,
    num_samples: int,
    num_channels: int,
    bit_depth: int,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    block_size: int = BLOCK_SIZE,
    silent: bool = False,
) -> torch.Tensor:
    """
    Decode token sequence in blocks from stream.

    Args:
        stream: Binary input stream.
        model: Trained GPTAudioLightningModule.
        num_samples: Number of samples per channel.
        num_channels: Number of channels (1 = mono, 2 = stereo).
        bit_depth: Audio bit depth.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        block_size: Block size (in samples). Defaults to BLOCK_SIZE.
        silent: If True, do not show progress bar. Defaults to False.

    Returns:
        Flat token tensor.
    """

    # get info
    bytes_per_sample = int(ceil(model_bit_depth / 8))
    num_tokens = num_samples * num_channels * bytes_per_sample
    effective_block_size = block_size * bytes_per_sample
    num_tokens_in_last_block = (num_tokens % effective_block_size) if num_tokens % effective_block_size != 0 else effective_block_size
    num_blocks = int(ceil(num_tokens / effective_block_size))
    logger.debug(
        "decode_blocks entry: num_samples=%s num_channels=%s bytes_per_sample=%s num_tokens=%s num_blocks=%s effective_block_size=%s",
        num_samples, num_channels, bytes_per_sample, num_tokens, num_blocks, effective_block_size,
    )

    # initialize tokens
    tokens = torch.zeros(
        num_tokens,
        dtype = torch.int64,
        device = model.device,
    )

    # create dummy tokens
    dummy_tokens = convert_waveform_to_tokens(
        waveform = torch.full(
            (1, 1), # shape (num_channels = 1, num_samples = 1)
            ARITHMETIC_CODER_BOS,
            dtype = tokens.dtype,
            device = model.device,
        ),
        bit_depth = bit_depth,
        model_bit_depth = model_bit_depth,
    ) # shape (model_bit_depth // 8,)
    num_dummy_tokens = len(dummy_tokens)

    # decode blocks
    block_range = range(num_blocks)
    if not silent:
        block_range = tqdm(
            iterable = block_range,
            desc = "Decoding blocks",
        )
    for block_idx in block_range:            

        # define input function for decoder
        bits = read_bits_from_stream(stream = stream)
        it = iter(bits)
        def input_fn():
            try:
                return next(it)
            except StopIteration:
                return None
        decoder = Decoder(
            base = ARITHMETIC_CODER_BASE,
            precision = ARITHMETIC_CODER_PRECISION,
            input_fn = input_fn,
        )

        # decode block
        is_last_block = (block_idx == num_blocks - 1) # determine if current block is the last block
        current_block_size = num_tokens_in_last_block if is_last_block else effective_block_size
        block_with_dummy = torch.cat([
            dummy_tokens, 
            torch.zeros(
                (current_block_size,), # shape (current_block_size,)
                dtype = tokens.dtype,
                device = model.device,
            ),
        ], dim = 0) # shape (current_block_size + num_dummy_tokens,)
        for i in range(current_block_size):
            current_token_idx = i + num_dummy_tokens
            with torch.no_grad():
                logits = model(
                    block_with_dummy[:current_token_idx].unsqueeze(dim = 0) # input of shape (batch_size = 1, i + num_dummy_tokens)
                ).logits # logits of shape (batch_size = 1, i + 1, vocab_size)
            pdf = torch.softmax(logits[0, -1, :], dim = -1).cpu().numpy() # (vocab_size,)
            token = decoder.decode(
                pdf = normalize_pdf_for_arithmetic_coding(pdf = pdf),
            )
            block_with_dummy[current_token_idx] = token

        # add block tokens to tokens
        block_start_idx = block_idx * effective_block_size
        tokens[block_start_idx:(block_start_idx + current_block_size)] = block_with_dummy[num_dummy_tokens:] # shape (current_block_size,), exclude dummy tokens
        logger.debug(
            "decode_blocks: decode block %s first_10_tokens=%s",
            block_idx, tokens[block_start_idx:(block_start_idx + 10)],
        )

    # detach tokens from computation graph
    tokens = tokens.detach()

    return tokens

##################################################


# WRAPPER FOR FILE I/O
##################################################

def decode_wrapper(
    path_input: str,
    path_output: str,
    model: GPTAudioLightningModule,
    model_bit_depth: int = MODEL_BIT_DEPTH,
    silent: bool = False,
) -> None:
    """
    Decode the input file.

    Args:
        path_input: Path to the input file (compressed .tlc file).
        path_output: Path to the output file (raw audio file).
        model: Trained GPTAudioLightningModule.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.
        silent: If True, do not show progress to stderr. Defaults to False.
    """

    # open input file
    with open(path_input, "rb") as stream:
        waveform, sample_rate, bit_depth = decode(
            stream = stream,
            model = model,
            model_bit_depth = model_bit_depth,
            silent = silent,
        )

    # save audio
    save_waveform(
        path = path_output,
        waveform = waveform,
        sample_rate = sample_rate,
        bit_depth = bit_depth,
    )

    return

##################################################
