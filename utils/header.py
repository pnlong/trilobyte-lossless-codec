# README
# Header

# Helper that implements logic for the header of 
# Trilobyte Lossless Codec (.tlc) files.

# IMPORTS
##################################################

# standard library
import logging
from typing import Dict, Any, BinaryIO
from math import log2

# utils
from utils.constants import (
    BLOCK_SIZE_BITS,
    NUM_SAMPLES_BITS,
    NUM_CHANNELS_BITS,
    SAMPLE_RATE_BITS,
    BIT_DEPTH_BITS,
    TOTAL_HEADER_BYTES,
)

# logging
logger = logging.getLogger(__name__) # get logger for the current module

##################################################


# HELPER FUNCTIONS
##################################################

def verify_power_of_two(
    n: int,
) -> bool:
    """
    Verify if a value is a power of two.

    Args:
        n: The value to verify.

    Returns:
        True if the value is a power of two, False otherwise.
    """
    if n <= 0:
        return False
    else:
        return log2(n).is_integer()

##################################################


# ENCODE HEADER
##################################################

def encode_header(
    header: Dict[str, Any],
    stream: BinaryIO,
) -> None:
    """
    Encode the header. Write the header to the output stream.

    Args:
        header: Header dictionary.
        stream: BinaryIO object.
    """

    # extract variables from header dictionary
    block_size = header["block_size"]
    num_samples = header["num_samples"]
    num_channels = header["num_channels"]
    sample_rate = header["sample_rate"]
    bit_depth = header["bit_depth"]

    # verify that block size and batch size are powers of two
    assert verify_power_of_two(n = block_size), "Block size must be a power of two"
    assert num_channels == 1 or num_channels == 2, "Number of channels must be 1 (mono) or 2 (stereo)"
    assert bit_depth % 8 == 0, "Bit depth must be a multiple of 8"
    
    # determine encoded values for header
    block_size_encoded = int(log2(block_size))
    num_samples_encoded = num_samples
    num_channels_encoded = num_channels - 1
    sample_rate_encoded = sample_rate
    bit_depth_encoded = int(bit_depth // 8)

    # assertions that the encoded values are within the allowed range
    assert block_size_encoded < (2 ** BLOCK_SIZE_BITS), "Block size must be less than 2 ** BLOCK_SIZE_BITS, configure BLOCK_SIZE_BITS in constants.py if you want to support larger block sizes"
    assert num_samples_encoded < (2 ** NUM_SAMPLES_BITS), "Number of samples must be less than 2 ** NUM_SAMPLES_BITS, configure NUM_SAMPLES_BITS in constants.py if you want to support larger number of samples"
    assert num_channels_encoded < (2 ** NUM_CHANNELS_BITS), "Number of channels must be less than 2 ** NUM_CHANNELS_BITS, configure NUM_CHANNELS_BITS in constants.py if you want to support larger number of channels"
    assert sample_rate_encoded < (2 ** SAMPLE_RATE_BITS), "Sample rate must be less than 2 ** SAMPLE_RATE_BITS, configure SAMPLE_RATE_BITS in constants.py if you want to support larger sample rates"
    assert bit_depth_encoded < (2 ** BIT_DEPTH_BITS), "Bit depth must be less than 2 ** BIT_DEPTH_BITS, configure BIT_DEPTH_BITS in constants.py if you want to support larger bit depths"

    # pack each value into its allocated bits and combine into a single integer (lsb first)
    shift = 0
    packed = 0
    packed |= (block_size_encoded & ((1 << BLOCK_SIZE_BITS) - 1)) << shift
    shift += BLOCK_SIZE_BITS
    packed |= (num_samples_encoded & ((1 << NUM_SAMPLES_BITS) - 1)) << shift
    shift += NUM_SAMPLES_BITS
    packed |= (num_channels_encoded & ((1 << NUM_CHANNELS_BITS) - 1)) << shift
    shift += NUM_CHANNELS_BITS
    packed |= (sample_rate_encoded & ((1 << SAMPLE_RATE_BITS) - 1)) << shift
    shift += SAMPLE_RATE_BITS
    packed |= (bit_depth_encoded & ((1 << BIT_DEPTH_BITS) - 1)) << shift

    # write as bytes
    stream.write(packed.to_bytes(
        length = TOTAL_HEADER_BYTES,
        byteorder = "little",
    ))

    return

##################################################


# DECODE HEADER
##################################################

def decode_header(
    stream: BinaryIO,
) -> Dict[str, Any]:
    """
    Decode the header.

    Args:
        stream: BinaryIO object.

    Returns:
        Decoded header dictionary with keys block_size, num_samples,
        num_channels, sample_rate, bit_depth.
    """

    # read bytes of header from stream into a single integer
    packed = int.from_bytes(
        stream.read(TOTAL_HEADER_BYTES), byteorder = "little",
    )

    # extract each field (same order as encode: lsb first)
    shift = 0
    block_size_encoded = (packed >> shift) & ((1 << BLOCK_SIZE_BITS) - 1)
    shift += BLOCK_SIZE_BITS
    num_samples_encoded = (packed >> shift) & ((1 << NUM_SAMPLES_BITS) - 1)
    shift += NUM_SAMPLES_BITS
    num_channels_encoded = (packed >> shift) & ((1 << NUM_CHANNELS_BITS) - 1)
    shift += NUM_CHANNELS_BITS
    sample_rate_encoded = (packed >> shift) & ((1 << SAMPLE_RATE_BITS) - 1)
    shift += SAMPLE_RATE_BITS
    bit_depth_encoded = (packed >> shift) & ((1 << BIT_DEPTH_BITS) - 1)

    # recompute original variables
    block_size = 2 ** block_size_encoded
    num_samples = num_samples_encoded
    num_channels = num_channels_encoded + 1
    sample_rate = sample_rate_encoded
    bit_depth = bit_depth_encoded * 8

    # return decoded header dictionary
    return {
        "block_size": block_size,
        "num_samples": num_samples,
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
    }

##################################################