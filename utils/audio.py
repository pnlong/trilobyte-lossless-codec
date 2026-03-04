# README
# Audio

# Helper for loading and saving audio files as signed integer PCM.

# IMPORTS
##################################################

# third party
import numpy as np
import soundfile as sf
import torch
import logging

# standard library
from typing import Tuple

# utils
from utils.constants import (
    BIT_DEPTH_TO_SUBTYPE,
    SUBTYPE_TO_BIT_DEPTH,
    MAX_BIT_DEPTH,
)

# logging
logger = logging.getLogger(__name__)  # get logger for the current module

##################################################


# LOAD
##################################################

def load_waveform(
    path: str,
) -> Tuple[torch.Tensor, int, int]:
    """
    Load audio from file as floating point.

    Supports formats readable by soundfile (WAV, FLAC, etc.).
    Parses sample rate and bit depth from file metadata.

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of (waveform, sample_rate, bit_depth).
        waveform: torch.Tensor of shape (num_channels, num_samples), 
            int32 in range [-2**31, 2**31 - 1].
        sample_rate: Sample rate in Hz.
        bit_depth: Audio bit depth (e.g. 8, 16, 24).
    """

    # get info
    info = sf.info(path)
    sample_rate = info.samplerate
    subtype = info.subtype
    if subtype not in SUBTYPE_TO_BIT_DEPTH.keys():
        raise ValueError(
            f"subtype {subtype} not supported; use {SUBTYPE_TO_BIT_DEPTH.keys()}"
        )
    bit_depth = SUBTYPE_TO_BIT_DEPTH[subtype]
    if bit_depth > MAX_BIT_DEPTH:
        raise ValueError(
            f"bit_depth {bit_depth} is greater than maximum supported bit depth {MAX_BIT_DEPTH}. " \
            f"Please use a bit depth less than or equal to {MAX_BIT_DEPTH}."
        )

    # read audio
    waveform, _ = sf.read(
        file = path,
        dtype = np.int32,
        always_2d = True,
    ) # in range [-1, 1]
    waveform = waveform.T # (num_channels, num_samples)
    waveform = torch.from_numpy(waveform)

    # debug: print stats
    logger.debug(
        "Loaded waveform with sample rate %s and bit depth %s (num_channels: %s, num_samples: %s)",
        sample_rate,
        bit_depth,
        waveform.shape[0],
        waveform.shape[1],
    )

    return waveform, sample_rate, bit_depth


def convert_waveform_to_unsigned_integers(
    waveform: torch.Tensor,
    bit_depth: int,
) -> torch.Tensor:
    """
    Convert floating point waveform to unsigned integers.

    Args:
        waveform: Waveform tensor of shape (num_channels, num_samples), 
            int32 in range [-2**31, 2**31 - 1].
        bit_depth: Audio bit depth (e.g. 8, 16, 24).

    Returns:
        Waveform tensor of shape (num_channels, num_samples), 
            int64 in range [0, (2 ** bit_depth) - 1].
    """

    # convert to unsigned integers
    shift = 32 - bit_depth # soundfile scales sub-32-bit formats to fill int32 range. The bit_depth-bit value is the top bit_depth bits.
    waveform = waveform.to(torch.int64) # convert to desired integer type
    if shift > 0:
        waveform = waveform >> shift # right shift to remove the bottom 32 - bit_depth bits
    waveform = waveform + (1 << (bit_depth - 1)) # add 2 ** (bit_depth - 1) for signed->unsigned conversion

    # debug: print stats
    logger.debug("Converted waveform to unsigned integers with min/max: %s/%s", waveform.min().item(), waveform.max().item())

    return waveform

##################################################


# SAVE
##################################################

def save_waveform(
    path: str,
    waveform: torch.Tensor,
    sample_rate: int,
    bit_depth: int,
) -> None:
    """
    Save waveform to file as signed integer PCM.

    Args:
        path: Path to the output file.
        waveform: Waveform tensor of shape (num_channels, num_samples), 
            int32 in range [-2**31, 2**31 - 1].
        sample_rate: Sample rate in Hz.
        bit_depth: Audio bit depth (e.g. 8, 16, 24).
    """

    # get info
    if bit_depth not in BIT_DEPTH_TO_SUBTYPE.keys():
        raise ValueError(
            f"bit_depth {bit_depth} not supported; use {BIT_DEPTH_TO_SUBTYPE.keys()}"
        )
    subtype = BIT_DEPTH_TO_SUBTYPE[bit_depth]

    # waveform is (channels, samples), but soundfile expects (samples, channels)
    data = waveform.cpu().numpy().T
    assert data.dtype == np.int32, f"Waveform dtype is {data.dtype}, but should be int32 in range [-2**31, 2**31 - 1]"
    
    # save waveform
    sf.write(
        file = path,
        data = data,
        samplerate = sample_rate,
        subtype = subtype,
    )

    # debug: print stats
    logger.debug("Saved waveform to %s with sample rate %s and bit depth %s", path, sample_rate, bit_depth)

    return


def convert_waveform_from_unsigned_integers(
    waveform: torch.Tensor,
    bit_depth: int,
) -> torch.Tensor:
    """
    Convert unsigned integers waveform to floating point.

    Args:
        waveform: Waveform tensor of shape (num_channels, num_samples), 
            int64 in range [0, (2 ** bit_depth) - 1].
        bit_depth: Audio bit depth (e.g. 8, 16, 24).

    Returns:
        Waveform tensor of shape (num_channels, num_samples), 
            int32 in range [-2**31, 2**31 - 1].
    """

    # convert to signed integers
    shift = 32 - bit_depth
    waveform = waveform - (1 << (bit_depth - 1)) # subtract 2 ** (bit_depth - 1) for unsigned->signed conversion
    if shift > 0:
        waveform = waveform << shift # left shift to add back bottom 32 - bit_depth bits
    waveform = waveform.to(torch.int32) # convert to int32

    # debug: print stats
    logger.debug("Converted waveform from unsigned integers to signed integers with min/max: %s/%s", waveform.min().item(), waveform.max().item())

    return waveform

##################################################
