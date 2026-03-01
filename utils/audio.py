# README
# Audio

# Helper for loading and saving audio files as signed integer PCM.

# IMPORTS
##################################################

# third party
import numpy as np
import soundfile as sf
import torch

# standard library
from typing import Tuple

# utils
from utils.constants import (
    BIT_DEPTH_TO_SUBTYPE,
    SUBTYPE_TO_BIT_DEPTH,
    MAX_BIT_DEPTH,
)

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
            float64 in range [-1, 1].
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
        dtype = np.float64,
        always_2d = True,
    ) # in range [-1, 1]
    waveform = waveform.T # (num_channels, num_samples)
    waveform = torch.from_numpy(waveform)

    return waveform, sample_rate, bit_depth


def convert_waveform_to_unsigned_integers(
    waveform: torch.Tensor,
    bit_depth: int,
) -> torch.Tensor:
    """
    Convert floating point waveform to unsigned integers.

    Args:
        waveform: Waveform tensor of shape (num_channels, num_samples), 
            float64 in range [-1, 1].
        bit_depth: Audio bit depth (e.g. 8, 16, 24).

    Returns:
        Waveform tensor of shape (num_channels, num_samples), 
            int64 in range [0, (2 ** bit_depth) - 1].
    """
    waveform = waveform + 1 # to range [0, 2]
    waveform = waveform * ((2 ** (bit_depth - 1)) - 0.5) # to range [0, (2 ** bit_depth) - 1]
    waveform = torch.round(waveform).to(torch.int64) # convert to integer
    waveform = torch.clamp( # clip waveform to bit depth, though realistically it should rarely happen
        input = waveform,
        min = 0,
        max = (2 ** bit_depth) - 1,
    )
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
            float64 in range [-1, 1].
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
    assert data.dtype == np.float64, f"Waveform dtype is {data.dtype}, but should be float64 in range [-1, 1]"
    
    # save waveform
    sf.write(
        file = path,
        data = data,
        samplerate = sample_rate,
        subtype = subtype,
    )

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
            float64 in range [-1, 1].
    """
    waveform = waveform.to(torch.float64) # convert to float64
    waveform = waveform / ((2 ** (bit_depth - 1)) - 0.5) # to range [0, 2]
    waveform = waveform - 1 # to range [-1, 1]
    waveform = torch.clamp(
        input = waveform,
        min = -1,
        max = 1,
    )
    return waveform

##################################################
