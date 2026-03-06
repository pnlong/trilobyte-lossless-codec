# README
# Encode/decode roundtrip tests

# Roundtrip tests using encode and decode with BytesIO.
# Covers different bit depths, mono/stereo, and sample counts not divisible by block size.

# IMPORTS
##################################################

# standard library
from io import BytesIO
from math import pi
from os.path import exists
import pytest
import torch

# utils
from encode import encode
from decode import decode
from utils.audio import (
    convert_waveform_from_unsigned_integers,
    convert_waveform_to_unsigned_integers,
)
from utils.constants import (
    BLOCK_SIZE,
    MODEL_PATH,
)
from utils.model import load_model

##################################################


# FIXTURE
##################################################


@pytest.fixture(scope = "module")
def model():
    """Load model once per module; skip if checkpoint missing."""
    if not exists(MODEL_PATH):
        pytest.skip(
            f"Model checkpoint not found at {MODEL_PATH}; run setup.sh to download"
        )
    return load_model(path_model = MODEL_PATH)


##################################################


# HELPERS
##################################################


def _roundtrip_stream(
    waveform: torch.Tensor,
    sample_rate: int,
    bit_depth: int,
    model,
    block_size: int = BLOCK_SIZE,
    silent: bool = True,
) -> torch.Tensor:
    """Encode waveform to BytesIO, decode back, return decoded waveform."""
    stream = BytesIO()
    encode(
        waveform = waveform,
        stream = stream,
        model = model,
        sample_rate = sample_rate,
        bit_depth = bit_depth,
        block_size = block_size,
        silent = silent,
    )
    stream.seek(0)
    decoded, decoded_sample_rate, decoded_bit_depth = decode(
        stream = stream,
        model = model,
        silent = silent,
    )
    assert decoded_sample_rate == sample_rate
    assert decoded_bit_depth == bit_depth
    return decoded


def _int32_waveform(shape, bit_depth: int) -> torch.Tensor:
    """Random int32 waveform (from random unsigned ints so roundtrip is exact)."""
    unsigned = torch.randint(0, 2**bit_depth, shape, dtype = torch.int64)
    return convert_waveform_from_unsigned_integers(unsigned, bit_depth)

##################################################


# ROUNDTRIP TESTS
##################################################

def test_roundtrip_mono_24bit_440hz_sine(model):
    """Mono 24-bit roundtrip on a 440 Hz sine waveform (quantized comparison)."""
    sample_rate = 44100
    num_samples = 1024
    bit_depth = 24
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    sine_float = torch.sin(2 * pi * 440 * t).unsqueeze(0)  # (1, num_samples), range [-1, 1]
    # Quantize to 24-bit signed then to int32 (soundfile format: 24-bit in top 24 bits of int32)
    half_scale = (1 << (bit_depth - 1)) - 1  # 2^23 - 1
    sine_24_signed = torch.clamp(
        torch.round(sine_float * half_scale).to(torch.int64),
        -(1 << (bit_depth - 1)),
        (1 << (bit_depth - 1)) - 1,
    )
    waveform = (sine_24_signed << (32 - bit_depth)).to(torch.int32)  # (1, num_samples)
    decoded = _roundtrip_stream(waveform, sample_rate, bit_depth, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_mono_24bit_two_blocks(model):
    """Mono 24-bit, ~2 blocks of tokens (341 samples * 3 bytes = 1023 tokens)."""
    num_samples = 341
    waveform = _int32_waveform((1, num_samples), 24)
    decoded = _roundtrip_stream(waveform, 44100, 24, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_mono_24bit_not_divisible_by_block(model):
    """Mono 24-bit, sample count such that token length is not divisible by block size."""
    num_samples = 100
    waveform = _int32_waveform((1, num_samples), 24)
    decoded = _roundtrip_stream(waveform, 48000, 24, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_stereo_24bit_two_blocks(model):
    """Stereo 24-bit, ~2 blocks (170 samples per channel)."""
    num_samples = 170
    waveform = _int32_waveform((2, num_samples), 24)
    decoded = _roundtrip_stream(waveform, 44100, 24, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_stereo_24bit_not_divisible_by_block(model):
    """Stereo 24-bit, sample count not filling whole blocks."""
    num_samples = 37
    waveform = _int32_waveform((2, num_samples), 24)
    decoded = _roundtrip_stream(waveform, 48000, 24, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_mono_16bit(model):
    """Mono 16-bit (mask token used for third byte in 24-bit model)."""
    num_samples = 150
    waveform = _int32_waveform((1, num_samples), 16)
    decoded = _roundtrip_stream(waveform, 44100, 16, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_mono_8bit(model):
    """Mono 8-bit (mask tokens for second and third bytes)."""
    num_samples = 120
    waveform = _int32_waveform((1, num_samples), 8)
    decoded = _roundtrip_stream(waveform, 44100, 8, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)


def test_roundtrip_mono_24bit_four_blocks(model):
    """Mono 24-bit, ~4 blocks worth of samples."""
    num_samples = 682
    waveform = _int32_waveform((1, num_samples), 24)
    decoded = _roundtrip_stream(waveform, 44100, 24, model)
    assert decoded.shape == waveform.shape
    torch.testing.assert_close(decoded, waveform)

##################################################
