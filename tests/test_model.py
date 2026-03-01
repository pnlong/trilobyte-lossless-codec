# README
# Model tests

# Roundtrip tests: waveform -> tokens -> waveform, assert we recover the original.
# Optional format tests for token layout.

# IMPORTS
##################################################

# standard library
from os.path import exists
import pytest
import torch

# utils (run from repo root: python -m pytest tests/)
from utils.constants import MODEL_PATH
from utils.model import (
    GPTAudioLightningModule,
    convert_tokens_to_waveform,
    convert_waveform_to_tokens,
    get_vocab_size,
    load_model,
)

##################################################


# ROUNDTRIP HELPER
##################################################

def _roundtrip(
    waveform: torch.Tensor,
    bit_depth: int = 24,
    model_bit_depth: int = 24,
) -> torch.Tensor:
    """encode waveform to tokens, decode back, return waveform."""
    tokens = convert_waveform_to_tokens(
        waveform=waveform,
        model_bit_depth=model_bit_depth,
        bit_depth=bit_depth,
    )
    num_channels = waveform.shape[0] if waveform.dim() > 1 else 1
    return convert_tokens_to_waveform(
        tokens=tokens,
        num_channels=num_channels,
        bit_depth=bit_depth,
        model_bit_depth=model_bit_depth,
    )

##################################################


# ROUNDTRIP TESTS (SEPARATE_BYTE_SUBVOCABULARIES=False default)
##################################################

def test_roundtrip_mono_24bit():
    """mono 24-bit unsigned waveform roundtrip (0 to 2**24 - 1)."""
    waveform = torch.randint(0, 2**24, size=(1, 100))
    decoded = _roundtrip(waveform=waveform, model_bit_depth=24, bit_depth=24)
    assert torch.equal(waveform, decoded)


def test_roundtrip_stereo_24bit():
    """stereo 24-bit unsigned waveform roundtrip (0 to 2**24 - 1)."""
    waveform = torch.randint(0, 2**24, size=(2, 100))
    decoded = _roundtrip(waveform=waveform, model_bit_depth=24, bit_depth=24)
    assert torch.equal(waveform, decoded)


def test_roundtrip_16bit_with_mask():
    """16-bit unsigned audio with 24-bit model; third token per sample is mask."""
    waveform = torch.randint(0, 2**16, size=(1, 50))
    decoded = _roundtrip(waveform=waveform, model_bit_depth=24, bit_depth=16)
    assert torch.equal(waveform, decoded)


def test_roundtrip_8bit_with_mask():
    """8-bit unsigned audio with 24-bit model; second and third tokens are mask."""
    waveform = torch.randint(0, 2**8, size=(1, 50))
    decoded = _roundtrip(waveform=waveform, model_bit_depth=24, bit_depth=8)
    assert torch.equal(waveform, decoded)


def test_roundtrip_1d_mono():
    """1d waveform treated as mono (unsigned 0 to 2**24 - 1)."""
    waveform = torch.randint(0, 2**24, size=(50,))
    decoded = _roundtrip(waveform=waveform, model_bit_depth=24, bit_depth=24)
    assert torch.equal(waveform, decoded.squeeze(0))

##################################################


# FORMAT TESTS (SEPARATE_BYTE_SUBVOCABULARIES=False)
##################################################

def test_format_all_bytes_in_range():
    """with separate=False, all tokens in [0,255] or mask 256."""
    # 0x123456 is positive in 24-bit signed
    waveform = torch.tensor([[0x123456]], dtype=torch.int64)
    tokens = convert_waveform_to_tokens(
        waveform=waveform,
        model_bit_depth=24,
        bit_depth=24,
    )
    assert tokens.shape == (3,)
    assert (tokens[:3] <= 256).all()
    assert (tokens[:3] >= 0).all()

##################################################


# LOAD MODEL TEST
##################################################

def test_load_model():
    """load_model returns GPTAudioLightningModule when checkpoint exists."""
    if not exists(MODEL_PATH):
        pytest.skip(f"Model checkpoint not found at {MODEL_PATH}; run setup.sh to download")
    model = load_model(path_model=MODEL_PATH)
    assert isinstance(model, GPTAudioLightningModule)
    assert hasattr(model, "model")
    assert hasattr(model, "forward")

##################################################
