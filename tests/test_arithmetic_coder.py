# README
# Arithmetic coder tests

# Roundtrip tests using the Encoder/Decoder class API and stream helpers.
# Encode symbols with a pdf, write bits to stream, read back, decode with same pdf; check we recover the original symbols.

# IMPORTS
##################################################

# standard library
from io import BytesIO
from typing import List, Union
import numpy as np
import pytest
import torch

# utils
from utils.arithmetic_coder import (
    Decoder,
    Encoder,
    normalize_pdf_for_arithmetic_coding,
    read_bits_from_stream,
    write_bits_to_stream,
)
from utils.constants import ARITHMETIC_CODER_BASE, ARITHMETIC_CODER_PRECISION

##################################################


# ROUNDTRIP HELPER
##################################################


def _roundtrip(
    symbols: List[int],
    pdf: Union[np.ndarray, torch.Tensor],
) -> List[int]:
    """Encode symbols with pdf (Encoder + terminate), write chunk to stream; read chunk, Decoder + decode; return decoded list."""
    pdf = np.asarray(pdf, dtype=np.float64).ravel()
    pdf_norm = normalize_pdf_for_arithmetic_coding(pdf)

    # Encode: collect bits with Encoder, then write one chunk
    bits = []
    encoder = Encoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        output_fn=bits.append,
    )
    for symbol in symbols:
        encoder.encode(pdf_norm, symbol)
    encoder.terminate()

    buf = BytesIO()
    write_bits_to_stream(bits, buf)
    buf.seek(0)

    # Decode: read one chunk, build input_fn, Decoder, decode same number of symbols
    bits_read = read_bits_from_stream(buf)
    it = iter(bits_read)

    def input_fn():
        try:
            return next(it)
        except StopIteration:
            return None

    decoder = Decoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        input_fn=input_fn,
    )
    decoded = [decoder.decode(pdf_norm) for _ in range(len(symbols))]
    return decoded


##################################################


# BASIC ROUNDTRIP TESTS
##################################################


def test_roundtrip_single_symbol():
    """Single symbol, deterministic."""
    pdf = np.array([1.0])
    symbols = [0]
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols


def test_roundtrip_two_symbols_uniform():
    """Two symbols, uniform pdf."""
    pdf = np.array([0.5, 0.5])
    symbols = [0, 1, 0, 1, 1, 0]
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols


def test_roundtrip_four_symbols_skewed():
    """Four symbols, skewed pdf."""
    pdf = np.array([0.5, 0.25, 0.15, 0.1])
    symbols = [0, 1, 2, 0, 3, 1, 0]
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols


def test_roundtrip_longer_sequence():
    """Longer sequence with small alphabet."""
    pdf = np.array([0.6, 0.3, 0.1])
    symbols = [0, 1, 2, 0, 0, 1, 2, 0, 1, 0] * 20
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols


def test_roundtrip_large_alphabet():
    """Alphabet size 64, short sequence."""
    n = 64
    pdf = np.ones(n) / n
    symbols = list(range(0, n, 4))  # 0, 4, 8, ..., 60
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols


def test_roundtrip_all_symbols_once():
    """Each symbol appears exactly once, arbitrary order."""
    pdf = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    symbols = [2, 0, 4, 1, 3]
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols

##################################################


# PDF AS TORCH TENSOR
##################################################


def test_roundtrip_pdf_torch_tensor():
    """Roundtrip works when pdf is a torch tensor."""
    pdf = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    symbols = [0, 1, 2, 0, 1]
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols

##################################################


# SYMBOLS AS NUMPY / TORCH
##################################################


def test_roundtrip_symbols_numpy():
    """Roundtrip works when symbols are passed as numpy array."""
    pdf = np.array([0.4, 0.6])
    symbols = np.array([0, 1, 1, 0, 1])
    assert _roundtrip(symbols=symbols.tolist(), pdf=pdf) == symbols.tolist()


def test_roundtrip_symbols_torch():
    """Roundtrip works when symbols are passed as torch tensor."""
    pdf = np.array([0.4, 0.6])
    symbols = torch.tensor([0, 1, 1, 0, 1])
    assert _roundtrip(symbols=symbols.tolist(), pdf=pdf) == symbols.tolist()

##################################################


# EDGE CASES
##################################################


def test_roundtrip_empty_sequence():
    """Empty symbol list: encode then decode zero symbols."""
    pdf = np.array([0.5, 0.5])
    symbols = []
    pdf_norm = normalize_pdf_for_arithmetic_coding(pdf)
    bits = []
    encoder = Encoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        output_fn=bits.append,
    )
    encoder.terminate()
    buf = BytesIO()
    write_bits_to_stream(bits, buf)
    buf.seek(0)
    bits_read = read_bits_from_stream(buf)
    it = iter(bits_read)

    def input_fn():
        try:
            return next(it)
        except StopIteration:
            return None

    decoder = Decoder(
        base=ARITHMETIC_CODER_BASE,
        precision=ARITHMETIC_CODER_PRECISION,
        input_fn=input_fn,
    )
    decoded = [decoder.decode(pdf_norm) for _ in range(0)]
    assert decoded == []


def test_roundtrip_single_symbol_repeated():
    """Same symbol repeated many times."""
    pdf = np.array([0.2, 0.8])  # symbol 1 much more likely
    symbols = [1] * 100
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols


def test_roundtrip_very_skewed_pdf():
    """Heavily skewed pdf (one symbol dominant)."""
    pdf = np.array([0.01, 0.01, 0.98])
    symbols = [2, 2, 2, 1, 2, 0, 2]
    assert _roundtrip(symbols=symbols, pdf=pdf) == symbols

##################################################
