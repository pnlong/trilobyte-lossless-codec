# README
# Header tests

# Roundtrip tests: encode then decode and check we recover the original header.
# Also verifies the encoded header has the expected byte size.

# IMPORTS
##################################################

# standard library
from io import BytesIO

# utils (run from repo root: python -m pytest tests/)
from utils.header import encode_header, decode_header
from utils.constants import TOTAL_HEADER_BYTES

##################################################


# ROUNDTRIP HELPER
##################################################

def _roundtrip(header):
    """encode header to stream, decode from stream, return decoded dict."""
    buf = BytesIO()
    encode_header(header=header, stream=buf)
    buf.seek(0)
    return decode_header(stream=buf)

##################################################


# ROUNDTRIP TESTS
##################################################

def test_roundtrip_basic():
    """basic mono, 44.1khz, 16-bit."""
    header = {
        "block_size": 512,
        "num_samples": 44100,
        "num_channels": 1,
        "sample_rate": 44100,
        "bit_depth": 16,
    }
    assert _roundtrip(header=header) == header


def test_roundtrip_stereo():
    """stereo, 48khz, 24-bit."""
    header = {
        "block_size": 256,
        "num_samples": 96000,
        "num_channels": 2,
        "sample_rate": 48000,
        "bit_depth": 24,
    }
    assert _roundtrip(header=header) == header


def test_roundtrip_minimal_block_batch():
    """minimum block and batch size (1)."""
    header = {
        "block_size": 1,
        "num_samples": 100,
        "num_channels": 1,
        "sample_rate": 8000,
        "bit_depth": 8,
    }
    assert _roundtrip(header=header) == header

##################################################


# HEADER SIZE TEST
##################################################

def test_header_expected_size():
    """encoded header is exactly TOTAL_HEADER_BYTES."""
    header = {
        "block_size": 512,
        "num_samples": 44100,
        "num_channels": 1,
        "sample_rate": 44100,
        "bit_depth": 16,
    }
    buf = BytesIO()
    encode_header(header=header, stream=buf)
    assert len(buf.getvalue()) == TOTAL_HEADER_BYTES

##################################################
