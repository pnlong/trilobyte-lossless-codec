# README
# Arithmetic Coder

# Helper that implements an arithmetic coder for
# blocks of audio data within Trilobyte Lossless
# Codec (.tlc) files.

# IMPORTS
##################################################

# standard library
import struct
import logging
from itertools import repeat
from math import ceil
from typing import BinaryIO, Callable, List, Optional, Union
import numpy as np
import torch

# utils
from utils.constants import (
    ARITHMETIC_CODER_BASE,
    ARITHMETIC_CODER_PRECISION,
    ARITHMETIC_CODER_EPS,
)

# logging
logger = logging.getLogger(__name__)  # get logger for the current module

##################################################


# HELPER FUNCTIONS
##################################################

def _pdf_to_numpy(
    pdf: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """
    Convert pdf to 1d numpy float64 array.

    Args:
        pdf: The probability distribution function to convert.

    Returns:
        The numpy array of the pdf.
    """
    if isinstance(pdf, torch.Tensor):
        pdf = pdf.detach().cpu().numpy()
    return np.asarray(pdf, dtype=np.float64).ravel()


def _normalize_pdf(
    pdf: np.ndarray,
    eps: float = ARITHMETIC_CODER_EPS,
) -> np.ndarray:
    """
    Normalize and ensure all entries > 0 for arithmetic coding.

    Args:
        pdf: The probability distribution function to normalize.
        eps: The epsilon value to ensure all entries > 0.

    Returns:
        The normalized pdf.
    """
    pdf = np.maximum(pdf, eps)
    pdf = pdf / np.sum(pdf)
    # keep sum slightly below 1 to avoid quantisation overflow
    n = pdf.size
    pdf = (1.0 - n * eps) * pdf + eps
    return pdf


def normalize_pdf_for_arithmetic_coding(
    pdf: Union[torch.Tensor, np.ndarray],
    eps: float = ARITHMETIC_CODER_EPS,
) -> np.ndarray:
    """
    Normalize a pdf for use with Encoder.encode / Decoder.decode (same as lnac pattern).
    Converts to numpy and ensures positive, sum <= 1.

    Args:
        pdf: The probability distribution function to normalize.
        eps: The epsilon value to ensure all entries > 0.

    Returns:
        The normalized pdf.
    """
    pdf = _pdf_to_numpy(pdf)
    return _normalize_pdf(pdf, eps=eps)


def _write_bits_to_stream(
    bits: List[int],
    stream: BinaryIO,
) -> None:
    """
    Write bit list to stream: 4-byte num_bits (le), then bits packed into
    full bytes (msb first). Bits are padded to the next byte boundary so
    the stream stays byte-aligned between consecutive writes.

    Args:
        bits: The list of bits to write.
        stream: The stream to write the bits to.
    """
    num_bits = len(bits)
    # pad to next byte boundary so the next write starts on a byte
    pad_bits = (8 - num_bits % 8) % 8
    padded = bits + [0] * pad_bits
    stream.write(struct.pack("<I", num_bits))
    for i in range(0, len(padded), 8):
        byte = sum(b << (7 - j) for j, b in enumerate(padded[i : i + 8]))
        stream.write(bytes([byte]))


def _read_bits_from_stream(
    stream: BinaryIO,
) -> List[int]:
    """
    Read bit list from stream (4-byte num_bits le, then packed bytes msb first).
    Reads exactly ceil(num_bits/8) bytes so the stream stays byte-aligned
    for the next chunk.

    Args:
        stream: The stream to read the bits from.

    Returns:
        The list of bits read from the stream (length num_bits; padding discarded).
    """
    num_bits = struct.unpack("<I", stream.read(4))[0]
    num_bytes = ceil(num_bits / 8)
    data = stream.read(num_bytes)
    bits = []
    for byte in data:
        bits.extend((byte >> (7 - j)) & 1 for j in range(8))
    return bits[:num_bits]

##################################################


# ARITHMETIC CODER CORE (base class and digit I/O)
##################################################

class _CoderBase:
    """
    Internal state for encode/decode; interval [low, high] in integer repr.

    Args:
        base: The base of the arithmetic coder.
        precision: The precision of the arithmetic coder.
        io_fn: The function to write the bits to.

    Attributes:
        _base: The base of the arithmetic coder.
        _base_to_pm1: The base to the power of -1.
        _base_to_pm2: The base to the power of -2.
        _io_fn: The function to write the bits to.
        _low: The low value of the interval.
        _high: The high value of the interval.
    """

    def __init__(
        self,
        base: int,
        precision: int,
        io_fn,
    ):
        self._base = base
        self._base_to_pm1 = base ** (precision - 1)
        self._base_to_pm2 = base ** (precision - 2)
        self._io_fn = io_fn
        self._low = 0
        self._high = base**precision - 1
        self._num_carry_digits = 0
        self._code = 0

    def _get_intervals(
        self,
        pdf: np.ndarray,
    ) -> np.ndarray:
        """
        Partition current interval according to pdf; return cumulative bounds.

        Args:
            pdf: The probability distribution function to partition the interval by.

        Returns:
            The cumulative bounds of the intervals.
        """
        width = self._high - self._low + 1
        cdf = np.insert(np.cumsum(pdf), 0, 0.0)
        qcpdf = (cdf * width).astype(np.int64)
        if np.any(qcpdf[1:] <= qcpdf[:-1]):
            raise ValueError("pdf has zero mass after quantisation; increase precision or eps")
        return self._low + qcpdf

    def _remove_matching_digits(
        self,
        low_pre_split: int,
        encoding: bool,
    ) -> None:
        """
        Output/consume matching msbs and renormalise interval.

        Args:
            low_pre_split: The low value of the interval before the split.
            encoding: Whether we are encoding or decoding.
        """

        def _shift_left(x: int) -> int:
            return (x % self._base_to_pm1) * self._base

        while self._low // self._base_to_pm1 == self._high // self._base_to_pm1:
            if encoding:
                low_msd = self._low // self._base_to_pm1
                self._io_fn(low_msd)
                carry_digit = (self._base - 1 + low_msd - low_pre_split // self._base_to_pm1) % self._base
                while self._num_carry_digits > 0:
                    self._io_fn(carry_digit)
                    self._num_carry_digits -= 1
            else:
                self._code = _shift_left(self._code) + self._io_fn()
            self._low = _shift_left(self._low)
            self._high = _shift_left(self._high) + self._base - 1

    def _remove_carry_digits(
        self, encoding: bool,
    ) -> None:
        """
        Handle near-underflow: second most significant digit prefix.

        Args:
            encoding: Whether we are encoding or decoding.
        """

        def _shift_left_keeping_msd(x: int) -> int:
            return x - (x % self._base_to_pm1) + (x % self._base_to_pm2) * self._base

        while self._low // self._base_to_pm2 + 1 == self._high // self._base_to_pm2:
            if encoding:
                self._num_carry_digits += 1
            else:
                self._code = _shift_left_keeping_msd(self._code) + self._io_fn()
            self._low = _shift_left_keeping_msd(self._low)
            self._high = _shift_left_keeping_msd(self._high) + self._base - 1

    def _process(
        self, 
        pdf: np.ndarray, 
        symbol: Optional[int],
    ) -> int:
        """
        One encode step (symbol not None) or decode step (symbol None).

        Args:
            pdf: The probability distribution function to process.
            symbol: The symbol to process.

        Returns:
            The symbol processed.
        """
        encoding = symbol is not None
        intervals = self._get_intervals(pdf)
        if not encoding:
            symbol = int(np.searchsorted(intervals, self._code, side="right") - 1)
        symbol = max(0, min(symbol, pdf.size - 1))
        low_pre_split = self._low
        self._low = int(intervals[symbol])
        self._high = int(intervals[symbol + 1]) - 1
        self._remove_matching_digits(low_pre_split=low_pre_split, encoding=encoding)
        self._remove_carry_digits(encoding=encoding)
        return symbol


def _raise_post_terminate_exception(*args: object, **kwargs: object) -> None:
    """Raise if encode/terminate is called after Encoder.terminate()."""
    raise ValueError(
        "Arithmetic encoder was terminated. "
        "Create a new instance for encoding more data."
    )


class Encoder(_CoderBase):
    """
    Arithmetic encoder (lnac-style API).
    Use output_fn to collect digits; then write bits to stream via write_bits_to_stream.
    """

    def __init__(self, base: int, precision: int, output_fn: Callable[[int], None]):
        super().__init__(base, precision, output_fn)

    def encode(self, pdf: np.ndarray, symbol: int) -> None:
        """Encode one symbol with the given distribution."""
        self._process(pdf, symbol)

    def terminate(self) -> None:
        """Finalize the arithmetic code (write final interval digits)."""
        self._io_fn(self._low // self._base_to_pm1)
        for _ in range(self._num_carry_digits):
            self._io_fn(self._base - 1)
        self.encode = _raise_post_terminate_exception  # type: ignore[assignment]
        self.terminate = _raise_post_terminate_exception  # type: ignore[assignment]


class Decoder(_CoderBase):
    """
    Arithmetic decoder (lnac-style API).
    input_fn() returns the next digit from {0, ..., base-1}, or None when exhausted
    (decoder will pad with base-1).
    """

    def __init__(self, base: int, precision: int, input_fn: Callable[[], Optional[int]]):
        trailing_digits = repeat(base - 1)

        def padded_input_fn() -> int:
            digit = input_fn()
            if digit is None:
                digit = next(trailing_digits)
            return int(digit)

        super().__init__(base, precision, padded_input_fn)
        for _ in range(precision):
            self._code = self._code * base + padded_input_fn()

    def decode(self, pdf: np.ndarray) -> int:
        """Decode one symbol with the given distribution."""
        return self._process(pdf, None)


##################################################


# STREAM HELPERS (public for use with Encoder/Decoder)
##################################################


def write_bits_to_stream(
    bits: List[int],
    stream: BinaryIO,
) -> None:
    """
    Write a list of bits to stream: 4-byte num_bits (le), then packed bytes (msb first), padded to byte boundary.

    Args:
        bits: The list of bits to write.
        stream: The stream to write the bits to.
    """
    _write_bits_to_stream(
        bits = bits,
        stream = stream,
    )


def read_bits_from_stream(
    stream: BinaryIO,
) -> List[int]:
    """
    Read one chunk of bits from stream (4-byte num_bits then packed bytes).

    Args:
        stream: The stream to read the bits from.
    """
    return _read_bits_from_stream(
        stream = stream,
    )


##################################################


# ARITHMETIC ENCODER (function API – implemented via Encoder)
##################################################

def encode_arithmetic(
    tokens: Union[List[int], np.ndarray, torch.Tensor],
    pdf: Union[torch.Tensor, np.ndarray],
    stream: BinaryIO,
    base: int = ARITHMETIC_CODER_BASE,
    precision: int = ARITHMETIC_CODER_PRECISION,
) -> None:
    """
    Encode a sequence of tokens with the given pdf(s) and write the bitstream to stream.
    Stream format: 4 bytes num_bits (little-endian), then bits packed into bytes
    (msb first), padded to the next byte boundary so the stream stays byte-aligned
    between calls.

    Args:
        tokens: The tokens to encode (length N).
        pdf: Either (vocab_size,) for a single shared pdf, or (N, vocab_size) for
            one pdf per token. Must match length of tokens when 2D.
        stream: The stream to write the encoded bits to.
        base: The base of the arithmetic coder.
        precision: The precision of the arithmetic coder.

    Raises:
        ValueError: If the pdf is not positive or the sum of the pdf is greater than 1 after normalisation.
    """
    # convert tokens to list of int
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy().ravel().tolist()
    else:
        tokens = list(tokens)
    n = len(tokens)

    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.ndim == 1:
        pdf = np.broadcast_to(pdf.reshape(1, -1), (n, pdf.size)).copy()
    if pdf.shape[0] != n:
        raise ValueError(f"pdf shape {pdf.shape} does not match token length {n} (expect (n, vocab_size) or (vocab_size,))")
    vocab_size = pdf.shape[1]
    # normalize each row
    for i in range(n):
        row = _normalize_pdf(pdf[i].ravel())
        if np.any(row <= 0) or np.sum(row) > 1.0 + 1e-9:
            raise ValueError("pdf must be positive and sum <= 1 after normalisation")
        pdf[i] = row.reshape(vocab_size)

    bits: List[int] = []

    def output_fn(digit: int) -> None:
        bits.append(digit)

    enc = _CoderBase(base=base, precision=precision, io_fn=output_fn)
    for i, token in enumerate(tokens):
        enc._process(pdf=pdf[i], symbol=token)

    # terminate: write final interval representative
    enc._io_fn(enc._low // enc._base_to_pm1)
    for _ in range(enc._num_carry_digits):
        enc._io_fn(enc._base - 1)

    _write_bits_to_stream(bits=bits, stream=stream)
    return

##################################################


# ARITHMETIC DECODER
##################################################

def decode_arithmetic(
    stream: BinaryIO,
    pdf: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_tokens: Optional[int] = None,
    get_pdf: Optional[Callable[[List[int]], np.ndarray]] = None,
    base: int = ARITHMETIC_CODER_BASE,
    precision: int = ARITHMETIC_CODER_PRECISION,
) -> List[int]:
    """
    Read the arithmetic-coded bitstream from stream and decode num_tokens tokens.
    Expects stream format from encode_arithmetic (4-byte num_bits then packed bits).

    Either (pdf, num_tokens) or (get_pdf, num_tokens) must be provided.
    - (pdf, num_tokens): use the same pdf for every symbol (one chunk read).
    - (get_pdf, num_tokens): read one chunk, then decode symbol i using get_pdf(decoded_so_far)
      so each symbol can have a different pdf (e.g. from a causal model).

    Args:
        stream: The stream to read the arithmetic-coded bitstream from.
        pdf: Single (vocab_size,) pdf for all tokens. Use with num_tokens.
        num_tokens: Number of tokens to decode.
        get_pdf: Callable(decoded_so_far) -> (vocab_size,) pdf for the next symbol.
        base: The base of the arithmetic coder.
        precision: The precision of the arithmetic coder.

    Returns:
        The decoded tokens.
    """
    if (pdf is None) == (get_pdf is None):
        raise ValueError("provide either pdf or get_pdf (not both, not neither)")
    if num_tokens is None:
        raise ValueError("num_tokens is required")

    if pdf is not None:
        pdf = _pdf_to_numpy(pdf)
        pdf = _normalize_pdf(pdf)

    bits = _read_bits_from_stream(stream=stream)
    it = iter(bits)

    def input_fn() -> int:
        try:
            return next(it)
        except StopIteration:
            return base - 1  # padding when exhausted

    dec = _CoderBase(base=base, precision=precision, io_fn=input_fn)
    for _ in range(precision):
        dec._code = dec._code * base + input_fn()

    out: List[int] = []
    for _ in range(num_tokens):
        if get_pdf is not None:
            pdf_i = _pdf_to_numpy(get_pdf(out))
            pdf_i = _normalize_pdf(pdf_i)
        else:
            pdf_i = pdf
        token = dec._process(pdf=pdf_i, symbol=None)
        out.append(token)
    return out

##################################################
