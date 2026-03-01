# README
# Constants

# Helper that implements constants for the Trilobyte Lossless Codec.

# IMPORTS
##################################################

# standard library
import logging
from os.path import dirname, join
from math import ceil

# logging
logger = logging.getLogger(__name__) # get logger for the current module

##################################################


# ENCODER CONSTANTS
##################################################

# default block size
BLOCK_SIZE = 512

##################################################


# MODEL CONSTANTS
##################################################

# default absolute filepath to the Trilobyte model checkpoint
MODEL_PATH = join(dirname(dirname(__file__)), "model.ckpt")

# default model bit depth
MODEL_BIT_DEPTH = 24

# vocab size per byte position
VOCAB_PER_BYTE = 256

# whether we use separate byte sub-vocabularies in the model
# if True, byte positions use separate sub-vocabularies (Byte 0: [0,255], Byte 1: [256,511], etc.)
# if False, all positions use [0,255], MASK=256
SEPARATE_BYTE_SUBVOCABULARIES = False

##################################################


# FILE CONSTANTS
##################################################

# default file extension for the output file
FILE_EXTENSION = "tlc"

##################################################


# HEADER CONSTANTS
##################################################

# number of bits for storing block size, where we store k for block size 2^k
BLOCK_SIZE_BITS = 4

# number of bits for storing number of samples, meaning we assume a maximum of 2 ** NUM_SAMPLES_BITS samples
NUM_SAMPLES_BITS = 32

# number of bits for storing the number of channels, where we store k for num_channels k + 1 (which conveniently allows us to represent mono as 0 and stereo as 1 in a single bit)
NUM_CHANNELS_BITS = 1

# number of bits for storing sample rate, meaning we assume a maximum of 2 ** SAMPLE_RATE_BITS sample rates
SAMPLE_RATE_BITS = 17

# number of bits for storing bit depth, where we store k for bit depth 8 * k
BIT_DEPTH_BITS = 2

# number of bytes for the header
TOTAL_HEADER_BYTES = int(ceil(
    (BLOCK_SIZE_BITS + NUM_SAMPLES_BITS + NUM_CHANNELS_BITS + SAMPLE_RATE_BITS + BIT_DEPTH_BITS) / 8
))

##################################################


# ARITHMETIC CODER CONSTANTS
##################################################

# output digits in {0, 1, ..., base - 1}; base 2 = bits
ARITHMETIC_CODER_BASE = 2

# internal state precision (digits in base); higher = less waste, larger ints
ARITHMETIC_CODER_PRECISION = 32

# minimum probability per symbol to avoid zero-mass after quantisation
ARITHMETIC_CODER_EPS = 1e-7

# BOS token
ARITHMETIC_CODER_BOS = 0

##################################################


# AUDIO CONSTANTS
##################################################

# soundfile subtype -> bit depth
SUBTYPE_TO_BIT_DEPTH = {
    "PCM_S8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
}

# bit depth -> soundfile subtype
BIT_DEPTH_TO_SUBTYPE = {
    bit_depth: subtype for subtype, bit_depth in SUBTYPE_TO_BIT_DEPTH.items()
}

# maximum supported bit depth
MAX_BIT_DEPTH = max(SUBTYPE_TO_BIT_DEPTH.values()) # 32

##################################################