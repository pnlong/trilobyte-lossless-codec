# README
# Trilobyte Lossless Codec

# Implements a CLI for the Trilobyte Lossless 
# Codec for encoding and decoding audio data to 
# and from Trilobyte Lossless Codec (.tlc) files.

# IMPORTS
##################################################

# standard library
import sys
import argparse
import logging
from os.path import exists, basename, dirname, splitext, getsize, join, abspath
from time import perf_counter
import torch

# utils
from utils.constants import (
    MODEL_PATH,
    MODEL_BIT_DEPTH,
    BLOCK_SIZE,
    FILE_EXTENSION,
)
from utils.header import verify_power_of_two
from utils.model import load_model
from encode import encode_wrapper
from decode import decode_wrapper

# logging
logger = logging.getLogger(__name__) # get logger for the current module

##################################################


# ARGUMENT PARSING
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "TLC", description = "Trilobyte Lossless Codec") # create argument parser

    # general options
    parser.add_argument(
        "infile",
        type = str,
        help = "Path to the input file.",
    )
    parser.add_argument(
        "-o", "--outfile",
        type = str,
        default = None,
        help = "Path to the output file.",
        required = False,
    )
    parser.add_argument(
        "--modelpath",
        type = str,
        default = None,
        help = f"Path to the Trilobyte model checkpoint. Defaults to {MODEL_PATH}.",
        required = False,
    )
    parser.add_argument(
        "--modeldepth",
        type = int,
        default = None,
        help = f"Model bit depth. Defaults to {MODEL_BIT_DEPTH}.",
        required = False,
    )
    parser.add_argument(
        "--gpu",
        action = "store_true",
        help = "Use GPU.",
        required = False,
    )
    parser.add_argument(
        "-s", "--silent",
        action = "store_true",
        help = "Silent mode (do not write runtime encode/decode statistics to stderr).",
        required = False,
    )
    parser.add_argument(
        "-v", "--verbose",
        action = "store_true",
        help = "Verbose mode (print detailed runtime encode/decode statistics to stderr).",
        required = False,
    )

    # decoding options
    parser.add_argument(
        "-d", "--decode",
        action = "store_true",
        help = "Decode a file (default is to encode).",
        required = False,
    )

    # encoding options
    parser.add_argument(
        "-b", "--blocksize",
        type = int,
        default = None,
        help = f"Block size. Defaults to {BLOCK_SIZE}.",
        required = False,
    )

    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace) # parse arguments

    # convert input file path to absolute path
    args.infile = abspath(args.infile)

    # infer output file path if not provided
    if args.outfile is None:
        args.outfile = join(abspath(dirname(__file__)), f"{splitext(basename(args.infile))[0]}.{FILE_EXTENSION}")
    else:
        args.outfile = abspath(args.outfile)

    # infer model path and bit depth if not provided
    if args.modelpath is None and args.modeldepth is None:
        args.modelpath = MODEL_PATH
        args.modeldepth = MODEL_BIT_DEPTH
    elif args.modelpath is not None and args.modeldepth is None:
        raise ValueError("Model bit depth must be set if model path is provided.")
    elif args.modelpath is None and args.modeldepth is not None:
        raise ValueError("Model path must be set if model bit depth is provided.")

    # check that input file exists
    if not exists(args.infile):
        raise FileNotFoundError(f"Input file does not exist: {args.infile}")

    # check that output file path is valid
    if not exists(dirname(args.outfile)):
        raise FileNotFoundError(f"Output file directory does not exist: {dirname(args.outfile)}")

    # ensure that log-level arguments are consistent
    if args.silent and args.verbose:
        raise ValueError("Cannot be both silent and verbose.")

    # ensure encoding arguments -- block size and batch size -- are consistent
    if args.decode and args.blocksize is not None:
        raise ValueError("Cannot specify block size when decoding.")
    elif not args.decode and args.blocksize is None:
        args.blocksize = BLOCK_SIZE

    # verify that block size and batch size are powers of two
    if not verify_power_of_two(n = args.blocksize):
        raise ValueError(f"Block size must be a power of two: {args.blocksize}")

    return args # return parsed arguments

##################################################


# MAIN METHOD
##################################################

def main(args):
    """
    Main method.

    Args:
        args: Arguments object.
    """

    # start timer
    start_time = perf_counter()

    # set up logging for the whole project: ensure root has a handler, then set level
    logging.basicConfig(
        format = "%(levelname)s\t%(name)s: %(message)s",
        level = logging.DEBUG, # handler level, root level set below
        force = True,
    )
    root = logging.getLogger()
    if args.silent:
        root.setLevel(logging.WARNING)
    elif args.verbose:
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)

    # load model, timing it
    start_time_model = perf_counter()
    logger.info(f"Loading model from {args.modelpath}...")
    model = load_model(path_model = args.modelpath)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu") # get device
    model.to(device) # move model to device
    model.eval() # set model to evaluation mode
    end_time_model = perf_counter()
    duration_model = end_time_model - start_time_model
    logger.info(f"Model loaded in {duration_model:.2f} seconds")

    # encode/decode branch, timing it
    start_time_code = perf_counter()
    if args.decode: # decode
        logger.info(f"Decoding {args.infile} to {args.outfile}...")
        decode_wrapper(
            path_input = args.infile,
            path_output = args.outfile,
            model = model, # device is inferred from model
            model_bit_depth = args.modeldepth,
            silent = args.silent,
        )
    else: # encode
        logger.info(f"Encoding {args.infile} to {args.outfile}...")
        encode_wrapper(
            path_input = args.infile,
            path_output = args.outfile,
            model = model, # device is inferred from model
            model_bit_depth = args.modeldepth,
            block_size = args.blocksize,
            silent = args.silent,
        )
        compression_rate = getsize(args.infile) / getsize(args.outfile)
        logger.info(f"Compression rate: {compression_rate:.2f}x")
    end_time_code = perf_counter()
    duration_code = end_time_code - start_time_code
    logger.info(f"{'Decoding' if args.decode else 'Encoding'} executed in {duration_code:.2f} seconds")

    # stop timer, note total runtime
    end_time = perf_counter()
    duration = end_time - start_time
    logger.info(f"Total time: {duration:.2f} seconds")

    # if we get here, we were successful
    sys.exit(0)

##################################################


# MAIN ENTRY POINT
##################################################

if __name__ == "__main__":

    args = parse_args()
    main(args = args)

##################################################
