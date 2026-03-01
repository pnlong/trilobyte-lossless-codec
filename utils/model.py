# README
# Model

# Helper that focuses on Trilobyte model logic 
# for encoding and decoding audio data within 
# Trilobyte Lossless Codec (.tlc) files.
# Based on https://github.com/ZacharyNovack/lnac/blob/main/train_gpt2.py

# IMPORTS
##################################################

# standard library
import math
import logging
from os.path import exists
from typing import Optional, Dict, Any
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

# utils
from utils.constants import (
    MODEL_PATH,
    MODEL_BIT_DEPTH,
    VOCAB_PER_BYTE,
    SEPARATE_BYTE_SUBVOCABULARIES,
)

# logging
logger = logging.getLogger(__name__)  # get logger for the current module

##################################################


# GPT AUDIO LIGHTNING MODULE (for loading checkpoints)
##################################################

# checkpoint-compatible constants (matches train_gpt2)
_BYTES_PER_SAMPLE_CKPT = 3
_VOCAB_SIZE_CKPT = _BYTES_PER_SAMPLE_CKPT * VOCAB_PER_BYTE + 1  # 769


class GPTAudioLightningModule(pl.LightningModule):
    """
    Lightning module for gpt2 audio model; used for loading trained checkpoints.

    Args:
        model_name: Name of the model to load.
        lr: Learning rate.
        weight_decay: Weight decay.
        warmup_steps: Warmup steps.
        max_steps: Maximum steps.
    """

    def __init__(
        self,
        model_name="gpt2",
        lr=3e-4,
        weight_decay=0.1,
        warmup_steps=1000,
        max_steps=-1,
        chunk_size=1024,
        stereo_interleave=False,
        pad_to_max_bytes=True,
        max_position_embeddings=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        config = GPT2Config.from_pretrained(model_name)
        config.vocab_size = _VOCAB_SIZE_CKPT
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings
        else:
            stereo_mult = 2 if stereo_interleave else 1
            config.max_position_embeddings = (
                chunk_size * _BYTES_PER_SAMPLE_CKPT * stereo_mult + 1
            )
        self.model = GPT2LMHeadModel(config)
        if kwargs.get("gradient_checkpointing"):
            self.model.gradient_checkpointing_enable()
        self.n_bytes = _BYTES_PER_SAMPLE_CKPT

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input ids.
            labels: Labels.

        Returns:
            Outputs.
        """
        return self.model(
            input_ids=input_ids,
            labels=labels,
        )

    def _per_byte_bpb(
        self,
        batch: torch.Tensor,
        logits: torch.Tensor,
        n_bytes: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute mean bpb for each byte position.

        Args:
            batch: Batch.
            logits: Logits.
            n_bytes: Number of bytes.
        """
        if n_bytes is None:
            n_bytes = self.n_bytes
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        token_bpb = nn.CrossEntropyLoss(reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape).mean(0) / np.log(2)
        labels = ["MSB", "MID", "LSB"][:n_bytes]
        return {label: float(token_bpb[i::n_bytes].mean()) for i, label in enumerate(labels)}

    def training_step(self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Batch.
            batch_idx: Batch index.

        Returns:
            Loss.
        """
        outputs = self(batch, labels=batch)
        bpb = outputs.loss / np.log(2)
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=True)
        self.log("train/bpb", bpb, on_step=True, on_epoch=True, prog_bar=True)
        if self.global_step % 500 == 0:
            with torch.no_grad():
                for label, val in self._per_byte_bpb(batch, outputs.logits).items():
                    self.log(f"train/bpb_{label}", val, on_step=True)
        return outputs.loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Validation step.

        Args:
            batch: Batch.
            batch_idx: Batch index.
            dataloader_idx: Data loader index.
        """
        outputs = self(
            input_ids=batch,
            labels=batch,
        )
        bpb = outputs.loss / np.log(2)
        dm = self.trainer.datamodule
        n_bytes = dm.val_n_bytes[dataloader_idx]
        per_byte = self._per_byte_bpb(batch, outputs.logits, n_bytes)
        self.validation_outputs[dataloader_idx].append({"bpb": float(bpb), **per_byte})

    def on_validation_epoch_start(
        self,
    ) -> None:
        """
        On validation epoch start.
        """
        dm = self.trainer.datamodule
        self.validation_outputs = {i: [] for i in range(len(dm.val_dataset_names))}

    def on_validation_epoch_end(
        self,
    ) -> None:
        """
        On validation epoch end.
        """
        dm = self.trainer.datamodule
        all_bpb = []
        for i, name in enumerate(dm.val_dataset_names):
            outputs = self.validation_outputs.get(i, [])
            if not outputs:
                continue
            mean_bpb = np.mean([o["bpb"] for o in outputs])
            self.log(f"val/bpb_{name}", mean_bpb, on_epoch=True)
            all_bpb.extend([o["bpb"] for o in outputs])
            for label in ["MSB", "MID", "LSB"][: dm.val_n_bytes[i]]:
                self.log(
                    f"val/bpb_{name}_{label}",
                    np.mean([o[label] for o in outputs]),
                    on_epoch=True,
                )
        if all_bpb:
            self.log("val/bpb", np.mean(all_bpb), on_epoch=True, prog_bar=True)
        self.validation_outputs = {}

    def configure_optimizers(
        self,
    ) -> Dict[str, Any]:
        """
        Configure optimizers.

        Returns:
            Optimizers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = (
            self.hparams.max_steps
            if self.hparams.max_steps > 0
            else self.trainer.estimated_stepping_batches
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


##################################################


# LOAD MODEL
##################################################

def load_model(
    path_model: str = MODEL_PATH,
) -> GPTAudioLightningModule:
    """
    Load the model.

    Args:
        path_model: Path to the model checkpoint.

    Returns:
        Trained GPTAudioLightningModule
    """

    # raise error if model checkpoint does not exist, give diretions on how to download it
    if not exists(path_model):
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {path_model}. " \
            f"Please run `bash setup.sh` to download it."
            )

    # Load checkpoint to infer architecture from state dict (max_position_embeddings)
    ckpt = torch.load(path_model, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    wpe_key = "model.transformer.wpe.weight"
    if wpe_key not in state:
        raise ValueError(
            f"Checkpoint missing '{wpe_key}'. Expected PyTorch Lightning checkpoint."
        )
    max_pos = state[wpe_key].shape[0]

    # Instantiate with architecture matching checkpoint, then load state
    model = GPTAudioLightningModule(max_position_embeddings=max_pos)
    model.load_state_dict(state, strict=True)

    return model

##################################################


# VOCAB SIZE
##################################################

def get_vocab_size(model_bit_depth: int = MODEL_BIT_DEPTH) -> int:
    """
    Return vocab size for token encoding (bytes 0-255 + mask).

    Args:
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.

    Returns:
        Vocab size (257 when SEPARATE_BYTE_SUBVOCABULARIES=False).
    """
    bytes_per_sample = int(math.ceil(model_bit_depth / 8))
    return (bytes_per_sample * VOCAB_PER_BYTE) + 1

##################################################


# CONVERT SAMPLES TO TOKENS
##################################################

def _get_mask_token(
    bytes_per_sample: int,
) -> int:
    """
    Get mask token.

    Args:
        bytes_per_sample: Bytes per sample.

    Returns:
        Mask token.
    """
    return bytes_per_sample * VOCAB_PER_BYTE


def convert_waveform_to_tokens(
    waveform: torch.Tensor,
    bit_depth: int = MODEL_BIT_DEPTH,
    model_bit_depth: int = MODEL_BIT_DEPTH,
) -> torch.Tensor:
    """
    Convert waveform to tokens.

    Args:
        waveform: Waveform tensor, shape (num_channels, num_samples).
        bit_depth: Bit depth. Defaults to MODEL_BIT_DEPTH.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.

    Returns:
        Flat token tensor of shape (num_channels * num_samples * bytes_per_sample,).
    """

    # raise error if audio bit depth is greater than model bit depth
    if bit_depth > model_bit_depth:
        raise ValueError(
            f"Audio bit depth {bit_depth} is greater than model bit depth {model_bit_depth}. " \
            f"Please use a model with a bit depth less than or equal to the audio bit depth."
        )

    # get constants
    bytes_per_sample = int(math.ceil(model_bit_depth / 8))
    n_bytes_audio = int(math.ceil(bit_depth / 8))
    mask_token = _get_mask_token(bytes_per_sample = bytes_per_sample)
    bits_per_sample = bit_depth

    # flatten channel-major: (C, N) or (N,) -> (C*N,)
    samples = waveform.reshape(-1).to(torch.int64)

    # convert to bytes
    byte_list = []
    for i in range(bytes_per_sample):
        if i < n_bytes_audio:
            shift = (bits_per_sample - 8) - (i * 8)
            if shift >= 0:
                byte_val = (samples >> shift) & 0xFF
            else:
                byte_val = (samples << (-shift)) & 0xFF
            if SEPARATE_BYTE_SUBVOCABULARIES:
                byte_list.append(byte_val + i * VOCAB_PER_BYTE)
            else:
                byte_list.append(byte_val)
        else:
            byte_list.append(torch.full_like(samples, mask_token, dtype = torch.int64))

    # stack bytes and reshape to flat token tensor
    tokens = torch.stack(byte_list, dim=-1).reshape(-1)

    return tokens

##################################################


# CONVERT TOKENS TO WAVEFORM
##################################################

def convert_tokens_to_waveform(
    tokens: torch.Tensor,
    num_channels: int,
    bit_depth: int = MODEL_BIT_DEPTH,
    model_bit_depth: int = MODEL_BIT_DEPTH,
) -> torch.Tensor:
    """
    Convert tokens to waveform.

    Args:
        tokens: Flat token sequence.
        num_channels: Number of channels (required to reshape).
        bit_depth: Original audio bit depth; when less than model_bit_depth, only
            that many bytes per sample are used to reconstruct. Defaults to MODEL_BIT_DEPTH.
        model_bit_depth: Model bit depth. Defaults to MODEL_BIT_DEPTH.

    Returns:
        Waveform tensor of shape (num_channels, num_samples).
    """

    # raise error if audio bit depth is greater than model bit depth
    if bit_depth > model_bit_depth:
        raise ValueError(
            f"Audio bit depth {bit_depth} is greater than model bit depth {model_bit_depth}. " \
            f"Please use a model with a bit depth greater than or equal to the audio bit depth."
        )

    # get constants
    bytes_per_sample = int(math.ceil(model_bit_depth / 8))
    n_bytes_audio = int(math.ceil(bit_depth / 8))
    mask_token = _get_mask_token(bytes_per_sample=bytes_per_sample)

    # raise error if token length is not divisible by bytes per sample
    if len(tokens) % bytes_per_sample != 0:
        raise ValueError(
            f"Token length {len(tokens)} not divisible by bytes_per_sample {bytes_per_sample}"
        )
    num_samples = len(tokens) // bytes_per_sample
    tokens = tokens.reshape(num_samples, bytes_per_sample)

    # convert to bytes
    bytes_arr = []
    for i in range(bytes_per_sample):
        t = tokens[:, i]
        if SEPARATE_BYTE_SUBVOCABULARIES:
            raw = (t - i * VOCAB_PER_BYTE) & 0xFF
        else:
            raw = t.clamp(0, 255)
        byte_val = torch.where(t == mask_token, torch.zeros_like(raw), raw)
        bytes_arr.append(byte_val)

    # reconstruct sample from bytes (msb first); only use n_bytes_audio significant bytes
    samples = bytes_arr[0].to(torch.int64)
    for i in range(1, n_bytes_audio):
        samples = (samples << 8) | bytes_arr[i].to(torch.int64)

    # reshape to waveform
    waveform = samples.reshape(num_channels, -1) # reshape to (num_channels, num_samples)
    waveform = waveform.to(torch.int64)

    return waveform

##################################################