# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte-Level Entropy Model for the Byte Latent Transformer (BLT).

Implements the predictability scoring model that maps raw UTF-8 byte sequences
to normalized, position-wise Shannon entropy values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from configs.base_config import IVERIConfig
from core.constants import BYTE_VOCAB_SIZE
from core.interfaces import BaseModule
from core.registry import register
from utils.validation import validate_shape


@register("blt_entropy_model")
class ByteEntropyModel(BaseModule):
    """Predictor network that computes entropy estimates over raw UTF-8 bytes.

    Maintains a configurable sequence prediction backend (CNN-MLP, LSTM, or Linear)
    and computes the normalized Shannon entropy over the resulting probability distribution.
    """

    cnn: nn.Conv1d | None
    lstm: nn.LSTM | None
    predictor: nn.Module

    def __init__(self, config: IVERIConfig, predictor_type: str = "cnn_mlp") -> None:
        """Initialize the byte entropy model.

        Args:
            config: Configuration object containing model parameters.
            predictor_type: Type of internal predictor ("cnn_mlp", "lstm", "linear").
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.predictor_type = predictor_type

        # Byte embedding layer
        self.embed = nn.Embedding(BYTE_VOCAB_SIZE, self.hidden_dim)
        self.cnn = None
        self.lstm = None

        # Configurable predictor backends
        if predictor_type == "cnn_mlp":
            # Causal convolution: left-pad only (no future-byte leakage).
            self.cnn = nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=0,
            )
            self._cnn_kernel_size = 3
            self.predictor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.hidden_dim, BYTE_VOCAB_SIZE),
            )
        elif predictor_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.predictor = nn.Linear(self.hidden_dim, BYTE_VOCAB_SIZE)
        elif predictor_type == "linear":
            self.predictor = nn.Linear(self.hidden_dim, BYTE_VOCAB_SIZE)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the model parameters."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if self.predictor_type == "cnn_mlp" and self.cnn is not None:
            nn.init.kaiming_uniform_(self.cnn.weight, nonlinearity="relu")
            if self.cnn.bias is not None:
                nn.init.zeros_(self.cnn.bias)
            if isinstance(self.predictor, nn.Sequential):
                for m in self.predictor:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        elif self.predictor_type == "lstm" and self.lstm is not None:
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            if isinstance(self.predictor, nn.Linear):
                nn.init.normal_(self.predictor.weight, mean=0.0, std=0.02)
                if self.predictor.bias is not None:
                    nn.init.zeros_(self.predictor.bias)
        elif self.predictor_type == "linear" and isinstance(self.predictor, nn.Linear):
            nn.init.normal_(self.predictor.weight, mean=0.0, std=0.02)
            if self.predictor.bias is not None:
                nn.init.zeros_(self.predictor.bias)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Compute the normalized entropy scores for the input byte sequence.

        Args:
            x: Input byte indices tensor of shape (B, S), values in [0, BYTE_VOCAB_SIZE).
            **kwargs: Extra arguments.

        Returns:
            Normalized entropy tensor of shape (B, S, 1) with values in [0.0, 1.0].
        """
        validate_shape(x, (-1, -1), name="input bytes x")
        batch_size, seq_len = x.shape

        # Handle empty sequences gracefully
        if seq_len == 0:
            return torch.zeros(batch_size, 0, 1, device=x.device, dtype=torch.float32)

        # 1. Embed bytes
        h = self.embed(x)  # (B, S, E)

        # 2. Run predictor
        if self.predictor_type == "cnn_mlp":
            # nn.Conv1d expects (B, E, S); left-pad for causal receptive field.
            assert self.cnn is not None
            h_trans = h.transpose(1, 2)
            k = getattr(self, "_cnn_kernel_size", 3)
            h_padded = F.pad(h_trans, (k - 1, 0))
            h_cnn = self.cnn(h_padded)
            h_features = h_cnn.transpose(1, 2)  # (B, S, E)
            logits = self.predictor(h_features)  # (B, S, 256)
        elif self.predictor_type == "lstm":
            assert self.lstm is not None
            h_lstm, _ = self.lstm(h)
            logits = self.predictor(h_lstm)  # (B, S, 256)
        else:  # "linear"
            logits = self.predictor(h)  # (B, S, 256)

        # 3. Calculate Shannon entropy over next-byte prediction logits
        # Numerical stability: use log_softmax to compute probabilities and log-probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Compute H_t = -sum(p * log2(p))
        # log2(p) = ln(p) / ln(2)
        ln_2 = 0.6931471805599453
        entropy = -torch.sum(probs * (log_probs / ln_2), dim=-1, keepdim=True)

        # 4. Normalize: maximum entropy for BYTE_VOCAB_SIZE outcomes
        max_entropy = math.log2(BYTE_VOCAB_SIZE)
        normalized_entropy = entropy / max_entropy

        # Clamp for safety against floating point inaccuracies
        normalized_entropy = torch.clamp(normalized_entropy, min=0.0, max=1.0)

        validate_shape(normalized_entropy, (batch_size, seq_len, 1), name="normalized_entropy")
        return normalized_entropy
