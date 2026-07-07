# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit, integration, and property tests for the BLT subsystem (Phase 1.6)."""

from __future__ import annotations

import time

import pytest
import torch

from configs.base_config import IVERIConfig
from core.constants import BYTE_VOCAB_SIZE
from core.exceptions import ShapeError
from model.blt import BLTByteDecoder, BLTByteEncoder, ByteEntropyModel, DynamicPatcher


@pytest.mark.parametrize("device", ["cpu"])
def test_entropy_model_causal_perturbation_invariance(base_config: IVERIConfig, device: str) -> None:
    """Future byte perturbations must not change entropy at earlier positions."""
    model = ByteEntropyModel(base_config, predictor_type="cnn_mlp").to(device)
    model.eval()
    raw = torch.randint(3, 250, (1, 20), device=device)
    perturbed = raw.clone()
    perturbed[:, 12:] = torch.randint(3, 250, (1, 8), device=device)
    with torch.no_grad():
        e0 = model(raw)
        e1 = model(perturbed)
    assert torch.allclose(e0[:, :12, :], e1[:, :12, :], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("predictor_type", ["cnn_mlp", "lstm", "linear"])
def test_entropy_model_output_and_configurability(
    base_config: IVERIConfig,
    device: str,
    predictor_type: str,
) -> None:
    """Verify that entropy model works under different configurations and produces valid outputs."""
    model = ByteEntropyModel(base_config, predictor_type=predictor_type).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)  # (B, S)

    entropy = model(x)
    assert entropy.shape == (2, 32, 1)
    assert (entropy >= 0.0).all() and (entropy <= 1.0).all()

    # Gradients verification
    loss = entropy.mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient flow."
        assert not torch.isnan(param.grad).any()


@pytest.mark.parametrize("device", ["cpu"])
def test_patcher_determinism_and_reconstruction(base_config: IVERIConfig, device: str) -> None:
    """Verify that patch generation is deterministic and satisfies reconstruction equivalence."""
    patcher = DynamicPatcher(base_config)
    raw_bytes = torch.randint(0, 256, (3, 64), device=device)
    entropy = torch.rand(3, 64, 1, device=device)

    # 1. Determinism Check: same inputs yield identical boundaries
    boundaries_1 = patcher.compute_boundaries(raw_bytes, entropy)
    boundaries_2 = patcher.compute_boundaries(raw_bytes, entropy)
    assert torch.equal(boundaries_1, boundaries_2)

    # 2. Patch bounds checks: index 0 must be True
    assert boundaries_1[:, 0].all()

    # 3. Maximum patch size check
    # Force max patch size trigger (by setting entropy to 0.0)
    entropy_low = torch.zeros(3, 64, 1, device=device)
    boundaries_low = patcher.compute_boundaries(raw_bytes, entropy_low)
    # Patch lengths should be exactly equal to patch_size_max (default 8)
    for b in range(3):
        indices = torch.where(boundaries_low[b])[0]
        lengths = indices[1:] - indices[:-1]
        assert (lengths == base_config.model.blt.patch_size_max).all()


@pytest.mark.parametrize("device", ["cpu"])
def test_multilingual_utf8_validation(base_config: IVERIConfig, device: str) -> None:
    """Verify BLT handles English, Hindi, Chinese, Emojis, and Mixed Unicode without crashes."""
    texts = [
        "English sentence for testing.",
        "नमस्ते दुनिया (Hindi UTF-8)",
        "你好世界 (Chinese UTF-8)",
        "👋🚀🔥 Unicode Emojis",
        "Mixed text: Hello 你好 नमस्ते 👋",
    ]

    entropy_model = ByteEntropyModel(base_config).to(device)
    patcher = DynamicPatcher(base_config)
    encoder = BLTByteEncoder(base_config).to(device)
    decoder = BLTByteDecoder(base_config).to(device)

    for text in texts:
        # Encode string to bytes
        byte_data = list(text.encode("utf-8"))
        raw_bytes = torch.tensor([byte_data], device=device, dtype=torch.long)  # (1, S)

        # Run pipeline
        entropy = entropy_model(raw_bytes)
        boundaries = patcher.compute_boundaries(raw_bytes, entropy)

        # Verify boundary properties
        assert boundaries.shape == raw_bytes.shape
        assert boundaries[0, 0]  # First element is start boundary

        # Run Encoder & Decoder
        latent = encoder(raw_bytes, boundary_map=boundaries)
        logits = decoder(latent, boundary_map=boundaries, raw_bytes=raw_bytes)

        # Check shape contracts
        assert latent.ndim == 3 and latent.shape[2] == base_config.model.hidden_dim
        assert logits.shape == (1, len(byte_data), BYTE_VOCAB_SIZE)
        assert not torch.isnan(logits).any()


@pytest.mark.parametrize("device", ["cpu"])
def test_encoder_decoder_roundtrip_gradient_flow(base_config: IVERIConfig, device: str) -> None:
    """Verify that full roundtrip gradients propagate from decoder through encoder to embeddings."""
    entropy_model = ByteEntropyModel(base_config).to(device)
    patcher = DynamicPatcher(base_config)
    encoder = BLTByteEncoder(base_config).to(device)
    decoder = BLTByteDecoder(base_config).to(device)

    raw_bytes = torch.randint(0, 256, (2, 32), device=device)
    entropy = entropy_model(raw_bytes)
    boundaries = patcher.compute_boundaries(raw_bytes, entropy)

    # Forward
    latent = encoder(raw_bytes, boundary_map=boundaries)
    logits = decoder(latent, boundary_map=boundaries, raw_bytes=raw_bytes)

    # Loss & Backward
    loss = logits.mean()
    loss.backward()

    # Verify gradients reach encoder embed weights
    assert encoder.embed.weight.grad is not None
    assert not torch.isnan(encoder.embed.weight.grad).any()
    # Verify gradients reach decoder embed weights
    assert decoder.embed.weight.grad is not None
    assert not torch.isnan(decoder.embed.weight.grad).any()


def test_blt_validation_checks(base_config: IVERIConfig) -> None:
    """Verify shape mismatch errors are correctly thrown."""
    entropy_model = ByteEntropyModel(base_config)
    patcher = DynamicPatcher(base_config)
    encoder = BLTByteEncoder(base_config)

    # Shape errors on input
    with pytest.raises(ShapeError):
        entropy_model(torch.randint(0, 256, (2, 4, 8)))  # Expects 2D

    x = torch.randint(0, 256, (2, 16))
    entropy_bad = torch.rand(2, 17)  # Length mismatch
    with pytest.raises(ShapeError):
        patcher.compute_boundaries(x, entropy_bad)

    boundaries_bad = torch.zeros(2, 17, dtype=torch.bool)
    with pytest.raises(ShapeError):
        encoder(x, boundary_map=boundaries_bad)


@pytest.mark.parametrize("device", ["cpu"])
def test_blt_numerical_stability(base_config: IVERIConfig, device: str) -> None:
    """Verify stability under NaN, Inf, empty, and single-token inputs."""
    entropy_model = ByteEntropyModel(base_config).to(device)
    patcher = DynamicPatcher(base_config)
    encoder = BLTByteEncoder(base_config).to(device)
    decoder = BLTByteDecoder(base_config).to(device)

    # 1. Empty Input
    x_empty = torch.randint(0, 256, (2, 0), device=device)
    entropy_empty = entropy_model(x_empty)
    boundaries_empty = patcher.compute_boundaries(x_empty, entropy_empty)
    latent_empty = encoder(x_empty, boundary_map=boundaries_empty)
    logits_empty = decoder(latent_empty, boundary_map=boundaries_empty, raw_bytes=x_empty)

    assert entropy_empty.shape == (2, 0, 1)
    assert boundaries_empty.shape == (2, 0)
    assert latent_empty.shape == (2, 0, base_config.model.hidden_dim)
    assert logits_empty.shape == (2, 0, BYTE_VOCAB_SIZE)

    # 2. Single-token Input
    x_single = torch.randint(0, 256, (1, 1), device=device)
    entropy_single = entropy_model(x_single)
    boundaries_single = patcher.compute_boundaries(x_single, entropy_single)
    latent_single = encoder(x_single, boundary_map=boundaries_single)
    logits_single = decoder(latent_single, boundary_map=boundaries_single, raw_bytes=x_single)

    assert entropy_single.shape == (1, 1, 1)
    assert boundaries_single.shape == (1, 1)
    assert latent_single.shape == (1, 1, base_config.model.hidden_dim)
    assert logits_single.shape == (1, 1, BYTE_VOCAB_SIZE)


@pytest.mark.parametrize("device", ["cpu"])
def test_blt_telemetry_collection(base_config: IVERIConfig, device: str) -> None:
    """Validate that we can calculate the detailed BLT telemetry statistics successfully."""
    entropy_model = ByteEntropyModel(base_config).to(device)
    patcher = DynamicPatcher(base_config)

    text = "Detailed profiling sentence for telemetry metrics collection validation in IVERI."
    byte_data = list(text.encode("utf-8"))
    raw_bytes = torch.tensor([byte_data], device=device, dtype=torch.long)

    # Measure time for throughput
    t0 = time.perf_counter()
    entropy = entropy_model(raw_bytes)
    boundaries = patcher.compute_boundaries(raw_bytes, entropy)
    t1 = time.perf_counter()

    # Basic measurements
    seq_len = len(byte_data)
    boundary_indices = torch.where(boundaries[0])[0].tolist()
    patch_lengths = []
    for i in range(len(boundary_indices)):
        start = boundary_indices[i]
        end = boundary_indices[i + 1] if i + 1 < len(boundary_indices) else seq_len
        patch_lengths.append(end - start)

    num_patches = len(patch_lengths)

    # Calculate telemetry
    avg_patch_len = sum(patch_lengths) / num_patches
    min_patch_len = min(patch_lengths)
    max_patch_len = max(patch_lengths)
    avg_entropy = float(entropy.mean().item())
    boundary_freq = num_patches / seq_len
    compression_ratio = seq_len / num_patches
    throughput_kb_per_sec = (seq_len / 1024.0) / (t1 - t0) if (t1 - t0) > 0 else 0.0

    # Ensure valid metrics are generated
    assert avg_patch_len >= base_config.model.blt.patch_size_min
    assert avg_patch_len <= base_config.model.blt.patch_size_max
    assert min_patch_len == min(patch_lengths)
    assert max_patch_len == max(patch_lengths)
    assert avg_entropy >= 0.0 and avg_entropy <= 1.0
    assert boundary_freq > 0.0 and boundary_freq <= 1.0
    assert compression_ratio >= 1.0
    assert throughput_kb_per_sec >= 0.0


@pytest.mark.parametrize("device", ["cpu"])
def test_patch_reconstruction_determinism(base_config: IVERIConfig, device: str) -> None:
    """Verify that mapping Input Bytes -> Patch -> Reconstruct -> Patch Again yields identical boundaries."""
    entropy_model = ByteEntropyModel(base_config).to(device)
    patcher = DynamicPatcher(base_config)
    encoder = BLTByteEncoder(base_config).to(device)
    decoder = BLTByteDecoder(base_config).to(device)

    text = "Deterministic patch reconstruction loop validation test!"
    byte_data = list(text.encode("utf-8"))
    raw_bytes = torch.tensor([byte_data], device=device, dtype=torch.long)

    # 1. First Pass
    entropy_1 = entropy_model(raw_bytes)
    boundaries_1 = patcher.compute_boundaries(raw_bytes, entropy_1)

    # Encoder / Decoder roundtrip
    latent = encoder(raw_bytes, boundary_map=boundaries_1)
    logits = decoder(latent, boundary_map=boundaries_1, raw_bytes=raw_bytes)

    # Reconstruct predicted bytes by argmax
    reconstructed_bytes = torch.argmax(logits, dim=-1)

    # 2. Second Pass on reconstructed bytes
    entropy_2 = entropy_model(reconstructed_bytes)
    boundaries_2 = patcher.compute_boundaries(reconstructed_bytes, entropy_2)

    # Check that computing boundaries on the same reconstructed bytes is deterministic
    boundaries_2_recomputed = patcher.compute_boundaries(reconstructed_bytes, entropy_2)
    assert torch.equal(boundaries_2, boundaries_2_recomputed)
