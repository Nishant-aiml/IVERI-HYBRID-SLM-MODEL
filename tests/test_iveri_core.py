# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration and numerical validation tests for the IVERI CORE Model (Phase 1.9)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from configs.base_config import IVERIConfig, get_base_config
from core.constants import ARCHITECTURE_VERSION, BYTE_VOCAB_SIZE
from core.exceptions import CheckpointError
from model.iveri_core import IVERIModel


@pytest.fixture
def test_config() -> IVERIConfig:
    """Create a mini 10M Nano configuration for testing."""
    cfg = get_base_config()
    cfg.model.hidden_dim = 64
    cfg.model.num_heads = 2
    cfg.model.num_layers = 2
    cfg.model.mamba_ratio = 2
    cfg.model.num_experts = 4
    cfg.model.num_active_experts = 2
    cfg.model.max_recursion_depth = 4
    cfg.model.titans_memory_dim = 32
    cfg.validate()
    return cfg


def test_model_initialization_and_forward(test_config: IVERIConfig) -> None:
    """Verify IVERIModel initializes properly and executes its forward pass."""
    model = IVERIModel(test_config)
    assert model.architecture_version == ARCHITECTURE_VERSION

    # Input: shape (B, S)
    B, S = 2, 16
    raw_bytes = torch.randint(0, 256, (B, S), dtype=torch.long)

    # 1. Forward returning logits directly
    logits = model(raw_bytes, return_dict=False)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (B, S, BYTE_VOCAB_SIZE)
    assert logits.dtype == torch.float32

    # 2. Forward returning dict schema
    out_dict = model(raw_bytes, return_dict=True)
    assert isinstance(out_dict, dict)
    expected_keys = {
        "logits",
        "byte_entropy",
        "patch_entropy",
        "boundary_map",
        "aux_loss",
        "telemetry",
    }
    assert set(out_dict.keys()) == expected_keys

    assert out_dict["logits"].shape == (B, S, BYTE_VOCAB_SIZE)
    assert out_dict["byte_entropy"].shape == (B, S, 1)
    assert out_dict["boundary_map"].shape == (B, S)
    assert out_dict["boundary_map"].dtype == torch.bool
    assert out_dict["aux_loss"].shape == ()
    assert isinstance(out_dict["telemetry"], dict)


def test_end_to_end_gradient_flow(test_config: IVERIConfig) -> None:
    """Verify that gradients propagate from the loss all the way to raw byte embeddings."""
    model = IVERIModel(test_config)

    B, S = 2, 8
    raw_bytes = torch.randint(0, 256, (B, S), dtype=torch.long)

    # Run in training mode
    model.train()

    # Forward
    out_dict = model(raw_bytes, return_dict=True)
    logits = out_dict["logits"]
    aux_loss = out_dict["aux_loss"]

    # Compute a mock loss
    target = torch.randint(0, 256, (B, S), dtype=torch.long)
    lm_loss = torch.nn.functional.cross_entropy(logits.view(-1, BYTE_VOCAB_SIZE), target.view(-1))
    total_loss = lm_loss + 0.01 * aux_loss

    # Backward
    total_loss.backward()

    # Verify gradients reach BLT Entropy Model Embeddings
    assert model.entropy_model.embed.weight.grad is not None
    assert not torch.isnan(model.entropy_model.embed.weight.grad).any()

    # Verify gradients reach BLT Encoder Embeddings
    assert model.encoder.embed.weight.grad is not None
    assert not torch.isnan(model.encoder.embed.weight.grad).any()

    # Verify gradients reach BLT Decoder Embeddings
    assert model.decoder.embed.weight.grad is not None
    assert not torch.isnan(model.decoder.embed.weight.grad).any()

    # Verify gradients reach Backbone parameters
    has_backbone_grad = False
    for p in model.backbone.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()
            has_backbone_grad = True
    assert has_backbone_grad, "Backbone parameters must receive gradients."


def test_tensor_signature_contract_validation(test_config: IVERIConfig) -> None:
    """Explicitly validate shape, dtype, device, and requires_grad contracts at each pipeline stage."""
    model = IVERIModel(test_config)
    model.train()

    B, S = 2, 12
    raw_bytes = torch.randint(0, 256, (B, S), dtype=torch.long)

    # 1. Byte Entropy Stage
    byte_entropy = model.entropy_model(raw_bytes)
    assert byte_entropy.shape == (B, S, 1)
    assert byte_entropy.dtype == torch.float32
    assert byte_entropy.requires_grad

    # 2. Boundary Map Stage
    boundary_map = model.patcher.compute_boundaries(raw_bytes, byte_entropy)
    assert boundary_map.shape == (B, S)
    assert boundary_map.dtype == torch.bool
    assert not boundary_map.requires_grad  # Discrete boundary mapping has no gradients

    # 3. Patch Embeddings Stage
    latent_patches = model.encoder.encode_with_boundaries(raw_bytes, boundary_map)
    P_max = latent_patches.shape[1]
    assert latent_patches.shape == (B, P_max, test_config.model.hidden_dim)
    assert latent_patches.dtype == torch.float32
    assert latent_patches.requires_grad

    # 4. Patch Entropy Stage
    patch_ids = torch.cumsum(boundary_map.long(), dim=-1) - 1
    patch_indices = torch.arange(P_max, device=raw_bytes.device).view(1, -1, 1)
    is_patch = patch_ids.unsqueeze(1) == patch_indices
    patch_lengths = is_patch.sum(dim=-1, keepdim=True)
    patch_lengths_clamped = torch.clamp(patch_lengths, min=1)
    M = is_patch.float() / patch_lengths_clamped.float()
    patch_entropy = torch.bmm(M, byte_entropy)
    assert patch_entropy.shape == (B, P_max, 1)
    assert patch_entropy.dtype == torch.float32
    assert patch_entropy.requires_grad

    # 5. Backbone Output Stage
    backbone_out = model.backbone(latent_patches, entropy=patch_entropy)
    assert backbone_out.shape == (B, P_max, test_config.model.hidden_dim)
    assert backbone_out.dtype == torch.float32
    assert backbone_out.requires_grad

    # 6. Decoded Logits Stage
    logits = model.decoder.decode_with_boundaries(backbone_out, boundary_map, raw_bytes)
    assert logits.shape == (B, S, BYTE_VOCAB_SIZE)
    assert logits.dtype == torch.float32
    assert logits.requires_grad


def test_multilingual_utf8_pipeline(test_config: IVERIConfig) -> None:
    """Verify that multi-lingual UTF-8 strings process without crashing."""
    model = IVERIModel(test_config)

    texts = [
        "IVERI CORE model test.",
        "यह एक परीक्षण वाक्य है।",
        "这是一个测试句子。",
        "👋 Unicode emojis check! 🚀🔥",
    ]

    for text in texts:
        byte_data = list(text.encode("utf-8"))
        raw_bytes = torch.tensor([byte_data], dtype=torch.long)  # (1, S)
        S = raw_bytes.shape[1]

        out = model(raw_bytes, return_dict=True)
        assert out["logits"].shape == (1, S, BYTE_VOCAB_SIZE)
        assert not torch.isnan(out["logits"]).any()


def test_boundary_conditions(test_config: IVERIConfig) -> None:
    """Stress test boundary cases including empty sequences and single tokens."""
    model = IVERIModel(test_config)

    # 1. Empty sequence
    empty_bytes = torch.zeros(2, 0, dtype=torch.long)
    out_empty = model(empty_bytes, return_dict=True)
    assert out_empty["logits"].shape == (2, 0, BYTE_VOCAB_SIZE)
    assert out_empty["byte_entropy"].shape == (2, 0, 1)
    assert out_empty["boundary_map"].shape == (2, 0)
    assert out_empty["aux_loss"].item() == 0.0

    # 2. Single token
    single_byte = torch.randint(0, 256, (1, 1), dtype=torch.long)
    out_single = model(single_byte, return_dict=True)
    assert out_single["logits"].shape == (1, 1, BYTE_VOCAB_SIZE)
    assert not torch.isnan(out_single["logits"]).any()


def test_checkpoint_save_and_load(test_config: IVERIConfig) -> None:
    """Verify checkpoint save and load restoration matches contract specifications."""
    model_src = IVERIModel(test_config)
    model_dst = IVERIModel(test_config)

    # Randomize source model weights to verify copying
    with torch.no_grad():
        for p in model_src.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Save to a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "checkpoint.pt"

        # Save
        model_src.save_checkpoint(
            path=checkpoint_path,
            step=12300,
            metrics={"loss": 1.25, "val_loss": 1.45},
            seed=999,
        )
        assert checkpoint_path.exists()

        # Load
        meta = model_dst.load_checkpoint(checkpoint_path)
        assert meta["step"] == 12300
        assert meta["random_seed"] == 999
        assert meta["metrics"] == {"loss": 1.25, "val_loss": 1.45}
        assert meta["checkpoint_version"] == "1.0"

        # Verify all parameters match bitwise
        for p_src, p_dst in zip(model_src.parameters(), model_dst.parameters(), strict=True):
            assert torch.equal(p_src, p_dst)


def test_checkpoint_incompatibility(test_config: IVERIConfig) -> None:
    """Verify that checkpoint loading rejects mismatched architecture configurations."""
    model = IVERIModel(test_config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "bad_checkpoint.pt"

        # Mock an incompatible checkpoint dict
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "architecture_version": "0.2.0-incompatible",
            "checkpoint_version": "1.0",
        }
        torch.save(checkpoint, checkpoint_path)

        # Restoring should fail due to mismatched version
        with pytest.raises(CheckpointError, match="Architecture version mismatch"):
            model.load_checkpoint(checkpoint_path)


def test_inference_and_determinism_consistency(test_config: IVERIConfig) -> None:
    """Assert that evaluation inference is strictly deterministic and stable across switches."""
    torch.manual_seed(42)
    model = IVERIModel(test_config)

    B, S = 2, 10
    raw_bytes = torch.randint(0, 256, (B, S), dtype=torch.long)

    # Execute twice in eval mode
    model.eval()
    with torch.no_grad():
        out1 = model(raw_bytes, return_dict=False)
        out2 = model(raw_bytes, return_dict=False)
    assert torch.equal(out1, out2)

    # Switch to training and back to eval to verify state preservation
    model.train()
    _ = model(raw_bytes)

    model.eval()
    with torch.no_grad():
        out3 = model(raw_bytes, return_dict=False)
    assert torch.equal(out1, out3)


def test_device_transfer_compatibility(test_config: IVERIConfig) -> None:
    """Sanity check verifying model registers on different devices successfully."""
    model = IVERIModel(test_config)

    # Verify parameter device migration
    model.to("cpu")
    for p in model.parameters():
        assert p.device.type == "cpu"


def test_memory_leak_sanity(test_config: IVERIConfig) -> None:
    """Confirm that successive forward steps do not leak tensor histories or build context memory."""
    model = IVERIModel(test_config)
    model.eval()

    B, S = 1, 8
    raw_bytes = torch.randint(0, 256, (B, S), dtype=torch.long)

    # Ensure no grad context prevents tracking history
    with torch.no_grad():
        for _ in range(5):
            out = model(raw_bytes, return_dict=True)
            assert out["logits"].grad_fn is None
