# Copyright 2026 IVERI Project
# Phase 1.9.1 — Stress Test Suite

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from configs.base_config import get_base_config
from core.constants import BOS_BYTE, BYTE_VOCAB_SIZE, EOS_BYTE, PAD_BYTE, RAW_BYTE_VOCAB_SIZE
from model.iveri_core import IVERIModel


@pytest.fixture
def model():
    cfg = get_base_config()
    m = IVERIModel(cfg)
    m.eval()
    return m


def test_empty_sequence(model):
    """Empty tensor must not crash."""
    raw = torch.zeros(2, 0, dtype=torch.long)
    out = model(raw, return_dict=True)
    assert out["logits"].shape == (2, 0, BYTE_VOCAB_SIZE)
    assert out["aux_loss"].item() == 0.0


def test_single_token(model):
    """Single byte must produce valid output."""
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (1, 1))
    out = model(raw, return_dict=True)
    assert out["logits"].shape == (1, 1, BYTE_VOCAB_SIZE)
    assert not torch.isnan(out["logits"]).any()


def test_large_batch(model):
    """Large batch size must work."""
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (8, 64))
    out = model(raw, return_dict=True)
    assert out["logits"].shape == (8, 64, BYTE_VOCAB_SIZE)
    assert not torch.isnan(out["logits"]).any()


def test_long_sequence(model):
    """Long sequences must not crash."""
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (1, 256))
    out = model(raw, return_dict=True)
    assert out["logits"].shape[0] == 1
    assert out["logits"].shape[2] == BYTE_VOCAB_SIZE
    assert not torch.isnan(out["logits"]).any()


@pytest.mark.parametrize(
    "text,lang",
    [
        (b"Hello, world! IVERI test.", "English"),
        (
            "\u092f\u0939 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0939\u0948".encode("utf-8"),
            "Hindi",
        ),
        ("\u6d4b\u8bd5\u53e5\u5b50".encode("utf-8"), "Chinese"),
        ("\u0645\u0631\u062d\u0628\u0627".encode("utf-8"), "Arabic"),
        ("\U0001f525\U0001f680\U0001f4a1".encode("utf-8"), "Emoji"),
        (
            "Hello \u0928\u092e\u0938\u094d\u0924\u0947 \u4f60\u597d \U0001f30d".encode("utf-8"),
            "Mixed",
        ),
    ],
)
def test_multilingual_utf8(model, text, lang):
    """Multilingual byte sequences must produce valid outputs."""
    byte_data = list(text)
    raw = torch.tensor([byte_data], dtype=torch.long)
    out = model(raw, return_dict=True)
    assert out["logits"].shape == (1, len(byte_data), BYTE_VOCAB_SIZE), f"{lang} shape mismatch"
    assert not torch.isnan(out["logits"]).any(), f"{lang} has NaN"
    assert not torch.isinf(out["logits"]).any(), f"{lang} has Inf"


def test_all_zeros(model):
    """All-zero (PAD) bytes must not crash."""
    raw = torch.zeros(2, 16, dtype=torch.long)
    out = model(raw, return_dict=True)
    assert not torch.isnan(out["logits"]).any()


def test_all_max_bytes(model):
    """All-255 bytes must not crash."""
    raw = torch.full((2, 16), 255, dtype=torch.long)
    out = model(raw, return_dict=True)
    assert not torch.isnan(out["logits"]).any()


def test_determinism_100_runs(model):
    """Eval mode must be bitwise deterministic across 100 runs."""
    torch.manual_seed(42)
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (2, 12))
    with torch.no_grad():
        ref = model(raw, return_dict=False).clone()
        for _ in range(99):
            out = model(raw, return_dict=False)
            assert torch.equal(ref, out), "Non-determinism detected!"


def test_repeated_inference_no_memory_leak(model):
    """50 forward passes must not accumulate gradients."""
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (1, 8))
    with torch.no_grad():
        for _ in range(50):
            out = model(raw, return_dict=True)
            assert out["logits"].grad_fn is None, "grad_fn found inside no_grad context"


def test_checkpoint_5_cycles():
    """5 save-load cycles must preserve bitwise identical weights."""
    cfg = get_base_config()
    m = IVERIModel(cfg)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    original_params = [p.clone() for p in m.parameters()]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.pt"
        for i in range(5):
            m.save_checkpoint(path, step=i, metrics={"cycle": i})
            m2 = IVERIModel(cfg)
            m2.load_checkpoint(path)
            for p_orig, p_loaded in zip(original_params, m2.parameters(), strict=False):
                assert torch.equal(p_orig, p_loaded), f"Cycle {i}: param mismatch"


def test_optimizer_compatibility():
    """Standard AdamW optimizer must work with the model."""
    cfg = get_base_config()
    m = IVERIModel(cfg)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (2, 8))
    out = m(raw, return_dict=True)
    loss = out["logits"].mean() + 0.01 * out["aux_loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # Verify model still runs after optimizer step
    with torch.no_grad():
        out2 = m(raw, return_dict=False)
    assert out2.shape == (2, 8, BYTE_VOCAB_SIZE)


def test_train_eval_mode_switch():
    """Switching between train/eval multiple times must remain stable."""
    cfg = get_base_config()
    m = IVERIModel(cfg)
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (2, 8))
    for _ in range(3):
        m.train()
        out_train = m(raw, return_dict=False)
        m.eval()
        with torch.no_grad():
            out_eval = m(raw, return_dict=False)
        assert not torch.isnan(out_train).any()
        assert not torch.isnan(out_eval).any()


def test_cpu_device_compatibility():
    """Model must work cleanly on CPU."""
    cfg = get_base_config()
    m = IVERIModel(cfg).to("cpu")
    for p in m.parameters():
        assert p.device.type == "cpu"
    raw = torch.randint(0, RAW_BYTE_VOCAB_SIZE, (1, 8))
    out = m(raw, return_dict=True)
    assert out["logits"].device.type == "cpu"
    assert not torch.isnan(out["logits"]).any()


def test_bos_eos_pad_bytes(model):
    """Collision-free special byte IDs must not crash."""
    for byte_val in [PAD_BYTE, BOS_BYTE, EOS_BYTE]:
        raw = torch.full((1, 8), byte_val, dtype=torch.long)
        out = model(raw, return_dict=True)
        assert not torch.isnan(out["logits"]).any(), f"NaN with byte={byte_val}"
