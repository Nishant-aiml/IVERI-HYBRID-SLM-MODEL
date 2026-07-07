# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for inference package (Phase 6.3.3)."""

from __future__ import annotations

from configs.base_config import get_base_config
from inference import ByteTokenizer, InferenceEngine, load_inference_model
from inference.benchmark import benchmark_inference


def _mini_model():
    cfg = get_base_config()
    cfg.model.hidden_dim = 32
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.mamba_ratio = 1
    cfg.model.num_experts = 2
    cfg.model.num_active_experts = 1
    cfg.model.max_recursion_depth = 2
    cfg.model.titans_memory_dim = 16
    cfg.hardware.device = "cpu"
    cfg.validate()
    return load_inference_model(config=cfg, device="cpu")


def test_byte_tokenizer_roundtrip() -> None:
    tok = ByteTokenizer()
    text = "Hello 世界"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_inference_engine_generate() -> None:
    model = _mini_model()
    engine = InferenceEngine(model)
    result = engine.generate("Hi", max_new_tokens=4)
    assert isinstance(result.text, str)
    assert result.latency_seconds >= 0.0


def test_inference_stream_yields_chunks() -> None:
    model = _mini_model()
    engine = InferenceEngine(model)
    chunks = list(engine.stream("A", max_new_tokens=3))
    assert len(chunks) >= 1


def test_benchmark_inference_cpu() -> None:
    model = _mini_model()
    engine = InferenceEngine(model)
    stats = benchmark_inference(engine, runs=2, warmup=1)
    assert "avg_tokens_per_second" in stats
