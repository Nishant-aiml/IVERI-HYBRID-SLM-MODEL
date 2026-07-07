# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 OBJ4 physical ablation runtime tests."""

from __future__ import annotations

from pathlib import Path

import torch

from configs.base_config import apply_ablation_overrides, get_base_config
from core.exceptions import ConfigError
from model.backbone import Backbone
from model.iveri_core import IVERIModel
from model.moe.router import SparseMoERouter
from core.constants import BYTE_VOCAB_SIZE
from research.ablation_audit import (
    collect_architecture_fingerprints,
    run_ablation_audit,
    verify_pairwise_distinct,
    write_ablation_verification_report,
)
from research.campaign_runner import ABLATION_CONFIG_OVERRIDES

REPORT_PATH = Path("reports/scientific_integrity_audit/Ablation_Verification.md")


def _mini_cfg():
    from research.ablation_audit import build_mini_config

    return build_mini_config()


def test_apply_ablation_overrides_sets_model_flags() -> None:
    cfg = get_base_config()
    apply_ablation_overrides(cfg, {"use_titans": False, "use_moe": False})
    assert cfg.model.use_titans is False
    assert cfg.model.use_moe is False


def test_unknown_ablation_field_raises() -> None:
    cfg = get_base_config()
    try:
        apply_ablation_overrides(cfg, {"use_nonexistent": False})
    except ConfigError:
        return
    raise AssertionError("Expected ConfigError for unknown ablation field")


def test_no_titans_skips_titans_module() -> None:
    cfg = _mini_cfg()
    cfg.model.use_titans = False
    backbone = Backbone(cfg)
    assert backbone.titans is None
    x = torch.randn(1, 4, cfg.model.hidden_dim)
    out = backbone(x, entropy=torch.rand(1, 4, 1))
    assert out.shape == x.shape


def test_no_blt_uses_byte_embed_path() -> None:
    cfg = _mini_cfg()
    cfg.model.use_blt = False
    model = IVERIModel(cfg)
    assert not hasattr(model, "entropy_model")
    assert hasattr(model, "byte_embed")
    out = model(torch.randint(0, 256, (2, 12)), return_dict=False)
    assert out.shape == (2, 12, BYTE_VOCAB_SIZE)


def test_no_moe_uses_dense_ffn() -> None:
    cfg = _mini_cfg()
    cfg.model.use_moe = False
    block = Backbone(cfg).blocks[0].sub_block
    assert not block.use_moe
    assert hasattr(block, "dense_ffn")
    assert not hasattr(block, "moe_router")


def test_no_entropy_routing_ignores_entropy() -> None:
    cfg = _mini_cfg()
    cfg.model.use_entropy_routing = False
    router = SparseMoERouter(cfg)
    router.eval()
    hidden = torch.randn(2, 5, cfg.model.hidden_dim)
    with torch.no_grad():
        la, _, _, _, _ = router._gating_logits(hidden, entropy=torch.zeros(2, 5, 1))
        lb, _, _, _, _ = router._gating_logits(hidden, entropy=torch.ones(2, 5, 1))
    assert torch.allclose(la, lb)


def test_campaign_ablation_keys_map_to_model_flags() -> None:
    for key, overrides in ABLATION_CONFIG_OVERRIDES.items():
        if key == "none":
            continue
        for field in overrides:
            assert hasattr(get_base_config().model, field), field


def test_ablation_architectures_pairwise_distinct() -> None:
    fingerprints = collect_architecture_fingerprints()
    distinct, collisions = verify_pairwise_distinct(fingerprints)
    assert distinct, collisions


def test_full_ablation_audit_pass() -> None:
    result = run_ablation_audit()
    assert result.production_verdict == "PASS"
    assert all(p.component_absent for p in result.probes)
    assert result.pairwise_distinct
    assert not any(a.severity == "CRITICAL" for a in result.antipatterns)


def test_write_ablation_verification_report() -> None:
    result = write_ablation_verification_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "Phase 6.3.1F" in text
    assert "Architecture Fingerprints" in text
    assert result.production_verdict == "PASS"
