import json
from pathlib import Path

import torch

from model.moe.router import SparseMoERouter
from research.entropy_routing_audit import (
    EntropyRoutingAuditResult,
    build_mini_config,
    probe_router_with_fixed_hidden,
    run_entropy_routing_audit,
    write_entropy_routing_report,
)


REPORT_PATH = Path("reports/scientific_integrity_audit/Entropy_Routing_Report.md")


def test_probe_router_entropy_conditions_routing() -> None:
    cfg = build_mini_config()
    result = probe_router_with_fixed_hidden(cfg)

    assert isinstance(result, EntropyRoutingAuditResult)
    assert result.production_verdict == "PASS"
    assert result.max_logit_diff_with_fixed_hidden > 0.0
    assert result.max_prob_diff_with_fixed_hidden > 0.0

    assert result.entropy_reaches_router is True
    assert result.entropy_reaches_routing_logits is True
    assert result.entropy_reaches_expert_probabilities is True


def test_router_fixed_hidden_routing_changes_with_entropy() -> None:
    cfg = build_mini_config()
    router = SparseMoERouter(cfg)
    router.eval()
    router.noise_enabled = False

    hidden = torch.randn(2, 6, cfg.model.hidden_dim)
    ent_low = torch.zeros(2, 6, 1)
    ent_high = torch.ones(2, 6, 1)

    with torch.no_grad():
        _, idx_low, _ = router(hidden, entropy=ent_low)
        _, idx_high, _ = router(hidden, entropy=ent_high)

    assert not torch.equal(idx_low, idx_high)


def test_run_entropy_routing_audit_smoke() -> None:
    result = run_entropy_routing_audit()
    assert result.protocol_version == "Phase-6.3.2-OBJ3"
    assert result.production_verdict == "PASS"
    assert isinstance(result.timestamp_utc, str)
    _ = json.dumps(result.to_dict())


def test_write_entropy_routing_report_creates_file(tmp_path: Path) -> None:
    local_report = tmp_path / "Entropy_Routing_Report.md"
    result = write_entropy_routing_report(local_report)

    assert local_report.exists()
    text = local_report.read_text(encoding="utf-8")

    assert "Phase 6.3.2 OBJ3" in text
    assert "Entropy \u2192 Routing Verification Report" in text
    assert "Fixed-Hidden Entropy Perturbation Experiment" in text
    assert "Raw JSON" in text
    assert result.production_verdict == "PASS"

    start = text.index("```json") + len("```json")
    end = text.index("```", start)
    embedded = json.loads(text[start:end].strip())
    assert embedded["protocol_version"] == result.protocol_version
