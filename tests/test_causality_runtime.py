# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 Objective 1 — runtime causality perturbation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.causality_probe import (
    CorpusType,
    build_mini_config,
    forward_intermediates,
    perturb_future_bytes,
    probe_corpus,
    run_causality_audit,
    write_causality_report,
)
from model.iveri_core import IVERIModel

REPORT_PATH = Path("reports/scientific_integrity_audit/Causality_Report.md")


@pytest.fixture(scope="module")
def causality_model() -> IVERIModel:
    model = IVERIModel(build_mini_config())
    model.eval()
    return model


def test_byte_entropy_causal_conv_no_future_leak() -> None:
    """Isolated check: perturbing future bytes does not change earlier entropy."""
    cfg = build_mini_config()
    model = IVERIModel(cfg)
    model.eval()
    raw = torch.randint(3, 250, (1, 24))
    gen = torch.Generator().manual_seed(99)
    perturbed = perturb_future_bytes(raw, cut_index=10, generator=gen)
    with torch.no_grad():
        e_ref = model.entropy_model(raw)
        e_pert = model.entropy_model(perturbed)
    assert torch.allclose(e_ref[:, :11, :], e_pert[:, :11, :], atol=1e-6, rtol=1e-5)


def test_end_to_end_causality_passes(causality_model: IVERIModel) -> None:
    summaries = probe_corpus(
        causality_model,
        CorpusType.RANDOM,
        seq_len=32,
        seed=42,
        position_step=2,
        trials_per_position=2,
    )
    decoder = next(s for s in summaries if s.module == "BLTByteDecoder")
    assert decoder.passed
    assert decoder.max_abs_error < 1e-5


@pytest.mark.parametrize("corpus", list(CorpusType))
def test_causality_all_corpora_pass(causality_model: IVERIModel, corpus: CorpusType) -> None:
    summaries = probe_corpus(
        causality_model,
        corpus,
        seq_len=28,
        seed=11,
        position_step=4,
        trials_per_position=1,
    )
    assert len(summaries) == 6
    for summary in summaries:
        assert summary.passed, f"{summary.module} leaked on {corpus.value}"


def test_decoder_logits_invariant_under_future_perturbation(causality_model: IVERIModel) -> None:
    raw = torch.randint(3, 250, (1, 32))
    gen = torch.Generator().manual_seed(7)
    perturbed = perturb_future_bytes(raw, cut_index=14, generator=gen)
    with torch.no_grad():
        ref = forward_intermediates(causality_model, raw)
        pert = forward_intermediates(causality_model, perturbed)
    assert torch.allclose(ref.logits[:, :15, :], pert.logits[:, :15, :], atol=1e-6, rtol=1e-5)


def test_generate_causality_report_md() -> None:
    result = write_causality_report(
        REPORT_PATH,
        seq_len=32,
        seed=42,
        position_step=2,
        trials_per_position=2,
    )
    assert REPORT_PATH.exists()
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert "Phase 6.3.2" in text
    assert "Perturbation protocol" in text
    assert result.end_to_end_verdict == "PASS"


def test_full_audit_json_fields() -> None:
    result = run_causality_audit(seq_len=24, seed=3, position_step=3, trials_per_position=1)
    assert result.protocol_version == "Phase-6.3.2-OBJ1"
    assert result.end_to_end_verdict == "PASS"
    assert len(result.module_summaries) == len(CorpusType) * 6
