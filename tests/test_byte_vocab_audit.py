# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 6.3.2 OBJ7 byte vocabulary runtime tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.byte_vocab import (
    ByteVocabularyError,
    remap_legacy_token_ids,
    validate_no_legacy_special_bytes,
)
from core.constants import (
    BOS_BYTE,
    BYTE_VOCAB_SIZE,
    EOS_BYTE,
    LEGACY_BOS_BYTE,
    LEGACY_EOS_BYTE,
    LEGACY_PAD_BYTE,
    PAD_BYTE,
    RAW_BYTE_VOCAB_SIZE,
    SPECIAL_BYTE_IDS,
)
from data.preprocessing import text_to_byte_ids
from research.byte_vocab_audit import run_byte_vocab_audit, write_byte_vocab_report

REPORT_PATH = Path("reports/scientific_integrity_audit/Byte_Vocabulary_Report.md")


def test_special_tokens_disjoint_from_raw_bytes() -> None:
    assert SPECIAL_BYTE_IDS.isdisjoint(range(RAW_BYTE_VOCAB_SIZE))
    assert BYTE_VOCAB_SIZE == RAW_BYTE_VOCAB_SIZE + len(SPECIAL_BYTE_IDS)
    assert (BOS_BYTE, PAD_BYTE, EOS_BYTE) == (256, 257, 258)


def test_nul_byte_in_content_does_not_collide_with_pad() -> None:
    ids = text_to_byte_ids("\x00", add_bos=False, add_eos=False)
    assert ids == [0]
    assert PAD_BYTE == 257
    assert ids[0] != PAD_BYTE


def test_control_bytes_in_content_use_raw_ids() -> None:
    ids = text_to_byte_ids("\x00\x01\x02", add_bos=False, add_eos=False)
    assert ids == [0, 1, 2]
    assert PAD_BYTE not in ids
    assert BOS_BYTE not in ids
    assert EOS_BYTE not in ids


def test_legacy_structural_encoding_detected() -> None:
    with pytest.raises(ByteVocabularyError):
        validate_no_legacy_special_bytes([LEGACY_BOS_BYTE, 65, LEGACY_EOS_BYTE])


def test_legacy_remap() -> None:
    assert remap_legacy_token_ids([LEGACY_BOS_BYTE, 65, LEGACY_PAD_BYTE]) == [
        BOS_BYTE,
        65,
        PAD_BYTE,
    ]


def test_byte_vocab_audit_pass() -> None:
    result = run_byte_vocab_audit()
    assert result.production_verdict == "PASS"
    assert all(g.passed for g in result.gates)


def test_write_byte_vocab_report(tmp_path: Path) -> None:
    out = tmp_path / "Byte_Vocabulary_Report.md"
    data = write_byte_vocab_report(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Phase 6.3.2 OBJ7" in text
    assert data["production_verdict"] == "PASS"


def test_regenerate_repo_byte_vocab_report() -> None:
    data = write_byte_vocab_report(REPORT_PATH)
    assert REPORT_PATH.exists()
    assert data["production_verdict"] == "PASS"
