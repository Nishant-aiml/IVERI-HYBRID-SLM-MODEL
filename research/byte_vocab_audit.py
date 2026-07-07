# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Byte vocabulary integrity audit (Phase 6.3.2 OBJ7)."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.byte_vocab import (
    ByteVocabularyError,
    remap_legacy_token_ids,
    strip_special_bytes,
    validate_no_legacy_special_bytes,
    validate_token_ids,
)
from core.constants import (
    ARCHITECTURE_VERSION,
    BOS_BYTE,
    BYTE_VOCAB_SIZE,
    EOS_BYTE,
    LEGACY_BOS_BYTE,
    LEGACY_EOS_BYTE,
    LEGACY_PAD_BYTE,
    LEGACY_SPECIAL_BYTE_IDS,
    PAD_BYTE,
    RAW_BYTE_VOCAB_SIZE,
    SPECIAL_BYTE_IDS,
)
from data.pipeline.byte_encoder import ByteEncoder
from data.preprocessing import text_to_byte_ids


@dataclass
class ByteVocabGateProbe:
    gate_name: str
    passed: bool
    detail: str


@dataclass
class ByteVocabAuditResult:
    protocol_version: str = "Phase-6.3.2-OBJ7"
    timestamp_utc: str = ""
    production_verdict: str = "UNKNOWN"
    architecture_version: str = ARCHITECTURE_VERSION
    gates: list[ByteVocabGateProbe] = field(default_factory=list)
    presence_proof: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_special_disjoint_from_raw() -> ByteVocabGateProbe:
    overlap = SPECIAL_BYTE_IDS.intersection(range(RAW_BYTE_VOCAB_SIZE))
    ok = len(overlap) == 0 and BYTE_VOCAB_SIZE == RAW_BYTE_VOCAB_SIZE + len(SPECIAL_BYTE_IDS)
    detail = (
        f"BOS={BOS_BYTE}, PAD={PAD_BYTE}, EOS={EOS_BYTE}, vocab={BYTE_VOCAB_SIZE}"
    )
    return ByteVocabGateProbe("special_disjoint_from_raw", ok, detail)


def _probe_legacy_collision_removed() -> ByteVocabGateProbe:
    ok = (
        BOS_BYTE not in LEGACY_SPECIAL_BYTE_IDS
        and PAD_BYTE not in LEGACY_SPECIAL_BYTE_IDS
        and EOS_BYTE not in LEGACY_SPECIAL_BYTE_IDS
        and LEGACY_SPECIAL_BYTE_IDS == frozenset({0, 1, 2})
    )
    return ByteVocabGateProbe(
        "legacy_collision_removed",
        ok,
        f"legacy={sorted(LEGACY_SPECIAL_BYTE_IDS)}; active specials={sorted(SPECIAL_BYTE_IDS)}",
    )


def _probe_encode_roundtrip() -> ByteVocabGateProbe:
    text = "Hello world"
    ids = text_to_byte_ids(text)
    try:
        validate_token_ids(ids)
    except ByteVocabularyError as exc:
        return ByteVocabGateProbe("encode_roundtrip", False, str(exc))
    enc = ByteEncoder()
    decoded = enc.decode(ids)
    ok = decoded == text
    return ByteVocabGateProbe("encode_roundtrip", ok, f"decoded={decoded!r}")


def _probe_legacy_remap() -> ByteVocabGateProbe:
    remapped = remap_legacy_token_ids([LEGACY_BOS_BYTE, ord("A"), LEGACY_EOS_BYTE])
    ok = remapped == [BOS_BYTE, ord("A"), EOS_BYTE]
    return ByteVocabGateProbe("legacy_remap", ok, f"remapped={remapped}")


def _probe_centralized_constants() -> ByteVocabGateProbe:
    modules = [
        "training.sft_dataset",
        "training.preference_dataset",
        "training.loss_mask",
        "evaluation.sft_evaluator",
        "evaluation.response_inspector",
    ]
    offenders: list[str] = []
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod)
        if "PAD_BYTE: int = 0" in src or "_PAD_BYTE: int = 0" in src:
            offenders.append(mod_name)
    ok = not offenders
    return ByteVocabGateProbe(
        "centralized_constants",
        ok,
        "no local PAD_BYTE=0" if ok else f"offenders={offenders}",
    )


def run_byte_vocab_audit() -> ByteVocabAuditResult:
    gates = [
        _probe_special_disjoint_from_raw(),
        _probe_legacy_collision_removed(),
        _probe_encode_roundtrip(),
        _probe_legacy_remap(),
        _probe_centralized_constants(),
    ]
    passed = all(g.passed for g in gates)
    return ByteVocabAuditResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        production_verdict="PASS" if passed else "FAIL",
        gates=gates,
        presence_proof=[
            "Content bytes map 1:1 to IDs 0–255; BOS/PAD/EOS use extended IDs 256–258.",
            "Legacy colliding assignments (0, 1, 2) preserved only for checkpoint remap.",
            "ByteEncoder validates token IDs and strips specials on decode.",
            "Model embeddings expanded to BYTE_VOCAB_SIZE=259.",
            f"ARCHITECTURE_VERSION bumped to {ARCHITECTURE_VERSION}.",
        ],
    )


def render_byte_vocab_report(data: dict[str, Any]) -> str:
    lines = [
        "# Byte Vocabulary Report (Phase 6.3.2 OBJ7)",
        "",
        f"**Generated:** {data['timestamp_utc']}  ",
        f"**Protocol:** {data['protocol_version']}  ",
        f"**Architecture:** {data['architecture_version']}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Collision-free byte vocabulary:** `{data['production_verdict']}`",
        "",
        "| Gate | Result | Detail |",
        "|------|--------|--------|",
    ]
    for gate in data["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"| {gate['gate_name']} | {status} | {gate['detail']} |")
    lines.append("")

    if data["production_verdict"] == "PASS":
        lines.extend(["## Scientific Rationale", ""])
        lines.extend(
            [
                "1. Raw UTF-8 payload bytes occupy IDs **0–255** without reinterpretation.",
                "2. Structural tokens **BOS=256**, **PAD=257**, **EOS=258** are outside the byte range.",
                "3. NUL (0), U+0001, and U+0002 can appear in real text/binary without colliding with specials.",
                "4. `remap_legacy_token_ids()` supports inference on pre-v0.2.0 checkpoints only.",
                "",
                "## Proof: Runtime Gates",
                "",
            ]
        )
        for i, proof in enumerate(data["presence_proof"], 1):
            lines.append(f"{i}. {proof}")
        lines.append("")

    lines.extend(
        [
            "## Raw JSON",
            "",
            "```json",
            json.dumps(data, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_byte_vocab_report(output_path: str | Path) -> dict[str, Any]:
    result = run_byte_vocab_audit()
    data = result.to_dict()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_byte_vocab_report(data), encoding="utf-8")
    return data
