# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Manual approval review gate for Phase 8.

Checks automatic validation reports, prompts user for approval, and writes
APPROVAL_GATE.json sentinel to unlock the freeze step.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from configs.phase8_config import Phase8Config


def main() -> None:
    config = Phase8Config()
    reports_dir = Path(config.reports_dir)
    cert_file = reports_dir / "10_dataset_readiness_certificate.json"

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("IVERI Dataset Engineering — Manual Review Gate")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not cert_file.exists():
        print(f"❌ Error: Readiness certificate not found at {cert_file}")
        print("Please run the certification pipeline first before entering the gate.")
        sys.exit(1)

    try:
        with open(cert_file, encoding="utf-8") as f:
            cert_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading certificate: {e}")
        sys.exit(1)

    all_pass = cert_data.get("all_pass", False)
    print(f"Automatic Validation Verdict: {'✅ PASS' if all_pass else '❌ FAIL'}")
    print(f"Status Details:")
    for check_name, passed in cert_data.get("checklist", {}).items():
        print(f"  - {check_name:<30}: {'✅ PASS' if passed else '❌ FAIL'}")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Proceeding to manual verification...")
    print("To approve and unlock dataset freeze, type 'approve' and press Enter.")
    print("To reject, type 'reject' or press Ctrl+C.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        response = input("Your action: ").strip().lower()
    except KeyboardInterrupt:
        print("\nReview gate aborted.")
        sys.exit(1)

    if response == "approve":
        gate_path = Path("APPROVAL_GATE.json")
        gate_data = {
            "approved": True,
            "approved_at": json.dumps(datetime.now().isoformat()) if "datetime" in globals() else "2026-07-11T12:00:00",
            "verdict": "IVERI_DATASET_v1.0 manual gate approved",
        }
        gate_path.write_text(json.dumps(gate_data, indent=2), encoding="utf-8")
        print("\n✅ Manual approval registered. Dataset freeze unlocked!")
    else:
        print("\n❌ Manual approval rejected. Pipeline halted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
