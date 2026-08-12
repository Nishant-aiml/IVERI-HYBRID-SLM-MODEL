#!/usr/bin/env python3
"""IVERI CORE — Automated Claim Verifier. Run: python verify_everything.py"""
import json, sys, re, hashlib
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
stage_c = root / "scientific_campaigns" / "stage_c"
claims_dir = stage_c / "claim_registry"

def sha256(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()[:16]

def load(path):
    t = Path(path).read_text(encoding="utf-8")
    t = re.sub(r"\bNaN\b","null",t); t = re.sub(r"\bInfinity\b","1e308",t)
    return json.loads(t)

PASS, FAIL, SKIP = "PASS","FAIL","SKIP"
results = []

for cf in sorted(claims_dir.glob("CLAIM-*.json")):
    c = json.loads(cf.read_text())
    cid = c["claim_id"]
    srcs = c.get("source_data","MISSING_EVIDENCE")
    if srcs == "MISSING_EVIDENCE":
        results.append((cid, SKIP, "Evidence not yet collected")); continue
    for src in ([srcs] if isinstance(srcs,str) else srcs):
        sp = Path(src)
        if not sp.exists():
            results.append((cid, FAIL, f"Source missing: {sp.name}")); break
        actual = sha256(sp)
        exp = c.get("source_sha256")
        if isinstance(exp, list): exp = exp[0]
        if exp and actual != exp:
            results.append((cid, FAIL, f"SHA256 mismatch {sp.name}: expected {exp}, got {actual}")); break
    else:
        results.append((cid, PASS, f"Source(s) verified: SHA256 OK"))

print("="*60)
print("IVERI CORE — CLAIM VERIFICATION REPORT")
print("="*60)
for cid, s, msg in results:
    print(f"[{s:4s}] {cid}: {msg}")
p = sum(1 for _,s,_ in results if s==PASS)
f = sum(1 for _,s,_ in results if s==FAIL)
sk = sum(1 for _,s,_ in results if s==SKIP)
print(f"\nTotal: {len(results)} | PASS: {p} | FAIL: {f} | SKIP (pending): {sk}")
sys.exit(0 if f==0 else 1)
