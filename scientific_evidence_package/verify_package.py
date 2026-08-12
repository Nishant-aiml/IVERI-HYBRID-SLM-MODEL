"""Verify evidence package contents"""
from pathlib import Path
import json

pkg = Path('scientific_evidence_package')
print('=== Evidence Package Contents ===')
for f in sorted(pkg.rglob('*')):
    if f.is_file():
        size = f.stat().st_size
        print('  ' + str(f.relative_to(pkg)) + ': ' + str(size) + ' bytes')

print()

print('=== Claim Registry Summary ===')
cr = pkg / 'claim_registry'
broken = 0
complete = 0
partial = 0
for f in sorted(cr.glob('*.json')):
    c = json.loads(f.read_text())
    broken_status = 'BROKEN CHAIN' if c.get('broken_chain') else 'OK'
    level = c.get('evidence_level', 'N/A')
    etype = c.get('evidence_type', 'UNKNOWN')
    print('  ' + c['claim_id'] + ': [' + etype + '] Level=' + level + ' ' + broken_status)
    if c.get('broken_chain'):
        broken += 1
    elif 'PARTIAL' in c.get('evidence_completeness', ''):
        partial += 1
    else:
        complete += 1

print()
print('  COMPLETE: ' + str(complete))
print('  PARTIAL:  ' + str(partial))
print('  BROKEN:   ' + str(broken))
print()

print('=== Missing Evidence (CRITICAL priority) ===')
me = json.loads((pkg / 'MISSING_EVIDENCE.json').read_text())
for item in me:
    if item.get('priority') == 'CRITICAL':
        print('  ' + item['evidence_id'] + ': ' + item['missing_item'])
        print('    Command: ' + item['execution_command'])
