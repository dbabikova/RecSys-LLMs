"""Prove the verifier rejects mutated artifacts; operate on temporary private copies."""
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def main():
    original = PROJECT / 'data/hw2/cbf20'
    private = PROJECT / 'data/hw2/stage2_diagnostics'
    private.mkdir(exist_ok=True)
    results = {}
    for case in ['swap_candidates', 'drop_user', 'alter_reference']:
        with tempfile.TemporaryDirectory(prefix='integrity_', dir=private) as temp:
            out = Path(temp)
            for name in ['candidates.jsonl', 'reference_predictions.csv', 'candidate_manifest.json']:
                shutil.copyfile(original / name, out / name)
            if case != 'alter_reference':
                records = [json.loads(s) for s in (out / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
                if case == 'swap_candidates':
                    records[0]['candidate_ids'][0], records[0]['candidate_ids'][1] = records[0]['candidate_ids'][1], records[0]['candidate_ids'][0]
                else:
                    records.pop()
                (out / 'candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            else:
                path = out / 'reference_predictions.csv'
                rows = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8'))))
                rows[0]['ISIN'] = rows[1]['ISIN']
                with path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(rows)
            result = subprocess.run([sys.executable, '-B', str(PROJECT / 'hw2/scripts/verify_candidates.py'),
                                     '--artifact-dir', str(out), '--report', str(out / 'report.json')], capture_output=True, text=True)
            expected = {'swap_candidates': 'Pool is not the exact CBF50 prefix',
                        'drop_user': 'Different user IDs/order',
                        'alter_reference': 'Frozen file hash failed: reference_predictions.csv'}[case]
            if result.returncode == 0 or expected not in result.stderr:
                raise ValueError(f'Integrity check failed: {case}')
            results[case] = {'rejected': True, 'reason': expected}
    report = {'status': 'passed', 'performed_by': 'Codex automated negative checks',
              'scope': 'Mutated private copies only; original frozen files unchanged', 'checks': results}
    (PROJECT / 'hw2/STAGE2_INTEGRITY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
