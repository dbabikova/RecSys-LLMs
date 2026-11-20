"""Early-data technical pilot only. No relevance labels and no June evaluation."""
import gzip
import csv
import io
import hashlib
import json
import time
import zipfile
from pathlib import Path

from input_contract import alias,canonical,digest
from llm_ranker import build_prompt,run_group
from local_llm import LocalQwen

ROOT=Path(__file__).resolve().parents[2]


def model_manifest(directory,cfg):
    hashes,sizes={},{}
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        meta=directory/'.cache/huggingface/download'/(p.name+'.metadata')
        if not meta.exists():
            raise ValueError('Missing download provenance: '+p.name)
        lines=meta.read_text().splitlines()
        if lines[0]!=cfg['model_revision']:
            raise ValueError('Wrong resolved model revision')
        with p.open('rb') as f:
            hashes[p.name]=hashlib.file_digest(f,'sha256').hexdigest()
        sizes[p.name]=p.stat().st_size
        if len(lines[1])==64 and hashes[p.name]!=lines[1]:
            raise ValueError('Model weight SHA-256 differs from repository etag')
    if not any(n.endswith('.safetensors') for n in hashes):
        raise ValueError('No model weights')
    result={'model_repository':cfg['model_repository'],'revision':cfg['model_revision'],
            'files_sha256':hashes,'file_sizes':sizes,'download_commit_metadata_checked':True,
            'weight_sha256_matches_remote_etags':True}
    (ROOT/'hw2/LLM_MODEL.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    started=time.perf_counter()
    cfg=json.loads((ROOT/'hw2/LLM_CONFIG.json').read_text())
    out=ROOT/'data/hw2/llm_pilot'
    out.mkdir(exist_ok=True)
    if (out/'calls.jsonl').exists():
        raise ValueError('Pilot evidence exists; do not overwrite or silently resume')
    model_dir=ROOT/'data/hw2/models/Qwen3-1.7B'
    model_manifest(model_dir,cfg)
    source=ROOT/'data/hw2/temporal'/cfg['pilot']['source']
    temporal=json.loads((ROOT/'hw2/TEMPORAL_DATA.json').read_text())
    expected=next(i['files_sha256'][source.name] for i in temporal['cuts'] if source.name in i['files_sha256'])
    if digest(source.read_bytes())!=expected:
        raise ValueError('Pilot historical input changed')
    payloads=[json.loads(s) for s in gzip.decompress(source.read_bytes()).decode().splitlines()]
    backend=LocalQwen(model_dir,cfg)
    sized=[]
    for p in payloads:
        count=int(backend.input_ids(build_prompt(p))['input_ids'].shape[1])
        sized.append((count,p['user_alias'],p))
    sized.sort(key=lambda x:(x[0],x[1]))
    chosen=sized[len(sized)//2]
    payload=chosen[2]
    pilot_tokens=[int(backend.input_ids(build_prompt(payload,s))['input_ids'].shape[1]) for s in [None,17,42,2026]]
    length_report={'early_groups_tokenized':len(sized),'min_input_tokens':sized[0][0],
                   'median_input_tokens':chosen[0],'max_input_tokens':sized[-1][0],
                   'groups_exceeding_context':sum(n+cfg['max_new_tokens']>cfg['context_limit_including_output'] for n,_,_ in sized),
                   'pilot_input_tokens_by_presentation':pilot_tokens}
    print(json.dumps(length_report),flush=True)
    (out/'pilot_payload.json').write_bytes(canonical(payload)+b'\n')
    # Only past metadata IDs are needed for original-ISIN tie-breaking. No labels
    # or temporal mapping/truth file is opened by this pilot.
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/FAR-Trans.zip') as z:
        metadata=z.read('FAR-Trans/asset_information.csv')
    source_hashes=json.loads((ROOT/'hw2/SOURCES.json').read_text(encoding='utf-8'))['source_csv_sha256']
    if digest(metadata)!=source_hashes['asset_information.csv']:
        raise ValueError('Metadata source changed')
    known={alias(r['ISIN']):r['ISIN'] for r in csv.DictReader(io.StringIO(metadata.decode('utf-8-sig')))
           if r['timestamp'][:10]<payload['cutoff']}
    alias_map={c['asset_alias']:known[c['asset_alias']] for c in payload['candidates']}
    try:
        branches=run_group(backend,payload,alias_map,cfg,out/'calls.jsonl')
    except Exception:
        (out/'incomplete.json').write_text(json.dumps({'status':'incomplete','no_quality_metrics':True})+'\n')
        raise
    calls=[json.loads(s) for s in (out/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(calls)!=6:
        raise ValueError('Wrong call count')
    result={'status':'completed_technical_pilot','config_sha256':digest((ROOT/'hw2/LLM_CONFIG.json').read_bytes()),
            'source_payloads_sha256':expected,'selection_rule':cfg['pilot']['selection'],
            'selected_payload_hash':digest(canonical(payload)), 'token_lengths':length_report,
            'calls':6,'valid_outputs':sum(not r['invalid'] for r in calls),'invalid_outputs':sum(r['invalid'] for r in calls),
            'fallback_calls':sum(r['fallback'] for r in calls),'backend':backend.metadata,
            'generation_seconds_total':sum(r['seconds'] for r in calls),
            'generation_seconds_mean':sum(r['seconds'] for r in calls)/6,
            'input_tokens_total':sum(r['input_tokens'] for r in calls),'output_tokens_total':sum(r['output_tokens'] for r in calls),
            'peak_working_set_bytes':max(r['peak_working_set_bytes'] for r in calls),
            'original_three_responses_identical':len({r['text'] for r in calls if r['branch']=='original'})==1,
            'aggregate_order_equal_across_branches':branches['original']['aggregate_aliases']==branches['permuted']['aggregate_aliases'],
            'full_600_call_linear_generation_hours':sum(r['seconds'] for r in calls)/6*600/3600,
            'runtime_estimate_limit':'One median-length early group only, no promise for full cohort; excludes downloads, setup and model loading',
            'whole_pilot_seconds':time.perf_counter()-started,'quality_metrics_computed':False,
            'june_evaluation_run':False,'fresh_calls_without_response_cache':True,
            'code_sha256':{n:digest((Path(__file__).parent/n).read_bytes()) for n in ['llm_ranker.py','local_llm.py','run_llm_pilot.py']}}
    (out/'aggregates.json').write_text(json.dumps(branches,indent=2)+'\n',encoding='utf-8')
    (out/'pilot_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (ROOT/'hw2/LLM_PILOT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
