"""Run early technical pilot then the frozen June cohort, with per-user checkpoints."""
import json
import time
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
from input_contract import digest
from gpu_llm import GPUQwen
from llm_ranker import run_group


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'bundle_manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if digest((root/name).read_bytes()) != expected:
            raise ValueError('Bundle hash mismatch: '+name)
    out = root/'results'
    out.mkdir(exist_ok=True)
    model = snapshot_download(cfg['model_repository'], revision=cfg['model_revision'],
                ignore_patterns=['*.bin','*.gguf','*.h5','*.msgpack'])
    expected_weights=json.loads((root/'model_manifest.json').read_text())['files_sha256']
    for name, expected in expected_weights.items():
        if name.endswith('.safetensors') and digest((Path(model)/name).read_bytes())!=expected:
            raise ValueError('Model weight hash mismatch')
    backend = GPUQwen(model,cfg)
    (out/'runtime.json').write_text(json.dumps(backend.metadata,indent=2))
    pilot = json.loads((root/'pilot_payload.json').read_text())
    pilot_map = json.loads((root/'pilot_mapping.json').read_text())
    if not (out/'pilot_complete.json').exists():
        if (out/'pilot_calls.jsonl').exists():
            raise ValueError('Partial pilot exists; preserve evidence before restarting')
        start=time.perf_counter()
        run_group(backend,pilot,pilot_map,cfg,out/'pilot_calls.jsonl')
        calls=[json.loads(s) for s in (out/'pilot_calls.jsonl').read_text().splitlines()]
        info={'seconds':time.perf_counter()-start,'calls':len(calls),
              'invalid':sum(c['invalid'] for c in calls), 'quality_metrics':False,
              'estimated_600_calls_minutes':sum(c['seconds'] for c in calls)/6*10}
        (out/'pilot_complete.json').write_text(json.dumps(info,indent=2))
        print('GPU PILOT',info,flush=True)
    payloads=[json.loads(s) for s in (root/'payloads.jsonl').read_text().splitlines()]
    maps={r['user_alias']:r['asset_alias_to_isin'] for r in
          (json.loads(s) for s in (root/'mappings.jsonl').read_text().splitlines())}
    if len(payloads)!=100 or len(maps)!=100:
        raise ValueError('Expected exact frozen 100-user cohort')
    for i,p in enumerate(payloads,1):
        uid=p['user_alias']
        target=out/(uid+'.json')
        if target.exists():
            continue
        log=out/(uid+'_calls.jsonl')
        if log.exists():
            # Resume only complete groups: retain interrupted requests explicitly.
            log.rename(out/(uid+'_interrupted_'+str(time.time_ns())+'.jsonl'))
        pool={c['asset_alias']:maps[uid][c['asset_alias']] for c in p['candidates']}
        branches=run_group(backend,p,pool,cfg,log)
        target.write_text(json.dumps({'user_alias':uid,'branches':branches},indent=2))
        print('COMPLETED',i,'/100',uid,flush=True)
        with zipfile.ZipFile(root/'ranker_results.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                z.write(f,'results/'+f.name)
    (out/'complete.json').write_text(json.dumps({'users':100,'calls':600,
             'bundle_manifest':manifest,'runtime':backend.metadata},indent=2))
    with zipfile.ZipFile(root/'ranker_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            z.write(f,'results/'+f.name)
    print('COMPLETE: 100 users, 600 June calls. Download ranker_results.zip',flush=True)


if __name__=='__main__':
    main()
