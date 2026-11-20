"""Audit real pilot logs without importing backend, parser or aggregation code."""
import csv
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')


def main():
    private=ROOT/'data/hw2/llm_pilot'
    result=json.loads((ROOT/'hw2/LLM_PILOT.json').read_text())
    cfg=json.loads((ROOT/'hw2/LLM_CONFIG.json').read_text())
    model=json.loads((ROOT/'hw2/LLM_MODEL.json').read_text())
    require(model['model_repository']==cfg['model_repository'] and model['revision']==cfg['model_revision'],'Pinned repository/revision mismatch')
    require(result['config_sha256']==sha((ROOT/'hw2/LLM_CONFIG.json').read_bytes()),'LLM config changed')
    model_dir=ROOT/'data/hw2/models/Qwen3-1.7B'
    for name,expected in model['files_sha256'].items():
        with (model_dir/name).open('rb') as f:
            require(hashlib.file_digest(f,'sha256').hexdigest()==expected,'Model file changed')
        metadata=(model_dir/'.cache/huggingface/download'/(name+'.metadata')).read_text().splitlines()
        require(metadata[0]==cfg['model_revision'],'Download commit metadata differs')
        if len(metadata[1])==64:
            require(metadata[1]==expected,'Weight hash differs from repository etag')
    payload=json.loads((private/'pilot_payload.json').read_text(encoding='utf-8'))
    require(sha(encoded(payload))==result['selected_payload_hash'],'Pilot payload changed')
    require(payload['cutoff']=='2021-04-01','Pilot is not early data')
    source=ROOT/'data/hw2/temporal'/cfg['pilot']['source']
    require(sha(source.read_bytes())==result['source_payloads_sha256'],'Source payloads changed')
    source_payloads=[json.loads(s) for s in gzip.decompress(source.read_bytes()).decode().splitlines()]
    calls=[json.loads(s) for s in (private/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(calls)==6 and [(r['branch'],r['call'],r['permutation_seed']) for r in calls]==
            [('original',1,None),('original',2,None),('original',3,None),('permuted',1,17),('permuted',2,42),('permuted',3,2026)],'Wrong call budget/order')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    candidates={c['asset_alias']:c for c in payload['candidates']}
    allowed=list(candidates)
    wire={a:f'A{i:02d}' for i,a in enumerate(sorted({c['asset_alias'] for c in payload['history']+payload['candidates']}),1)}
    inverse={v:k for k,v in wire.items()}
    prelude=calls[0]['prompt'].split('\n',1)[0]
    template=json.loads(calls[0]['prompt'].split('\n',1)[1])
    lengths=[]
    for source_payload in source_payloads:
        data={k:source_payload[k] for k in ['user_alias','cutoff','repeat_purchases_allowed','profile','history_summary']}
        for kind,source_key in [('history','history'),('candidate','candidates')]:
            columns=template[kind+'_columns']
            data[kind+'_columns']=columns
            source_wire={a:f'A{i:02d}' for i,a in enumerate(sorted({c['asset_alias'] for c in source_payload['history']+source_payload['candidates']}),1)}
            data[kind+'_rows']=[[source_wire[item[k]] if k=='asset_alias' else item['attributes'][k] if k in item['attributes'] else item[k] for k in columns] for item in source_payload[source_key]]
        prompt=prelude+'\n'+encoded(data).decode()
        chat=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        lengths.append((len(tokenizer(chat,truncation=False)['input_ids']),source_payload['user_alias'],sha(encoded(source_payload))))
    lengths.sort()
    require(lengths[len(lengths)//2][2]==result['selected_payload_hash'],'Pilot is not median-length group')
    outputs={'original':[],'permuted':[]}
    invalid=0
    for record in calls:
        require(record['user_alias']==payload['user_alias'] and record['generated_response_cache_hit'] is False,'Wrong user/cache flag')
        require(record['prompt_sha256']==sha(record['prompt'].encode()),'Prompt digest mismatch')
        prefix,text=record['prompt'].split('\n',1)
        require(prefix==prelude,'Instruction changed between calls')
        data=json.loads(text)
        require({k:data[k] for k in ['user_alias','cutoff','repeat_purchases_allowed','profile','history_summary']}==
                {k:payload[k] for k in ['user_alias','cutoff','repeat_purchases_allowed','profile','history_summary']},'User context changed')
        reconstructed={}
        for kind in ['history','candidate']:
            rows=[]
            for r in data[kind+'_rows']:
                flat=dict(zip(data[kind+'_columns'],r))
                attributes={f:flat.pop(f) for f in ['assetCategory','assetSubCategory','sector','industry','marketID']}
                flat['asset_alias']=inverse[flat['asset_alias']]
                rows.append(dict(flat,attributes=attributes))
            reconstructed[kind]=rows
        require(reconstructed['history']==payload['history'],'History lost in serialization')
        require({c['asset_alias']:c for c in reconstructed['candidate']}==candidates,'Candidate fields changed')
        order=[c['asset_alias'] for c in reconstructed['candidate']]
        seed=record['permutation_seed']
        expected=allowed if seed is None else sorted(allowed,key=lambda a:(sha(f'{seed}:{a}'.encode()),a))
        require(order==expected,'Presentation order incorrect')
        chat=tokenizer.apply_chat_template([{'role':'user','content':record['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        count=len(tokenizer(chat,truncation=False)['input_ids'])
        require(count==record['input_tokens'] and count+cfg['max_new_tokens']<=cfg['context_limit_including_output'],'Input token limit mismatch')
        try:
            value=json.loads(record['text'].strip())
            valid=type(value) is list and len(value)==10 and all(type(a) is str for a in value) and len(set(value))==10 and set(value)<={wire[a] for a in candidates}
        except (ValueError,TypeError):
            valid=False
        expected=[inverse[a] for a in value] if valid else allowed[:10]
        require(record['invalid']==(not valid) and record['fallback']==(not valid),'Invalid/fallback flags mismatch')
        require(record['ranked_aliases']==expected,'Parser/fallback ranking mismatch')
        invalid+=not valid
        outputs[record['branch']].append(expected)
        require(record['seconds']>0 and record['output_tokens']>0,'Not a real completed generation')
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/FAR-Trans.zip') as z:
        raw=z.read('FAR-Trans/asset_information.csv')
    reverse={'A'+sha(r['ISIN'].encode())[:12]:r['ISIN'] for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))) if r['timestamp'][:10]<'2021-04-01'}
    saved=json.loads((private/'aggregates.json').read_text())
    for branch,rankings in outputs.items():
        points={a:0 for a in allowed}
        for ranking in rankings:
            for index,a in enumerate(ranking):
                points[a]+=10-index
        expected=sorted(allowed,key=lambda a:(-points[a],reverse[a]))[:10]
        require(saved[branch]['aggregate_aliases']==expected,'Aggregate/ISIN tie-break incorrect')
    require(result['invalid_outputs']==invalid and result['calls']==6,'Pilot aggregate counts incorrect')
    require(result['quality_metrics_computed'] is False and result['june_evaluation_run'] is False,'Unexpected quality or June evaluation')
    report={'status':'passed','performed_by':'Codex automated check; Daria review pending',
            'real_calls_audited':6,'valid_outputs':6-invalid,'fallback_calls':invalid,
            'checks':{k:True for k in ['pinned_model_file_hashes','early_source_payload_membership',
                     'median_length_selection_without_labels','six_real_calls_no_response_cache','lossless_payload_serialization','only_candidate_order_changes',
                     'token_budget_no_truncation','strict_parser_and_original_fallback','independent_rank_point_aggregation']},
            'quality_metrics_computed':False,'june_evaluation_run':False}
    require(any(sha(encoded(p))==result['selected_payload_hash'] for p in source_payloads),'Chosen payload absent from source')
    (ROOT/'hw2/STAGE6_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
