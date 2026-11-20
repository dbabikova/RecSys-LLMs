import json
import unittest
import tempfile
from pathlib import Path
from llm_ranker import aggregate,build_prompt,parse_or_fallback,parse_response,prompt_data,run_group,wire_aliases
from input_contract import common_payload
import test_input_contract


class LLMTests(unittest.TestCase):
    def payload(self):
        assets,counts,dates,profile=test_input_contract.ContractTests().fixture()
        return common_payload('U001','2021-06-01',assets,profile,counts,dates,list(assets)[:20])

    def test_table_round_trip_and_only_order_changes(self):
        p=self.payload()
        original=prompt_data(p)
        inverse={v:k for k,v in wire_aliases(p).items()}
        for kind,source in [('history',p['history']),('candidate',p['candidates'])]:
            restored=[]
            for row in original[kind+'_rows']:
                flat=dict(zip(original[kind+'_columns'],row))
                fields=['assetCategory','assetSubCategory','sector','industry','marketID']
                attributes={k:flat.pop(k) for k in fields}
                flat['asset_alias']=inverse[flat['asset_alias']]
                restored.append(dict(flat,attributes=attributes))
            self.assertEqual(restored,source)
        for seed in [17,42,2026]:
            other=prompt_data(p,seed)
            self.assertEqual({k:v for k,v in original.items() if k!='candidate_rows'},
                             {k:v for k,v in other.items() if k!='candidate_rows'})
            self.assertEqual(sorted(original['candidate_rows']),sorted(other['candidate_rows']))
            self.assertEqual(build_prompt(p,seed),build_prompt(p,seed))

    def test_strict_parser_rejects_repairs_duplicates_and_outside(self):
        allowed=[f'A{i:03d}' for i in range(1,21)]
        valid=json.dumps(allowed[:10])
        self.assertEqual(parse_response(valid,allowed),allowed[:10])
        for raw in ['```json\n'+valid+'\n```','Answer: '+valid,json.dumps(allowed[:9]),
                    json.dumps([allowed[0]]*10),json.dumps(['OUTSIDE']+allowed[:9]),
                    json.dumps(list(range(10))),json.dumps({'ranking':allowed[:10]})]:
            with self.assertRaises(ValueError):
                parse_response(raw,allowed)

    def test_fallback_is_original_order_for_all_calls(self):
        p=self.payload()
        expected=[c['asset_alias'] for c in p['candidates'][:10]]
        for _ in [None,17,42,2026]:
            # Caller always supplies the original payload to fallback, not presentation.
            result=parse_or_fallback('bad output',p)
            self.assertEqual(result['ranked_aliases'],expected)
            self.assertTrue(result['fallback'])

    def test_aggregation_uses_original_isin_tie_break(self):
        aliases=[f'A{i:02d}' for i in range(20)]
        mapping={a:f'I{19-i:02d}' for i,a in enumerate(aliases)}
        result=aggregate([aliases[:10],list(reversed(aliases[:10])),aliases[10:]],mapping)
        self.assertEqual(result,list(reversed(aliases[:10])))
        with self.assertRaises(ValueError):
            aggregate([aliases[:10]],mapping)

    def test_six_fresh_calls_and_per_call_fallback(self):
        p=self.payload()
        ids=[c['asset_alias'] for c in p['candidates']]
        wire=wire_aliases(p)
        class FakeBackend:
            def __init__(self):
                self.calls=[]
            def generate(self,prompt):
                self.calls.append(prompt)
                return {'text':'bad' if len(self.calls)==4 else json.dumps([wire[a] for a in ids[:10]])}
        backend=FakeBackend()
        cfg={'original_order_calls':3,'permutation_seeds':[17,42,2026]}
        mapping={a:f'I{i:02d}' for i,a in enumerate(ids)}
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'calls.jsonl'
            output=run_group(backend,p,mapping,cfg,log)
            rows=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(backend.calls),6)
        self.assertEqual(len(set(backend.calls[:3])),1)
        self.assertTrue(rows[3]['fallback'])
        self.assertEqual(rows[3]['ranked_aliases'],ids[:10])
        self.assertTrue(all(not r['generated_response_cache_hit'] for r in rows))
        self.assertEqual(set(output),{'original','permuted'})

    def test_backend_failure_is_incomplete_not_fallback(self):
        p=self.payload()
        ids=[c['asset_alias'] for c in p['candidates']]
        class BrokenBackend:
            def generate(self,prompt):
                raise ValueError('Context overflow')
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'calls.jsonl'
            with self.assertRaisesRegex(ValueError,'Context overflow'):
                run_group(BrokenBackend(),p,{a:f'I{i}' for i,a in enumerate(ids)},
                          {'original_order_calls':3,'permutation_seeds':[17,42,2026]},log)
            rows=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'fatal_incomplete')
        self.assertNotIn('fallback',rows[0])


if __name__=='__main__':
    unittest.main()
