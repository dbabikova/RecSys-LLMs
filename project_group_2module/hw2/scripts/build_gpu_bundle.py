"""Build a small input-only Colab notebook. No June truth or raw user IDs."""
import base64
import csv
import io
import json
import zipfile
from pathlib import Path
from input_contract import alias, digest

ROOT=Path(__file__).resolve().parents[2]


def main():
    output=ROOT/'data/hw2/colab'
    output.mkdir(exist_ok=True)
    files={name:(Path(__file__).parent/name).read_bytes() for name in
           ['input_contract.py','llm_ranker.py','gpu_llm.py','run_gpu_ranker.py']}
    files['config.json']=(ROOT/'hw2/LLM_CONFIG.json').read_bytes()
    files['model_manifest.json']=(ROOT/'hw2/LLM_MODEL.json').read_bytes()
    files['payloads.jsonl']=(ROOT/'data/hw2/common_inputs/payloads.jsonl').read_bytes()
    maps=[json.loads(s) for s in (ROOT/'data/hw2/common_inputs/private_mapping.jsonl').read_text().splitlines()]
    files['mappings.jsonl']=('\n'.join(json.dumps({k:r[k] for k in
                         ['user_alias','asset_alias_to_isin']}) for r in maps)+'\n').encode()
    files['pilot_payload.json']=(ROOT/'data/hw2/llm_pilot/pilot_payload.json').read_bytes()
    pilot=json.loads(files['pilot_payload.json'])
    with zipfile.ZipFile(ROOT/'data/hw2/inputs/FAR-Trans.zip') as z:
        rows=csv.DictReader(io.StringIO(z.read('FAR-Trans/asset_information.csv').decode('utf-8-sig')))
        known={alias(r['ISIN']):r['ISIN'] for r in rows if r['timestamp'][:10]<pilot['cutoff']}
    files['pilot_mapping.json']=json.dumps({c['asset_alias']:known[c['asset_alias']] for c in pilot['candidates']}).encode()
    manifest={'june_truth_included':False,'raw_user_ids_included':False,'users':100,
              'files_sha256':{n:digest(b) for n,b in files.items()},
              'runtime_variant':'CUDA float16; frozen model, prompts, candidates and 600 calls'}
    files['bundle_manifest.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr(name,data)
    (output/'ranker_gpu_bundle.zip').write_bytes(buffer.getvalue())
    encoded=base64.b64encode(buffer.getvalue()).decode()
    cells=[]
    def cell(kind,text):
        item={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code': item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','# Daria HW2: frozen CBF20 ranker experiment\nSelect Runtime → Change runtime type → T4 GPU, then Run all.\n100 fixed users; no June labels in this notebook. CUDA float16 runtime variant.\nKeep this tab open; download the result archive after completion.')
    cell('code',"import torch\nassert torch.cuda.is_available(), 'Select T4 GPU runtime first'\nprint(torch.cuda.get_device_name())\n%pip -q install transformers==4.57.6 huggingface_hub==0.36.0 hf_xet==1.2.0 psutil==7.2.2\n")
    cell('code',"import base64, io, zipfile\nfrom pathlib import Path\narchive = '"+encoded+"'\nfolder=Path('/content/daria_ranker'); folder.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode(archive))) as z: z.extractall(folder)\nprint('Input-only bundle ready; no truth labels.')\n")
    cell('code',"!python -u /content/daria_ranker/run_gpu_ranker.py\n")
    cell('code',"from google.colab import files\nfiles.download('/content/daria_ranker/ranker_results.zip')\n")
    notebook={'cells':cells,'metadata':{'accelerator':'GPU','colab':{'name':'Daria_HW2_GPU.ipynb'},
              'kernelspec':{'name':'python3','display_name':'Python 3'}},'nbformat':4,'nbformat_minor':5}
    (output/'Daria_HW2_GPU.ipynb').write_text(json.dumps(notebook,ensure_ascii=False),encoding='utf-8')
    (ROOT/'hw2/GPU_BUNDLE.json').write_text(json.dumps({**manifest,
        'notebook_sha256':digest((output/'Daria_HW2_GPU.ipynb').read_bytes()),
        'bundle_zip_bytes':len(buffer.getvalue())},indent=2)+'\n')
    print(output/'Daria_HW2_GPU.ipynb')
    print('Bundle bytes:',len(buffer.getvalue()))


if __name__=='__main__': main()
