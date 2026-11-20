"""Download only public pinned Qwen files into the ignored workspace model folder."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HOME']=str(ROOT/'data/hw2/hf_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'


def main():
    from huggingface_hub import snapshot_download
    cfg=json.loads((ROOT/'hw2/LLM_CONFIG.json').read_text())
    target=ROOT/'data/hw2/models/Qwen3-1.7B'
    snapshot_download(repo_id=cfg['model_repository'],revision=cfg['model_revision'],
                      local_dir=target,cache_dir=ROOT/'data/hw2/hf_cache/hub',token=False,max_workers=1,
                      allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
                                      'model*.safetensors','model.safetensors.index.json','vocab.json','merges.txt','LICENSE'])
    print('Pinned public model files downloaded; no user data sent and no inference run.')


if __name__=='__main__':
    main()
