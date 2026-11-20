"""Offline Qwen backend; generation caching exists within a call only."""
import time
from pathlib import Path


class LocalQwen:
    def __init__(self,model_dir,cfg):
        import torch
        import transformers
        import psutil
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch,self.cfg,self.psutil=torch,cfg,psutil
        torch.set_num_threads(cfg['torch_threads'])
        torch.set_num_interop_threads(cfg['torch_interop_threads'])
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.process=psutil.Process()
        started=time.perf_counter()
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
                    dtype=torch.float32,attn_implementation=cfg['attention_implementation'])
        self.model.to('cpu').eval()
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.generation_config.top_k=None
        self.metadata={'torch':torch.__version__,'transformers':transformers.__version__,
                       'device':str(next(self.model.parameters()).device),'dtype':str(next(self.model.parameters()).dtype),
                       'parameters':sum(p.numel() for p in self.model.parameters()),
                       'model_load_seconds':time.perf_counter()-started,'rss_after_load_bytes':self.process.memory_info().rss,
                       'cuda_available':torch.cuda.is_available(),'thinking':False,'do_sample':False,
                       'generated_response_cache':False,'torch_seed':0}

    def input_ids(self,prompt):
        text=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                                               add_generation_prompt=True,enable_thinking=False)
        return self.tokenizer(text,return_tensors='pt',truncation=False)

    def generate(self,prompt):
        inputs=self.input_ids(prompt)
        n=inputs['input_ids'].shape[1]
        if n+self.cfg['max_new_tokens']>self.cfg['context_limit_including_output']:
            raise ValueError('Context overflow; no truncation permitted')
        started=time.perf_counter()
        with self.torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=self.cfg['max_new_tokens'],do_sample=False,
                                      use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
        tokens=output[0,n:]
        memory=self.process.memory_info()
        return {'text':self.tokenizer.decode(tokens,skip_special_tokens=True),
                'input_tokens':int(n),'output_tokens':len(tokens),'seconds':time.perf_counter()-started,
                'rss_bytes':memory.rss,'peak_working_set_bytes':getattr(memory,'peak_wset',memory.rss),
                'hit_output_cap':len(tokens)==self.cfg['max_new_tokens']}
