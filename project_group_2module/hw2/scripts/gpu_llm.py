"""Explicit CUDA/fp16 runtime variant; frozen prompts, weights and decoding."""
import time


class GPUQwen:
    def __init__(self, directory, cfg):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required: select T4 runtime; CPU is too slow')
        self.torch, self.cfg = torch, cfg
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
                            dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.metadata = {'torch': torch.__version__, 'transformers': transformers.__version__,
                         'device': torch.cuda.get_device_name(), 'dtype': 'float16',
                         'runtime_change': 'CPU float32 -> CUDA float16; no quality-based tuning'}

    def generate(self, prompt):
        text = self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=False).to('cuda')
        n = inputs['input_ids'].shape[1]
        if n + self.cfg['max_new_tokens'] > self.cfg['context_limit_including_output']:
            raise ValueError('Context overflow')
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.cfg['max_new_tokens'],
                       do_sample=False, use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        self.torch.cuda.synchronize()
        tokens = output[0, n:]
        return {'text':self.tokenizer.decode(tokens, skip_special_tokens=True),
                'input_tokens':int(n), 'output_tokens':len(tokens),
                'seconds':time.perf_counter()-start,
                'gpu_peak_allocated_bytes':self.torch.cuda.max_memory_allocated(),
                'hit_output_cap':len(tokens)==self.cfg['max_new_tokens']}
