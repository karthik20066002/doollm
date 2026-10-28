#!/usr/bin/env python3

import argparse
import os
import time
import resource
os.environ.setdefault('HF_HOME', str(__import__('pathlib').Path(__file__).resolve().parents[1]/'build/hf'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from common import MODEL, BUILD, DEFAULT_SYSTEM, prompt_ids, write_json

QUERIES = ['What is the BFG 9000?', "What is planet Mars's colour?", 'What is 7 times 8?',
           'Reply with exactly the word: READY', 'Translate hello into Hindi.',
           'What is the capital of France?', 'Name the largest planet in the Solar System.',
           'Explain why ice floats on water in one sentence.']


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-tokens',type=int,default=48)
    ap.add_argument('--query',action='append')
    ap.add_argument('--name',default='baseline')
    args=ap.parse_args()
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
    print('Loading official model in float32 on CPU',flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
                                               dtype=torch.float32,attn_implementation='eager').eval()
    results=[]
    for qi,query in enumerate(args.query or QUERIES):
        ids=prompt_ids(query)
        template=tok.apply_chat_template([{'role':'system','content':DEFAULT_SYSTEM},{'role':'user','content':query}],
                                        tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(template,'keys'):template=template['input_ids']
        assert ids==template, ('chat template differs',ids,template)
        inp=torch.tensor([ids])
        captures={}
        handles=[]
        if qi==0:
            for li in (0,1,2,4,23):
                def hook(mod, inputs, output, index=li):
                    captures[f'layer_{index}']=output[0].detach().numpy().copy()
                handles.append(model.model.layers[li].register_forward_hook(hook))
        t0=time.monotonic()
        with torch.inference_mode():
            out=model(input_ids=inp,use_cache=False)
        for h in handles:h.remove()
        logits=out.logits[0,-1].detach().numpy()
        del out
        np.savez(BUILD/f'{"official" if args.name=="baseline" else args.name}_{qi}.npz',ids=np.array(ids),logits=logits,**captures)
        with torch.inference_mode():
            generated=model.generate(inp,attention_mask=torch.ones_like(inp),
                generation_config=GenerationConfig(max_new_tokens=args.max_tokens,do_sample=False,
                    repetition_penalty=1.0,eos_token_id=124900,pad_token_id=124893,use_cache=True))
        new=generated[0,len(ids):].tolist()
        row={'query':query,'prompt_ids':ids,'token_ids':new,'text':tok.decode(new,skip_special_tokens=True),
             'seconds':time.monotonic()-t0,'first_top_ids':np.argsort(logits)[-10:][::-1].tolist()}
        results.append(row)
        print(query, '=>',row['text'],flush=True)
        write_json(BUILD/(args.name+'.json'),{'precision':'original weights, float32 arithmetic',
            'decoding':'greedy, repetition_penalty=1.0, thinking disabled','results':results,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':main()
