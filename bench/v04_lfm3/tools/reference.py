#!/usr/bin/env python3

import argparse
import json
import math
import time
import numpy as np
import torch
from safetensors import safe_open
from common import MODEL, DATA, BUILD, CONFIG, DEFAULT_SYSTEM, prompt_ids, tokenizer, write_json

D, I, HEADS, KV, HD = 2048, 7680, 16, 4, 128
EPS = 1e-5


class Weights:
    def __init__(self, quantized=False):
        self.quantized=quantized
        self.small={}
        if quantized:
            self.spec={x['name']:x for x in json.loads((DATA/'weights.json').read_text())['tensors']}
            self.raw={}
            self.scales={}
        else:
            self.file=safe_open(MODEL/'model.safetensors',framework='pt',device='cpu')

    def tensor(self,name):
        if name in self.small:return self.small[name]
        if self.quantized:
            spec=self.spec[name]
            if spec['format']=='u8-row':
                if name not in self.raw:
                    a=np.memmap(DATA/(spec['stem']+'.bin'),dtype='u1',mode='c',shape=tuple(spec['shape']))
                    self.raw[name]=torch.from_numpy(a)
                    self.scales[name]=torch.from_numpy(np.fromfile(DATA/(spec['stem']+'.scale'),dtype='<f8'))
                return self.raw[name]
            v=torch.from_numpy(np.fromfile(DATA/(spec['stem']+'.bin'),dtype='<f8').reshape(spec['shape']))
        else:
            v=self.file.get_tensor(name)
        if v.ndim!=2:self.small[name]=v.double()
        return self.small.get(name,v)

    def linear(self,name,x):
        w=self.tensor(name)
        pieces=[]
        for start in range(0,w.shape[0],1024):
            block=w[start:start+1024].double()
            if self.quantized:
                y=x @ (block-128.).T
                y=y*self.scales[name][start:start+1024]
            else:
                y=x @ block.T
            pieces.append(y)
        return torch.cat(pieces,dim=-1)

    def embed(self,ids):
        name='model.embed_tokens.weight'
        w=self.tensor(name)[ids].double()
        if self.quantized:w=(w-128.)*self.scales[name][ids,None]
        return w


def rms(x,w):
    return x*torch.rsqrt((x*x).mean(dim=-1,keepdim=True)+EPS)*w


class Reference:
    def __init__(self,quantized=False):
        self.w=Weights(quantized)
        self.reset()

    def reset(self):
        self.pos=0
        self.keys={};self.values={};self.conv={}

    def rope(self,x,kind,positions):
        params=CONFIG['rope_parameters'][kind]
        count=int(HD*params.get('partial_rotary_factor',1.))//2
        inv=torch.zeros(HD//2,dtype=torch.float64)
        inv[:count]=params['rope_theta']**(-torch.arange(0,count*2,2,dtype=torch.float64)/HD)
        angle=positions[:,None]*inv[None,:]
        c,s=angle.cos()[:,None,:],angle.sin()[:,None,:]
        a,b=x[...,:HD//2],x[...,HD//2:]
        return torch.cat([a*c-b*s,a*s+b*c],dim=-1)

    @torch.inference_mode()
    def forward(self,ids,capture=False,layer_indices=None,hidden=None,project_logits=True):
        ids=torch.as_tensor(ids,dtype=torch.long)
        n=len(ids)
        positions=torch.arange(self.pos,self.pos+n,dtype=torch.float64)
        x=self.w.embed(ids) if hidden is None else hidden.double()
        taps={}
        for li in (range(24) if layer_indices is None else layer_indices):
            kind=CONFIG['layer_types'][li]
            p=f'model.layers.{li}.'
            get=lambda suffix:self.w.tensor(p+suffix).double()
            linear=lambda suffix,v:self.w.linear(p+suffix,v)
            h=rms(x,get('operator_norm.weight'))
            if kind=='conv':
                b,c,v=linear('conv.in_proj.weight',h).chunk(3,dim=-1)
                gated=b*v
                old=self.conv.get(li,torch.zeros(6,D,dtype=torch.float64))
                history=torch.cat([old,gated],dim=0)
                kernel=get('conv.conv.weight').reshape(D,7)
                y=torch.zeros_like(gated)
                for k in range(7):y+=history[k:k+n]*kernel[:,k]
                self.conv[li]=history[-6:].clone()
                out=linear('conv.out_proj.weight',c*y)
            else:
                q=linear('self_attn.q_proj.weight',h).reshape(n,HEADS,HD)
                k=linear('self_attn.k_proj.weight',h).reshape(n,KV,HD)
                v=linear('self_attn.v_proj.weight',h).reshape(n,KV,HD)
                q=self.rope(rms(q,get('self_attn.q_norm.weight')),kind,positions)
                k=self.rope(rms(k,get('self_attn.k_norm.weight')),kind,positions)
                keys=torch.cat([self.keys.get(li,k[:0]),k],dim=0)
                values=torch.cat([self.values.get(li,v[:0]),v],dim=0)
                self.keys[li]=keys;self.values[li]=values
                kk=keys.repeat_interleave(HEADS//KV,dim=1)
                vv=values.repeat_interleave(HEADS//KV,dim=1)
                score=torch.einsum('thd,uhd->htu',q,kk)/math.sqrt(HD)
                past=torch.arange(len(keys))[None,:]
                current=positions[:,None]
                mask=past>current
                if kind=='sliding_attention':mask=mask | (past<=current-CONFIG['sliding_window'])
                score.masked_fill_(mask[None,:,:],float('-inf'))
                probs=torch.softmax(score,dim=-1)
                att=torch.einsum('htu,uhd->thd',probs,vv)
                gate=torch.sigmoid(linear('self_attn.gate_proj.weight',h))
                out=linear('self_attn.o_proj.weight',(att*gate[:,:,None]).reshape(n,D))
            x=x+rms(out,get('post_operator_norm.weight'))
            h=rms(x,get('ffn_norm.weight'))
            a=linear('feed_forward.w1.weight',h)
            b=linear('feed_forward.w3.weight',h)
            out=linear('feed_forward.w2.weight',torch.nn.functional.silu(a)*b)
            x=x+rms(out,get('post_ffn_norm.weight'))
            if capture:taps[f'layer_{li}']=x.numpy().copy()
        if not project_logits:
            self.pos+=n
            return x,taps
        h=rms(x[-1:],self.w.tensor('model.embedding_norm.weight').double())
        logits=self.w.linear('model.embed_tokens.weight',h)[0]
        self.pos+=n
        return logits,taps

    def generate(self,ids,max_tokens=48,capture=False):
        self.reset()
        logits,taps=self.forward(ids,capture)
        first=logits.numpy().copy()
        result=[]
        for i in range(max_tokens):
            tid=int(logits.argmax())
            result.append(tid)
            if tid==124900 or i+1==max_tokens:break
            logits,_=self.forward([tid])
        return result,first,taps


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--quantized',action='store_true')
    ap.add_argument('--query',default='What is the BFG 9000?')
    ap.add_argument('--system',default=DEFAULT_SYSTEM)
    ap.add_argument('--thinking',action='store_true')
    ap.add_argument('--max-tokens',type=int,default=48)
    ap.add_argument('--baseline',type=int)
    ap.add_argument('--output',default='reference')
    args=ap.parse_args()
    torch.set_num_threads(4)
    model=Reference(args.quantized)
    ids=prompt_ids(args.query,args.system,args.thinking)
    t=time.monotonic()
    generated,first,taps=model.generate(ids,args.max_tokens,capture=True)
    report={'query':args.query,'thinking':args.thinking,'system':args.system,'prompt_ids':ids,'token_ids':generated,
            'text':tokenizer().decode(generated,skip_special_tokens=True),'quantized':args.quantized,
            'seconds':time.monotonic()-t}
    if args.baseline is not None:
        official=np.load(BUILD/f'official_{args.baseline}.npz')
        assert ids==official['ids'].tolist()
        metrics={}
        for key in official.files:
            if key=='ids':continue
            ref=official[key];actual=first if key=='logits' else taps[key]
            metrics[key]={'max_abs':float(np.abs(ref-actual).max()),
                          'rms_error':float(np.sqrt(np.mean((ref-actual)**2))),
                          'relative_l2':float(np.linalg.norm(ref-actual)/np.linalg.norm(ref))}
            if not args.quantized:
                assert np.allclose(ref,actual,atol=0.003,rtol=0.003),(key,metrics[key])
        report['official_comparison']=metrics
    write_json(BUILD/(args.output+'.json'),report)
    np.savez(BUILD/(args.output+'.npz'),logits=first,**taps)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
