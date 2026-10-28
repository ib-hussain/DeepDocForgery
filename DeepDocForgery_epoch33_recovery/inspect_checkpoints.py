#!/usr/bin/env python3
"""Read-only audit of DeepDocForgery epoch-33 emergency and native checkpoints.
Run from repository root with your project's torch environment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    # Files must be created locally by your trusted training process.
    return torch.load(path,map_location='cpu',weights_only=False)


def structural(v):
    if isinstance(v,dict):
        keys=list(v.keys())
        return {'type':'dict','size':len(v),'keys_sample':[str(x) for x in keys[:20]]}
    if isinstance(v,torch.Tensor):
        return {'type':'tensor','shape':list(v.shape),'dtype':str(v.dtype)}
    return {'type':type(v).__name__,'value':str(v)[:180] if isinstance(v,(str,int,float,bool,type(None))) else None}


def inspect(label,path):
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    print(f'\n=== {label}: {p} ({p.stat().st_size:,} bytes) ===',flush=True)
    print('sha256:',digest(p),flush=True)
    c=load(p)
    if not isinstance(c,dict):
        raise TypeError(f'Expected dict checkpoint; got {type(c).__name__}')
    print('TOP LEVEL KEYS:',[str(k) for k in c.keys()],flush=True)
    for k,v in c.items():
        if k in ('model','model_state_dict','state_dict','model_state','optimizer','optimizer_state_dict','scheduler','scheduler_state_dict','scaler','scaler_state_dict','rng_state','cuda_rng_state'):
            print(f'  {k}:',structural(v),flush=True)
        elif not isinstance(v,(dict,list,tuple,torch.Tensor)):
            print(f'  {k}: {str(v)[:180]}',flush=True)
        elif isinstance(v,dict) and len(v)<=15:
            print(f'  {k}:',structural(v),flush=True)
    model_keys=('model_state_dict','model','state_dict','model_state')
    model_key=next((key for key in model_keys if isinstance(c.get(key),dict) and any(isinstance(v,torch.Tensor) for v in c[key].values())),None)
    stats={'model_key':model_key,'total_tensors':0,'nonfinite_tensors':[],'model_entries':0}
    if model_key:
        state=c[model_key]
        stats['model_entries']=len(state)
        for key,t in state.items():
            if isinstance(t,torch.Tensor):
                stats['total_tensors']+=1
                if (t.is_floating_point() or t.is_complex()) and not bool(torch.isfinite(t).all()):
                    stats['nonfinite_tensors'].append(str(key))
    print('MODEL FINITENESS:',json.dumps(stats,indent=2),flush=True)
    return c,stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native',default='output/model/cuda-full/last.pt')
    p.add_argument('--emergency',default='output/model/cuda-full/epoch33_EMERGENCY.pt')
    args=p.parse_args()
    native,n=inspect('NATIVE EPOCH-32',args.native)
    emergency,e=inspect('RESCUED EPOCH-33',args.emergency)
    if n['model_key'] and e['model_key']:
        a=native[n['model_key']]; b=emergency[e['model_key']]
        missing=set(a)-set(b)
        unexpected=set(b)-set(a)
        shape_mismatch=[str(k) for k in set(a)&set(b) if isinstance(a[k],torch.Tensor) and isinstance(b[k],torch.Tensor) and a[k].shape!=b[k].shape]
        print('\n=== MODEL COMPATIBILITY ===')
        print('Missing model keys:',sorted(map(str,missing))[:30])
        print('Unexpected model keys:',sorted(map(str,unexpected))[:30])
        print('Shape mismatches:',shape_mismatch[:30])
        if not missing and not unexpected and not shape_mismatch:
            changed=sum(1 for k in a if isinstance(a[k],torch.Tensor) and isinstance(b[k],torch.Tensor) and not torch.equal(a[k],b[k]))
            print(f'Model tensors changed relative to epoch 32: {changed}/{len(a)}')
        else:
            print('STOP: state dicts require reconciliation before resuming.')
    print('\nREAD-ONLY AUDIT COMPLETE. No checkpoint files were changed.')

if __name__=='__main__':
    main()
