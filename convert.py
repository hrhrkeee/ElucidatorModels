#!/usr/bin/env python3
"""Export the pinned Google image tower for the optional Elucidator knowledge model."""
import argparse, json, pathlib, time
import numpy as np
import torch
from transformers import AutoModel
import coremltools as ct

class ImageEncoder(torch.nn.Module):
    def __init__(self, vision): super().__init__();self.vision=vision
    def forward(self,pixels):
        features=self.vision(pixel_values=pixels,return_dict=False)[1]
        return torch.nn.functional.normalize(features,dim=-1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='KnowledgeImageEncoder.mlpackage');args=p.parse_args()
    model=AutoModel.from_pretrained('google/siglip2-base-patch16-224',revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2',trust_remote_code=False,use_safetensors=True,attn_implementation='eager').eval()
    revision=getattr(model.config,'_commit_hash',None);encoder=ImageEncoder(model.vision_model).eval();del model
    torch.manual_seed(20261105);example=torch.rand(1,3,224,224)*2-1
    with torch.inference_mode():traced=torch.jit.trace(encoder,example,strict=False)
    converted=ct.convert(traced,inputs=[ct.TensorType(name='pixels',shape=(1,3,224,224),dtype=np.float32)],outputs=[ct.TensorType(name='embedding',dtype=np.float32)],minimum_deployment_target=ct.target.iOS18,compute_precision=ct.precision.FLOAT16)
    converted.author='Google; converted for Elucidator'
    converted.license='Apache-2.0; https://huggingface.co/google/siglip2-base-patch16-224'
    converted.short_description='SigLIP 2 Base image encoder; RGB 224x224 scaled to [-1,1]; normalized embedding; FP16 v1'
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);converted.save(str(path))
    with torch.inference_mode():expected=encoder(example).numpy()
    times=[];actual=None
    for _ in range(12):
        started=time.monotonic();actual=converted.predict({'pixels':example.numpy()})['embedding'];times.append((time.monotonic()-started)*1000)
    meta={'model':'google/siglip2-base-patch16-224','revision':revision,'license':'Apache-2.0','imageTowerParameters':sum(x.numel() for x in encoder.parameters()),'packageBytes':sum(f.stat().st_size for f in path.rglob('*') if f.is_file()),'maximumAbsoluteConversionError':float(np.max(np.abs(expected-actual))),'macWarmP95Milliseconds':float(np.percentile(times[1:],95)),'torch':torch.__version__,'numpy':np.__version__,'coremltools':ct.__version__,'limitations':['Mac warm inference, not iPhone end-to-end latency','Random input conversion check; task accuracy requires image-level re-evaluation','Image tower only; optional download; no text encoder']}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
