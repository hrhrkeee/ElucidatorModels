#!/usr/bin/env python3
"""Compare a converted image tower with the pinned official weights on real images."""
import argparse, json, pathlib
import numpy as np
import torch
import coremltools as ct
from PIL import Image, ImageOps
from transformers import AutoModel

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--images',nargs='+',required=True);p.add_argument('--embedding-output');p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4)
source=AutoModel.from_pretrained('google/siglip2-base-patch16-224',revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2',trust_remote_code=False,use_safetensors=True,attn_implementation='eager').eval().vision_model
model=ct.models.MLModel(a.model)
reference=[];converted=[]
for filename in a.images:
    im=ImageOps.exif_transpose(Image.open(filename)).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    pixels=(np.asarray(im,dtype=np.float32)/127.5-1).transpose(2,0,1)[None].copy()
    with torch.inference_mode():
        v=source(pixel_values=torch.from_numpy(pixels),return_dict=False)[1]
        v=torch.nn.functional.normalize(v,dim=-1).numpy()[0]
    actual=model.predict({'pixels':pixels})['embedding'].reshape(-1)
    if actual.shape!=(768,) or not np.isfinite(actual).all(): raise ValueError('invalid embedding')
    reference.append(v);converted.append(actual)
x=np.asarray(reference);y=np.asarray(converted)
x/=np.linalg.norm(x,axis=1,keepdims=True);y/=np.linalg.norm(y,axis=1,keepdims=True)
cos=np.sum(x*y,axis=1);rx=x@x.T;ry=y@y.T
np.fill_diagonal(rx,-np.inf);np.fill_diagonal(ry,-np.inf)
report={'imageCount':len(x),'minimumEmbeddingCosine':float(cos.min()),'meanEmbeddingCosine':float(cos.mean()),'maximumAbsoluteError':float(np.abs(x-y).max()),'nearestOtherImageAgreement':float(np.mean(rx.argmax(1)==ry.argmax(1))),'scope':'Conversion parity on real images, not independent retrieval quality or iPhone performance'}
if a.embedding_output: pathlib.Path(a.embedding_output).write_text(json.dumps([{'path':str(pathlib.Path(f).resolve()),'vector':v.tolist()} for f,v in zip(a.images,x)]))
pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if cos.min()<0.999 or report['nearestOtherImageAgreement']<0.95: raise SystemExit('conversion parity failed')
