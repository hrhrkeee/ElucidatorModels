#!/usr/bin/env python3
"""Extract local original-image embeddings using the production image contract on macOS.
The manifest must contain images with path, sha256, imageID, knowledgeID, split,
cohort, and groupID. Source images and per-image features must not be published.
"""
import argparse, hashlib, json, pathlib, time
import coremltools as ct
import numpy as np
from PIL import Image, ImageOps

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True);p.add_argument('--manifest',required=True)
p.add_argument('--root',required=True);p.add_argument('--output-prefix',required=True)
a=p.parse_args();manifest=json.loads(pathlib.Path(a.manifest).read_text());root=pathlib.Path(a.root)
model=ct.models.MLModel(a.model);rows={'development':[],'final':[]}
for i,r in enumerate(manifest['images']):
    source=root/r['path']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Source hash mismatch')
    image=ImageOps.exif_transpose(Image.open(source)).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    pixels=(np.asarray(image,dtype=np.float32)/127.5-1).transpose(2,0,1)[None].copy()
    started=time.monotonic();v=model.predict({'pixels':pixels})['embedding'].reshape(-1)
    if v.shape!=(768,) or not np.isfinite(v).all():raise ValueError('Invalid embedding')
    v=v/np.linalg.norm(v)
    rows[r['cohort']].append(dict(imageID=r['imageID'].upper(),knowledgeID=r['knowledgeID'].upper(),groupID=r['groupID'],
        sourceSHA256=r['sha256'],split=r['split'],encoder='siglip2',view='original',vector=v.tolist(),milliseconds=(time.monotonic()-started)*1000))
    if i%100==0:print(f'Extracted {i}/{len(manifest["images"])}',flush=True)
for cohort,features in rows.items():
    pathlib.Path(a.output_prefix+'-'+cohort+'.json').write_text(json.dumps(dict(rows=features,
        preprocessing='upright RGB PIL antialiased bilinear 224; NCHW [-1,1]; fp16 Core ML; normalized embedding',
        modelRevision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2')))
print({k:len(v) for k,v in rows.items()},flush=True)
