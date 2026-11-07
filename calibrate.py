#!/usr/bin/env python3
"""Exploratory native embedding gates. Existing observations are development data, not sealed validation."""
import argparse, json, numpy as np
p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rows=[r for r in json.load(open(a.features))['rows'] if r['encoder']=='siglip2' and r['view']=='original']
gallery=[r for r in rows if r['split']=='gallery'];known=[r for r in rows if r['split']=='validation'];unknown=[r for r in rows if r['split']=='unknown']
def unit(x):
 x=np.asarray(x,dtype=np.float64);return x/np.linalg.norm(x,axis=-1,keepdims=True)
x=unit([r['vector'] for r in gallery]);q=unit([r['vector'] for r in known]);u=unit([r['vector'] for r in unknown]);kd=np.maximum(0,2-2*q@x.T);ud=np.maximum(0,2-2*u@x.T).min(1)
# Five monotonic operating points. Never copy native-Vision thresholds.
limits=[]
for rate in [0,0.005,0.01,0.025,0.05]:
 permitted=int(len(ud)*rate)
 limits.append(float(np.nextafter(np.sort(ud)[min(permitted,len(ud)-1)],-np.inf)))
report={'version':1,'metric':'squared-l2-unit-embedding','galleryImages':len(x),'knownCalibrationImages':len(q),'unknownDevelopmentImages':len(u),'maximumDistances':limits,'knownNearestCorrect':float(np.mean([gallery[i]['knowledgeID']==r['knowledgeID'] for i,r in zip(kd.argmin(1),known)])),'knownPassingGate':[float(np.mean(kd.min(1)<=t)) for t in limits],'unknownPassingGate':[float(np.mean(ud<=t)) for t in limits],'limitations':['Previously observed public dataset features reused for development','Native distance gates only; not final Ridge/VLM acceptance or independent quality validation','No iPhone performance measurement']}
open(a.output,'w').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
