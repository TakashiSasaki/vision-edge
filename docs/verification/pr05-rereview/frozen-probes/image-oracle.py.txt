"""Check rendered screenshots against fixed source colors and geometry.

Neither this code nor browser-oracle imports the production coordinate helper.
Screenshots use CSS pixel output even at high DPR.
"""
import json
import math
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent
observations=[]
def near(image, x, y, color, radius=3, tolerance=45):
    candidates=[]
    for yy in range(round(y)-radius,round(y)+radius+1):
        for xx in range(round(x)-radius,round(x)+radius+1):
            pixel=image.getpixel((xx,yy))[:3]
            distance=max(abs(a-b) for a,b in zip(pixel,color))
            candidates.append((distance,xx,yy,pixel))
    best=min(candidates)
    return dict(pass_=best[0]<=tolerance,distance=best[0],location=best[1:3],pixel=best[3])

def overlay(image,x,y,color):
    # Subpixel strokes are antialiased with either known adjacent source color.
    # Require a >=20% stroke contribution, and <=10 channel error to that line.
    candidates=[]
    for yy in range(round(y)-3,round(y)+4):
        for xx in range(round(x)-3,round(x)+4):
            pixel=image.getpixel((xx,yy))[:3]
            for base in [(12,34,56),(240,180,40)]:
                for percent in range(20,101):
                    alpha=percent/100
                    predicted=[alpha*a+(1-alpha)*b for a,b in zip(color,base)]
                    candidates.append((max(abs(a-b)for a,b in zip(pixel,predicted)),xx,yy,pixel,alpha,base))
    best=min(candidates)
    return dict(pass_=best[0]<=10,distance=best[0],location=best[1:3],pixel=best[3],alpha=best[4],base=best[5])

for phase in ['before','after']:
    for row in json.loads((ROOT/phase/'results.json').read_text()):
        name=f"{row['name']}-{row['dpr']}"
        before=Image.open(ROOT/phase/(name+'-before.png')).convert('RGB')
        after=Image.open(ROOT/phase/(name+'-after.png')).convert('RGB')
        ox,oy=row['origin']; sx,sy=row['scale']
        # Source (140,85) avoids the baseline's misplaced ripple/text.
        samples={}
        for label,(x,y,color) in {'object':(ox+140*sx,oy+85*sy,(240,180,40)), 'background':(ox+300*sx,oy+300*sy,(12,34,56))}.items():
            pixel=before.getpixel((round(x),round(y)))
            samples[label]=dict(point=[x,y],pixel=pixel,expected=color,pass_=max(abs(a-b)for a,b in zip(pixel,color))<=3)
        # Rendering of old/new source is already aligned on both commits.
        # The changed handler must put a cyan ring at the clicked visible point.
        ring=near(after, math.floor(row['point']['x'])+24*sx,math.floor(row['point']['y']),(56,189,248),radius=4,tolerance=45)
        bbox=overlay(before,ox+64*sx,oy+100*sy,(99,102,241) if phase=='after' else (255,255,255))
        selected_bbox=overlay(after,ox+64*sx,oy+100*sy,(255,255,255))
        result=dict(phase=phase,name=name,samples=samples,bbox=bbox,selected_bbox=selected_bbox,ripple=ring)
        result['pass']=all(x['pass_'] for x in samples.values()) and bbox['pass_'] and (phase=='before' or (ring['pass_'] and selected_bbox['pass_']))
        observations.append(result)
(ROOT/'image-results.json').write_text(json.dumps(observations,indent=2)+'\n')
failures=[x for x in observations if not x['pass']]
print(f'{len(observations)} image cases, {len(failures)} unexpected failures')
print(f"Baseline rings at intended click: {[r['ripple']['pass_'] for r in observations if r['phase']=='before']} (expected false)")
if failures:
    print(json.dumps(failures,indent=2))
    raise SystemExit(1)
