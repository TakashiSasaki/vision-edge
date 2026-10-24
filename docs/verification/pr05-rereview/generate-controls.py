"""New reviewer controls: exact rational rectangles, once-rounded edges.

Unlike ideal normalized fractions, expected normalized values use the actual
representable client point and once-rounded reference edges. No product code
or author generator is used.
"""
from fractions import Fraction as F
import json
import math
from pathlib import Path

sources=[(640,480),(1280,720),(1920,1080),(3840,2160),(853,480),(480,640),(600,600),(1365,768),(8192,4320)]
boxes=[(F(61441,64),F(34591,64)),(F(51263,64),F(38401,64)),(F(20305,64),F(41027,64)),(F(1920),F(1080)),(F(1,64),F(1,64)),(F(541,64),F(117,64))]
origins=[(F(149,4),F(165,2)),(F(1,64),F(-16063,64)),(F(-1000000),F(1000000)),(F(2**24-1024),F(-2**24+256))]
rows=[]
for vw,vh in sources:
    for width,height in boxes:
        scale=min(width/vw,height/vh);cw,ch=vw*scale,vh*scale
        for ox,oy in origins:
            edges=[float(ox+(width-cw)/2),float(oy+(height-ch)/2),float(ox+(width+cw)/2),float(oy+(height+ch)/2)]
            l,t,r,b=edges;mx=float(ox+width/2);my=float(oy+height/2)
            points=[('corner',(x,y))for x in [l,r]for y in [t,b]]
            points += [('edge',(l,my)),('edge',(r,my)),('edge',(mx,t)),('edge',(mx,b)),('center',(mx,my))]
            for axis,edge,other,direction in [(0,l,my,-1),(0,r,my,1),(1,t,mx,-1),(1,b,mx,1)]:
                for kind,coordinate in [('adjacent-outside',math.nextafter(edge,direction*math.inf)),('adjacent-inside',math.nextafter(edge,-direction*math.inf)),('outside-1e-9',edge+direction*1e-9),('outside-0.001',edge+direction*0.001)]:
                    # Large origins can round an epsilon back onto the edge.
                    if coordinate==edge:continue
                    point=(coordinate,other) if axis==0 else (other,coordinate)
                    points.append((kind,point))
            points += [('adjacent-outside-corner',(math.nextafter(x,xd*math.inf),math.nextafter(y,yd*math.inf)))for x,xd in [(l,-1),(r,1)]for y,yd in [(t,-1),(b,1)]]
            for dpr in [1,1.25,2,3]:
                for kind,(x,y) in points:
                    expected=None if 'outside' in kind else [(x-l)/(r-l),(y-t)/(b-t)]
                    rows.append(dict(video=[vw,vh],rect=dict(left=float(ox),top=float(oy),width=float(width),height=float(height)),dpr=dpr,point=[x,y],kind=kind,referenceEdges=edges,expected=expected))
p=Path(__file__).resolve().parent
(p/'independent-vectors.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
print(len(rows),'independent representable-boundary vectors')
