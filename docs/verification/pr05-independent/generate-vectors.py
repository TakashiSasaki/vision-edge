"""Generate once-rounded numeric references with exact rational arithmetic."""
from fractions import Fraction as F
from pathlib import Path
import json

cases=[
    ('pillarbox',[640,480],[960,540],[120,0,720,540]),
    ('letterbox',[1280,720],[800,600],[0,75,800,450]),
    ('equal',[1280,720],[960,540],[0,0,960,540]),
    ('square-source',[600,600],[960,540],[210,0,540,540]),
    ('square-box',[1280,720],[600,600],[0,F(525,4),600,F(675,2)]),
    ('fractional',[640,480],[F(3843,4),F(2161,4)],[F(2885,24),0,F(2161,3),F(2161,4)]),
]
rows=[]
for name,video,box,content in cases:
    x,y,w,h=map(F,content)
    points=[(x+w*u,y+h*v,[float(u),float(v)],'boundary' if u in [0,1] or v in [0,1] else 'inside')for u,v in [(0,0),(1,0),(0,1),(1,1),(0,F(1,2)),(1,F(1,2)),(F(1,2),0),(F(1,2),1),(F(1,2),F(1,2)),(F(1,4),F(3,4))]]
    e=F(1,1000)
    points += [(x+e,y+h/2,[float(e/w),0.5],'inside'),(x+w-e,y+h/2,[float(1-e/w),0.5],'inside'),(x+w/2,y+e,[0.5,float(e/h)],'inside'),(x+w/2,y+h-e,[0.5,float(1-e/h)],'inside')]
    points += [(xx,yy,None,'outside')for xx,yy in [(x-e,y+h/2),(x+w+e,y+h/2),(x+w/2,y-e),(x+w/2,y+h+e),(x-e,y-e),(x+w+e,y-e),(x-e,y+h+e),(x+w+e,y+h+e)]]
    for left,top in [(F(149,4),F(165,2)),(F(500),F(250)),(F(-20),F(-250))]:
        for dpr in [1,1.25,2,3]:
            for px,py,expected,kind in points:
                rows.append(dict(name=name,video=video,rect=dict(left=float(left),top=float(top),width=float(box[0]),height=float(box[1])),dpr=dpr,point=[float(left+px),float(top+py)],expected=expected,kind=kind))
out=Path(__file__).resolve().parent/'numeric-vectors.json'
out.write_text(json.dumps(rows,separators=(',',':'))+'\n')
print(len(rows),'exact-rational coordinate vectors')
