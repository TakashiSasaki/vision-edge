"""Explicit candidate NMS hypothesis; unchanged learned outputs, no native parity claim."""
import numpy as np
from perception_diagnostics import decode, sigmoid, box_iou


def decode_policy(outputs, meta, nms_iou):
    if nms_iou is None:return decode(outputs,meta)
    if nms_iou!=0.5:raise ValueError('only the prespecified0.5NMS hypothesis is allowed')
    raw=outputs['detections'][0];result={}
    scale=np.asarray(meta['coordinate_scale'],dtype=np.float64)
    if scale.shape!=(4,) or not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('invalid inverse scale')
    for kind,channel,limit in [('objects',0,64),('text',1,32)]:
        heat=sigmoid(raw[...,channel]);pad=np.pad(heat,1,constant_values=-np.inf)
        peak=np.maximum.reduce([pad[i:i+64,j:j+64] for i in range(3) for j in range(3)])
        valid=(np.arange(64)[:,None]*8<meta['h'])&(np.arange(64)[None,:]*8<meta['w'])
        cells=np.argwhere((heat==peak)&(heat>=.3)&valid)
        order=sorted(cells.tolist(),key=lambda c:(-float(heat[tuple(c)]),c[0],c[1]))
        selected=[];boxes=[]
        for i,j in order:
            width,height=abs(float(raw[i,j,2+2*channel])),abs(float(raw[i,j,3+2*channel]))
            cx,cy=j+float(raw[i,j,6+2*channel]),i+float(raw[i,j,7+2*channel])
            box=np.array([cx-width/2,cy-height/2,cx+width/2,cy+height/2])/64
            box[[0,2]]=box[[0,2]].clip(0,meta['w']/512);box[[1,3]]=box[[1,3]].clip(0,meta['h']/512)
            if any(box_iou(box,old)>nms_iou for old in boxes):continue
            boxes.append(box)
            p={'bbox':(box*scale*1000).round().astype(int).tolist(),'score':float(heat[i,j]),'cell':[i,j]}
            if kind=='objects':
                logit=np.einsum('r,rhw->hw',outputs['coefficients'][0,:,i,j],outputs['prototypes'][0])
                centers=(np.arange(128)+.5)/128
                inside=(centers[:,None]>=box[1])&(centers[:,None]<=box[3])&(centers[None,:]>=box[0])&(centers[None,:]<=box[2])
                inside[min(127,int((box[1]+box[3])*64)),min(127,int((box[0]+box[2])*64))]=True
                mask=(sigmoid(logit)*inside)>=.5
                rows=np.minimum(127,((np.arange(128)+.5)/128*meta['h']/4).astype(int));cols=np.minimum(127,((np.arange(128)+.5)/128*meta['w']/4).astype(int))
                p['_mask']=mask[np.ix_(rows,cols)]
            selected.append(p)
            if len(selected)>=limit:break
        result[kind]=selected
    return result
