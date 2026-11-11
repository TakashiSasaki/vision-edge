"""Immutable new source images and a predeclared development/reserved protocol."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import urllib.error

from PIL import Image
from verify_checkpoint import identity, verify, read_json

ROOT=Path(__file__).parent/'fixtures'
MANIFEST=ROOT/'native-corpus.json'
PROTOCOL=ROOT/'native-protocol.json'


def cases():
    rows=read_json(MANIFEST.read_bytes())['cases']
    names=[c['name'] for c in rows]
    if len(rows)!=12 or len(set(names))!=12 or len({c['rgb_sha256'] for c in rows})!=12:
        raise ValueError('invalid image corpus identity set')
    protocol=read_json(PROTOCOL.read_bytes())
    for split,count in [('development',8),('reserved_validation',4)]:
        selected=[c['name'] for c in rows if c['split']==split]
        if len(selected)!=count or selected!=protocol[split]:raise ValueError('corpus/protocol split mismatch')
    for c in rows:
        if c['name']!=f"coco_{c['image_id']:012d}" or c['file']!=c['name']+'.jpg' or c['url']!=f"https://images.cocodataset.org/val2017/{c['image_id']:012d}.jpg":
            raise ValueError('unexpected image source')
    return {c['name']:c for c in rows}


def selected_cases(names,split):
    records=cases()
    if not 1<=len(names)<=4 or len(set(names))!=len(names) or split not in ('development','reserved_validation'):
        raise ValueError('one to four distinct cases in a known split required')
    if any(n not in records or records[n]['split']!=split for n in names):raise ValueError('unknown/cross-split case')
    return [records[n] for n in names]


def verify_image(path,record):
    verify(path,(record['size'],record['sha256']))
    with Image.open(path) as source:
        orientation=source.getexif().get(274)
        image=source.convert('RGB')
    # Use the same decoded pixels, without an implicit EXIF transform, in both runtimes.
    if [image.height,image.width,3]!=record['shape'] or hashlib.sha256(image.tobytes()).hexdigest()!=record['rgb_sha256'] or orientation!=record['exif_orientation']:
        raise ValueError('decoded RGB or orientation identity mismatch')
    return image


def acquire(cache,verify_only):
    cache.mkdir(parents=True,exist_ok=True);result=[]
    for c in cases().values():
        target=cache/c['file'];row={'name':c['name'],'url':c['url']}
        try:
            blocked=cache/(c['name']+'-http403.json')
            if not target.exists():
                if verify_only:raise ValueError('image absent; verify-only does not download')
                if blocked.exists():
                    row['http_status']=403
                    raise ValueError('recorded HTTP403 is not retried')
                try:
                    request=urllib.request.Request(c['url'],headers={'User-Agent':'MirageResearch/1.0'})
                    with urllib.request.urlopen(request,timeout=20) as response:
                        data=response.read(10*1024*1024+1);row['http_status']=response.status;row['final_url']=response.url
                    if len(data)!=c['size'] or hashlib.sha256(data).hexdigest()!=c['sha256']:
                        raise ValueError('download identity mismatch')
                    # Only commit checksum-verified bytes. Existing bad files are never silently replaced.
                    with target.open('xb') as f:f.write(data)
                except urllib.error.HTTPError as error:
                    row['http_status']=error.code
                    if error.code==403:blocked.write_text(json.dumps({'url':c['url'],'status':403})+'\n')
                    raise
            verify_image(target,c);row.update(status='VERIFIED',image=identity(target))
        except (ValueError,OSError) as error:row.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        result.append(row)
        print(c['name'],row['status'],flush=True)
        if row.get('http_status')==403:break
    return {'source':identity(Path(__file__)),'manifest':identity(MANIFEST),'protocol':identity(PROTOCOL),'cases':result,
            'completed':len(result)==12,'status':'FAIL' if len(result)!=12 or any(c['status']=='FAIL' for c in result) else 'VERIFIED'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    result=acquire(a.cache_dir,a.verify_only)
    with a.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(1 if result['status']=='FAIL' else 0)


if __name__=='__main__':main()
