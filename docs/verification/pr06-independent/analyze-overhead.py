import json,statistics,math,pathlib,sys
root=pathlib.Path(__file__).resolve().parent
src=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else root.parent/'pr06/overhead.json')
data=json.load(open(src));trials=data['trials'];groups={}
def stats(v):
 return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),p95=sorted(v)[math.ceil(.95*len(v))-1],variance=statistics.pvariance(v),min=min(v),max=max(v))
for t in trials:
 n=t['iterations'];c=t['counts'];rgba=t['path']=='rgba';enabled=t['enabled'];seg=t['model']=='selfie-segmenter'
 assert c['processed']==c['sdk']==c['drawn']==n
 assert c['captured']==n*3 and c['compared']==(n*5-3 if rgba else 0)
 assert c['released']==(0 if rgba else n*3)
 assert c['maskRead']==c['maskClosed']==(n if seg else 0)
 assert c['snapshot']==(n//8 if enabled else 0)
 assert c['metricClock']>0 if enabled else c['metricClock']==0
 for stage in ['engine','draw','total','hud']:
  x=t['samples'][stage];s=stats(x)
  assert len(x)==(n//8 if stage=='hud' else n)
  for key in ['mean','median','p95','variance']:assert math.isclose(s[key],t[stage][key],abs_tol=1e-8), (stage,key,s[key],t[stage][key])
 key=f"{t['width']}x{t['height']}/{t['path']}/{t['model']}";groups.setdefault(key,[]).append(t)
rows={}
for key,ts in groups.items():
 pairs={}
 for t in ts:pairs.setdefault(t['pair'],{})[t['enabled']]=t
 assert all(set(p)=={False,True} for p in pairs.values())
 assert [t['enabled'] for t in ts]==sum(([False,True] if i%2==0 else [True,False] for i in range(len(pairs))),[])
 rows[key]={stage:{'disabled':stats([x for t in ts if not t['enabled'] for x in t['samples'][stage]]),'enabled':stats([x for t in ts if t['enabled'] for x in t['samples'][stage]]),'paired_mean_delta':stats([statistics.mean(p[True]['samples'][stage])-statistics.mean(p[False]['samples'][stage]) for p in pairs.values()])} for stage in ['total','engine','draw','hud']}
output={'input':str(src),'configurations':len(rows),'trials':len(trials),'warmups':len(data['warmup']),'jobs':sum(t['counts']['processed'] for t in trials),'all_work_and_raw_statistics_valid':True,'groups':rows}
out=root/(sys.argv[2] if len(sys.argv)>2 else 'author-raw-independent-analysis.json');out.write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k!='groups'}))
for key,v in rows.items():print(key,'total paired median/p95',v['total']['paired_mean_delta']['median'],v['total']['paired_mean_delta']['p95'])
