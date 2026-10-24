import json,statistics,math,pathlib,sys
root=pathlib.Path(__file__).resolve().parent
source=pathlib.Path(sys.argv[1]);data=json.loads(source.read_text());trials=data['trials']
def stats(x):
 s=sorted(x);return {'n':len(x),'mean':statistics.mean(x),'median':statistics.median(x),'p95':s[math.ceil(len(x)*.95)-1],'variance':statistics.pvariance(x),'min':s[0],'max':s[-1]}
groups={}
for t in trials:
 n=t.get('iterations',t.get('n'));assert n==32
 assert t.get('snapshots',t.get('snapshotCalls'))==n and t.get('renders',t.get('renderCalls'))==n
 if 'readCalls' in t:assert t['readCalls']==(n if t['version']=='after' else 0)
 for values in t['samples'].values():assert len(values)==n and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in values)
 for snap,prep,total in zip(t['samples']['snapshot'],t['samples']['prepare'],t['samples']['total']):assert math.isclose(snap+prep,total,abs_tol=1e-8)
 if 'read' in t['samples']:
  for rd,rend,prep in zip(t['samples']['read'],t['samples']['render'],t['samples']['prepare']):assert math.isclose(rd+rend,prep,abs_tol=1e-8)
 key=f"{t['profile']}/{'open' if t['open'] else 'closed'}";groups.setdefault(key,[]).append(t)
output={'input':str(source),'trials':len(trials),'updates':sum(t.get('n',t.get('iterations')) for t in trials),'warmups':len(data['warmup']),'groups':{}}
for key,ts in groups.items():
 pairs={}
 for t in ts:pairs.setdefault(t['pair'],{})[t['version']]=t
 assert all(set(p)=={'before','after'} for p in pairs.values())
 assert [t['version'] for t in ts]==sum((['before','after'] if i%2==0 else ['after','before'] for i in range(len(pairs))),[])
 stages=list(ts[0]['samples']);output['groups'][key]={stage:{'before':stats([x for t in ts if t['version']=='before' for x in t['samples'][stage]]),'after':stats([x for t in ts if t['version']=='after' for x in t['samples'][stage]]),'pairedMeanDelta':stats([statistics.mean(p['after']['samples'][stage])-statistics.mean(p['before']['samples'][stage]) for p in pairs.values()])} for stage in stages}
(root/sys.argv[2]).write_text(json.dumps(output,indent=2)+'\n');print({k:v for k,v in output.items() if k!='groups'})
for k,v in output['groups'].items():print(k,'prepare delta',v['prepare']['pairedMeanDelta'],'total delta',v['total']['pairedMeanDelta'])
