# Identical checker-only zero-call payload, original/new actual summarizers. No SDK.
import pathlib,tempfile,subprocess,json,hashlib,sys
source=pathlib.Path(sys.argv[1]).resolve();old=subprocess.check_output(['git','-C',str(source),'show','31721975d4eeef0ff31d29074aadafc801efae88:scripts/real-sdk/summarize.mjs'])
with tempfile.TemporaryDirectory(prefix='pr07-summary-compare-') as tmp:
 t=pathlib.Path(tmp);p=t/'old.mjs';p.write_bytes(old);rows=[]
 for label,module in [('reviewed-old',p),('corrected',source/'scripts/real-sdk/summarize.mjs')]:
  run=t/label;(run/'browser/failed').mkdir(parents=True)
  fixture={'runtime':{'scope':'CHECKER ONLY; no SDK/model call'},'evidence':{'probe':{'raw':[],'creates':[{'model':'selfie-segmenter','error':'initialization failed'}],'snapshots':[],'rawDropped':0}}}
  (run/'browser/failed/evidence.json').write_text(json.dumps(fixture))
  code="const {summarizeRun}=await import(process.argv[1]); await summarizeRun(process.argv[2]);"
  c=subprocess.run(['node','--input-type=module','-e',code,module.as_uri(),str(run)],capture_output=True,text=True,timeout=10);assert c.returncode==0,c.stderr
  actual=json.loads((run/'measurements.json').read_text());rows.append({'label':label,'sourceSha256':hashlib.sha256(module.read_bytes()).hexdigest(),'exit':c.returncode,'fixture':fixture,'actual':actual})
 assert rows[0]['actual']['inferenceExecuted'] is True
 assert rows[1]['actual']['inferenceExecuted'] is False and rows[1]['actual']['populations']==[]
 print(json.dumps(rows,indent=2))
