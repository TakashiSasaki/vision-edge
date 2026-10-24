# New independent controls on isolated ignored cache; no model/network acquisition.
import pathlib,subprocess,os,json,hashlib
root=pathlib.Path('/tmp/pr07-rereview-current');asset=root/'.cache/real-sdk-assets/vision_wasm_internal.wasm';original=asset.read_bytes();records=[]
def run(argv,timeout=30,extra=None):
 env=dict(os.environ);env.update(extra or {});p=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=timeout)
 row={'argv':argv,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr};records.append(row);return row
try:
 asset.write_bytes(bytes([original[0]^1])+original[1:])
 a=run(['node','scripts/real-sdk/assets.mjs']);assert a['exit']==1
 b=run(['node','scripts/real-sdk/run.mjs']);assert b['exit']==1
 last=pathlib.Path((root/'.cache/real-sdk-last-run.txt').read_text().strip());matrix=json.loads((last/'matrix.json').read_text());assert matrix['childExit'] is None;assert all(r['status']=='fail' for r in matrix['matrix'][:6]);b['matrix']=matrix
finally: asset.write_bytes(original)
backup=asset.with_suffix('.wasm.independent-backup');assert not backup.exists();asset.rename(backup)
try:
 a=run(['node','scripts/real-sdk/assets.mjs']);assert a['exit']==2
 b=run(['node','node_modules/@playwright/test/cli.js','test','--config','playwright.real-sdk.config.ts','--grep','asset diagnostic selfie-segmenter'],150,{'PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH':'/usr/bin/chromium','REAL_SDK_RUN_DIR':'/tmp/pr07-rereview-unrelated-startup'})
 assert b['exit']==1
 report=json.loads(pathlib.Path('/tmp/pr07-rereview-unrelated-startup/playwright.json').read_text());assert report['stats']['expected']==0 and report['stats']['unexpected']==1;b['stats']=report['stats']
finally: backup.rename(asset)
assert asset.read_bytes()==original
print(json.dumps({'scope':'NEW cache/integrity and failure-diagnostic controls ONLY; no model acquisition/inference','originalSha256':hashlib.sha256(original).hexdigest(),'restored':True,'records':records},indent=2))
