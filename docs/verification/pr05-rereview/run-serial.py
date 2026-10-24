"""Re-run unchanged browser checks without overlapping Chromium test suites."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
REPLAY=Path('/workspace/pr05-rereview-probes');FROZEN=REPLAY/'docs/verification/pr05-independent'
env=dict(os.environ,npm_config_cache='/workspace/npm-cache',PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH='/usr/bin/chromium')
commands=[
    (ROOT,['npm','run','test:e2e'],'serial-standard-browser.txt'),
    (REPLAY,['node','docs/verification/pr05-independent/browser-oracle.mjs'],'serial-original-browser.txt'),
    (REPLAY,['node','docs/verification/pr05-independent/intrinsic-oracle.mjs'],'serial-original-intrinsic.txt'),
    (REPLAY,['python3','docs/verification/pr05-independent/image-oracle.py'],'serial-original-image.txt'),
]
results=[]
for cwd,cmd,name in commands:
    start=time.time()
    with (OUT/'logs'/name).open('w')as stream:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    results.append(dict(command=cmd,cwd=str(cwd),exit=r.returncode,seconds=time.time()-start,output=name))
    (OUT/'logs/serial.json').write_text(json.dumps(results,indent=2)+'\n');print(name,r.returncode,flush=True)
    if cwd==ROOT:
        with zipfile.ZipFile(OUT/'standard-serial.zip','w',zipfile.ZIP_DEFLATED)as z:
            for directory in ['playwright-report','test-results']:
                for f in (ROOT/directory).rglob('*'):
                    if f.is_file():z.write(f,str(f.relative_to(ROOT)))
with zipfile.ZipFile(OUT/'frozen-replay-serial.zip','w',zipfile.ZIP_DEFLATED)as z:
    for directory in ['after','before','before-visible-fresh']:
        for f in (FROZEN/directory).rglob('*'):
            if f.is_file():z.write(f,str(f.relative_to(FROZEN)))
    for name in ['numeric-vectors.json','numeric-results.json','numeric-failures.json','intrinsic-results.json','image-results.json']:
        z.write(FROZEN/name,name)
for name in ['intrinsic-results.json','image-results.json']:
    shutil.copy2(FROZEN/name,OUT/name)
shutil.copy2(FROZEN/'after/results.json',OUT/'browser-results.json')
if any(r['exit']for r in results):raise SystemExit(1)
