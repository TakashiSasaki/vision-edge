import tempfile,pathlib,shutil,subprocess,json,sys
src=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/pr07-review-current')
results=[]
for name,manifest in [('empty-assets',{'schema':1,'assets':[]}),('wrong-schema',{'schema':999,'assets':[]}),('missing-required-models',{'schema':1,'assets':[a for a in json.loads((src/'e2e-real-sdk/assets.lock.json').read_text())['assets'] if a.get('packageFile')]}),('invalid-json',None)]:
 with tempfile.TemporaryDirectory(prefix='pr07-invalid-manifest-') as tmp:
  root=pathlib.Path(tmp); (root/'scripts/real-sdk').mkdir(parents=True); (root/'e2e-real-sdk').mkdir()
  shutil.copy(src/'scripts/real-sdk/assets.mjs',root/'scripts/real-sdk/assets.mjs')
  (root/'node_modules').symlink_to(src/'node_modules',target_is_directory=True)
  (root/'e2e-real-sdk/assets.lock.json').write_text(json.dumps(manifest) if manifest is not None else '{invalid')
  p=subprocess.run(['node','scripts/real-sdk/assets.mjs'],cwd=root,capture_output=True,text=True,timeout=10)
  results.append({'case':name,'manifest':manifest,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'expected':'invalid manifest must fail before asset acceptance; never exit0'})
print(json.dumps(results,indent=2))
