import pathlib,hashlib,json,subprocess,sys
root=pathlib.Path(__file__).resolve().parent
target=pathlib.Path(sys.argv[1]).resolve()
head=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
assert head in ['04a3530c6ec9755bb7b89c3e31c1c3e15c9b1ea9','7056c67cd1f59c8d3e2237b84df05c223d99a25e'],head
folder=target/'docs/verification/pr06-independent';folder.mkdir(parents=True,exist_ok=True)
for name,digest in json.loads((root/'original-probe-bindings.json').read_text()).items():
 content=(root/'original-probes'/(name+'.txt')).read_bytes();assert hashlib.sha256(content).hexdigest()==digest;(folder/name).write_bytes(content)
print('Exact original probes materialized; target',head)
