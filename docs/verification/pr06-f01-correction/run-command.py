import datetime, json, os, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parent
name, cwd, *command = sys.argv[1:]
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
log = root / 'logs' / (name + '.txt')
log.write_bytes(result.stdout)
receipt = dict(name=name, cwd=cwd, command=command, start=start, end=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit=result.returncode, log=str(log.relative_to(root)), browser=os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH'))
with (root / 'commands.jsonl').open('a') as stream: stream.write(json.dumps(receipt) + '\n')
print(f'{name}: exit={result.returncode}', flush=True)
sys.exit(result.returncode)
