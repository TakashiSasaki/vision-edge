"""Author replay of unchanged independent vectors; not independent closure."""
import hashlib
import json
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).resolve().parents[4]
replay = pathlib.Path('/workspace/vision-edge-pr04-before')
evidence = pathlib.Path('/workspace/scratch/pr04-review-evidence/docs/verification/pr04-independent')
logs = root / 'docs/verification/pr04-corrections/logs'
products = ['src/services/frameScheduler.ts', 'src/components/VisionCanvas.tsx', 'src/services/visionEngine.ts', 'src/types/vision.ts']
for product in products:
    shutil.copy2(root / product, replay / product)
vectors = {}
for vector in (evidence / 'vectors').iterdir():
    relative = vector.relative_to(evidence)
    original = subprocess.check_output(['git', 'show', 'c632cf0cbce82b48503540651120c88fff75c8ef:docs/verification/pr04-independent/' + relative.as_posix()], cwd=root)
    assert vector.read_bytes() == original
    assert (replay / 'docs/verification/pr04-independent' / relative).read_bytes() == original
    vectors[relative.as_posix()] = hashlib.sha256(original).hexdigest()
binding = {'review_evidence_head': 'c632cf0cbce82b48503540651120c88fff75c8ef', 'corrective_commit': '89e444820b6befbd333236e65697b5bcf8f5e15a', 'unchanged_vectors': vectors, 'copied_corrected_products': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in products}}
(logs / 'replay-source-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
commands = [
    ['node', 'node_modules/vitest/vitest.mjs', 'run', '--config', 'docs/verification/pr04-independent/vectors/vitest.config.mjs'],
    ['node', 'docs/verification/pr04-independent/vectors/native-first-frame.mjs'],
    ['node', 'docs/verification/pr04-independent/vectors/native-frame-race.mjs'],
]
records = []
for i, cmd in enumerate(commands):
    result = subprocess.run(cmd, cwd=replay, capture_output=True, text=True)
    (logs / f'after-{i:02d}.txt').write_text((result.stdout + result.stderr).rstrip() + '\n')
    records.append({'command': cmd, 'cwd': str(replay), 'exit': result.returncode, 'log': f'after-{i:02d}.txt'})
    (logs / 'after-checks.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'{cmd}: exit {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
