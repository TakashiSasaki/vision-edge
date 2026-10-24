"""Author replays PR #9 with only a Canvas-capable SDK observation boundary."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

root = Path(__file__).resolve().parents[4]
replay = Path('/workspace/vision-edge-pr04-r2')
evidence = Path('/workspace/scratch/pr04-rereview-evidence/docs/verification/pr04-rereview')
logs = root / 'docs/verification/pr04-compatibility/logs'
products = ['src/services/frameScheduler.ts', 'src/services/visionEngine.ts', 'src/types/vision.ts', 'src/components/VisionCanvas.tsx']
for product in products:
    shutil.copy2(root / product, replay / product)
changes = []
for name in ['native-boundaries.mjs', 'forced-native-race.mjs', 'independentSdk.mjs']:
    original = (evidence / 'vectors' / name).read_text()
    # TexImageSource includes Canvas. WebCodecs requires an explicit timestamp
    # when an observation frame is created from a Canvas; it is not source PTS.
    adapted = original.replace('new NativeFrame(input)', 'new NativeFrame(input,input instanceof HTMLCanvasElement?{timestamp:0}:undefined)').replace('new Frame(input)', 'new Frame(input,input instanceof HTMLCanvasElement?{timestamp:0}:undefined)')
    assert adapted != original
    original_assertions = [line for line in original.splitlines() if 'expect(' in line or 'process.exitCode' in line]
    assert original_assertions == [line for line in adapted.splitlines() if 'expect(' in line or 'process.exitCode' in line]
    (replay / 'docs/verification/pr04-rereview/vectors' / name).write_text(adapted)
    changes.append({'file': name, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(), 'adapted_sha256': hashlib.sha256(adapted.encode()).hexdigest(), 'change': 'Canvas observation-frame constructor timestamp argument only; assertions unchanged'})
(logs / 'observer-adaptation.json').write_text(json.dumps(changes, indent=2) + '\n')
commands = [
    ['node', 'node_modules/vitest/vitest.mjs', 'run', '--config', 'docs/verification/pr04-rereview/vectors/vitest.config.mjs'],
    ['node', 'docs/verification/pr04-rereview/vectors/native-boundaries.mjs'],
    ['node', 'node_modules/@playwright/test/cli.js', 'test', '--config', 'docs/verification/pr04-rereview/vectors/playwright.config.mjs'],
    ['node', 'docs/verification/pr04-rereview/vectors/forced-native-race.mjs'],
]
records = []
for i, command in enumerate(commands):
    result = subprocess.run(command, cwd=replay, text=True, capture_output=True)
    text = result.stdout + result.stderr
    (logs / f'after-{i:02d}.txt').write_text('\n'.join(line.rstrip() for line in text.splitlines()).rstrip() + '\n')
    records.append({'command': command, 'exit': result.returncode, 'log': f'after-{i:02d}.txt'})
    (logs / 'after-checks.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'{i}: exit {result.returncode}', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
