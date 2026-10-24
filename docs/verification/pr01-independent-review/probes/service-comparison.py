#!/usr/bin/env python3
"""Compare real base/target services with synthetic detector boundary inputs."""
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'logs'
SOURCE = """
import { getLabelInfo, getCategoryColor } from './src/services/dictionary';
import { visionEngine } from './src/services/visionEngine';
let detections = [];
visionEngine.mediaPipeDetector = { detectForVideo: () => ({ detections }) };
const video = { videoWidth: 640, videoHeight: 480, readyState: 2 };
const cases = [];
for (const model of ['mediapipe', 'yolo-seg', 'segformer', 'selfie-segmenter']) {
  for (const populated of [true, false, false, false]) {
    detections = populated ? [{categories: [{categoryName: 'dog', score: 0.9}], boundingBox: {originX: 10, originY: 20, width: 100, height: 80}}] : [];
    cases.push({model, populated, objects: visionEngine.detectVideoFrame(video, model, 0.35).objects});
  }
}
console.log(JSON.stringify({dog: getLabelInfo(' DOG '), unknown: getLabelInfo('QUANTUM WIDGET'), carpet: getLabelInfo('carpet'), constructorType: typeof getLabelInfo('constructor'), color: getCategoryColor('dog'), cases}));
"""

if __name__ == '__main__':
    results = {}
    for label, sha in [('base', '90323c00a6cd9a8157bf3eec853dbe5a857f07a1'), ('target', 'f20474be156522ee6141cde9c2b3a958db8d2457')]:
        with tempfile.TemporaryDirectory(prefix='pr01-review-services-') as directory:
            probe = Path(directory)
            with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', sha], cwd=ROOT))) as tar:
                tar.extractall(probe, filter='data')
            (probe / 'node_modules').symlink_to(ROOT / 'node_modules', target_is_directory=True)
            (probe / 'probe.ts').write_text(SOURCE)
            output = subprocess.check_output([str(ROOT / 'node_modules/.bin/tsx'), 'probe.ts'], cwd=probe, text=True)
            results[label] = json.loads(output)
    results['identical_observed_outputs'] = results['base'] == results['target']
    (OUT / 'service-comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('Base/target service outputs identical:', results['identical_observed_outputs'])
    print('Dictionary:', {k: v for k, v in results['target'].items() if k != 'cases'})
