import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
const summary = values => {
  if (!values.length) return { count: 0, mean: null, median: null, p95: null, max: null, populationVariance: null };
  const sorted = [...values].sort((a, b) => a - b); const n = sorted.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  return { count: n, mean, median: sorted[Math.floor((n - 1) / 2)] / 2 + sorted[Math.floor(n / 2)] / 2, p95: sorted[Math.ceil(n * 0.95) - 1], max: sorted[n - 1], populationVariance: values.reduce((s, v) => s + (v - mean) ** 2, 0) / n };
};
export async function summarizeRun(run) {
  const files = [];
  async function walk(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true }).catch(() => [])) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(file); else if (entry.name === 'evidence.json') files.push(file);
    }
  }
  await walk(path.join(run, 'browser'));
  const populations = [];
  for (const file of files) {
    const { evidence, runtime } = JSON.parse(await readFile(file, 'utf8')); const probe = evidence.probe;
    for (const model of ['mediapipe', 'selfie-segmenter']) {
      const raw = probe.raw.filter(row => row.model === model && !row.error);
      const positive = raw.filter(row => row.scene === (model === 'mediapipe' ? 'coffee.png' : 'astronaut.png'));
      populations.push({ file: path.relative(run, file), model, runtime,
        initialization: probe.creates.filter(row => row.model === model),
        retainedSdkCalls: raw.length, droppedSdkCalls: probe.rawDropped,
        warmup: positive[0] ?? null,
        steadyStateSdkWall: summary(positive.slice(1).map(row => row.ms).filter(Number.isFinite)),
        missingSdkDurations: positive.slice(1).filter(row => !Number.isFinite(row.ms)).length,
        coverage: probe.rawDropped ? 'partial; bounded latest80 retained, first-ever warmup unknown, retained-first shown' : 'all retained actual SDK calls; first positive excluded as warmup',
        snapshots: probe.snapshots,
      });
    }
  }
  const output = { units: 'milliseconds JavaScript wall time; observation overhead present in product metrics, original-call-only raw.ms excludes later observation', inferenceExecuted: files.length > 0, populations,
    limits: 'No physical GPU duration/utilization, CPU utilization, camera FPS, camera-to-display latency or real-device guarantee; overlapping stages must not be summed. No timing pass threshold.' };
  await writeFile(path.join(run, 'measurements.json'), JSON.stringify(output, null, 2) + '\n');
}
