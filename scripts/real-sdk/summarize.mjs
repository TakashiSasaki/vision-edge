import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { linkedReadyResults } from '../../e2e-real-sdk/acceptance.ts';
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
      if (entry.isDirectory()) await walk(file); else if (entry.name === 'evidence.json' || entry.name === 'diagnostic.json') files.push(file);
    }
  }
  await walk(path.join(run, 'browser'));
  const populations = []; const execution = [];
  for (const file of files) {
    const data = JSON.parse(await readFile(file, 'utf8'));
    const probe = data.evidence?.probe ?? data.observation;
    const runtime = data.runtime;
    const raw = Array.isArray(probe?.raw) ? probe.raw : [];
    const valid = raw.filter(row => Number.isSafeInteger(row.id) && row.id > 0 && ['mediapipe', 'selfie-segmenter'].includes(row.model));
    const errors = [];
    if (!probe || !Array.isArray(probe.raw)) errors.push('missing SDK observations');
    if (valid.length !== raw.length || new Set(valid.map(row => row.id)).size !== valid.length) errors.push('invalid or duplicate SDK call records');
    const count = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
    const attempts = count(probe?.calls);
    const sdkSuccesses = count(probe?.sdkSuccesses);
    const dropped = count(probe?.rawDropped);
    const successes = valid.filter(row => row.sdkSucceeded === true);
    const failed = valid.filter(row => row.sdkSucceeded === false);
    if (attempts !== null && dropped !== null && attempts !== valid.length + dropped) errors.push('attempt/retention counters do not match SDK records');
    if (sdkSuccesses !== null && dropped !== null && sdkSuccesses > successes.length + dropped) errors.push('success counter exceeds observed/evicted SDK records');
    if (attempts !== null && attempts < valid.length) errors.push('attempt counter smaller than retained records');
    if (sdkSuccesses !== null && (attempts === null || sdkSuccesses > attempts || sdkSuccesses < successes.length)) errors.push('inconsistent successful SDK counter');
    const observations = { raw: valid, processed: probe?.processed ?? [], adoptions: probe?.adoptions ?? [], draws: probe?.draws ?? [] };
    const ready = linkedReadyResults(observations);
    const adopted = new Set(observations.adoptions.filter(row => ready.some(result => result.id === row.id)).map(row => row.id));
    const rendered = new Set(observations.draws.filter(row => adopted.has(row.id) && row.width > 0 && row.height > 0 && Number.isFinite(row.alphaPixels) && row.alphaPixels >= 0).map(row => row.id));
    const eligible = errors.length === 0;
    execution.push({ file: path.relative(run, file), evidenceErrors: errors,
      attempts: attempts ?? valid.length, retainedAttempts: valid.length,
      successfulSdkCalls: sdkSuccesses ?? successes.length, retainedSuccessfulSdkCalls: successes.length,
      failedSdkCallsRetained: failed.length, unknownSdkOutcomesRetained: valid.length - successes.length - failed.length,
      readyProcessedResultsRetained: ready.length, adoptedResultsRetained: adopted.size, renderedResultsRetained: rendered.size,
      inferenceAttempted: eligible && ((attempts ?? valid.length) > 0),
      inferenceExecuted: eligible && ((sdkSuccesses ?? successes.length) > 0),
      resultAdopted: eligible && adopted.size > 0, resultRendered: eligible && rendered.size > 0,
      coverage: (probe?.rawDropped ?? 0) > 0 ? 'partial retained records; lifetime attempt/success counters where available' : 'retained observations; unknown outcomes are not successes',
      initialization: probe?.creates ?? [],
    });
    for (const model of ['mediapipe', 'selfie-segmenter']) {
      const modelRaw = successes.filter(row => row.model === model && !row.error);
      if (!eligible || !modelRaw.length) continue;
      const positive = modelRaw.filter(row => row.stamp?.scene === (model === 'mediapipe' ? 'coffee.png' : 'astronaut.png'));
      populations.push({ file: path.relative(run, file), model, runtime,
        initialization: (probe.creates ?? []).filter(row => row.model === model),
        retainedSdkCalls: modelRaw.length, droppedSdkCalls: probe.rawDropped ?? 0,
        warmup: positive[0] ?? null,
        steadyStateSdkWall: summary(positive.slice(1).map(row => row.ms).filter(value => typeof value === 'number' && Number.isFinite(value) && value >= 0)),
        missingSdkDurations: positive.slice(1).filter(row => typeof row.ms !== 'number' || !Number.isFinite(row.ms) || row.ms < 0).length,
        coverage: probe.rawDropped ? 'partial; bounded latest80 retained, first-ever warmup unknown, retained-first shown' : 'all retained actual SDK calls; first positive excluded as warmup',
        snapshots: probe.snapshots ?? [],
      });
    }
  }
  const output = { units: 'milliseconds JavaScript wall time; observation overhead present in product metrics, original-call-only raw.ms excludes later observation',
    inferenceAttempted: execution.some(row => row.inferenceAttempted), inferenceExecuted: execution.some(row => row.inferenceExecuted),
    resultAdopted: execution.some(row => row.resultAdopted), resultRendered: execution.some(row => row.resultRendered), execution, populations,
    limits: 'inferenceExecuted means an observed successful SDK return, not a semantic/model acceptance verdict. Attempts, processing, adoption and rendering are separate. No physical timing/utilization guarantee; overlapping stages must not be summed.' };
  await writeFile(path.join(run, 'measurements.json'), JSON.stringify(output, null, 2) + '\n');
}
