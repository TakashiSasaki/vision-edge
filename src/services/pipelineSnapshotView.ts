import { PIPELINE_EVENT_CAP, PIPELINE_JOB_CAP, PIPELINE_QUANTILE_CAP, PIPELINE_STAGES, type AttemptOutcome, type PipelineSnapshot, type PipelineStage, type ResultOutcome, type StageStatistics } from './pipelineMetrics';

// A measurement result is runtime data, even when its producer has a TS type.
// Copy only bounded known fields; React must never index/coerce the raw payload.
export interface PipelineSnapshotView {
  stages: Record<PipelineStage, Record<AttemptOutcome, StageStatistics | null>>;
  results: PipelineSnapshot['results'] | null;
  resultsTruncated: boolean | null;
  pendingResults: number | null;
  unclocked: number | null;
  captureDutyPercent: number | null;
}
const outcomes: readonly AttemptOutcome[] = ['success', 'failed', 'cancelled', 'stale'];
const resultOutcomes: readonly ResultOutcome[] = ['first-draw', 'overwritten', 'cancelled', 'hidden', 'draw-failed', 'expired', 'capacity', 'unbound'];
const record = (value: unknown): Record<string, unknown> | null => typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null;
const count = (value: unknown, limit: number): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 && value <= limit;
const duration = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const flag = (value: unknown): boolean | null => typeof value === 'boolean' ? value : null;
function row(value: unknown): StageStatistics | null {
  try {
    const source = record(value); if (!source) return null;
    const { count: n, measured, missing, quantileCount, truncated } = source;
    if (!count(n, PIPELINE_EVENT_CAP) || !count(measured, n) || !count(missing, n) || measured + missing !== n
      || !count(quantileCount, Math.min(measured, PIPELINE_QUANTILE_CAP)) || typeof truncated !== 'boolean') return null;
    return {
      count: n, measured, missing, quantileCount, truncated,
      // Null statistics, including overflowed sum/mean, are valid partial data.
      // Unexecuted stages/quantiles cannot acquire an invented zero duration.
      totalMs: measured ? duration(source.totalMs) : null,
      mean: measured ? duration(source.mean) : null,
      max: measured ? duration(source.max) : null,
      median: quantileCount ? duration(source.median) : null,
      p95: quantileCount ? duration(source.p95) : null,
    };
  } catch { return null; }
}
function results(value: unknown): PipelineSnapshot['results'] | null {
  const source = record(value); if (!source) return null;
  const result: PipelineSnapshot['results'] = {};
  let total = 0;
  for (const outcome of resultOutcomes) {
    if (!Object.hasOwn(source, outcome)) continue;
    const n = source[outcome]; if (!count(n, PIPELINE_EVENT_CAP)) return null;
    result[outcome] = n; total += n;
  }
  return total <= PIPELINE_EVENT_CAP ? result : null;
}
export function readPipelineSnapshot(value: unknown): PipelineSnapshotView | null {
  try {
    const source = record(value); if (!source) return null;
    const rawStages = record(source.stages);
    const stages = {} as PipelineSnapshotView['stages'];
    for (const stage of PIPELINE_STAGES) {
      const rawOutcomes = rawStages ? record(rawStages[stage]) : null;
      stages[stage] = {} as Record<AttemptOutcome, StageStatistics | null>;
      for (const outcome of outcomes) stages[stage][outcome] = row(rawOutcomes?.[outcome]);
    }
    const rawUnclocked = source.unclocked;
    const rawPending = source.pendingResults;
    const unclocked = count(rawUnclocked, Number.MAX_SAFE_INTEGER) ? rawUnclocked : null;
    const captures = outcomes.map(outcome => stages.capture[outcome]);
    const captureCovered = unclocked === 0 && captures.every(s => s !== null && !s.truncated && s.missing === 0 && (s.measured === 0 || s.totalMs !== null))
      && captures.some(s => s !== null && s.measured > 0);
    return {
      stages, results: results(source.results), resultsTruncated: flag(source.resultsTruncated),
      pendingResults: count(rawPending, PIPELINE_JOB_CAP) ? rawPending : null,
      unclocked, captureDutyPercent: captureCovered ? duration(source.captureDutyPercent) : null,
    };
  } catch { return null; } // Includes throwing getters/proxies; never a product failure.
}
