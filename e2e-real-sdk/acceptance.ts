// Checker-only data contract. These functions never call or replace an SDK.
export interface Stamp { scene: string; revision: number; stream: number }
export interface RawObservation { id: number; model: string; input: string; requestId: number | null; stamp: Stamp | null; sdkSucceeded: boolean; error?: string; ms: number; scene?: string }
export interface ProcessedObservation { id: number; rawId: number; model: string; input: string; requestId: number; stamp: Stamp | null; status: string; objects: Array<{ label: string }> }
export interface AdoptionObservation { id: number }
export interface DrawObservation { id: number; alphaPixels: number; width: number; height: number }
export interface Observations { raw: RawObservation[]; processed: ProcessedObservation[]; adoptions: AdoptionObservation[]; draws: DrawObservation[] }
export interface SceneBoundary { afterCallId: number; afterResultId: number; model: string; input: string; requestId: number; stamp: Stamp }
const identity = (value: number) => Number.isSafeInteger(value) && value > 0;
const sameStamp = (a: Stamp | null, b: Stamp | null) => !!a && !!b && a.scene === b.scene && a.revision === b.revision && a.stream === b.stream;
const bound = (raw: RawObservation, result: ProcessedObservation) => raw.sdkSucceeded === true && !raw.error && raw.id === result.rawId && raw.model === result.model && raw.input === result.input && raw.requestId === result.requestId && sameStamp(raw.stamp, result.stamp);
export function linkedReadyResults(probe: Observations) {
  if (new Set(probe.raw.map(row => row.id)).size !== probe.raw.length || new Set(probe.processed.map(row => row.id)).size !== probe.processed.length) return [];
  return probe.processed.filter(result => identity(result.id) && identity(result.rawId) && result.status === 'ready' && probe.raw.some(raw => bound(raw, result)));
}
export function checkNegativeScene(probe: Observations, boundary: SceneBoundary, expected: string) {
  const fail = (reason: string) => ({ status: 'fail' as const, reason });
  const wait = (reason: string) => ({ status: 'waiting' as const, reason });
  if (!boundary || !Number.isSafeInteger(boundary.afterCallId) || !Number.isSafeInteger(boundary.afterResultId) || !boundary.stamp) return fail('invalid scene boundary');
  const attempts = probe.raw.filter(row => row.id > boundary.afterCallId);
  if (attempts.length && attempts.some((row, index) => row.id !== boundary.afterCallId + index + 1)) return fail('negative-scene attempt coverage missing/truncated');
  if (attempts.some(row => typeof row.sdkSucceeded !== 'boolean')) return fail('unknown SDK outcome after scene boundary');
  if (attempts.some(row => row.error || row.sdkSucceeded === false)) return fail('new SDK/observation failure after scene boundary');
  if (attempts.some(row => row.model !== boundary.model || row.input !== boundary.input || row.requestId !== boundary.requestId)) return fail('unexpected model/input/generation after scene boundary');
  const successful = attempts.filter(row => row.sdkSucceeded === true && !row.error && sameStamp(row.stamp, boundary.stamp));
  if (new Set(successful.map(row => row.id)).size !== successful.length) return fail('duplicate raw call identity');
  if (successful.length < 5) return wait('need five fresh successful SDK calls for the presented scene');
  // Existing product retention allows the first two empty observations to retain
  // a prior object. Every later eligible processed result must be genuinely empty.
  const afterRetention = new Set(successful.slice(2).map(row => row.id));
  const results = linkedReadyResults(probe).filter(row => row.id > boundary.afterResultId && afterRetention.has(row.rawId));
  if (new Set(results.map(row => row.rawId)).size !== results.length) return fail('duplicate processed raw identity');
  if (results.some(row => row.objects.some(object => object.label === expected))) return fail('expected object remains after retention');
  if (results.length < 3) return wait('need three fresh processed results after retention');
  const latest = results.find(row => row.rawId === successful.at(-1)!.id);
  if (!latest) return wait('latest successful SDK call has no matching ready result');
  if (!probe.adoptions.some(row => row.id === latest.id)) return wait('latest ready result was not adopted');
  if (!probe.draws.some(row => row.id === latest.id && row.width > 0 && row.height > 0 && Number.isFinite(row.alphaPixels) && row.alphaPixels >= 0)) return wait('latest adopted result has no matching visible Canvas copy');
  return { status: 'pass' as const, reason: 'fresh SDK → processed → adopted → rendered scene', rawId: latest.rawId, resultId: latest.id, successfulCalls: successful.length, processedAfterRetention: results.length };
}
