import '../../../../src/index.css';
import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import { PipelineDetails as Before } from './PipelineDetails-before';
import { PipelineDetails as After } from '../../../../src/components/PipelineDetails';
import { readPipelineSnapshot } from '../../../../src/services/pipelineSnapshotView';
import { PIPELINE_STAGES, PipelineMetrics } from '../../../../src/services/pipelineMetrics';
const root = createRoot(document.getElementById('hud')!);
const profiles = new Map<string, PipelineMetrics>();
for (const [name, count] of [['fresh', 0], ['300', 300], ['truncated', 2051]] as const) {
  const metrics = new PipelineMetrics(() => 0);
  for (const stage of PIPELINE_STAGES) for (const outcome of ['success', 'failed', 'cancelled', 'stale'] as const)
    for (let i = 0; i < count; i++) metrics.record({ at: 0, stage, outcome, ms: i % 5 ? i / 100 : null });
  profiles.set(name, metrics);
}
function trial({ version, profile, open, iterations }: { version: 'before' | 'after'; profile: string; open: boolean; iterations: number }) {
  const Component = version === 'before' ? Before : After;
  const metrics = profiles.get(profile)!;
  flushSync(() => root.render(<Component snapshot={null} />));
  document.querySelector('details')!.open = open;
  const snapshot: number[] = []; const prepare: number[] = []; const total: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now(); const value = metrics.snapshot(); const computed = performance.now();
    flushSync(() => root.render(version === 'before' ? <Before snapshot={value} /> : <After snapshot={readPipelineSnapshot(value)} />)); const end = performance.now();
    snapshot.push(computed - start); prepare.push(end - computed); total.push(end - start);
  }
  return { version, profile, open, iterations, snapshots: iterations, renders: iterations, samples: { snapshot, prepare, total } };
}
function equivalent(profile: string) {
  const metrics = profiles.get(profile)!;
  flushSync(() => root.render(<Before snapshot={metrics.snapshot()} />));
  document.querySelector('details')!.open = false;
  const before = document.getElementById('hud')!.innerHTML;
  flushSync(() => root.render(<After snapshot={readPipelineSnapshot(metrics.snapshot())} />));
  document.querySelector('details')!.open = false;
  return before === document.getElementById('hud')!.innerHTML;
}
window.hudCost = { trial, equivalent };
declare global { interface Window { hudCost: { trial: typeof trial; equivalent: typeof equivalent } } }
