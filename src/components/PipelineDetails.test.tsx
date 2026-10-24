import { cleanup, render } from '@testing-library/react';
import { afterEach, expect, it } from 'vitest';
import { PipelineDetails } from './PipelineDetails';
import { readPipelineSnapshot } from '../services/pipelineSnapshotView';
import { PipelineMetrics } from '../services/pipelineMetrics';
afterEach(cleanup);
function source() {
  const metrics = new PipelineMetrics(() => 0);
  metrics.record({ stage: 'capture', outcome: 'success', at: 0, ms: 0 });
  metrics.record({ stage: 'sdk', outcome: 'success', at: 0, ms: 7 });
  return metrics.snapshot()!;
}
function cells(container: HTMLElement, stage: string) {
  const row = [...container.querySelectorAll('table[aria-label="success interval statistics"] tbody tr')].find(row => row.querySelector('th')!.textContent === stage)!;
  return [...row.querySelectorAll('td')].map(cell => cell.textContent);
}
it.each([null, undefined, { stages: null }, { stages: {} }, { stages: { capture: null } }, { stages: { capture: { success: {} } } }])('renders malformed or unavailable snapshot %j without a React exception', snapshot => {
  const { container } = render(<PipelineDetails snapshot={readPipelineSnapshot(snapshot)} />);
  expect(cells(container, 'capture')).toEqual(Array(7).fill('未計測'));
  expect(container.textContent).not.toContain('[object Object]');
});
it('renders observed zero and valid neighbors while invalid values stay unavailable, and recovers on a valid snapshot', () => {
  const valid = source(); const { container, rerender } = render(<PipelineDetails snapshot={readPipelineSnapshot(valid)} />);
  expect(cells(container, 'capture')).toEqual(['1 / 1 / 0', '0.00', '0.00', '0.00', '0.00', '0.00', '1 / 1 / 保持窓全体']);
  const invalid = source(); Object.assign(invalid.stages.sdk.success, { mean: Infinity, p95: { toFixed: () => { throw Error('bad statistic'); } } });
  rerender(<PipelineDetails snapshot={readPipelineSnapshot({ ...invalid, results: null, pendingResults: {} })} />);
  expect(cells(container, 'sdk')).toEqual(['1 / 1 / 0', '未計測', '7.00', '未計測', '7.00', '7.00', '1 / 1 / 保持窓全体']);
  expect(container.textContent).toContain('結果 outcomes (件 / 5秒): 未計測');
  expect(container.textContent).toContain('未描画保持: 未計測');
  rerender(<PipelineDetails snapshot={readPipelineSnapshot(valid)} />);
  expect(cells(container, 'sdk')[1]).toBe('7.00');
});
it('keeps truncated and missing-duration partial statistics visible', () => {
  const metrics = new PipelineMetrics(() => 0);
  for (let i = 0; i < 2050; i++) metrics.record({ stage: 'comparison', outcome: 'success', at: 0, ms: i % 2 ? 0 : null });
  const { container } = render(<PipelineDetails snapshot={readPipelineSnapshot(metrics.snapshot())} />);
  expect(cells(container, 'comparison')).toEqual(['2048 / 1024 / 1024', '0.00', '0.00', '0.00', '0.00', '0.00', '300 / 1024 / 切捨あり・部分集計']);
});
it('contains getters which throw or change after reading without leaking raw React children', () => {
  const snapshot = source(); let reads = 0;
  Object.defineProperty(snapshot, 'unclocked', { get: () => ++reads === 1 ? 0 : {} });
  const { container, rerender } = render(<PipelineDetails snapshot={readPipelineSnapshot(snapshot)} />);
  expect(container.textContent).toContain('時刻欠測 (世代累計): 0');
  expect(reads).toBe(1);
  rerender(<PipelineDetails snapshot={readPipelineSnapshot(new Proxy({}, { get() { throw Error('payload'); } }))} />);
  expect(cells(container, 'capture')[1]).toBe('未計測');
});
