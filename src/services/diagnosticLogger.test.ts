import { describe, it, expect, vi, beforeEach } from 'vitest';
import { diagnosticLogger } from './diagnosticLogger';

describe('diagnosticLogger', () => {
  beforeEach(() => {
    diagnosticLogger.clear();
  });

  it('records log entries and notifies subscribers', () => {
    const subscriber = vi.fn();
    const unsub = diagnosticLogger.subscribe(subscriber);

    diagnosticLogger.log('info', 'resolve', 'mirage', 'mirage-onnx-exp-wasm', 'Asset resolved', {
      details: { url: '/api/model-cache/mirage/...' },
    });

    expect(subscriber).toHaveBeenCalledTimes(1);
    const logs = diagnosticLogger.getLogs();
    expect(logs).toHaveLength(1);
    expect(logs[0].message).toBe('Asset resolved');
    expect(logs[0].stage).toBe('resolve');
    expect(logs[0].modelId).toBe('mirage');

    unsub();
  });

  it('captures errors with stack traces and returns getLatestError', () => {
    diagnosticLogger.log('info', 'fetch', 'mirage', 'mirage-onnx-exp-wasm', 'Fetching...');
    const fakeError = new Error('HTTP 503 Service Unavailable');
    diagnosticLogger.log('error', 'fetch', 'mirage', 'mirage-onnx-exp-wasm', 'Fetch failed', {
      error: fakeError,
      details: { status: 503 },
    });

    const latest = diagnosticLogger.getLatestError();
    expect(latest).toBeDefined();
    expect(latest?.level).toBe('error');
    expect(latest?.error?.message).toBe('HTTP 503 Service Unavailable');
    expect(latest?.error?.name).toBe('Error');
    expect(latest?.details).toEqual({ status: 503 });
  });

  it('generates structured diagnostic reports', async () => {
    diagnosticLogger.log('warn', 'ort-session', 'mirage', 'mirage-onnx-exp-webgpu', 'WebGPU fallback to WASM');

    const report = await diagnosticLogger.generateReport({
      activeModel: {
        model: 'mirage',
        variantId: 'mirage-onnx-exp-webgpu',
        status: 'ready',
      },
    });

    expect(report.generatedAt).toBeDefined();
    expect(report.environment).toBeDefined();
    expect(report.activeModel?.variantId).toBe('mirage-onnx-exp-webgpu');
    expect(report.logs).toHaveLength(1);
    expect(report.logs[0].level).toBe('warn');
  });

  it('allows unbound function extraction of getLogs and subscribe for React useSyncExternalStore', () => {
    const { getLogs, subscribe } = diagnosticLogger;
    expect(() => getLogs()).not.toThrow();
    expect(getLogs()).toEqual([]);

    const listener = vi.fn();
    let unsub!: () => void;
    expect(() => {
      unsub = subscribe(listener);
    }).not.toThrow();

    diagnosticLogger.log('info', 'system', 'test', 'var', 'Unbound test');
    expect(listener).toHaveBeenCalledTimes(1);
    expect(getLogs()).toHaveLength(1);
    unsub();
  });
});
