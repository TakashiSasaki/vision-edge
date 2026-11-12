export type DiagnosticLogLevel = 'info' | 'warn' | 'error';

export type DiagnosticStage =
  | 'resolve'
  | 'fetch'
  | 'verify'
  | 'ort-session'
  | 'inference'
  | 'self-test'
  | 'system'
  | 'wasm'
  | 'task';

export interface DiagnosticLogEntry {
  id: string;
  timestamp: string; // ISO-8601
  elapsedMs: number;
  level: DiagnosticLogLevel;
  modelId: string;
  variantId: string;
  stage: DiagnosticStage;
  message: string;
  details?: Record<string, unknown>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
}

export interface SelfTestResult {
  step:
    | 'server-status'
    | 'asset-fetch'
    | 'sha256-verify'
    | 'ort-wasm'
    | 'webgpu-adapter'
    | 'ort-webgpu'
    | 'ort-webgpu-inference'
    | 'ort-webgpu-parity';
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped';
  durationMs?: number;
  message?: string;
  details?: Record<string, unknown>;
}

export interface FullDiagnosticReport {
  generatedAt: string;
  environment: {
    userAgent: string;
    crossOriginIsolated: boolean;
    hardwareConcurrency: number;
    deviceMemory?: number;
    webGpuAvailable: boolean;
    webGpuAdapterInfo?: Record<string, string>;
  };
  activeModel?: {
    model: string;
    variantId: string;
    status: string;
    errorCause?: string;
  };
  selfTestResults?: SelfTestResult[];
  logs: DiagnosticLogEntry[];
}

const MAX_LOG_ENTRIES = 100;

class DiagnosticLogger {
  private serverTelemetryEnabled = false;
  private logs: readonly DiagnosticLogEntry[] = [];
  private readonly listeners = new Set<() => void>();
  private readonly startTime = Date.now();

  public enableServerTelemetry = (enabled = true): void => {
    this.serverTelemetryEnabled = enabled;
  };

  public isServerTelemetryEnabled = (): boolean => {
    return this.serverTelemetryEnabled;
  };

  public log = (
    level: DiagnosticLogLevel,
    stage: DiagnosticStage,
    modelId: string,
    variantId: string,
    message: string,
    options?: {
      details?: Record<string, unknown>;
      error?: unknown;
      sendToServer?: boolean;
    }
  ): DiagnosticLogEntry => {
    const elapsedMs = Date.now() - this.startTime;

    let errorObj: DiagnosticLogEntry['error'];
    if (options?.error) {
      if (options.error instanceof Error) {
        errorObj = {
          name: options.error.name,
          message: options.error.message,
          stack: options.error.stack,
        };
      } else {
        errorObj = {
          name: 'Error',
          message: String(options.error),
        };
      }
    }

    const entry: DiagnosticLogEntry = {
      id: `diag-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      timestamp: new Date().toISOString(),
      elapsedMs,
      level,
      modelId,
      variantId,
      stage,
      message,
      details: options?.details,
      error: errorObj,
    };

    const nextLogs = [...this.logs, entry];
    this.logs = nextLogs.length > MAX_LOG_ENTRIES ? nextLogs.slice(-MAX_LOG_ENTRIES) : nextLogs;

    this.notify();

    // Console mirror
    const prefix = `[Diagnostics][${stage}][${modelId}:${variantId}]`;
    if (level === 'error') {
      console.error(prefix, message, options?.details ?? '', options?.error ?? '');
    } else if (level === 'warn') {
      console.warn(prefix, message, options?.details ?? '', options?.error ?? '');
    } else {
      console.log(prefix, message, options?.details ?? '');
    }

    // Server-side transmission if explicitly requested and telemetry enabled
    if (options?.sendToServer && this.serverTelemetryEnabled) {
      this.sendToServer(entry).catch(() => {
        // Silently ignore server logging network errors to prevent loops
      });
    }

    return entry;
  };

  public getLogs = (): readonly DiagnosticLogEntry[] => {
    return this.logs;
  };

  public getLatestError = (): DiagnosticLogEntry | undefined => {
    for (let i = this.logs.length - 1; i >= 0; i--) {
      if (this.logs[i].level === 'error') return this.logs[i];
    }
    return undefined;
  };

  public clear = (): void => {
    this.logs = [];
    this.notify();
  };

  public subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private notify(): void {
    for (const listener of this.listeners) {
      try {
        listener();
      } catch (err) {
        console.warn('DiagnosticLogger listener error', err);
      }
    }
  }

  public sendToServer = async (entry: DiagnosticLogEntry): Promise<void> => {
    if (typeof fetch !== 'function') return;
    if (
      typeof window === 'undefined' ||
      !(
        window as unknown as { __CONTAINER_MODEL_CACHE_ENABLED__?: boolean }
      ).__CONTAINER_MODEL_CACHE_ENABLED__
    ) {
      return;
    }
    try {
      await fetch('/api/model-cache/diagnostics/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry),
        keepalive: true,
      });
    } catch {
      // Non-blocking telemetry
    }
  }

  public generateReport = async (extra?: {
    activeModel?: { model: string; variantId: string; status: string; errorCause?: string };
    selfTestResults?: SelfTestResult[];
  }): Promise<FullDiagnosticReport> => {
    let webGpuAvailable = false;
    let webGpuAdapterInfo: Record<string, string> | undefined;

    if (typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu) {
      webGpuAvailable = true;
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const typedAdapter = adapter as GPUAdapter & {
            info?: GPUAdapterInfo;
            isFallbackAdapter?: boolean;
            requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
          };
          const info =
            typeof typedAdapter.requestAdapterInfo === 'function'
              ? await typedAdapter.requestAdapterInfo()
              : typedAdapter.info;
          if (info) {
            webGpuAdapterInfo = {
              vendor: info.vendor || '',
              architecture: info.architecture || '',
              device: info.device || '',
              description: info.description || '',
              isFallbackAdapter: String(Boolean(typedAdapter.isFallbackAdapter)),
            };
          }
        }
      } catch {
        // WebGPU adapter info probe failed
      }
    }

    return {
      generatedAt: new Date().toISOString(),
      environment: {
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
        crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : false,
        hardwareConcurrency: typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 1 : 1,
        deviceMemory: typeof navigator !== 'undefined' ? (navigator as { deviceMemory?: number }).deviceMemory : undefined,
        webGpuAvailable,
        webGpuAdapterInfo,
      },
      activeModel: extra?.activeModel,
      selfTestResults: extra?.selfTestResults,
      logs: [...this.logs],
    };
  }
}

export const diagnosticLogger = new DiagnosticLogger();
