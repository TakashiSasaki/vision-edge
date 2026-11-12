import React, { useState, useEffect, useSyncExternalStore, useCallback } from 'react';
import {
  X,
  Play,
  Copy,
  Check,
  Trash2,
  AlertTriangle,
  AlertCircle,
  Info,
  ChevronDown,
  ChevronRight,
  Cpu,
  Zap,
} from 'lucide-react';
import {
  diagnosticLogger,
  type SelfTestResult,
} from '../services/diagnosticLogger';
import {
  MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
  fetchModelBuffer,
  verifyMirageExperimentalOnnxBytes,
} from '../services/mirageExperimentalOnnxService';
import { MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL } from '../services/modelAssetResolver';
import * as ort from 'onnxruntime-web';

interface DiagnosticsModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeModel?: string;
  activeVariant?: string;
  onRetryModel?: () => void;
}

export function DiagnosticsModal({
  isOpen,
  onClose,
  activeModel,
  activeVariant,
  onRetryModel,
}: DiagnosticsModalProps) {
  const logs = useSyncExternalStore(
    diagnosticLogger.subscribe,
    diagnosticLogger.getLogs,
    diagnosticLogger.getLogs
  );

  const [filterLevel, setFilterLevel] = useState<'all' | 'errors'>('all');
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selfTestRunning, setSelfTestRunning] = useState(false);
  const [selfTestResults, setSelfTestResults] = useState<SelfTestResult[]>([]);

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleCopyReport = useCallback(async () => {
    try {
      const report = await diagnosticLogger.generateReport({
        activeModel: activeModel
          ? {
              model: activeModel,
              variantId: activeVariant ?? 'default',
              status: 'diagnostics-inspection',
            }
          : undefined,
        selfTestResults:
          selfTestResults.length > 0 ? selfTestResults : undefined,
      });
      await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Failed to copy diagnostic report', err);
    }
  }, [activeModel, activeVariant, selfTestResults]);

  const runSelfTest = useCallback(async () => {
    if (selfTestRunning) return;
    setSelfTestRunning(true);

    const initialSteps: SelfTestResult[] = [
      {
        step: 'server-status',
        name: '1. サーバー状態・キャッシュAPI確認 (/api/model-cache/status)',
        status: 'pending',
      },
      {
        step: 'asset-fetch',
        name: '2. モデルファイル配信通信確認 (cact-dequantized-reference-order.onnx)',
        status: 'pending',
      },
      {
        step: 'sha256-verify',
        name: '3. WebCrypto SHA-256 チェックサム整合性検証',
        status: 'pending',
      },
      {
        step: 'ort-wasm',
        name: '4. ONNX Runtime WASM セッション生成 & 基準推論テスト',
        status: 'pending',
      },
      {
        step: 'webgpu-adapter',
        name: '5. WebGPU アダプター取得テスト (navigator.gpu)',
        status: 'pending',
      },
      {
        step: 'ort-webgpu',
        name: '6. ONNX Runtime WebGPU セッション生成テスト (WASMフォールバック禁止)',
        status: 'pending',
      },
      {
        step: 'ort-webgpu-inference',
        name: '7. WebGPU 実推論テスト (非ゼロ決定論入力)',
        status: 'pending',
      },
      {
        step: 'ort-webgpu-parity',
        name: '8. WebGPU vs WASM 出力数値一致検証 (atol=rtol=2e-4)',
        status: 'pending',
      },
    ];

    setSelfTestResults([...initialSteps]);

    const updateStep = (index: number, patch: Partial<SelfTestResult>) => {
      setSelfTestResults((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], ...patch };
        return next;
      });
    };

    let downloadedBuffer: ArrayBuffer | null = null;
    const deterministicInput = new Float32Array(1 * 3 * 512 * 512);
    for (let i = 0; i < deterministicInput.length; i++) {
      deterministicInput[i] = Math.fround(((i % 251) - 125) / 127.5);
    }
    let wasmReferenceHeads: {
      detections: Float32Array;
      prototypes: Float32Array;
      coefficients: Float32Array;
    } | null = null;

    // Step 1: Server Status
    try {
      updateStep(0, { status: 'running' });
      const t0 = performance.now();
      const res = await fetch('/api/model-cache/status');
      const ms = Math.round(performance.now() - t0);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} ${res.statusText}`);
      }
      const data = await res.json();
      updateStep(0, {
        status: 'passed',
        durationMs: ms,
        message: `サーバー応答正常 (カタログ数: ${data.catalogCount ?? '不明'}, WASM: MP=${data.wasm?.mediapipe?.available}, ORT=${data.wasm?.ort?.available})`,
        details: { data },
      });
    } catch (err) {
      updateStep(0, {
        status: 'failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }

    // Step 2: Asset Fetch
    try {
      updateStep(1, { status: 'running' });
      const t0 = performance.now();
      downloadedBuffer = await fetchModelBuffer(
        MIRAGE_EXPERIMENTAL_ONNX_LOCAL_URL,
        {
          expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
          onChunk: (loaded, total) => {
            updateStep(1, {
              message: `受信中... ${loaded.toLocaleString()} / ${total.toLocaleString()} bytes (${Math.round((loaded / total) * 100)}%)`,
            });
          },
        }
      );
      const totalMs = Math.round(performance.now() - t0);
      updateStep(1, {
        status: 'passed',
        durationMs: totalMs,
        message: `受信完了: ${downloadedBuffer.byteLength.toLocaleString()} bytes (${totalMs}ms 完了)`,
        details: {
          byteLength: downloadedBuffer.byteLength,
          expectedSize: MIRAGE_EXPERIMENTAL_ONNX_EXPECTED_SIZE,
        },
      });
    } catch (err) {
      updateStep(1, {
        status: 'failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }

    // Step 3: SHA-256 Verification
    if (downloadedBuffer) {
      try {
        updateStep(2, { status: 'running' });
        const t0 = performance.now();
        const verified = await verifyMirageExperimentalOnnxBytes(
          downloadedBuffer
        );
        const ms = Math.round(performance.now() - t0);
        updateStep(2, {
          status: 'passed',
          durationMs: ms,
          message: `SHA-256 完全一致: ${verified.sha256.substring(0, 16)}... (${ms}ms)`,
          details: { verified },
        });
      } catch (err) {
        updateStep(2, {
          status: 'failed',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    } else {
      updateStep(2, {
        status: 'skipped',
        message: 'モデルデータ未取得のためスキップ',
      });
    }

    // Step 4: ORT WASM Session + Reference Forward Pass
    if (downloadedBuffer) {
      try {
        updateStep(3, { status: 'running' });
        const t0 = performance.now();
        const session = await ort.InferenceSession.create(
          new Uint8Array(downloadedBuffer),
          {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            logSeverityLevel: 3,
          }
        );
        const createMs = Math.round(performance.now() - t0);
        const inNames = session.inputNames;
        const outNames = session.outputNames;
        const runT0 = performance.now();
        const imgTensor = new ort.Tensor(
          'float32',
          deterministicInput,
          [1, 3, 512, 512]
        );
        const hwTensor = new ort.Tensor(
          'int64',
          new BigInt64Array([BigInt(512), BigInt(512)]),
          [2]
        );
        try {
          const out = await session.run(
            { image: imgTensor, valid_hw: hwTensor },
            ['detections', 'prototypes', 'coefficients']
          );
          const runMs = Math.round(performance.now() - runT0);
          wasmReferenceHeads = {
            detections: new Float32Array(out.detections.data as Float32Array),
            prototypes: new Float32Array(out.prototypes.data as Float32Array),
            coefficients: new Float32Array(out.coefficients.data as Float32Array),
          };
          for (const val of Object.values(out)) {
            (val as { dispose?: () => void })?.dispose?.();
          }
          updateStep(3, {
            status: 'passed',
            durationMs: createMs + runMs,
            message: `WASM 初期化 (${createMs}ms) & 推論成功 (${runMs}ms, 出力数: ${outNames.length})`,
            details: {
              createMs,
              runMs,
              inputNames: inNames,
              outputCount: outNames.length,
            },
          });
        } finally {
          imgTensor.dispose?.();
          hwTensor.dispose?.();
          await session.release();
        }
      } catch (err) {
        updateStep(3, {
          status: 'failed',
          message: err instanceof Error ? err.message : String(err),
        });
      }
    } else {
      updateStep(3, {
        status: 'skipped',
        message: 'モデルデータ未取得のためスキップ',
      });
    }

    // Step 5: WebGPU adapter capability
    let adapterAvailable = false;
    try {
      updateStep(4, { status: 'running' });
      if (
        typeof navigator === 'undefined' ||
        !('gpu' in navigator) ||
        !navigator.gpu
      ) {
        updateStep(4, {
          status: 'skipped',
          message:
            'ブラウザに WebGPU API (navigator.gpu) がありません',
        });
      } else {
        const t0 = performance.now();
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error('WebGPU adapter request returned null');
        }
        adapterAvailable = true;
        const typedAdapter = adapter as GPUAdapter & {
          info?: GPUAdapterInfo;
          isFallbackAdapter?: boolean;
          requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
        };
        const rawInfo =
          typeof typedAdapter.requestAdapterInfo === 'function'
            ? await typedAdapter.requestAdapterInfo()
            : typedAdapter.info;
        const info = rawInfo
          ? {
              vendor: rawInfo.vendor || '',
              architecture: rawInfo.architecture || '',
              device: rawInfo.device || '',
              description: rawInfo.description || '',
              isFallbackAdapter: Boolean(typedAdapter.isFallbackAdapter),
            }
          : undefined;
        const ms = Math.round(performance.now() - t0);
        updateStep(4, {
          status: 'passed',
          durationMs: ms,
          message: `WebGPU アダプター検出: ${info?.description || info?.vendor || 'Generic WebGPU'} (${ms}ms)`,
          details: { info },
        });
      }
    } catch (err) {
      updateStep(4, {
        status: 'failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }

    // Steps 6, 7, 8: WebGPU Session Creation (no fallback), WebGPU Inference, and WASM Parity
    if (!downloadedBuffer || !adapterAvailable) {
      const skipMsg = !downloadedBuffer
        ? 'モデルデータ未取得のためスキップ'
        : 'WebGPU アダプター未検出のためスキップ';
      updateStep(5, { status: 'skipped', message: skipMsg });
      updateStep(6, { status: 'skipped', message: skipMsg });
      updateStep(7, { status: 'skipped', message: skipMsg });
    } else {
      let webgpuSession: ort.InferenceSession | null = null;
      try {
        updateStep(5, { status: 'running' });
        const t0 = performance.now();
        webgpuSession = await ort.InferenceSession.create(
          new Uint8Array(downloadedBuffer),
          {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
            logSeverityLevel: 3,
          }
        );
        const createMs = Math.round(performance.now() - t0);
        updateStep(5, {
          status: 'passed',
          durationMs: createMs,
          message: `WebGPU セッション生成成功 (${createMs}ms · executionProviders: ['webgpu'])`,
          details: {
            createMs,
            executionProviders: ['webgpu'],
            inputNames: webgpuSession.inputNames,
            outputCount: webgpuSession.outputNames.length,
          },
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        updateStep(5, {
          status: 'failed',
          message: `WebGPU セッション生成失敗: ${msg}`,
        });
        updateStep(6, {
          status: 'skipped',
          message: 'WebGPU セッション生成失敗のためスキップ',
        });
        updateStep(7, {
          status: 'skipped',
          message: 'WebGPU セッション生成失敗のためスキップ',
        });
      }

      if (webgpuSession) {
        let webgpuHeads: {
          detections: Float32Array;
          prototypes: Float32Array;
          coefficients: Float32Array;
        } | null = null;
        try {
          updateStep(6, { status: 'running' });
          const runT0 = performance.now();
          const imgTensor = new ort.Tensor(
            'float32',
            deterministicInput,
            [1, 3, 512, 512]
          );
          const hwTensor = new ort.Tensor(
            'int64',
            new BigInt64Array([BigInt(512), BigInt(512)]),
            [2]
          );
          try {
            const out = await webgpuSession.run(
              { image: imgTensor, valid_hw: hwTensor },
              ['detections', 'prototypes', 'coefficients']
            );
            const runMs = Math.round(performance.now() - runT0);
            webgpuHeads = {
              detections: new Float32Array(out.detections.data as Float32Array),
              prototypes: new Float32Array(out.prototypes.data as Float32Array),
              coefficients: new Float32Array(
                out.coefficients.data as Float32Array
              ),
            };
            for (const val of Object.values(out)) {
              (val as { dispose?: () => void })?.dispose?.();
            }
            updateStep(6, {
              status: 'passed',
              durationMs: runMs,
              message: `WebGPU 推論成功 (${runMs}ms · 3ヘッド出力完了)`,
              details: {
                runMs,
                outputLengths: [
                  webgpuHeads.detections.length,
                  webgpuHeads.prototypes.length,
                  webgpuHeads.coefficients.length,
                ],
              },
            });
          } finally {
            imgTensor.dispose?.();
            hwTensor.dispose?.();
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          updateStep(6, {
            status: 'failed',
            message: `WebGPU 推論失敗: ${msg}`,
          });
          updateStep(7, {
            status: 'skipped',
            message: 'WebGPU 推論失敗のためスキップ',
          });
        } finally {
          await webgpuSession.release();
        }

        if (webgpuHeads && wasmReferenceHeads) {
          updateStep(7, { status: 'running' });
          const t0 = performance.now();
          const compareHead = (a: Float32Array, b: Float32Array) => {
            let maxAbsDiff = 0;
            let sumAbsDiff = 0;
            let violations = 0;
            for (let i = 0; i < a.length; i++) {
              const diff = Math.abs(a[i] - b[i]);
              sumAbsDiff += diff;
              if (diff > maxAbsDiff) maxAbsDiff = diff;
              if (diff > 2e-4 + 2e-4 * Math.abs(b[i])) violations++;
            }
            return {
              maxAbsDiff,
              meanAbsDiff: sumAbsDiff / a.length,
              violationsAt2e4: violations,
              allcloseAt2e4: violations === 0,
            };
          };
          const detDiff = compareHead(
            webgpuHeads.detections,
            wasmReferenceHeads.detections
          );
          const protoDiff = compareHead(
            webgpuHeads.prototypes,
            wasmReferenceHeads.prototypes
          );
          const coeffDiff = compareHead(
            webgpuHeads.coefficients,
            wasmReferenceHeads.coefficients
          );
          const allPass =
            detDiff.allcloseAt2e4 &&
            protoDiff.allcloseAt2e4 &&
            coeffDiff.allcloseAt2e4;
          const ms = Math.round(performance.now() - t0);
          updateStep(7, {
            status: allPass ? 'passed' : 'failed',
            durationMs: ms,
            message: allPass
              ? `全3ヘッド一致 (maxAbsDiff: det=${detDiff.maxAbsDiff.toExponential(2)}, proto=${protoDiff.maxAbsDiff.toExponential(2)}, coeff=${coeffDiff.maxAbsDiff.toExponential(2)})`
              : `許容差超過 (違反数: det=${detDiff.violationsAt2e4}, proto=${protoDiff.violationsAt2e4}, coeff=${coeffDiff.violationsAt2e4})`,
            details: {
              detections: detDiff,
              prototypes: protoDiff,
              coefficients: coeffDiff,
            },
          });
        } else if (webgpuHeads && !wasmReferenceHeads) {
          updateStep(7, {
            status: 'skipped',
            message: 'WASM 基準出力が未取得のため比較スキップ',
          });
        }
      }
    }

    setSelfTestRunning(false);
  }, [selfTestRunning]);

  if (!isOpen) return null;

  const filteredLogs =
    filterLevel === 'all'
      ? logs
      : logs.filter((l) => l.level === 'error' || l.level === 'warn');

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="diagnostics-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150"
    >
      <div className="relative w-full max-w-4xl max-h-[92vh] flex flex-col rounded-xl border border-slate-800 bg-[#0c1222] text-slate-100 shadow-2xl overflow-hidden font-sans">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-800 bg-[#090e1a]">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-cyan-950/60 border border-cyan-800/40 text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2
                id="diagnostics-modal-title"
                className="text-base font-semibold text-slate-100 tracking-tight"
              >
                モデル診断・ログインスペクター
              </h2>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span>
                  対象: {activeModel ?? '未選択'} ({activeVariant ?? 'default'})
                </span>
                <span aria-hidden="true">·</span>
                <span>全 {logs.length} 件記録</span>
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="閉じる"
            className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Action Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-2.5 px-5 py-2.5 border-b border-slate-800/80 bg-slate-900/50">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={runSelfTest}
              disabled={selfTestRunning}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-white bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 transition-colors shadow-sm"
            >
              <Play className="w-3.5 h-3.5" />
              {selfTestRunning
                ? 'セルフテスト実行中...'
                : '段階別セルフテスト実行'}
            </button>
            {onRetryModel && (
              <button
                type="button"
                onClick={() => {
                  onRetryModel();
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg text-slate-200 bg-slate-800 hover:bg-slate-700 transition-colors"
              >
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                モデル初期化を再試行
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Filter */}
            <div className="flex items-center p-0.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
              <button
                type="button"
                onClick={() => setFilterLevel('all')}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  filterLevel === 'all'
                    ? 'bg-slate-800 text-slate-100 font-medium'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                すべて ({logs.length})
              </button>
              <button
                type="button"
                onClick={() => setFilterLevel('errors')}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  filterLevel === 'errors'
                    ? 'bg-rose-950/80 text-rose-300 font-medium'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                警告・エラーのみ (
                {logs.filter((l) => l.level !== 'info').length})
              </button>
            </div>

            <button
              type="button"
              onClick={handleCopyReport}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-lg text-slate-300 bg-slate-800 hover:bg-slate-700 transition-colors"
              title="環境情報・ログ・テスト結果をJSONでコピー"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              <span>{copied ? 'コピー完了' : '診断JSONをコピー'}</span>
            </button>

            <button
              type="button"
              onClick={() => diagnosticLogger.clear()}
              className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-slate-800/80 rounded-lg transition-colors"
              title="ログ消去"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 min-h-0 overflow-y-auto p-5 space-y-4">
          {/* Section: Self Test Results if any */}
          {selfTestResults.length > 0 && (
            <div className="rounded-lg border border-slate-800 bg-[#090e1a] p-3.5 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  段階別セルフテスト検証結果
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {selfTestResults.filter((r) => r.status === 'passed').length}
                  /{selfTestResults.length} 合格
                </span>
              </div>
              <div className="space-y-1.5">
                {selfTestResults.map((result, idx) => (
                  <div
                    key={result.step || idx}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 p-2 rounded bg-slate-900/60 border border-slate-800/60 text-xs"
                  >
                    <div className="flex items-center gap-2">
                      {result.status === 'passed' && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                          合格
                        </span>
                      )}
                      {result.status === 'failed' && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-rose-950 text-rose-400 border border-rose-800/60">
                          失敗
                        </span>
                      )}
                      {result.status === 'running' && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/60 animate-pulse">
                          実行中
                        </span>
                      )}
                      {result.status === 'skipped' && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-400">
                          スキップ
                        </span>
                      )}
                      {result.status === 'pending' && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-900 text-slate-500">
                          待機
                        </span>
                      )}
                      <span className="text-slate-200 font-medium">
                        {result.name}
                      </span>
                    </div>
                    {result.message && (
                      <span
                        className={`text-right truncate max-w-md ${
                          result.status === 'failed'
                            ? 'text-rose-300 font-mono'
                            : 'text-slate-400'
                        }`}
                      >
                        {result.message}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Section: Environment Overview */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
              <span className="text-slate-400 block mb-0.5">WASM スレッド</span>
              <span className="font-mono text-slate-200">
                {typeof crossOriginIsolated !== 'undefined' &&
                crossOriginIsolated
                  ? '分離有効 (Multi-thread)'
                  : '単一 (Single-thread)'}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
              <span className="text-slate-400 block mb-0.5">論理コア数</span>
              <span className="font-mono text-slate-200">
                {typeof navigator !== 'undefined'
                  ? navigator.hardwareConcurrency || 1
                  : '—'}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
              <span className="text-slate-400 block mb-0.5">WebGPU API</span>
              <span className="font-mono text-slate-200">
                {typeof navigator !== 'undefined' &&
                'gpu' in navigator &&
                navigator.gpu
                  ? '利用可能'
                  : '非対応 (WASM推奨)'}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
              <span className="text-slate-400 block mb-0.5">ONNX ランタイム</span>
              <span className="font-mono text-slate-200">
                ort-web v1.30.0
              </span>
            </div>
          </div>

          {/* Section: Live Diagnostic Log Stream */}
          <div className="rounded-lg border border-slate-800 bg-[#090e1a] overflow-hidden">
            <div className="px-3.5 py-2.5 border-b border-slate-800/80 bg-slate-900/40 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                実行タイムライン & イベントログ ({filteredLogs.length} 件)
              </span>
              <span className="text-[11px] text-slate-500">
                クリックで詳細・スタックトレースを展開
              </span>
            </div>

            {filteredLogs.length === 0 ? (
              <div className="p-8 text-center text-xs text-slate-500">
                ログはまだ記録されていません。モデルを選択するか「段階別セルフテスト実行」をクリックしてください。
              </div>
            ) : (
              <div className="divide-y divide-slate-800/50 max-h-[380px] overflow-y-auto font-mono text-xs">
                {filteredLogs.map((entry) => {
                  const isExpanded = expandedLogId === entry.id;
                  const hasDetails =
                    entry.details ||
                    (entry.error && entry.error.stack) ||
                    (entry.error && entry.error.message);

                  let levelColor = 'text-slate-400 bg-slate-800/40';
                  let levelIcon = <Info className="w-3.5 h-3.5 text-cyan-400" />;
                  if (entry.level === 'warn') {
                    levelColor =
                      'text-amber-300 bg-amber-950/40 border border-amber-800/40';
                    levelIcon = (
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                    );
                  } else if (entry.level === 'error') {
                    levelColor =
                      'text-rose-300 bg-rose-950/40 border border-rose-800/40';
                    levelIcon = (
                      <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                    );
                  }

                  return (
                    <div
                      key={entry.id}
                      className="hover:bg-slate-900/50 transition-colors"
                    >
                      <div
                        onClick={() =>
                          hasDetails &&
                          setExpandedLogId(isExpanded ? null : entry.id)
                        }
                        className={`flex items-start gap-2.5 px-3.5 py-2 cursor-pointer ${
                          hasDetails ? 'hover:text-slate-100' : ''
                        }`}
                      >
                        <span className="text-slate-500 tabular-nums shrink-0 pt-0.5">
                          +{entry.elapsedMs}ms
                        </span>

                        <span
                          className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider shrink-0 ${levelColor}`}
                        >
                          {levelIcon}
                          {entry.level}
                        </span>

                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700/60 shrink-0">
                          {entry.stage}
                        </span>

                        <div className="flex-1 min-w-0">
                          <p
                            className={`break-words ${
                              entry.level === 'error'
                                ? 'text-rose-300 font-semibold'
                                : entry.level === 'warn'
                                  ? 'text-amber-300'
                                  : 'text-slate-300'
                            }`}
                          >
                            {entry.message}
                          </p>
                        </div>

                        {hasDetails && (
                          <div className="text-slate-500 shrink-0 pt-0.5">
                            {isExpanded ? (
                              <ChevronDown className="w-3.5 h-3.5" />
                            ) : (
                              <ChevronRight className="w-3.5 h-3.5" />
                            )}
                          </div>
                        )}
                      </div>

                      {/* Expandable JSON / Stack Trace */}
                      {isExpanded && (
                        <div className="px-4 py-3 bg-black/40 border-t border-slate-800 text-[11px] space-y-2">
                          {entry.error && (
                            <div>
                              <span className="text-rose-400 font-semibold block mb-1">
                                {entry.error.name}: {entry.error.message}
                              </span>
                              {entry.error.stack && (
                                <pre className="p-2 rounded bg-slate-950 text-rose-200/90 overflow-x-auto whitespace-pre font-mono text-[10px] leading-relaxed border border-rose-950">
                                  {entry.error.stack}
                                </pre>
                              )}
                            </div>
                          )}

                          {entry.details && (
                            <div>
                              <span className="text-slate-400 block mb-1">
                                診断詳細ペイロード:
                              </span>
                              <pre className="p-2 rounded bg-slate-950 text-slate-300 overflow-x-auto whitespace-pre font-mono text-[10px] leading-relaxed border border-slate-800">
                                {JSON.stringify(entry.details, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-slate-800 bg-[#090e1a]">
          <span className="text-xs text-slate-400">
            診断ログはブラウザ内リングバッファ (最新100件)
            およびサーバーエンドポイントに記録されます
          </span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium rounded-lg text-slate-200 bg-slate-800 hover:bg-slate-700 transition-colors"
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
}
