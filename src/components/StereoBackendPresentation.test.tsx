import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ControlPanel } from './ControlPanel';
import type { OnnxBackendInfo, StereoBackendPreference } from '../types/vision';
vi.mock('../services/speechService', () => ({ speechService: { getVoices: () => [], speak: vi.fn() } }));

function show(variant: string, preference: StereoBackendPreference, backend: OnnxBackendInfo) {
  render(<ControlPanel activeModel="stereo-depth" activeVariant={variant}
    modelState={{ status: 'ready', model: 'stereo-depth', variantId: variant, requestId: 1, backend }}
    inferenceParams={{ backendPreference: preference }}
    renderOptions={{ showBoxes: true, showMasks: true, showLabels: true, showConfidence: true, maskOpacity: .35, confidenceThreshold: .35 }}
    onChangeRenderOptions={() => {}} ttsConfig={{ enabled: false, autoReadout: false, cooldownSec: 2, rate: 1, pitch: 1, volume: 1, voiceURI: '' }}
    onChangeTTSConfig={() => {}} detectedObjects={[]} hasCurrentFrame={false} onSelectObject={() => {}} isSpeaking={false} />);
  fireEvent.click(screen.getByRole('button', { name: 'モデル調整' }));
}
const gpu: OnnxBackendInfo = {
  runtime: 'onnxruntime-web', requestedProvider: 'webgpu', configuredProviders: ['webgpu', 'wasm'],
  actualBackend: 'unmeasured', fallbackReason: null,
};
it('distinguishes GPU preference, graph preparation and unmeasured execution', () => {
  show('lightstereo-s', 'webgpu', { ...gpu, configurationNote: 'Concat分割でWebGPU向けにモデルを準備' });
  expect(screen.getByRole('button', { name: 'WebGPU優先' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.queryByRole('button', { name: /強制/ })).not.toBeInTheDocument();
  expect(screen.getByText(/GPU関連エラー時はCPUへ切り替えます/)).toBeInTheDocument();
  expect(screen.getByText(/各演算の実行先・物理GPUの使用状況は未計測/)).toBeInTheDocument();
  expect(screen.getByText(/設定の説明: Concat分割/)).toBeInTheDocument();
  expect(screen.queryByText(/フォールバック理由:/)).not.toBeInTheDocument();
});
it('explains CPU selection in auto mode for MobileStereoNet', () => {
  show('mobilestereonet-2d', 'auto', { ...gpu, requestedProvider: 'auto', configuredProviders: ['wasm'] });
  expect(screen.getByText('このモデルは互換性のため、自動設定ではWASM（CPU）を使用します。')).toBeInTheDocument();
});
it('shows an actual fallback reason separately from preparation', () => {
  show('lightstereo-s', 'webgpu', {
    ...gpu, configuredProviders: ['wasm'], configurationNote: 'Concat分割済み',
    fallbackReason: 'WebGPU kernel runtime failure; fell back to WASM',
  });
  expect(screen.getByText(/設定の説明: Concat分割済み/)).toBeInTheDocument();
  expect(screen.getByText(/フォールバック理由: WebGPU kernel runtime failure/)).toBeInTheDocument();
});
