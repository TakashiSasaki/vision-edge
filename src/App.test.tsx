import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from './App';
import { VisionEngine, type SegmenterTask, type VisionTaskFactory } from './services/visionEngine';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(yes => { resolve = yes; });
  return { promise, resolve };
}
function fixture() {
  const detector = { detectForVideo: vi.fn(() => ({ detections: [] })), close: vi.fn() };
  const segmenter: SegmenterTask = { segmentForVideo: vi.fn(), close: vi.fn() };
  const yoloTask = { detectForVideo: vi.fn(async () => []), close: vi.fn() };
  const segformerTask = { detectForVideo: vi.fn(async () => []), close: vi.fn() };
  const mirageTask = { detectForVideo: vi.fn(async () => []), close: vi.fn() };
  const stereoTask = { estimateForVideo: vi.fn(async () => ({}) as never), close: vi.fn() };
  const factory: VisionTaskFactory = {
    load: vi.fn(async () => ({ wasmLoaderPath: 'loader', wasmBinaryPath: 'wasm' })),
    detector: vi.fn(async () => detector), segmenter: vi.fn(async () => segmenter),
    yoloSeg: vi.fn(async () => yoloTask),
    segformer: vi.fn(async () => segformerTask),
    mirage: vi.fn(async () => mirageTask),
    stereoDepth: vi.fn(async () => stereoTask),
  };
  return { engine: new VisionEngine(factory), factory, detector, segmenter, yoloTask, segformerTask, mirageTask, stereoTask };
}
beforeEach(() => {
  window.localStorage?.clear();
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getUserMedia: vi.fn().mockRejectedValue(new DOMException('Test camera denied', 'NotAllowedError')),
  } });
});
afterEach(() => Reflect.deleteProperty(navigator, 'mediaDevices'));

it('uses the implemented choices in both UI paths and shows read-only backend settings', async () => {
  const f = fixture();
  const user = userEvent.setup();
  render(<App engine={f.engine} />);
  await screen.findByText('モデル準備完了（カメラの準備・推論成功とは別です）');
  const header = screen.getByRole('banner');
  const model = within(header).getByRole('combobox', { name: 'モデル' });
  const enabled = within(model).getAllByRole('option').filter(option => !(option as HTMLOptionElement).disabled);
  expect(enabled.map(option => (option as HTMLOptionElement).value)).toEqual(['selfie-segmenter', 'mediapipe', 'yolo-seg', 'segformer', 'mirage', 'stereo-depth', 'image-classifier']);
  expect(within(header).getAllByRole('combobox')).toHaveLength(2);
  const variantSelect = within(header).getByRole('combobox', { name: 'バリアント' });
  expect(variantSelect).toHaveValue('general');
  expect(within(header).getByLabelText('バックエンド状態')).toHaveTextContent('GPU delegate 指定・初期化済み（GPU実動作は未計測）');
  await user.selectOptions(model, 'yolo-seg');
  await waitFor(() => expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'yolo-seg', variantId: 'yolov8n-seg' }));
  expect(model).toHaveValue('yolo-seg');
  expect(variantSelect).toHaveValue('yolov8n-seg');
  await user.selectOptions(model, 'segformer');
  await waitFor(() => expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'segformer', variantId: 'b0' }));
  expect(model).toHaveValue('segformer');
  expect(variantSelect).toHaveValue('b0');
  await user.selectOptions(model, 'mirage');
  await waitFor(() => expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mirage', variantId: 'mirage-51m' }));
  expect(model).toHaveValue('mirage');
  expect(variantSelect).toHaveValue('mirage-51m');
  expect(f.factory.mirage).toHaveBeenCalledTimes(1);
  expect(f.factory.mirage).toHaveBeenCalledWith(expect.objectContaining({ variantId: 'mirage-51m' }));
  expect(within(header).getByLabelText('バックエンド状態')).toHaveTextContent('Cactus WebAssembly (CPU / Web Worker 非同期推論)');
  await user.selectOptions(model, 'mediapipe');
  await waitFor(() => expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'mediapipe', variantId: 'lite0' }));
  await user.click(within(header).getByRole('button', { name: '仕様' }));
  const dialog = screen.getByRole('dialog');
  for (const name of [/General/, /EfficientDet-Lite0/, /YOLOv8n-seg/, /SegFormer-B0/, /Mirage 51\.3M/]) {
    const button = within(dialog).getByRole('button', { name });
    expect(button).not.toBeDisabled();
  }
  expect(within(dialog).queryByText('稼働中')).not.toBeInTheDocument();
  await user.click(within(dialog).getByRole('button', { name: /General/ }));
  expect(model).toHaveValue('selfie-segmenter');
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.queryByText('現在しきい値を超える物体はありません')).not.toBeInTheDocument();
  expect(screen.getByLabelText('推論FPS')).toHaveTextContent('推論: 未計測');
  expect(document.body).not.toHaveTextContent(/\d+\s*FPS|\d+ms|稼働中/);
});

it('renders WASM failure separately from camera failure, and retries only on the explicit button', async () => {
  const f = fixture();
  vi.mocked(f.factory.load).mockRejectedValueOnce(new Error('WASM download failed'));
  const user = userEvent.setup();
  render(<App engine={f.engine} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('モデル実行環境の読み込みに失敗しました');
  expect(await screen.findByRole('heading', { name: 'カメラエラー' })).toBeInTheDocument();
  expect(screen.getByLabelText('バックエンド状態')).toHaveTextContent('バックエンド未準備');
  expect(f.factory.load).toHaveBeenCalledTimes(1);
  expect(screen.queryByText('現在しきい値を超える物体はありません')).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'モデルを再試行' }));
  await screen.findByText('モデル準備完了（カメラの準備・推論成功とは別です）');
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(f.factory.load).toHaveBeenCalledTimes(2);
  expect(f.factory.segmenter).toHaveBeenCalledTimes(1);
  expect(f.factory.detector).not.toHaveBeenCalled();
  expect(screen.getByRole('heading', { name: 'カメラエラー' })).toBeInTheDocument();
});

it('shows task creation failure and leaves camera stop available', async () => {
  const f = fixture();
  vi.mocked(f.factory.segmenter).mockRejectedValueOnce(new Error('model cannot be created'));
  vi.mocked(navigator.mediaDevices.getUserMedia).mockReturnValue(new Promise(() => {}));
  const user = userEvent.setup();
  render(<App engine={f.engine} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('モデルの生成に失敗しました');
  expect(f.engine.getState()).toMatchObject({ status: 'error', stage: 'task' });
  await user.click(within(screen.getByRole('banner')).getByRole('button', { name: '停止' }));
  expect(within(screen.getByRole('banner')).getByRole('button', { name: 'カメラ開始' })).toBeInTheDocument();
  expect(screen.getByRole('alert')).toHaveTextContent('モデルの生成に失敗しました');
});

it('keeps loading distinct from successful empty inference and from the GPU ready label', async () => {
  const f = fixture();
  const task = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValue(task.promise);
  render(<App engine={f.engine} />);
  expect(screen.getByRole('status')).toHaveTextContent('モデル読込中・推論待機');
  expect(screen.getByLabelText('バックエンド状態')).toHaveTextContent('バックエンド未準備');
  expect(screen.queryByText('現在しきい値を超える物体はありません')).not.toBeInTheDocument();
  expect(f.segmenter.segmentForVideo).not.toHaveBeenCalled();
  await act(async () => { task.resolve(f.segmenter); });
  expect(screen.getByRole('status')).toHaveTextContent('モデル準備完了');
  expect(screen.queryByText('現在しきい値を超える物体はありません')).not.toBeInTheDocument();
});

it('does not let delayed A overwrite B in the real application UI', async () => {
  const f = fixture();
  const a = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValue(a.promise);
  const user = userEvent.setup();
  render(<App engine={f.engine} />);
  await waitFor(() => expect(f.factory.segmenter).toHaveBeenCalledTimes(1));
  await user.selectOptions(screen.getByRole('combobox', { name: 'モデル' }), 'mediapipe');
  await screen.findByText('モデル準備完了（カメラの準備・推論成功とは別です）');
  const ready = f.engine.getState();
  await act(async () => { a.resolve(f.segmenter); });
  expect(f.engine.getState()).toBe(ready);
  expect(screen.getByRole('combobox', { name: 'モデル' })).toHaveValue('mediapipe');
  expect(f.segmenter.close).toHaveBeenCalledTimes(1);
  expect(f.detector.close).not.toHaveBeenCalled();
});

it('disposes pending task ownership on unmount and releases its late completion', async () => {
  const f = fixture();
  const a = deferred<SegmenterTask>();
  vi.mocked(f.factory.segmenter).mockReturnValue(a.promise);
  const { unmount, container } = render(<App engine={f.engine} />);
  await waitFor(() => expect(f.factory.segmenter).toHaveBeenCalledTimes(1));
  unmount();
  await act(async () => { a.resolve(f.segmenter); });
  expect(f.engine.getState().status).toBe('idle');
  expect(f.segmenter.close).toHaveBeenCalledTimes(1);
  expect(container).toBeEmptyDOMElement();
});

it('remembers the last selected model and variant across remounts', async () => {
  const first = fixture();
  const user = userEvent.setup();
  const view1 = render(<App engine={first.engine} />);
  await screen.findByText('モデル準備完了（カメラの準備・推論成功とは別です）');

  const header1 = screen.getByRole('banner');
  await user.selectOptions(within(header1).getByRole('combobox', { name: 'モデル' }), 'stereo-depth');
  await waitFor(() =>
    expect(first.engine.getState()).toMatchObject({
      status: 'ready',
      model: 'stereo-depth',
      variantId: 'stereo-sad-baseline',
    })
  );

  await user.selectOptions(within(header1).getByRole('combobox', { name: 'バリアント' }), 'lightstereo-s');
  await waitFor(() =>
    expect(first.engine.getState()).toMatchObject({
      status: 'ready',
      model: 'stereo-depth',
      variantId: 'lightstereo-s',
    })
  );

  view1.unmount();

  const second = fixture();
  render(<App engine={second.engine} />);
  await waitFor(() =>
    expect(second.engine.getState()).toMatchObject({
      status: 'ready',
      model: 'stereo-depth',
      variantId: 'lightstereo-s',
    })
  );
  const header2 = screen.getByRole('banner');
  expect(within(header2).getByRole('combobox', { name: 'モデル' })).toHaveValue('stereo-depth');
  expect(within(header2).getByRole('combobox', { name: 'バリアント' })).toHaveValue('lightstereo-s');
});

it('allows switching Stereo Depth execution backend preference (auto / webgpu / wasm) per model in ControlPanel', async () => {
  const f = fixture();
  const user = userEvent.setup();
  render(<App engine={f.engine} />);
  await screen.findByText('モデル準備完了（カメラの準備・推論成功とは別です）');

  const header = screen.getByRole('banner');
  await user.selectOptions(within(header).getByRole('combobox', { name: 'モデル' }), 'stereo-depth');
  await waitFor(() =>
    expect(f.factory.stereoDepth).toHaveBeenCalledWith(
      expect.objectContaining({
        variantId: 'stereo-sad-baseline',
        backendPreference: 'auto',
      })
    )
  );

  await user.click(screen.getByRole('button', { name: 'モデル調整' }));
  const backendGroup = screen.getByRole('group', { name: 'ステレオ実行バックエンド' });
  expect(within(backendGroup).getByRole('button', { name: '自動（互換性優先）' })).toBeInTheDocument();
  expect(screen.getByText('SAD BaselineはCPU専用です。GPUを使うには別の方式を選択してください。')).toBeInTheDocument();
  await user.click(within(backendGroup).getByRole('button', { name: 'WebGPU優先' }));
  await waitFor(() =>
    expect(f.factory.stereoDepth).toHaveBeenLastCalledWith(
      expect.objectContaining({
        variantId: 'stereo-sad-baseline',
        backendPreference: 'webgpu',
      })
    )
  );

  await user.click(within(backendGroup).getByRole('button', { name: 'WASM (CPU)' }));
  await waitFor(() =>
    expect(f.factory.stereoDepth).toHaveBeenLastCalledWith(
      expect.objectContaining({
        variantId: 'stereo-sad-baseline',
        backendPreference: 'wasm',
      })
    )
  );
});

it('defaults Stereo Depth real camera 180-degree rotation ON, allows toggling, and shows disabled status during SBS video playback', async () => {
  const f = fixture();
  const dummyCanvas = document.createElement('canvas');
  f.stereoTask.estimateForVideo.mockResolvedValue({
    width: 16,
    height: 9,
    disparity: new Float32Array(16 * 9).fill(8),
    minDisparity: 8,
    maxDisparity: 8,
    medianDisparity: 8,
    scaleMinDisparity: 8,
    scaleMaxDisparity: 16,
    validRatio: 1,
    overlayCanvas: dummyCanvas,
    leftCanvas: dummyCanvas,
    rightCanvas: dummyCanvas,
    colormap: 'turbo',
    viewMode: 'left-overlay',
    swapLeftRight: false,
    rotate180: false,
    sourceEyeWidth: 480,
    sourceEyeHeight: 270,
  } as never);
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn(() => ({ width: 20 })),
    beginPath: vi.fn(),
    rect: vi.fn(),
    clip: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
  } as unknown as CanvasRenderingContext2D);

  const user = userEvent.setup();
  render(<App engine={f.engine} />);
  await screen.findByText('モデル準備完了（カメラの準備・推論成功とは別です）');

  const header = screen.getByRole('banner');
  await user.selectOptions(within(header).getByRole('combobox', { name: 'モデル' }), 'stereo-depth');
  await waitFor(() =>
    expect(f.engine.getState()).toMatchObject({ status: 'ready', model: 'stereo-depth' })
  );

  await user.click(screen.getByRole('button', { name: 'モデル調整' }));
  const rotateCheckbox = screen.getByRole('checkbox', {
    name: '実カメラ入力を180°回転 (逆さ設置補正)',
  });
  expect(rotateCheckbox).toBeChecked();
  expect(screen.getByText('実カメラ適用中')).toBeInTheDocument();

  await user.click(rotateCheckbox);
  expect(rotateCheckbox).not.toBeChecked();
  expect(screen.getByText('OFF (正立入力)')).toBeInTheDocument();

  // Re-enable before switching to a synthetic SBS video preset
  await user.click(rotateCheckbox);
  expect(rotateCheckbox).toBeChecked();

  await user.click(screen.getByRole('button', { name: '合成: 移動ブロック' }));
  expect(screen.getByText('動画・静止画では無効')).toBeInTheDocument();
  expect(rotateCheckbox).toBeChecked();
});
