import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from './App';

// Keep the app and vision engine real; replace only the external MediaPipe loader.
vi.mock('@mediapipe/tasks-vision', () => ({
  FilesetResolver: { forVisionTasks: vi.fn().mockRejectedValue(new Error('Offline unit test')) },
  ObjectDetector: { createFromOptions: vi.fn() },
  ImageSegmenter: { createFromOptions: vi.fn() },
}));

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn().mockRejectedValue(new DOMException('Test camera denied', 'NotAllowedError')) },
  });
});
afterEach(() => Reflect.deleteProperty(navigator, 'mediaDevices'));

it('renders the app with denied camera and unavailable external models, then switches model UI', async () => {
  const user = userEvent.setup();
  render(<App />);
  const header = screen.getByRole('banner');
  expect(within(header).getByText('VisionStudio AI')).toBeInTheDocument();
  await waitFor(() => expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalled());
  expect(await screen.findByText('カメラエラー')).toBeInTheDocument();

  const [model, backend] = within(header).getAllByRole('combobox');
  expect(model).toHaveValue('selfie-segmenter');
  expect(within(model).getAllByRole('option')).toHaveLength(4);
  await user.selectOptions(model, 'mediapipe');
  expect(model).toHaveValue('mediapipe');
  await user.selectOptions(backend, 'wasm-simd');
  expect(backend).toHaveValue('wasm-simd');

  await user.click(within(header).getByRole('button', { name: '仕様' }));
  expect(screen.getByRole('heading', { name: '物体認識・セグメンテーションモデル仕様' })).toBeInTheDocument();
  await user.click(screen.getByText('YOLO-seg (YOLOv8/11-seg)'));
  expect(model).toHaveValue('yolo-seg');
  expect(screen.queryByRole('heading', { name: '物体認識・セグメンテーションモデル仕様' })).not.toBeInTheDocument();
});
