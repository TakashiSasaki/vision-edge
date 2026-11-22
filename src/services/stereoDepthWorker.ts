import * as ort from 'onnxruntime-web';
import {
  createStereoWorkerProcessorWithOrt,
  type OrtWorkerRuntimeLike,
  type StereoWorkerRequest,
  type StereoWorkerResponse,
  type StereoWorkerSessionCreator,
  type WebGpuStereoComputeEngine,
} from './stereoDepthWorkerCore';

export * from './stereoDepthWorkerCore';

export function createStereoWorkerProcessor(
  sessionCreator?: StereoWorkerSessionCreator,
  wgslEngineFactory?: () => Promise<WebGpuStereoComputeEngine>
) {
  return createStereoWorkerProcessorWithOrt(
    ort as unknown as OrtWorkerRuntimeLike,
    sessionCreator,
    wgslEngineFactory
  );
}

const workerScope =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function' &&
  typeof window === 'undefined'
    ? (self as unknown as {
        onmessage: ((ev: MessageEvent<StereoWorkerRequest>) => void) | null;
        postMessage: (msg: StereoWorkerResponse, transfer?: Transferable[]) => void;
      })
    : null;

if (workerScope) {
  const processor = createStereoWorkerProcessor();
  workerScope.onmessage = (event: MessageEvent<StereoWorkerRequest>) => {
    void processor.handleMessage(event.data, (res, transfer) => {
      if (transfer && transfer.length > 0) {
        workerScope.postMessage(res, transfer);
      } else {
        workerScope.postMessage(res);
      }
    });
  };
}
