import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import type { CameraFacingMode } from '../types/vision';

type FailureStage = 'acquisition' | 'playback' | 'ended';
type CameraState = {
  status: 'idle' | 'requesting' | 'ready';
  facingMode: CameraFacingMode;
  requestId: number;
} | {
  status: 'error'; facingMode: CameraFacingMode; requestId: number;
  message: string; stage: FailureStage; cause: unknown;
};

function errorMessage(stage: FailureStage, cause: unknown): string {
  if (stage === 'playback') return 'カメラ映像を再生できません。再度開始してください。';
  if (stage === 'ended') return 'カメラ映像が終了しました。再度開始してください。';
  const name = cause instanceof Error || cause instanceof DOMException ? cause.name : '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'カメラへのアクセスが拒否されました。ブラウザの設定でカメラ許可をオンにしてください。';
  if (name === 'NotSupportedError') return 'カメラを利用できません。対応ブラウザの HTTPS または localhost で開いてください。';
  return 'カメラデバイスが見つからないか使用中です。';
}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) {
    try { track.stop(); }
    catch (cause) { console.warn('Camera track cleanup failed', { cause }); }
  }
}

// Every effect owns only its own stream. Callback identity and model/UI state
// cannot restart acquisition; only enabled/facing/target-ref changes can do so.
export function useCameraStream(videoRef: RefObject<HTMLVideoElement | null>, enabled: boolean, facingMode: CameraFacingMode, onStop: () => void) {
  const onStopRef = useRef(onStop);
  useLayoutEffect(() => { onStopRef.current = onStop; }, [onStop]);
  const generation = useRef(0);
  const [state, setState] = useState<CameraState>({ status: 'idle', facingMode, requestId: 0 });

  useEffect(() => {
    const requestId = ++generation.current;
    const video = videoRef.current;
    if (!enabled || !video) {
      // Keep a denied-camera error visible after the App switches to stopped.
      setState(previous => previous.status === 'error' ? previous : { status: 'idle', facingMode, requestId });
      return;
    }
    let cancelled = false;
    let failed = false;
    let owned: MediaStream | null = null;
    const current = () => !cancelled && !failed && videoRef.current === video;
    const releaseOwned = () => {
      const stream = owned;
      owned = null;
      if (!stream) return;
      for (const track of stream.getTracks()) track.removeEventListener('ended', onEnded);
      if (video.srcObject === stream) {
        video.pause();
        video.srcObject = null;
      }
      stopTracks(stream);
    };
    const fail = (stage: FailureStage, cause: unknown) => {
      if (!current()) return;
      failed = true;
      releaseOwned();
      console.warn('Camera request failed', { requestId, facingMode, stage, cause });
      setState({ status: 'error', facingMode, requestId, stage, cause, message: errorMessage(stage, cause) });
      onStopRef.current();
    };
    const onEnded = () => fail('ended', new DOMException('Camera track ended', 'NotReadableError'));

    setState({ status: 'requesting', facingMode, requestId });
    void (async () => {
      let stage: FailureStage = 'acquisition';
      try {
        const devices = navigator.mediaDevices;
        if (!devices?.getUserMedia) throw new DOMException('Camera API unavailable', 'NotSupportedError');
        const stream = await devices.getUserMedia({
          video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
        });
        if (!current()) { stopTracks(stream); return; }
        owned = stream;
        for (const track of stream.getTracks()) track.addEventListener('ended', onEnded);
        video.srcObject = stream;
        stage = 'playback';
        await video.play();
        if (!current()) { releaseOwned(); return; }
        setState({ status: 'ready', facingMode, requestId });
      } catch (cause) {
        fail(stage, cause);
      }
    })();

    return () => {
      cancelled = true;
      releaseOwned();
    };
  }, [videoRef, enabled, facingMode]);

  const clearError = useCallback(() => {
    setState(previous => previous.status === 'error'
      ? { status: 'idle', facingMode: previous.facingMode, requestId: previous.requestId }
      : previous);
  }, []);
  return {
    state, clearError,
    cameraError: state.status === 'error' ? state.message : null,
    cameraReady: enabled && state.status === 'ready' && state.facingMode === facingMode,
    cameraRequestId: state.requestId,
  };
}
