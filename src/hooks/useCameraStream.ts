import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import type { CameraFacingMode } from '../types/vision';

export type CameraAcquisitionMode = 'standard' | 'stereo-sbs';

export const STEREO_CAMERA_WIDTH = 3840;
export const STEREO_CAMERA_HEIGHT = 1080;

type FailureStage = 'acquisition' | 'playback' | 'ended';
type CameraState = {
  status: 'idle' | 'requesting' | 'ready';
  facingMode: CameraFacingMode;
  mode: CameraAcquisitionMode;
  requestId: number;
} | {
  status: 'error'; facingMode: CameraFacingMode; mode: CameraAcquisitionMode; requestId: number;
  message: string; stage: FailureStage; cause: unknown;
};

function createStereoCameraNotFoundError(cause?: unknown): DOMException {
  return new DOMException(
    `Stereo camera (${STEREO_CAMERA_WIDTH}x${STEREO_CAMERA_HEIGHT}) is not attached` +
      (cause instanceof Error && cause.message ? `: ${cause.message}` : ''),
    'StereoCameraNotFoundError'
  );
}

function getCauseName(cause: unknown): string {
  return cause instanceof Error || cause instanceof DOMException ? cause.name : '';
}

function isPermissionDeniedError(cause: unknown): boolean {
  const name = getCauseName(cause);
  return name === 'NotAllowedError' || name === 'PermissionDeniedError';
}

function errorMessage(stage: FailureStage, cause: unknown, mode: CameraAcquisitionMode): string {
  if (stage === 'playback') return 'カメラ映像を再生できません。再度開始してください。';
  if (stage === 'ended') return 'カメラ映像が終了しました。再度開始してください。';
  const name = getCauseName(cause);
  if (isPermissionDeniedError(cause)) return 'カメラへのアクセスが拒否されました。ブラウザの設定でカメラ許可をオンにしてください。';
  if (name === 'NotSupportedError') return 'カメラを利用できません。対応ブラウザの HTTPS または localhost で開いてください。';
  if (
    mode === 'stereo-sbs' &&
    (name === 'StereoCameraNotFoundError' || name === 'OverconstrainedError' || name === 'NotFoundError')
  ) {
    return `横${STEREO_CAMERA_WIDTH}×縦${STEREO_CAMERA_HEIGHT}ピクセルに対応したステレオカメラが装着されていません。ステレオカメラを接続するか、テスト用の合成SBS動画・フリー素材SBS動画・SBS画像をご利用ください。`;
  }
  return 'カメラデバイスが見つからないか使用中です。';
}

function stopTracks(stream: MediaStream): void {
  for (const track of stream.getTracks()) {
    try { track.stop(); }
    catch (cause) { console.warn('Camera track cleanup failed', { cause }); }
  }
}

function stereoConstraints(deviceId?: string): MediaStreamConstraints {
  return {
    video: deviceId
      ? {
          deviceId: { exact: deviceId },
          width: { exact: STEREO_CAMERA_WIDTH },
          height: { exact: STEREO_CAMERA_HEIGHT },
        }
      : {
          width: { exact: STEREO_CAMERA_WIDTH },
          height: { exact: STEREO_CAMERA_HEIGHT },
        },
    audio: false,
  };
}

function matchesStereoTrackResolution(stream: MediaStream): boolean {
  const videoTrack = stream.getVideoTracks?.()[0] ?? stream.getTracks?.()[0];
  const settings = videoTrack?.getSettings?.();
  if (settings && typeof settings.width === 'number' && typeof settings.height === 'number') {
    return settings.width === STEREO_CAMERA_WIDTH && settings.height === STEREO_CAMERA_HEIGHT;
  }
  return true;
}

async function acquireStereoCameraStream(
  devices: MediaDevices,
  isCurrent: () => boolean
): Promise<MediaStream | null> {
  const tryStereo = async (deviceId?: string): Promise<MediaStream | null> => {
    const stream = await devices.getUserMedia(stereoConstraints(deviceId));
    if (!isCurrent()) {
      stopTracks(stream);
      return null;
    }
    if (!matchesStereoTrackResolution(stream)) {
      stopTracks(stream);
      throw new DOMException(
        `Camera track does not provide ${STEREO_CAMERA_WIDTH}x${STEREO_CAMERA_HEIGHT}`,
        'OverconstrainedError'
      );
    }
    return stream;
  };

  const tryCandidateIds = async (deviceIds: string[]): Promise<MediaStream | null> => {
    let lastError: unknown = null;
    for (const deviceId of deviceIds) {
      if (!isCurrent()) return null;
      try {
        return await tryStereo(deviceId);
      } catch (err) {
        if (isPermissionDeniedError(err)) throw err;
        lastError = err;
      }
    }
    throw createStereoCameraNotFoundError(lastError);
  };

  const listVideoInputs = async (): Promise<MediaDeviceInfo[]> => {
    if (typeof devices.enumerateDevices !== 'function') return [];
    const enumerated = await devices.enumerateDevices();
    return enumerated.filter((device) => device.kind === 'videoinput');
  };

  const initialInputs = await listVideoInputs();
  if (!isCurrent()) return null;

  const knownDeviceIds = initialInputs.map((d) => d.deviceId).filter((id) => Boolean(id));
  if (knownDeviceIds.length > 0) {
    return tryCandidateIds(knownDeviceIds);
  }

  try {
    return await tryStereo(undefined);
  } catch (defaultErr) {
    if (isPermissionDeniedError(defaultErr)) throw defaultErr;

    // If multiple videoinput devices exist but their deviceIds were hidden before permission,
    // briefly unlock enumeration, stop the probe stream, then inspect each camera for 3840x1080.
    if (initialInputs.length >= 2) {
      let unlockedStream: MediaStream | null = null;
      try {
        unlockedStream = await devices.getUserMedia({ video: true, audio: false });
      } catch (unlockErr) {
        if (isPermissionDeniedError(unlockErr)) throw unlockErr;
      } finally {
        if (unlockedStream) stopTracks(unlockedStream);
      }
      if (!isCurrent()) return null;

      const unlockedInputs = await listVideoInputs();
      if (!isCurrent()) return null;
      const unlockedIds = unlockedInputs.map((d) => d.deviceId).filter((id) => Boolean(id));
      if (unlockedIds.length > 0) {
        return tryCandidateIds(unlockedIds);
      }
    }

    throw createStereoCameraNotFoundError(defaultErr);
  }
}

// Every effect owns only its own stream. Callback identity and model/UI state
// cannot restart acquisition; only enabled/facing/mode/target-ref changes can do so.
export function useCameraStream(
  videoRef: RefObject<HTMLVideoElement | null>,
  enabled: boolean,
  facingMode: CameraFacingMode,
  onStop: () => void,
  mode: CameraAcquisitionMode = 'standard'
) {
  const onStopRef = useRef(onStop);
  useLayoutEffect(() => { onStopRef.current = onStop; }, [onStop]);
  const generation = useRef(0);
  const [state, setState] = useState<CameraState>({ status: 'idle', facingMode, mode, requestId: 0 });

  useEffect(() => {
    const requestId = ++generation.current;
    const video = videoRef.current;
    if (!enabled || !video) {
      // Keep a denied-camera error visible after the App switches to stopped.
      setState(previous => previous.status === 'error' ? previous : { status: 'idle', facingMode, mode, requestId });
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
      console.warn('Camera request failed', { requestId, facingMode, mode, stage, cause });
      setState({ status: 'error', facingMode, mode, requestId, stage, cause, message: errorMessage(stage, cause, mode) });
      onStopRef.current();
    };
    const onEnded = () => fail('ended', new DOMException('Camera track ended', 'NotReadableError'));

    setState({ status: 'requesting', facingMode, mode, requestId });
    void (async () => {
      let stage: FailureStage = 'acquisition';
      try {
        const devices = navigator.mediaDevices;
        if (!devices?.getUserMedia) throw new DOMException('Camera API unavailable', 'NotSupportedError');
        const stream =
          mode === 'stereo-sbs'
            ? await acquireStereoCameraStream(devices, current)
            : await devices.getUserMedia({
                video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false,
              });
        if (!stream) return;
        if (!current()) { stopTracks(stream); return; }
        owned = stream;
        for (const track of stream.getTracks()) track.addEventListener('ended', onEnded);
        video.srcObject = stream;
        stage = 'playback';
        await video.play();
        if (!current()) { releaseOwned(); return; }
        if (
          mode === 'stereo-sbs' &&
          video.videoWidth > 0 &&
          video.videoHeight > 0 &&
          (video.videoWidth !== STEREO_CAMERA_WIDTH || video.videoHeight !== STEREO_CAMERA_HEIGHT)
        ) {
          throw createStereoCameraNotFoundError(
            new Error(`Received ${video.videoWidth}x${video.videoHeight} instead of ${STEREO_CAMERA_WIDTH}x${STEREO_CAMERA_HEIGHT}`)
          );
        }
        setState({ status: 'ready', facingMode, mode, requestId });
      } catch (cause) {
        fail(stage, cause);
      }
    })();

    return () => {
      cancelled = true;
      releaseOwned();
    };
  }, [videoRef, enabled, facingMode, mode]);

  const clearError = useCallback(() => {
    setState(previous => previous.status === 'error'
      ? { status: 'idle', facingMode: previous.facingMode, mode: previous.mode, requestId: previous.requestId }
      : previous);
  }, []);
  return {
    state, clearError,
    cameraError: state.status === 'error' ? state.message : null,
    cameraReady: enabled && state.status === 'ready' && state.facingMode === facingMode && state.mode === mode,
    cameraRequestId: state.requestId,
  };
}
