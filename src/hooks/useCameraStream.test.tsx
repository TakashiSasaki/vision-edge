import { StrictMode } from 'react';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useCameraStream } from './useCameraStream';
import type { CameraFacingMode } from '../types/vision';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (cause: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function fakeStream() {
  const tracks = [0, 1].map(() => {
    let readyState: 'live' | 'ended' = 'live';
    const track = Object.assign(new EventTarget(), { stop: vi.fn(() => { readyState = 'ended'; }) });
    return Object.defineProperty(track, 'readyState', { get: () => readyState });
  });
  const stream = { getTracks: () => tracks } as unknown as MediaStream;
  return { stream, tracks };
}
function fixture() {
  const requests: ReturnType<typeof deferred<MediaStream>>[] = [];
  const getUserMedia = vi.fn<(constraints: MediaStreamConstraints) => Promise<MediaStream>>(() => { const request = deferred<MediaStream>(); requests.push(request); return request.promise; });
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } });
  const video = Object.assign(document.createElement('video'), { srcObject: null });
  const videoRef = { current: video };
  const onStop = vi.fn();
  const initialProps = { enabled: true, facingMode: 'user' as CameraFacingMode, onStop };
  const view = renderHook(props => useCameraStream(videoRef, props.enabled, props.facingMode, props.onStop), { initialProps });
  return { ...view, getUserMedia, requests, video, videoRef, onStop, initialProps };
}
async function resolveRequest(request: ReturnType<typeof deferred<MediaStream>>, stream: MediaStream) {
  await act(async () => { request.resolve(stream); });
}
beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
});
afterEach(() => { Reflect.deleteProperty(navigator, 'mediaDevices'); });

it('attaches only the requested stream and waits for playback before camera ready', async () => {
  const f = fixture();
  const playback = deferred<void>();
  vi.mocked(f.video.play).mockReturnValue(playback.promise);
  const s = fakeStream();
  expect(f.getUserMedia).toHaveBeenCalledWith({ video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  expect(f.result.current.cameraReady).toBe(false);
  await resolveRequest(f.requests[0], s.stream);
  expect(f.video.srcObject).toBe(s.stream);
  expect(f.result.current.cameraReady).toBe(false);
  await act(async () => { playback.resolve(); });
  expect(f.result.current.cameraReady).toBe(true);
  expect(f.onStop).not.toHaveBeenCalled();
});

it('does not reacquire for parent rerenders or a changed notification callback; uses the latest callback on failure', async () => {
  const f = fixture();
  const nextStop = vi.fn();
  f.rerender({ ...f.initialProps, onStop: nextStop });
  expect(f.getUserMedia).toHaveBeenCalledTimes(1);
  await act(async () => { f.requests[0].reject(new DOMException('denied', 'NotAllowedError')); });
  expect(f.onStop).not.toHaveBeenCalled();
  expect(nextStop).toHaveBeenCalledTimes(1);
  expect(f.result.current.state).toMatchObject({ status: 'error', stage: 'acquisition' });
  expect(f.result.current.cameraError).toContain('アクセスが拒否');
  f.rerender({ ...f.initialProps, onStop: vi.fn() });
  expect(f.getUserMedia).toHaveBeenCalledTimes(1);
});

it.each(['stop', 'unmount'] as const)('releases an acquired stream exactly once on %s and detaches the video', async action => {
  const f = fixture(); const s = fakeStream();
  await resolveRequest(f.requests[0], s.stream);
  expect(f.result.current.cameraReady).toBe(true);
  if (action === 'stop') f.rerender({ ...f.initialProps, enabled: false });
  else f.unmount();
  expect(f.video.srcObject).toBeNull();
  expect(f.video.pause).toHaveBeenCalledTimes(1);
  s.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  f.unmount();
  s.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
});

it.each(['stop', 'unmount'] as const)('immediately releases a late acquisition after %s without attaching or playing it', async action => {
  const f = fixture(); const s = fakeStream();
  if (action === 'stop') f.rerender({ ...f.initialProps, enabled: false });
  else f.unmount();
  await resolveRequest(f.requests[0], s.stream);
  s.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  expect(f.video.srcObject).toBeNull();
  expect(f.video.play).not.toHaveBeenCalled();
  expect(f.onStop).not.toHaveBeenCalled();
});

it('keeps the new facing stream when the old request resolves afterwards', async () => {
  const f = fixture(); const old = fakeStream(); const next = fakeStream();
  f.rerender({ ...f.initialProps, facingMode: 'environment' });
  expect(f.getUserMedia).toHaveBeenCalledTimes(2);
  expect(f.getUserMedia.mock.calls[1][0]).toMatchObject({ video: { facingMode: { ideal: 'environment' } } });
  await resolveRequest(f.requests[1], next.stream);
  const ready = f.result.current.state;
  await resolveRequest(f.requests[0], old.stream);
  expect(f.result.current.state).toBe(ready);
  expect(f.video.srcObject).toBe(next.stream);
  expect(f.video.play).toHaveBeenCalledTimes(1);
  old.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  next.tracks.forEach(track => expect(track.stop).not.toHaveBeenCalled());
});

it('ignores an obsolete rejection rather than stopping the newer camera', async () => {
  const f = fixture(); const next = fakeStream();
  f.rerender({ ...f.initialProps, facingMode: 'environment' });
  await resolveRequest(f.requests[1], next.stream);
  await act(async () => { f.requests[0].reject(new Error('old camera failed')); });
  expect(f.result.current.cameraReady).toBe(true);
  expect(f.result.current.cameraError).toBeNull();
  expect(f.onStop).not.toHaveBeenCalled();
  next.tracks.forEach(track => expect(track.stop).not.toHaveBeenCalled());
});

it('releases the previous ready facing stream before requesting a replacement', async () => {
  const f = fixture(); const old = fakeStream(); const next = fakeStream();
  await resolveRequest(f.requests[0], old.stream);
  f.rerender({ ...f.initialProps, facingMode: 'environment' });
  expect(f.result.current.cameraReady).toBe(false);
  expect(f.video.srcObject).toBeNull();
  old.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  await resolveRequest(f.requests[1], next.stream);
  expect(f.video.srcObject).toBe(next.stream);
  expect(f.result.current.cameraReady).toBe(true);
});

it('stop/start during pending acquisition cannot let the older result overwrite the new stream', async () => {
  const f = fixture(); const old = fakeStream(); const next = fakeStream();
  f.rerender({ ...f.initialProps, enabled: false });
  f.rerender(f.initialProps);
  await resolveRequest(f.requests[1], next.stream);
  await resolveRequest(f.requests[0], old.stream);
  expect(f.video.srcObject).toBe(next.stream);
  expect(f.result.current.cameraReady).toBe(true);
  old.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  next.tracks.forEach(track => expect(track.stop).not.toHaveBeenCalled());
});

it('a late old play rejection neither stops nor detaches the newer stream', async () => {
  const f = fixture(); const old = fakeStream(); const next = fakeStream();
  const oldPlay = deferred<void>();
  vi.mocked(f.video.play).mockReturnValueOnce(oldPlay.promise);
  await resolveRequest(f.requests[0], old.stream);
  f.rerender({ ...f.initialProps, facingMode: 'environment' });
  await resolveRequest(f.requests[1], next.stream);
  const state = f.result.current.state;
  await act(async () => { oldPlay.reject(new DOMException('old pause interrupted play', 'AbortError')); });
  expect(f.result.current.state).toBe(state);
  expect(f.video.srcObject).toBe(next.stream);
  expect(f.onStop).not.toHaveBeenCalled();
  old.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  next.tracks.forEach(track => expect(track.stop).not.toHaveBeenCalled());
});

it('a current play rejection releases tracks and reports playback failure once', async () => {
  const f = fixture(); const s = fakeStream();
  const cause = new DOMException('cannot play', 'NotAllowedError');
  vi.mocked(f.video.play).mockRejectedValueOnce(cause);
  await resolveRequest(f.requests[0], s.stream);
  expect(f.result.current.state).toMatchObject({ status: 'error', stage: 'playback', cause });
  expect(f.result.current.cameraError).toContain('映像を再生できません');
  expect(f.video.srcObject).toBeNull();
  s.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  expect(f.onStop).toHaveBeenCalledTimes(1);
  f.unmount();
  s.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
});

it('can explicitly restart after denied acquisition without an automatic retry loop', async () => {
  const f = fixture();
  await act(async () => { f.requests[0].reject(new DOMException('denied', 'NotAllowedError')); });
  f.rerender(f.initialProps);
  expect(f.getUserMedia).toHaveBeenCalledTimes(1);
  f.rerender({ ...f.initialProps, enabled: false });
  expect(f.result.current.cameraError).toContain('アクセスが拒否');
  act(() => { f.result.current.clearError(); });
  expect(f.result.current.cameraError).toBeNull();
  f.rerender(f.initialProps);
  const next = fakeStream();
  await resolveRequest(f.requests[1], next.stream);
  expect(f.result.current.cameraReady).toBe(true);
});

it('handles an unavailable camera API without an unhandled TypeError', async () => {
  Reflect.deleteProperty(navigator, 'mediaDevices');
  const onStop = vi.fn();
  const videoRef = { current: document.createElement('video') };
  const f = renderHook(() => useCameraStream(videoRef, true, 'user', onStop));
  await act(async () => {});
  expect(f.result.current.cameraError).toContain('HTTPS');
  expect(f.result.current.state).toMatchObject({ status: 'error', stage: 'acquisition' });
  expect(onStop).toHaveBeenCalledTimes(1);
});

it('handles a synchronous getUserMedia exception as an acquisition failure', async () => {
  const cause = new TypeError('invalid constraints');
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(() => { throw cause; }) } });
  const onStop = vi.fn();
  const videoRef = { current: document.createElement('video') };
  const f = renderHook(() => useCameraStream(videoRef, true, 'user', onStop));
  await act(async () => {});
  expect(f.result.current.state).toMatchObject({ status: 'error', stage: 'acquisition', cause });
  expect(onStop).toHaveBeenCalledTimes(1);
});

it('an ended current track releases the remaining stream and cannot later publish play success', async () => {
  const f = fixture(); const s = fakeStream(); const play = deferred<void>();
  vi.mocked(f.video.play).mockReturnValueOnce(play.promise);
  await resolveRequest(f.requests[0], s.stream);
  act(() => { s.tracks[0].dispatchEvent(new Event('ended')); });
  expect(f.result.current.state).toMatchObject({ status: 'error', stage: 'ended' });
  expect(f.onStop).toHaveBeenCalledTimes(1);
  expect(f.video.srcObject).toBeNull();
  s.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  await act(async () => { play.resolve(); });
  expect(f.result.current.cameraReady).toBe(false);
  expect(f.result.current.state.status).toBe('error');
});

it('StrictMode cleanup releases the first late stream without harming the second setup', async () => {
  const requests: ReturnType<typeof deferred<MediaStream>>[] = [];
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn(() => {
    const request = deferred<MediaStream>(); requests.push(request); return request.promise;
  }) } });
  const videoRef = { current: Object.assign(document.createElement('video'), { srcObject: null }) };
  const onStop = vi.fn();
  const f = renderHook(() => useCameraStream(videoRef, true, 'user', onStop), { wrapper: StrictMode });
  expect(requests).toHaveLength(2);
  const old = fakeStream(); const next = fakeStream();
  await resolveRequest(requests[1], next.stream);
  await resolveRequest(requests[0], old.stream);
  expect(videoRef.current.srcObject).toBe(next.stream);
  expect(f.result.current.cameraReady).toBe(true);
  old.tracks.forEach(track => expect(track.stop).toHaveBeenCalledTimes(1));
  next.tracks.forEach(track => expect(track.stop).not.toHaveBeenCalled());
});
