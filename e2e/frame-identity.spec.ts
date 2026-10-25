import { expect, test, type BrowserContext } from '@playwright/test';
import { createHash } from 'node:crypto';

interface NativeFrameProbe {
  requests: number;
  first: Array<{ mediaTime: number; presentedFrames: number }>;
  streams: MediaStream[];
  source: HTMLVideoElement | null;
  releaseModel: () => void;
  drawNext: () => void;
  hold: boolean;
  contend: boolean;
  blocked: number;
  reservedPixels: number[];
  reservedFrames: number;
  reservedImageTimestamp: number;
  delayed: number;
  dispose: () => void;
}
declare global { interface Window { nativeFrameProbe: NativeFrameProbe } }
test.use({ baseURL: 'http://127.0.0.1:4174' });

async function installProbe(context: BrowserContext, fallback: boolean, kind: 'still' | 'decoded') {
  await context.route('**/*', route => new URL(route.request().url()).origin === 'http://127.0.0.1:4174' ? route.continue() : route.abort());
  await context.addInitScript(({ fallback, kind }) => {
    const nativeTimeout = window.setTimeout.bind(window);
    const nativeRaf = window.requestAnimationFrame.bind(window);
    const nativeRvfc = HTMLVideoElement.prototype.requestVideoFrameCallback;
    const nativePlay = HTMLMediaElement.prototype.play;
    const urls: string[] = [];
    const probe: NativeFrameProbe = { requests: 0, first: [], streams: [], source: null, releaseModel() {}, drawNext() {}, hold: false, contend: false, blocked: 0, delayed: 0, reservedPixels: [], reservedFrames: 0, reservedImageTimestamp: -1,
      dispose() { probe.streams.forEach(stream => stream.getTracks().forEach(track => track.stop())); probe.source?.pause(); urls.forEach(URL.revokeObjectURL); } };
    window.nativeFrameProbe = probe;
    window.visionSdkGate = new Promise<void>(resolve => { probe.releaseModel = resolve; });
    // Hold the first real production timer until a native image advances.
    // Deliver that image's notification just after the queued work, reproducing
    // decoder progress before callback delivery without substituting a scheduler.
    let held: { callback: (...args: unknown[]) => void; args: unknown[]; id: number } | null = null;
    window.setTimeout = ((callback: TimerHandler, timeout?: number, ...args: unknown[]) => {
      if (timeout === 0 && probe.hold && kind === 'decoded' && typeof callback === 'function') {
        probe.hold = false; probe.delayed++;
        held = { callback: callback as (...args: unknown[]) => void, args, id: nativeTimeout(() => {
          if (held) throw new Error('Native input did not advance before the held job');
        }, 4000) };
        return held.id;
      }
      if (timeout === 0 && probe.hold) { probe.hold = false; probe.delayed++; return nativeTimeout(callback, 120, ...args); }
      return nativeTimeout(callback, timeout, ...args);
    }) as typeof window.setTimeout;
    const releaseBeforeArrival = (arrival: () => void) => {
      if (!held || probe.reservedPixels.length === 0) return false;
      const video = document.querySelector('video')!;
      const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 64;
      const context = canvas.getContext('2d', { willReadFrequently: true })!;
      context.drawImage(video, 0, 0);
      const pixels = context.getImageData(0, 0, 64, 64).data;
      if (pixels.every((value, i) => value === probe.reservedPixels[i])) return false;
      const job = held; held = null; clearTimeout(job.id);
      probe.source?.pause();
      // Drain in-flight MediaStream presentation before inferring the final
      // stalled image; pausing the consumer alone does not drain its producer.
      nativeRaf(() => nativeRaf(() => nativeTimeout(() => { job.callback(...job.args); nativeTimeout(arrival, 0); }, 100)));
      return true;
    };
    const contend = () => {
      if (!probe.contend || probe.delayed === 0) return;
      const video = document.querySelector('video')!;
      const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 64;
      const context = canvas.getContext('2d', { willReadFrequently: true })!;
      const image = new VideoFrame(video);
      try {
        context.drawImage(image, 0, 0);
        probe.reservedPixels = Array.from(context.getImageData(0, 0, 64, 64).data);
        probe.reservedImageTimestamp = image.timestamp;
      } finally { image.close(); }
      probe.reservedFrames = video.getVideoPlaybackQuality().totalVideoFrames;
      probe.contend = false; probe.blocked++;
      const end = performance.now() + 320;
      while (performance.now() < end) { /* Native decoding continues during main-thread contention. */ }
    };
    Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', { configurable: true, value: fallback ? undefined : function(this: HTMLVideoElement, callback: VideoFrameRequestCallback) {
      return nativeRvfc.call(this, (now, metadata) => {
        const deliver = () => { callback(now, metadata); contend(); };
        if (!releaseBeforeArrival(deliver)) deliver();
      });
    } });
    if (fallback) window.requestAnimationFrame = callback => nativeRaf(now => {
      const deliver = () => { callback(now); contend(); };
      if (!releaseBeforeArrival(deliver)) deliver();
    });
    HTMLMediaElement.prototype.play = function() {
      if (this instanceof HTMLVideoElement && this.srcObject instanceof MediaStream) {
        nativeRvfc.call(this, (_now, metadata) => { probe.first.push({ mediaTime: metadata.mediaTime, presentedFrames: metadata.presentedFrames }); });
        if (kind === 'still') {
          const track = this.srcObject.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
          nativeTimeout(() => track.requestFrame(), 20);
          // Third scene: race a second presentation against the bootstrap timer.
          if (probe.requests === 3) nativeTimeout(() => probe.drawNext(), 60);
        }
      }
      return nativePlay.call(this);
    };
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: async () => {
      probe.requests++;
      const canvas = document.createElement('canvas'); canvas.width = 64; canvas.height = 64;
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#154a73'; ctx.fillRect(0, 0, 64, 64);
      if (kind === 'still') {
        if (probe.requests === 3) probe.hold = true;
        const stream = canvas.captureStream(0);
        probe.drawNext = () => { ctx.fillStyle = '#d49227'; ctx.fillRect(0, 0, 64, 64); (stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack).requestFrame(); };
        probe.streams.push(stream); return stream;
      }
      // Encode a changing clip in-browser, then let the native decoder progress
      // independently of the blocked JS thread. No ffmpeg or downloaded fixture.
      const recordingStream = canvas.captureStream(20); probe.streams.push(recordingStream);
      const recorder = new MediaRecorder(recordingStream, { mimeType: 'video/webm;codecs=vp8' });
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = event => { chunks.push(event.data); };
      const recorded = new Promise<void>(resolve => { recorder.onstop = () => resolve(); });
      recorder.start();
      for (let frame = 0; frame < 30; frame++) {
        ctx.fillStyle = `rgb(${frame * 7},${255 - frame * 5},${frame * 3})`; ctx.fillRect(0, 0, 64, 64);
        await new Promise<void>(resolve => nativeTimeout(resolve, 50));
      }
      recorder.stop(); await recorded; recordingStream.getTracks().forEach(track => track.stop());
      const source = document.createElement('video'); source.muted = true; source.playsInline = true; source.loop = true;
      source.src = URL.createObjectURL(new Blob(chunks, { type: recorder.mimeType })); urls.push(source.src); probe.source = source;
      await nativePlay.call(source);
      const stream = (source as HTMLVideoElement & { captureStream: () => MediaStream }).captureStream(); probe.streams.push(stream); return stream;
    } } });
  }, { fallback, kind });
}

for (const fallback of [false, true]) {
  test(`F01 actual canvas/engine consumes one decoded image after delayed execution (${fallback ? 'fallback' : 'rVFC'})`, async ({ context, page }, testInfo) => {
    await installProbe(context, fallback, 'decoded');
    const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto('/');
    try {
      await expect.poll(() => page.evaluate(() => window.nativeFrameProbe.first.length)).toBe(1);
      await expect(page.getByLabel('カメラ状態')).toHaveText('カメラ映像を取得しました');
      await page.evaluate(() => {
        window.visionSdkProbe.observeInputs = true;
        window.visionSdkProbe.afterInput = video => { video.pause(); window.nativeFrameProbe.source!.pause(); };
        window.nativeFrameProbe.hold = true; window.nativeFrameProbe.contend = true;
        window.nativeFrameProbe.releaseModel();
      });
      await expect.poll(() => page.evaluate(() => window.visionSdkProbe.calls)).toBeGreaterThanOrEqual(1);
      await page.waitForTimeout(500);
      const result = await page.evaluate(() => ({ inputs: window.visionSdkProbe.inputs, calls: window.visionSdkProbe.calls,
        reservedImageTimestamp: window.nativeFrameProbe.reservedImageTimestamp, reservedPixels: window.nativeFrameProbe.reservedPixels, reservedFrames: window.nativeFrameProbe.reservedFrames, delayed: window.nativeFrameProbe.delayed, blocked: window.nativeFrameProbe.blocked, requests: window.nativeFrameProbe.requests }));
      const evidence = { ...result, samePixelsAcrossInputs: result.inputs.slice(1).map(input => input.pixels.every((value, i) => value === result.inputs[0].pixels[i])), reservedPixels: undefined, reservedPixelSha256: createHash('sha256').update(Buffer.from(result.reservedPixels)).digest('hex'), inputs: result.inputs.map(input => ({ sourceKind: input.sourceKind, time: input.time,
        qualityFrames: input.qualityFrames, pixelSha256: createHash('sha256').update(Buffer.from(input.pixels)).digest('hex') })) };
      await testInfo.attach('consumed-native-input.json', { body: JSON.stringify(evidence, null, 2), contentType: 'application/json' });
      expect(result.delayed).toBe(1); expect(result.blocked).toBe(1);
      expect(result.calls).toBe(1); expect(result.inputs).toHaveLength(1);
      expect(result.inputs[0].pixels).toHaveLength(64 * 64 * 4);
      expect(result.inputs[0].sourceKind).toBe('video-frame');
      expect(result.inputs[0].time! * 1000000).toBeGreaterThan(result.reservedImageTimestamp);
      expect(result.inputs[0].qualityFrames).toBeGreaterThanOrEqual(result.reservedFrames);
      expect(result.inputs[0].pixels.every((value, i) => value === result.reservedPixels[i])).toBe(false);
      expect(result.requests).toBe(1);
      await expect(page.getByLabel('推論FPS')).toHaveText('推論: 未計測');
      await expect(page.getByLabel('フレーム処理時間')).toContainText(/\d+\.\d+ ms/);
      expect(errors).toEqual([]);
    } finally { await page.evaluate(() => window.nativeFrameProbe.dispose()); }
  });
  test(`F02 pre-presented first frame, equal-time restart and bootstrap/arrival race (${fallback ? 'fallback' : 'rVFC'})`, async ({ context, page }, testInfo) => {
    await installProbe(context, fallback, 'still');
    await page.goto('/');
    try {
      await expect.poll(() => page.evaluate(() => window.nativeFrameProbe.first.length)).toBe(1);
      await expect(page.getByLabel('カメラ状態')).toHaveText('カメラ映像を取得しました');
      expect(await page.evaluate(() => window.visionSdkProbe.calls)).toBe(0);
      await page.evaluate(() => { window.visionSdkProbe.observeInputs = true; window.nativeFrameProbe.releaseModel(); });
      await expect.poll(() => page.evaluate(() => window.visionSdkProbe.calls)).toBe(1);
      await page.waitForTimeout(400); expect(await page.evaluate(() => window.visionSdkProbe.calls)).toBe(1);
      const creates = await page.evaluate(() => window.visionSdkProbe.creates);
      for (const session of [2, 3]) {
        await page.getByRole('banner').getByRole('button', { name: '停止', exact: true }).click();
        await page.getByRole('banner').getByRole('button', { name: 'カメラ開始', exact: true }).click();
        await expect.poll(() => page.evaluate(() => window.visionSdkProbe.calls)).toBe(session);
        await page.waitForTimeout(400); expect(await page.evaluate(() => window.visionSdkProbe.calls)).toBe(session);
      }
      const result = await page.evaluate(() => ({ inputs: window.visionSdkProbe.inputs.map(input => ({ time: input.time, qualityFrames: input.qualityFrames })),
        first: window.nativeFrameProbe.first, creates: window.visionSdkProbe.creates, requests: window.nativeFrameProbe.requests, delayed: window.nativeFrameProbe.delayed }));
      // currentTime is a running playhead; mediaTime identifies the source's
      // first image and is zero again in the new camera session.
      expect(result.first.slice(0, 2).map(first => first.mediaTime)).toEqual([0, 0]);
      expect(result.inputs[2].qualityFrames).toBeGreaterThanOrEqual(2);
      expect(result.creates).toBe(creates); expect(result.requests).toBe(3); expect(result.delayed).toBe(1);
      await expect(page.getByLabel('推論FPS')).toHaveText('推論: 未計測');
      await testInfo.attach('first-frame-scenes.json', { body: JSON.stringify(result, null, 2), contentType: 'application/json' });
    } finally { await page.evaluate(() => window.nativeFrameProbe.dispose()); }
  });
}
