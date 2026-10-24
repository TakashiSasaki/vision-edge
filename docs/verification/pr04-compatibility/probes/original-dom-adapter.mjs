// The original abstract unit fixtures have no readable canvas/image data.
// Preserve their logical source-frame markers as an explicit video capability;
// native capability-absence probes run separately without this adapter.
import { vi } from 'vitest';
vi.mock('../../../../src/services/frameScheduler', async importOriginal => {
  const actual = await importOriginal();
  return { ...actual, startFrameScheduler(options) {
    const video = options.video;
    if (typeof VideoFrame !== 'function' && !video.getVideoPlaybackQuality) {
      let coarse = 0, previousTime, previousCount;
      const request = video.requestVideoFrameCallback?.bind(video);
      if (request) video.requestVideoFrameCallback = callback => request((now, metadata) => {
        if (metadata.mediaTime === previousTime && metadata.presentedFrames !== previousCount) coarse++;
        previousTime = metadata.mediaTime; previousCount = metadata.presentedFrames;
        callback(now, metadata);
      });
      video.getVideoPlaybackQuality = () => ({ totalVideoFrames: Math.floor(video.currentTime * 1000000) + coarse + 1 });
    }
    return actual.startFrameScheduler(options);
  } };
});
