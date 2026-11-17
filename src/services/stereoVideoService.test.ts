import { describe, expect, it, vi } from 'vitest';
import {
  STEREO_VIDEO_PRESETS,
  SYNTHETIC_STEREO_VIDEO_PRESET_IDS,
  FREE_STEREO_VIDEO_PRESET_IDS,
  getStereoVideoReferenceRegions,
  isStereoVideoPresetId,
  renderStereoVideoPresetFrame,
  clampVideoTime,
  stepVideoTime,
  formatVideoTimecode,
} from './stereoVideoService';
import { disparityToDepthMeters } from './stereoDepthService';

function createMockCanvasWithContext(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  const mockCtx = {
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    rect: vi.fn(),
    clip: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    clearRect: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    ellipse: vi.fn(),
    fillText: vi.fn(),
    createLinearGradient: vi.fn(() => ({
      addColorStop: vi.fn(),
    })),
    createRadialGradient: vi.fn(() => ({
      addColorStop: vi.fn(),
    })),
    setLineDash: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
  } as unknown as CanvasRenderingContext2D;

  vi.spyOn(canvas, 'getContext').mockReturnValue(mockCtx as never);
  return canvas;
}

describe('stereoVideoService', () => {
  it('defines synthetic and free real-world filmed SBS stereo video presets with valid metadata', () => {
    expect(SYNTHETIC_STEREO_VIDEO_PRESET_IDS).toEqual([
      'synth_moving_blocks',
      'synth_corridor',
      'synth_obstacles',
    ]);
    expect(FREE_STEREO_VIDEO_PRESET_IDS).toEqual([
      'free_outdoor_drive',
      'free_nature_park',
      'free_tabletop_lab',
    ]);

    const allPresets = Object.values(STEREO_VIDEO_PRESETS);
    expect(allPresets).toHaveLength(6);

    for (const preset of allPresets) {
      expect(preset.fps).toBeGreaterThan(0);
      expect(preset.durationSec).toBeGreaterThan(0);
      expect(preset.eyeWidth).toBe(480);
      expect(preset.eyeHeight).toBe(270);
      expect(preset.recommendedBaselineMm).toBeGreaterThan(0);
      expect(preset.recommendedFocalPx).toBeGreaterThan(0);
      expect(isStereoVideoPresetId(preset.id)).toBe(true);
    }

    for (const freeId of FREE_STEREO_VIDEO_PRESET_IDS) {
      const freePreset = STEREO_VIDEO_PRESETS[freeId];
      expect(freePreset.videoUrl).toMatch(/^\/stereo-videos\/.+\.webm$/);
      expect(freePreset.upstreamUrl).toMatch(/^https:\/\/commons\.wikimedia\.org\/wiki\/File:/);
      const refRegions = getStereoVideoReferenceRegions(freeId, 1.0);
      expect(refRegions.length).toBeGreaterThanOrEqual(3);
    }

    expect(isStereoVideoPresetId('unknown_preset')).toBe(false);
  });

  it('renders Side-by-Side stereo frames and ground-truth disparity regions for every preset', () => {
    for (const preset of Object.values(STEREO_VIDEO_PRESETS)) {
      const canvas = createMockCanvasWithContext();
      const result = renderStereoVideoPresetFrame(preset.id, 2.4, canvas);

      expect(result.canvas).toBe(canvas);
      expect(canvas.width).toBe(preset.eyeWidth * 2);
      expect(canvas.height).toBe(preset.eyeHeight);
      expect(result.eyeWidth).toBe(preset.eyeWidth);
      expect(result.eyeHeight).toBe(preset.eyeHeight);
      expect(result.regions.length).toBeGreaterThanOrEqual(2);

      for (const region of result.regions) {
        expect(region.disparityPx).toBeGreaterThan(0);
        expect(region.normX).toBeGreaterThan(0);
        expect(region.normX).toBeLessThan(1);
        expect(region.normY).toBeGreaterThan(0);
        expect(region.normY).toBeLessThan(1);

        const depthMeters = disparityToDepthMeters(
          region.disparityPx,
          preset.recommendedBaselineMm,
          preset.recommendedFocalPx
        );
        expect(depthMeters).not.toBeNull();
        expect(depthMeters!).toBeGreaterThan(0.2);
      }
    }
  });

  it('updates ground-truth target positions and disparities dynamically over time', () => {
    const canvas = createMockCanvasWithContext();
    const frameAt0 = renderStereoVideoPresetFrame('synth_obstacles', 0.2, canvas);
    const frameAt2 = renderStereoVideoPresetFrame('synth_obstacles', 2.2, canvas);

    expect(frameAt0.frameIndex).not.toBe(frameAt2.frameIndex);
    expect(frameAt0.regions[0].disparityPx).not.toBe(frameAt2.regions[0].disparityPx);
  });

  it('clamps, steps, and formats video timecodes accurately', () => {
    expect(clampVideoTime(14.5, 12, true)).toBeCloseTo(2.5, 4);
    expect(clampVideoTime(14.5, 12, false)).toBe(12);
    expect(clampVideoTime(-1.5, 12, false)).toBe(0);

    const nextFrame = stepVideoTime(1.0, 1, 30, 12, true);
    expect(nextFrame).toBeCloseTo(1.0 + 1 / 30, 5);

    const wrappedPrevFrame = stepVideoTime(0, -1, 30, 10, true);
    expect(wrappedPrevFrame).toBeCloseTo(10 - 1 / 30, 4);

    expect(formatVideoTimecode(2.5, 30)).toEqual({
      timeText: '00:02.50',
      frameIndex: 75,
    });
  });
});
