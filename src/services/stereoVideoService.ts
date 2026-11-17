export type StereoVideoPresetId =
  | 'synth_moving_blocks'
  | 'synth_corridor'
  | 'synth_obstacles'
  | 'free_outdoor_drive'
  | 'free_nature_park'
  | 'free_tabletop_lab';

export type StereoVideoSourceKind = 'synthetic' | 'free-preset' | 'uploaded-file';

export type StereoPlaybackRate = 0.25 | 0.5 | 1.0 | 1.5;

export interface StereoVideoPresetMetadata {
  id: StereoVideoPresetId;
  kind: 'synthetic' | 'free-preset';
  label: string;
  shortLabel: string;
  description: string;
  license: string;
  attribution: string;
  upstreamUrl: string;
  /** Real-world filmed Side-by-Side video asset path under public/ */
  videoUrl?: string;
  fps: number;
  durationSec: number;
  eyeWidth: number;
  eyeHeight: number;
  recommendedBaselineMm: number;
  recommendedFocalPx: number;
}

export interface StereoVideoGroundTruthRegion {
  id: string;
  label: string;
  disparityPx: number;
  /** Normalized [0..1] center X in the left eye */
  normX: number;
  /** Normalized [0..1] center Y in the left eye */
  normY: number;
}

export interface StereoVideoFrameRenderResult {
  canvas: HTMLCanvasElement;
  eyeWidth: number;
  eyeHeight: number;
  frameIndex: number;
  timeSec: number;
  regions: StereoVideoGroundTruthRegion[];
}

export const STEREO_VIDEO_PRESETS: Record<StereoVideoPresetId, StereoVideoPresetMetadata> = {
  synth_moving_blocks: {
    id: 'synth_moving_blocks',
    kind: 'synthetic',
    label: '合成SBS動画: 移動立体ブロック',
    shortLabel: '合成: 移動ブロック',
    description:
      '遠景・中景・近景のテクスチャ付き立体ブロックが異なる周期と奥行き（視差 6px〜38px）で往復移動する動的キャリブレーション映像。',
    license: 'Built-in Synthetic (MIT)',
    attribution: 'Procedural Ground-Truth Stereo Generator',
    upstreamUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API',
    fps: 30,
    durationSec: 12,
    eyeWidth: 480,
    eyeHeight: 270,
    recommendedBaselineMm: 60,
    recommendedFocalPx: 500,
  },
  synth_corridor: {
    id: 'synth_corridor',
    kind: 'synthetic',
    label: '合成SBS動画: 奥行き回廊ウォークスルー',
    shortLabel: '合成: 奥行き回廊',
    description:
      '前進するロボット・移動体視点で左右のピラー・床面グリッド・距離マーカーが遠方から手前へ連続的に流れる奥行き回廊シーン。',
    license: 'Built-in Synthetic (MIT)',
    attribution: 'Procedural Corridor Stereo Simulator',
    upstreamUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API',
    fps: 30,
    durationSec: 10,
    eyeWidth: 480,
    eyeHeight: 270,
    recommendedBaselineMm: 65,
    recommendedFocalPx: 520,
  },
  synth_obstacles: {
    id: 'synth_obstacles',
    kind: 'synthetic',
    label: '合成SBS動画: 接近・交差する障害物',
    shortLabel: '合成: 接近障害物',
    description:
      '歩行者・移動カート・横断ターゲットが遠方（約4.3m）から近距離（約0.85m）へ接近・交差し、視差と実距離の追従性を検証できるシーン。',
    license: 'Built-in Synthetic (MIT)',
    attribution: 'Procedural Dynamic Obstacle Stereo Scene',
    upstreamUrl: 'https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API',
    fps: 30,
    durationSec: 12,
    eyeWidth: 480,
    eyeHeight: 270,
    recommendedBaselineMm: 60,
    recommendedFocalPx: 500,
  },
  free_outdoor_drive: {
    id: 'free_outdoor_drive',
    kind: 'free-preset',
    label: 'フリー実写SBS動画: ヴッパータール空中鉄道 (Wuppertaler Schwebebahn)',
    shortLabel: 'フリー実写: 空中鉄道',
    description:
      'ドイツ・ヴッパータールで実写ステレオ撮影された懸垂式モノレールとトラス鉄橋・河川のサイドバイサイド映像（無限遠オフセット補正済）。',
    license: 'CC BY-SA 4.0 (Wikimedia Commons)',
    attribution: 'Gradzeichen — File:Wuppertaler schwebebahn.sbs.webm',
    upstreamUrl: 'https://commons.wikimedia.org/wiki/File:Wuppertaler_schwebebahn.sbs.webm',
    videoUrl: '/stereo-videos/wuppertal-schwebebahn-sbs.webm',
    fps: 24,
    durationSec: 8,
    eyeWidth: 480,
    eyeHeight: 270,
    recommendedBaselineMm: 65,
    recommendedFocalPx: 520,
  },
  free_nature_park: {
    id: 'free_nature_park',
    kind: 'free-preset',
    label: 'フリー実写SBS動画: 踏切遮断機と通過貨物列車 (Bahnübergang Laatzen)',
    shortLabel: 'フリー実写: 踏切と列車',
    description:
      '手前の架線柱・踏切設備と、中景を通過する2階建て自動車運搬列車・遠景の樹林を実写ステレオカメラで捉えたサイドバイサイド映像。',
    license: 'CC BY-SA 4.0 (Wikimedia Commons)',
    attribution: 'Gradzeichen — File:Bahnübergang latzen.sbs.webm',
    upstreamUrl: 'https://commons.wikimedia.org/wiki/File:Bahn%C3%BCbergang_latzen.sbs.webm',
    videoUrl: '/stereo-videos/bahn-latzen-crossing-sbs.webm',
    fps: 24,
    durationSec: 8,
    eyeWidth: 480,
    eyeHeight: 270,
    recommendedBaselineMm: 65,
    recommendedFocalPx: 500,
  },
  free_tabletop_lab: {
    id: 'free_tabletop_lab',
    kind: 'free-preset',
    label: 'フリー実写SBS動画: 広場セグウェイ群と高速鉄道ICE車内 (Segway & ICE 1134)',
    shortLabel: 'フリー実写: 広場とICE車内',
    description:
      '建築広場を移動するセグウェイ群と歩行者（前半）および時速250kmで走行する高速鉄道ICE 1134車内の近距離速度表示盤と客室通路（後半）の実写ステレオ映像。',
    license: 'CC BY-SA 4.0 (Wikimedia Commons)',
    attribution: 'Gradzeichen — File:Segway.sbs.webm & File:Ice geschwindigkeit.sbs.webm',
    upstreamUrl: 'https://commons.wikimedia.org/wiki/File:Ice_geschwindigkeit.sbs.webm',
    videoUrl: '/stereo-videos/segway-ice-real-sbs.webm',
    fps: 24,
    durationSec: 9,
    eyeWidth: 480,
    eyeHeight: 270,
    recommendedBaselineMm: 60,
    recommendedFocalPx: 500,
  },
};

export const SYNTHETIC_STEREO_VIDEO_PRESET_IDS: readonly StereoVideoPresetId[] = [
  'synth_moving_blocks',
  'synth_corridor',
  'synth_obstacles',
];

export const FREE_STEREO_VIDEO_PRESET_IDS: readonly StereoVideoPresetId[] = [
  'free_outdoor_drive',
  'free_nature_park',
  'free_tabletop_lab',
];

export function isStereoVideoPresetId(value: string): value is StereoVideoPresetId {
  return Object.prototype.hasOwnProperty.call(STEREO_VIDEO_PRESETS, value);
}

export function clampVideoTime(timeSec: number, durationSec: number, loop = true): number {
  if (!Number.isFinite(timeSec) || durationSec <= 0) return 0;
  if (loop) {
    const mod = timeSec % durationSec;
    return mod < 0 ? mod + durationSec : mod;
  }
  return Math.max(0, Math.min(durationSec, timeSec));
}

export function stepVideoTime(
  currentTimeSec: number,
  deltaFrames: number,
  fps: number,
  durationSec: number,
  loop = true
): number {
  const safeFps = fps > 0 ? fps : 30;
  const currentFrame = Math.round(currentTimeSec * safeFps);
  const targetTime = (currentFrame + deltaFrames) / safeFps;
  return clampVideoTime(targetTime, durationSec, loop);
}

export function formatVideoTimecode(timeSec: number, fps: number): {
  timeText: string;
  frameIndex: number;
} {
  const safeTime = Math.max(0, Number.isFinite(timeSec) ? timeSec : 0);
  const safeFps = fps > 0 ? fps : 30;
  const mins = Math.floor(safeTime / 60);
  const secs = safeTime - mins * 60;
  const mm = String(mins).padStart(2, '0');
  const ss = secs.toFixed(2).padStart(5, '0');
  return {
    timeText: `${mm}:${ss}`,
    frameIndex: Math.floor(safeTime * safeFps + 1e-4),
  };
}

function drawRichTexturedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  baseRgb: readonly [number, number, number],
  accentRgb: readonly [number, number, number],
  highlightRgb: readonly [number, number, number],
  seed: number,
  label?: string,
  patternShiftX = 0
): void {
  const rx = Math.round(x);
  const ry = Math.round(y);
  const rw = Math.round(w);
  const rh = Math.round(h);
  if (rw <= 2 || rh <= 2) return;

  ctx.fillStyle = `rgb(${baseRgb[0]}, ${baseRgb[1]}, ${baseRgb[2]})`;
  ctx.fillRect(rx, ry, rw, rh);

  const cell = 8;
  for (let py = 0; py < rh; py += cell) {
    const gy = Math.floor(py / cell);
    for (let px = 0; px < rw; px += cell) {
      const gx = Math.floor((px + patternShiftX) / cell);
      const hash = ((gx * 73 + gy * 179 + seed * 43) >>> 0) % 11;
      if (hash <= 3) {
        ctx.fillStyle = `rgb(${accentRgb[0]}, ${accentRgb[1]}, ${accentRgb[2]})`;
        ctx.fillRect(rx + px + 1, ry + py + 1, Math.min(cell - 2, rw - px - 1), Math.min(cell - 2, rh - py - 1));
      } else if (hash === 4 || hash === 5) {
        ctx.fillStyle = `rgb(${highlightRgb[0]}, ${highlightRgb[1]}, ${highlightRgb[2]})`;
        ctx.fillRect(rx + px + 2, ry + py + 2, Math.min(cell - 4, rw - px - 2), Math.min(cell - 4, rh - py - 2));
      }
    }
  }

  ctx.strokeStyle = 'rgba(248, 250, 252, 0.72)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(rx + 1, ry + 1, rw - 2, rh - 2);

  if (label && rw >= 44 && rh >= 22) {
    const boxW = Math.min(rw - 8, label.length * 7 + 12);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.fillRect(rx + 5, ry + 5, boxW, 17);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 10px monospace';
    ctx.fillText(label, rx + 9, ry + 17);
  }
}

function drawMovingBlocksEye(
  ctx: CanvasRenderingContext2D,
  offsetX: number,
  eyeSign: 0 | 1,
  eyeW: number,
  eyeH: number,
  t: number
): StereoVideoGroundTruthRegion[] {
  const phase = (t / 12) * Math.PI * 2;

  // 1. Background wall at constant disparity = 4px
  const bgDisp = 4;
  const bgShift = eyeSign === 1 ? -bgDisp : 0;
  drawRichTexturedRect(
    ctx,
    offsetX + bgShift - 20,
    -12,
    eyeW + 40,
    eyeH + 24,
    [32, 42, 62],
    [52, 66, 94],
    [76, 94, 128],
    11
  );

  // 2. Far block: oscillates gently horizontally, disparity around 9..13px
  const farDisp = Math.round(11 + Math.sin(phase) * 2);
  const farX = Math.round(290 + Math.cos(phase * 0.8) * 34);
  const farY = Math.round(32 + Math.sin(phase * 1.2) * 14);
  const farW = 135;
  const farH = 110;
  drawRichTexturedRect(
    ctx,
    offsetX + farX + (eyeSign === 1 ? -farDisp : 0),
    farY,
    farW,
    farH,
    [42, 98, 145],
    [78, 148, 198],
    [125, 192, 238],
    23,
    `FAR d=${farDisp}px`
  );

  // 3. Mid block: moves vertically and horizontally, disparity oscillates 17..23px
  const midDisp = Math.round(20 + Math.cos(phase * 1.5) * 3);
  const midX = Math.round(165 + Math.sin(phase) * 42);
  const midY = Math.round(74 + Math.cos(phase) * 20);
  const midW = 132;
  const midH = 130;
  drawRichTexturedRect(
    ctx,
    offsetX + midX + (eyeSign === 1 ? -midDisp : 0),
    midY,
    midW,
    midH,
    [38, 138, 106],
    [82, 192, 152],
    [138, 232, 194],
    37,
    `MID d=${midDisp}px`
  );

  // 4. Near block: approaches and recedes between d=28px and d=38px
  const nearDisp = Math.round(33 + Math.sin(phase * 2) * 5);
  const nearX = Math.round(52 + Math.cos(phase * 1.1) * 28);
  const nearY = Math.round(114 + Math.sin(phase * 0.9) * 16);
  const nearW = 122;
  const nearH = 124;
  drawRichTexturedRect(
    ctx,
    offsetX + nearX + (eyeSign === 1 ? -nearDisp : 0),
    nearY,
    nearW,
    nearH,
    [192, 78, 42],
    [238, 134, 78],
    [252, 188, 128],
    53,
    `NEAR d=${nearDisp}px`
  );

  return [
    {
      id: 'near_block',
      label: '近景ブロック',
      disparityPx: nearDisp,
      normX: (nearX + nearW * 0.5) / eyeW,
      normY: (nearY + nearH * 0.5) / eyeH,
    },
    {
      id: 'mid_block',
      label: '中景ブロック',
      disparityPx: midDisp,
      normX: (midX + midW * 0.5) / eyeW,
      normY: (midY + midH * 0.5) / eyeH,
    },
    {
      id: 'far_block',
      label: '遠景ブロック',
      disparityPx: farDisp,
      normX: (farX + farW * 0.5) / eyeW,
      normY: (farY + farH * 0.5) / eyeH,
    },
  ];
}

function drawCorridorEye(
  ctx: CanvasRenderingContext2D,
  offsetX: number,
  eyeSign: 0 | 1,
  eyeW: number,
  eyeH: number,
  t: number
): StereoVideoGroundTruthRegion[] {
  // Forward walkthrough cycle [0..1)
  const travel = (t * 0.35) % 1;

  // Back wall at disparity = 5px
  const backDisp = 5;
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -backDisp : 0) - 16,
    -12,
    eyeW + 32,
    eyeH + 24,
    [26, 34, 52],
    [46, 58, 86],
    [68, 86, 122],
    7
  );

  // Perspective corridor stages (from far to near)
  const numStages = 4;
  const stages: Array<{ depthProgress: number }> = [];
  for (let i = 0; i < numStages; i++) {
    const p = (i / numStages + travel) % 1;
    stages.push({ depthProgress: p });
  }
  stages.sort((a, b) => a.depthProgress - b.depthProgress);

  // Center portal at far end (d = 7px)
  drawRichTexturedRect(
    ctx,
    offsetX + 188 + (eyeSign === 1 ? -7 : 0),
    82,
    104,
    106,
    [48, 82, 138],
    [82, 126, 192],
    [126, 172, 232],
    17,
    'PORTAL 7px'
  );

  const regions: StereoVideoGroundTruthRegion[] = [];

  for (let idx = 0; idx < stages.length; idx++) {
    const p = stages[idx].depthProgress; // 0 (far) -> 1 (near)
    const disp = Math.round(8 + p * 28); // 8px .. 36px
    const shift = eyeSign === 1 ? -disp : 0;

    // Left & right corridor pillars move outward and grow taller as p -> 1
    const spread = Math.round(52 + p * 145);
    const pillarW = Math.round(38 + p * 44);
    const pillarH = Math.round(96 + p * 146);
    const pillarY = Math.round((eyeH - pillarH) / 2);

    const leftPillarX = Math.round(eyeW * 0.5 - spread - pillarW * 0.5);
    const rightPillarX = Math.round(eyeW * 0.5 + spread - pillarW * 0.5);

    const baseColor: [number, number, number] =
      p > 0.66 ? [184, 112, 38] : p > 0.33 ? [42, 132, 144] : [58, 92, 148];
    const accColor: [number, number, number] =
      p > 0.66 ? [232, 166, 72] : p > 0.33 ? [82, 186, 198] : [96, 142, 206];
    const hiColor: [number, number, number] =
      p > 0.66 ? [252, 212, 124] : p > 0.33 ? [138, 228, 238] : [148, 192, 244];

    drawRichTexturedRect(
      ctx,
      offsetX + leftPillarX + shift,
      pillarY,
      pillarW,
      pillarH,
      baseColor,
      accColor,
      hiColor,
      31 + idx * 7,
      `${disp}px`
    );
    drawRichTexturedRect(
      ctx,
      offsetX + rightPillarX + shift,
      pillarY,
      pillarW,
      pillarH,
      baseColor,
      accColor,
      hiColor,
      43 + idx * 7,
      `${disp}px`
    );

    if (idx === stages.length - 1) {
      regions.push({
        id: 'corridor_near_pillar',
        label: '手前ピラー',
        disparityPx: disp,
        normX: Math.max(0.05, Math.min(0.95, (leftPillarX + pillarW * 0.5) / eyeW)),
        normY: 0.5,
      });
    }
  }

  // Floor guide beacon moving along the center floor
  const beaconP = (travel * 1.2) % 1;
  const beaconDisp = Math.round(12 + beaconP * 24);
  const beaconW = Math.round(64 + beaconP * 42);
  const beaconH = Math.round(44 + beaconP * 38);
  const beaconX = Math.round((eyeW - beaconW) * 0.5);
  const beaconY = Math.round(145 + beaconP * 75);
  drawRichTexturedRect(
    ctx,
    offsetX + beaconX + (eyeSign === 1 ? -beaconDisp : 0),
    beaconY,
    beaconW,
    beaconH,
    [192, 62, 68],
    [240, 112, 116],
    [254, 168, 172],
    71,
    `FLOOR ${beaconDisp}px`
  );

  regions.push({
    id: 'corridor_floor_beacon',
    label: '床面ビーコン',
    disparityPx: beaconDisp,
    normX: (beaconX + beaconW * 0.5) / eyeW,
    normY: (beaconY + beaconH * 0.5) / eyeH,
  });

  return regions;
}

function drawObstaclesEye(
  ctx: CanvasRenderingContext2D,
  offsetX: number,
  eyeSign: 0 | 1,
  eyeW: number,
  eyeH: number,
  t: number
): StereoVideoGroundTruthRegion[] {
  const phase = (t / 12) * Math.PI * 2;

  // Textured backdrop & floor horizon
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -4 : 0) - 16,
    0,
    eyeW + 32,
    140,
    [32, 44, 64],
    [52, 68, 96],
    [78, 98, 134],
    13
  );
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -9 : 0) - 16,
    140,
    eyeW + 32,
    eyeH - 140,
    [44, 54, 68],
    [68, 82, 102],
    [98, 116, 142],
    19
  );

  // Obstacle 1: Approaching Cart/Vehicle on the left-center (d = 10px .. 34px)
  const cartProgress = 0.5 - 0.5 * Math.cos(phase); // 0 -> 1 -> 0
  const cartDisp = Math.round(10 + cartProgress * 24);
  const cartW = Math.round(88 + cartProgress * 52);
  const cartH = Math.round(74 + cartProgress * 46);
  const cartX = Math.round(62 + cartProgress * 28);
  const cartY = Math.round(92 + cartProgress * 42);
  drawRichTexturedRect(
    ctx,
    offsetX + cartX + (eyeSign === 1 ? -cartDisp : 0),
    cartY,
    cartW,
    cartH,
    [42, 128, 168],
    [84, 178, 222],
    [144, 218, 250],
    41,
    `CART d=${cartDisp}px`
  );

  // Obstacle 2: Pedestrian / Upright obstacle approaching on right-center (d = 12px .. 36px)
  const pedProgress = 0.5 + 0.5 * Math.sin(phase);
  const pedDisp = Math.round(12 + pedProgress * 24);
  const pedW = Math.round(58 + pedProgress * 34);
  const pedH = Math.round(115 + pedProgress * 55);
  const pedX = Math.round(295 - pedProgress * 36);
  const pedY = Math.round(68 + pedProgress * 24);
  drawRichTexturedRect(
    ctx,
    offsetX + pedX + (eyeSign === 1 ? -pedDisp : 0),
    pedY,
    pedW,
    pedH,
    [198, 82, 46],
    [244, 138, 86],
    [254, 192, 138],
    59,
    `PED d=${pedDisp}px`
  );

  // Obstacle 3: Crossing lateral hazard in mid-foreground (d = 22px)
  const crossDisp = 22;
  const crossX = Math.round(185 + Math.sin(phase * 1.5) * 85);
  const crossY = 158;
  const crossW = 86;
  const crossH = 78;
  drawRichTexturedRect(
    ctx,
    offsetX + crossX + (eyeSign === 1 ? -crossDisp : 0),
    crossY,
    crossW,
    crossH,
    [48, 148, 96],
    [94, 202, 144],
    [152, 238, 192],
    73,
    `HAZARD ${crossDisp}px`
  );

  return [
    {
      id: 'approaching_pedestrian',
      label: '接近歩行者ターゲット',
      disparityPx: pedDisp,
      normX: (pedX + pedW * 0.5) / eyeW,
      normY: (pedY + pedH * 0.5) / eyeH,
    },
    {
      id: 'approaching_cart',
      label: '接近ビークル',
      disparityPx: cartDisp,
      normX: (cartX + cartW * 0.5) / eyeW,
      normY: (cartY + cartH * 0.5) / eyeH,
    },
    {
      id: 'crossing_hazard',
      label: '横断障害物',
      disparityPx: crossDisp,
      normX: (crossX + crossW * 0.5) / eyeW,
      normY: (crossY + crossH * 0.5) / eyeH,
    },
  ];
}

function drawOutdoorDriveEye(
  ctx: CanvasRenderingContext2D,
  offsetX: number,
  eyeSign: 0 | 1,
  eyeW: number,
  eyeH: number,
  t: number
): StereoVideoGroundTruthRegion[] {
  const phase = (t / 15) * Math.PI * 2;
  const scroll = Math.round(t * 48);

  // Sky & distant cityscape (d = 3px)
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -3 : 0) - 16,
    0,
    eyeW + 32,
    115,
    [58, 86, 124],
    [82, 114, 156],
    [118, 152, 194],
    101
  );

  // Roadside buildings left & right (d = 9px)
  drawRichTexturedRect(
    ctx,
    offsetX + 12 + (eyeSign === 1 ? -9 : 0),
    24,
    110,
    115,
    [76, 84, 98],
    [104, 114, 132],
    [138, 148, 168],
    107,
    undefined,
    scroll >> 1
  );
  drawRichTexturedRect(
    ctx,
    offsetX + 358 + (eyeSign === 1 ? -9 : 0),
    20,
    110,
    119,
    [72, 82, 94],
    [98, 112, 128],
    [132, 146, 164],
    109,
    undefined,
    scroll >> 1
  );

  // Road surface in 3 stepped depth bands (far d=8px, mid d=16px, near d=26px)
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -8 : 0) - 24,
    115,
    eyeW + 48,
    46,
    [54, 60, 70],
    [78, 86, 98],
    [108, 118, 134],
    113,
    undefined,
    scroll
  );
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -16 : 0) - 28,
    161,
    eyeW + 56,
    52,
    [48, 54, 64],
    [74, 82, 96],
    [112, 122, 138],
    127,
    undefined,
    scroll * 2
  );
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -26 : 0) - 36,
    213,
    eyeW + 72,
    57,
    [42, 48, 58],
    [68, 78, 92],
    [116, 128, 146],
    131,
    undefined,
    scroll * 3
  );

  // Leading vehicle in center lane (d oscillates 16..28px)
  const leadCarDisp = Math.round(22 + Math.sin(phase) * 6);
  const leadCarW = Math.round(112 + Math.sin(phase) * 18);
  const leadCarH = Math.round(82 + Math.sin(phase) * 12);
  const leadCarX = Math.round(184 + Math.cos(phase * 0.7) * 18);
  const leadCarY = Math.round(108 + Math.sin(phase) * 10);
  drawRichTexturedRect(
    ctx,
    offsetX + leadCarX + (eyeSign === 1 ? -leadCarDisp : 0),
    leadCarY,
    leadCarW,
    leadCarH,
    [36, 112, 164],
    [76, 162, 218],
    [138, 208, 248],
    137,
    `VEHICLE ${leadCarDisp}px`
  );

  // Roadside sign / pole on right foreground (d = 32px)
  const signDisp = 32;
  const signX = Math.round(342 - ((t * 28) % 45));
  const signY = 86;
  drawRichTexturedRect(
    ctx,
    offsetX + signX + (eyeSign === 1 ? -signDisp : 0),
    signY,
    74,
    118,
    [182, 96, 38],
    [232, 148, 76],
    [252, 198, 128],
    149,
    `SIGN ${signDisp}px`
  );

  return [
    {
      id: 'lead_vehicle',
      label: '先行車両',
      disparityPx: leadCarDisp,
      normX: (leadCarX + leadCarW * 0.5) / eyeW,
      normY: (leadCarY + leadCarH * 0.5) / eyeH,
    },
    {
      id: 'roadside_sign',
      label: '路側標識',
      disparityPx: signDisp,
      normX: (signX + 37) / eyeW,
      normY: (signY + 59) / eyeH,
    },
  ];
}

function drawNatureParkEye(
  ctx: CanvasRenderingContext2D,
  offsetX: number,
  eyeSign: 0 | 1,
  eyeW: number,
  eyeH: number,
  t: number
): StereoVideoGroundTruthRegion[] {
  const pan = Math.round(Math.sin((t / 14) * Math.PI * 2) * 24);
  const flow = Math.round(t * 36);

  // Distant forest canopy (d = 5px)
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -5 : 0) - 24,
    0,
    eyeW + 48,
    135,
    [34, 72, 54],
    [54, 108, 78],
    [86, 148, 108],
    163,
    undefined,
    pan
  );

  // Flowing stream in lower half (d = 15px)
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -15 : 0) - 24,
    135,
    eyeW + 48,
    135,
    [34, 88, 118],
    [64, 138, 174],
    [118, 194, 224],
    173,
    undefined,
    flow
  );

  // Midground boulder formation (d = 19px)
  const rockDisp = 19;
  const rockX = 195 + pan;
  const rockY = 92;
  drawRichTexturedRect(
    ctx,
    offsetX + rockX + (eyeSign === 1 ? -rockDisp : 0),
    rockY,
    136,
    108,
    [88, 92, 98],
    [126, 132, 142],
    [168, 176, 186],
    181,
    `BOULDER ${rockDisp}px`
  );

  // Foreground tree trunk & foliage on left (d = 31px)
  const treeDisp = 31;
  const treeX = 48 + Math.round(pan * 1.4);
  const treeY = 28;
  drawRichTexturedRect(
    ctx,
    offsetX + treeX + (eyeSign === 1 ? -treeDisp : 0),
    treeY,
    98,
    212,
    [96, 68, 44],
    [142, 104, 68],
    [186, 144, 98],
    191,
    `TREE ${treeDisp}px`
  );

  return [
    {
      id: 'foreground_tree',
      label: '前景樹木',
      disparityPx: treeDisp,
      normX: (treeX + 49) / eyeW,
      normY: (treeY + 106) / eyeH,
    },
    {
      id: 'midground_boulder',
      label: '中景岩場',
      disparityPx: rockDisp,
      normX: (rockX + 68) / eyeW,
      normY: (rockY + 54) / eyeH,
    },
  ];
}

function drawTabletopLabEye(
  ctx: CanvasRenderingContext2D,
  offsetX: number,
  eyeSign: 0 | 1,
  eyeW: number,
  eyeH: number,
  t: number
): StereoVideoGroundTruthRegion[] {
  const angle = (t / 10) * Math.PI * 2;

  // Lab wall backdrop (d = 6px)
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -6 : 0) - 16,
    0,
    eyeW + 32,
    150,
    [38, 44, 56],
    [58, 68, 84],
    [84, 96, 118],
    211
  );

  // Tabletop mat surface (d = 16px)
  drawRichTexturedRect(
    ctx,
    offsetX + (eyeSign === 1 ? -16 : 0) - 24,
    150,
    eyeW + 48,
    120,
    [46, 62, 74],
    [72, 96, 114],
    [104, 136, 158],
    223
  );

  // Two objects orbiting on a turntable: as one rotates to the front (d ~ 34px), the other rotates to the back (d ~ 14px)
  const items = [
    {
      id: 'turntable_box_a',
      name: 'BOX-A',
      theta: angle,
      w: 104,
      h: 108,
      base: [194, 78, 52] as const,
      acc: [242, 134, 92] as const,
      hi: [254, 192, 148] as const,
      seed: 229,
    },
    {
      id: 'turntable_cyl_b',
      name: 'CYL-B',
      theta: angle + Math.PI,
      w: 96,
      h: 122,
      base: [42, 138, 116] as const,
      acc: [86, 194, 166] as const,
      hi: [146, 236, 208] as const,
      seed: 239,
    },
  ].sort((a, b) => Math.sin(a.theta) - Math.sin(b.theta));

  const regions: StereoVideoGroundTruthRegion[] = [];

  for (const item of items) {
    const depthFactor = 0.5 + 0.5 * Math.sin(item.theta); // 0 (back) .. 1 (front)
    const disp = Math.round(14 + depthFactor * 20); // 14px .. 34px
    const cx = Math.round(eyeW * 0.5 + Math.cos(item.theta) * 108);
    const cy = Math.round(132 + Math.sin(item.theta) * 26);
    const x = Math.round(cx - item.w * 0.5);
    const y = Math.round(cy - item.h * 0.5);

    drawRichTexturedRect(
      ctx,
      offsetX + x + (eyeSign === 1 ? -disp : 0),
      y,
      item.w,
      item.h,
      item.base,
      item.acc,
      item.hi,
      item.seed,
      `${item.name} ${disp}px`
    );

    regions.push({
      id: item.id,
      label: item.name,
      disparityPx: disp,
      normX: cx / eyeW,
      normY: cy / eyeH,
    });
  }

  return regions;
}

export function renderStereoVideoPresetFrame(
  presetId: StereoVideoPresetId,
  timeSec: number,
  targetCanvas?: HTMLCanvasElement
): StereoVideoFrameRenderResult {
  const meta = STEREO_VIDEO_PRESETS[presetId] ?? STEREO_VIDEO_PRESETS.synth_moving_blocks;
  const eyeW = meta.eyeWidth;
  const eyeH = meta.eyeHeight;
  const clampedTime = clampVideoTime(timeSec, meta.durationSec, true);
  const frameIndex = Math.floor(clampedTime * meta.fps + 1e-4);
  const quantizedTime = frameIndex / meta.fps;

  const canvas = targetCanvas ?? document.createElement('canvas');
  if (canvas.width !== eyeW * 2 || canvas.height !== eyeH) {
    canvas.width = eyeW * 2;
    canvas.height = eyeH;
  }

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return {
      canvas,
      eyeWidth: eyeW,
      eyeHeight: eyeH,
      frameIndex,
      timeSec: quantizedTime,
      regions: [],
    };
  }

  let leftRegions: StereoVideoGroundTruthRegion[] = [];

  for (const eyeSign of [0, 1] as const) {
    const offsetX = eyeSign === 0 ? 0 : eyeW;
    ctx.save();
    ctx.beginPath();
    ctx.rect(offsetX, 0, eyeW, eyeH);
    ctx.clip();

    let regions: StereoVideoGroundTruthRegion[];
    switch (presetId) {
      case 'synth_corridor':
        regions = drawCorridorEye(ctx, offsetX, eyeSign, eyeW, eyeH, quantizedTime);
        break;
      case 'synth_obstacles':
        regions = drawObstaclesEye(ctx, offsetX, eyeSign, eyeW, eyeH, quantizedTime);
        break;
      case 'free_outdoor_drive':
        regions = drawOutdoorDriveEye(ctx, offsetX, eyeSign, eyeW, eyeH, quantizedTime);
        break;
      case 'free_nature_park':
        regions = drawNatureParkEye(ctx, offsetX, eyeSign, eyeW, eyeH, quantizedTime);
        break;
      case 'free_tabletop_lab':
        regions = drawTabletopLabEye(ctx, offsetX, eyeSign, eyeW, eyeH, quantizedTime);
        break;
      case 'synth_moving_blocks':
      default:
        regions = drawMovingBlocksEye(ctx, offsetX, eyeSign, eyeW, eyeH, quantizedTime);
        break;
    }

    if (eyeSign === 0) {
      leftRegions = regions;
    }
    ctx.restore();
  }

  return {
    canvas,
    eyeWidth: eyeW,
    eyeHeight: eyeH,
    frameIndex,
    timeSec: quantizedTime,
    regions: leftRegions,
  };
}

export function getStereoVideoReferenceRegions(
  presetId: StereoVideoPresetId,
  timeSec: number
): StereoVideoGroundTruthRegion[] {
  switch (presetId) {
    case 'free_outdoor_drive':
      return [
        {
          id: 'wuppertal_truss_post',
          label: '鉄橋トラス支柱 (近景)',
          disparityPx: 9,
          normX: 0.24,
          normY: 0.62,
        },
        {
          id: 'wuppertal_train',
          label: '懸垂式モノレール車両 (中景)',
          disparityPx: 6,
          normX: 0.48,
          normY: 0.45,
        },
        {
          id: 'wuppertal_buildings',
          label: '背景ビル群 (遠景)',
          disparityPx: 3,
          normX: 0.78,
          normY: 0.26,
        },
      ];
    case 'free_nature_park':
      return [
        {
          id: 'latzen_mast',
          label: '手前の架線柱 279-0 (近景)',
          disparityPx: 10,
          normX: 0.65,
          normY: 0.56,
        },
        {
          id: 'latzen_freight_train',
          label: '通過する自動車運搬列車 (中景)',
          disparityPx: 6,
          normX: 0.28,
          normY: 0.58,
        },
        {
          id: 'latzen_trees',
          label: '背景樹林 (遠景)',
          disparityPx: 3,
          normX: 0.84,
          normY: 0.52,
        },
      ];
    case 'free_tabletop_lab':
      if (timeSec >= 4.5) {
        return [
          {
            id: 'ice_monitor',
            label: 'ICE 1134 車内速度表示盤 (近景)',
            disparityPx: 14,
            normX: 0.36,
            normY: 0.36,
          },
          {
            id: 'ice_arch',
            label: '客室天井アーチ (中景)',
            disparityPx: 8,
            normX: 0.74,
            normY: 0.58,
          },
          {
            id: 'ice_corridor',
            label: '客室奥の通路 (遠景)',
            disparityPx: 3,
            normX: 0.50,
            normY: 0.82,
          },
        ];
      }
      return [
        {
          id: 'segway_pedestrian_near',
          label: '広場手前の歩行者 (近景)',
          disparityPx: 9,
          normX: 0.82,
          normY: 0.58,
        },
        {
          id: 'segway_riders_mid',
          label: '移動するセグウェイ群 (中景)',
          disparityPx: 6,
          normX: 0.44,
          normY: 0.52,
        },
        {
          id: 'segway_facade_far',
          label: '建築ファサード (遠景)',
          disparityPx: 3,
          normX: 0.65,
          normY: 0.25,
        },
      ];
    default:
      return [];
  }
}

