import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  Camera,
  AlertCircle,
  SwitchCamera,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Repeat,
  Film,
} from 'lucide-react';
import {
  DetectionFrame,
  ModelState,
  DetectedObject,
  RunnableModelType,
  CameraFacingMode,
  RenderOptions,
  TTSConfig,
  StereoDepthMapData,
  StereoDepthProbeResult,
  StereoPinnedProbe,
  StereoColormap,
} from '../types/vision';
import { speechService } from '../services/speechService';
import { VisionEngine, type ExtraInferenceOptions } from '../services/visionEngine';
import { JobLock, startFrameScheduler } from '../services/frameScheduler';
import { HUD_INTERVAL_MS, PerformanceMetrics, type PerformanceSnapshot } from '../services/performanceMetrics';
import { PipelineMetrics, PipelineProbe, telemetry, type InferenceMeasurement } from '../services/pipelineMetrics';
import { readPipelineSnapshot, type PipelineSnapshotView } from '../services/pipelineSnapshotView';
import { modelMetadata } from '../services/modelRegistry';
import { useCameraStream } from '../hooks/useCameraStream';
import { drawCanvasFrame } from '../services/canvasRenderer';
import { PipelineDetails } from './PipelineDetails';
import { clientPointToVideo } from '../services/videoCoordinates';
import { confidenceDescription } from '../services/confidenceDescription';
import { disparityToDepthMeters, getColormapLut } from '../services/stereoDepthService';
import {
  STEREO_VIDEO_PRESETS,
  clampVideoTime,
  formatVideoTimecode,
  getStereoVideoReferenceRegions,
  renderStereoVideoPresetFrame,
  stepVideoTime,
  type StereoPlaybackRate,
  type StereoVideoGroundTruthRegion,
  type StereoVideoPresetId,
} from '../services/stereoVideoService';

export interface StillImageSource {
  id: string;
  label: string;
  canvas: HTMLCanvasElement;
}

export type StereoVideoSource =
  | {
      kind: 'preset';
      presetId: StereoVideoPresetId;
      id: string;
      label: string;
    }
  | {
      kind: 'uploaded-file';
      id: string;
      label: string;
      fileUrl: string;
      fileName: string;
    };

function buildLegendGradientCss(colormap: StereoColormap): string {
  const lut = getColormapLut(colormap);
  const stops: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const idx = Math.min(255, Math.round((i / 8) * 255)) * 3;
    stops.push(`rgb(${lut[idx]}, ${lut[idx + 1]}, ${lut[idx + 2]}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

interface VisionCanvasProps {
  activeModel: RunnableModelType;
  engine: VisionEngine;
  modelState: ModelState;
  facingMode: CameraFacingMode;
  onToggleFacingMode: () => void;
  renderOptions: RenderOptions;
  inferenceParams?: ExtraInferenceOptions;
  ttsConfig: TTSConfig;
  isCameraActive: boolean;
  onCameraStart: () => void;
  onCameraStop: () => void;
  onSelectObject: (obj: DetectedObject) => void;
  selectedObjectId?: string;
  onFrame: (frame: DetectionFrame | null) => void;
  stillImageSource?: StillImageSource | null;
  onClearStillImage?: () => void;
  stereoVideoSource?: StereoVideoSource | null;
  onClearStereoVideo?: () => void;
  onStereoVideoRegions?: (regions: StereoVideoGroundTruthRegion[]) => void;
  stereoHoverProbe?: StereoDepthProbeResult | null;
  onStereoHoverProbe?: (probe: StereoDepthProbeResult | null) => void;
  stereoPinnedProbes?: StereoPinnedProbe[];
  onPinStereoProbe?: (probe: StereoDepthProbeResult) => void;
}

const PLAYBACK_RATES: readonly StereoPlaybackRate[] = [0.25, 0.5, 1.0, 1.5];

export const VisionCanvas: React.FC<VisionCanvasProps> = ({
  activeModel,
  engine,
  modelState,
  facingMode,
  onToggleFacingMode,
  renderOptions,
  inferenceParams,
  ttsConfig,
  isCameraActive,
  onCameraStart,
  onCameraStop,
  onSelectObject,
  selectedObjectId,
  onFrame,
  stillImageSource = null,
  onClearStillImage,
  stereoVideoSource = null,
  onClearStereoVideo,
  onStereoVideoRegions,
  stereoHoverProbe = null,
  onStereoHoverProbe,
  stereoPinnedProbes = [],
  onPinStereoProbe,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const cameraMode = activeModel === 'stereo-depth' ? 'stereo-sbs' : 'standard';
  const { cameraError, clearError, cameraReady, cameraRequestId } = useCameraStream(
    videoRef,
    isCameraActive && !stillImageSource && !stereoVideoSource,
    facingMode,
    onCameraStop,
    cameraMode
  );
  const [clickRipple, setClickRipple] = useState<{ x: number; y: number; label: string } | null>(null);
  const modelReady = modelState.status === 'ready' && modelState.model === activeModel;
  const requestId = modelState.requestId;
  const threshold = renderOptions.confidenceThreshold;

  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stereoVideoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const uploadedVideoElRef = useRef<HTMLVideoElement | null>(null);
  const latestFrameRef = useRef<DetectionFrame | null>(null);
  // Pointer inspection follows the frame actually copied to the Canvas. HUD
  // publication is throttled and can still describe an older accepted result.
  const drawnFrameRef = useRef<DetectionFrame | null>(null);
  const lockRef = useRef(new JobLock());
  const schedulerGeneration = useRef(0);
  const previousScheduler = useRef<{
    engine: VisionEngine;
    activeModel: RunnableModelType;
    requestId: number;
    cameraRequestId: number;
    threshold: number;
    retry: number;
  } | null>(null);
  const [inferenceError, setInferenceError] = useState(false);
  const [fault, setFault] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [stats, setStats] = useState<PerformanceSnapshot>({
    inferenceFps: null,
    drawFps: null,
    processingMs: null,
    stalled: false,
  });
  const [pipelineStats, setPipelineStats] = useState<PipelineSnapshotView | null>(null);

  // Stereo Video Transport State
  const [videoPlaying, setVideoPlaying] = useState<boolean>(true);
  const [videoCurrentTime, setVideoCurrentTime] = useState<number>(0);
  const [videoDuration, setVideoDuration] = useState<number>(12);
  const [videoFps, setVideoFps] = useState<number>(30);
  const [videoPlaybackRate, setVideoPlaybackRate] = useState<StereoPlaybackRate>(1.0);
  const [videoLoop, setVideoLoop] = useState<boolean>(true);
  const [uploadedVideoReady, setUploadedVideoReady] = useState<boolean>(false);
  const [presetVideoSettled, setPresetVideoSettled] = useState<boolean>(true);
  const [uploadedVideoError, setUploadedVideoError] = useState<string | null>(null);
  const [customSourceInferring, setCustomSourceInferring] = useState<boolean>(false);
  const videoTimeRef = useRef<number>(0);
  const stereoVideoInFlightRef = useRef<boolean>(false);
  const pendingStereoVideoEvalRef = useRef<number | null>(null);
  const stereoVideoMetricsRef = useRef<PerformanceMetrics>(new PerformanceMetrics());
  const videoTransportRef = useRef({
    videoDuration,
    videoFps,
    videoPlaybackRate,
    videoLoop,
    videoPlaying,
  });
  useLayoutEffect(() => {
    videoTransportRef.current = {
      videoDuration,
      videoFps,
      videoPlaybackRate,
      videoLoop,
      videoPlaying,
    };
  });

  const stereoEstimationKey =
    activeModel === 'stereo-depth'
      ? `${inferenceParams?.swapLeftRight ?? false}:${inferenceParams?.cameraRotate180 ?? true}:${inferenceParams?.maxDisparity ?? 64}:${inferenceParams?.sadWindowSize ?? 7}`
      : JSON.stringify(inferenceParams ?? {});

  const latest = useRef({
    onFrame,
    onStereoHoverProbe,
    onStereoVideoRegions,
    renderOptions,
    inferenceParams,
    ttsConfig,
    modelReady,
    requestId,
    cameraRequestId,
    isCameraActive,
    cameraReady,
    activeModel,
    stillImageSource,
    stereoVideoSource,
  });
  useLayoutEffect(() => {
    latest.current = {
      onFrame,
      onStereoHoverProbe,
      onStereoVideoRegions,
      renderOptions,
      inferenceParams,
      ttsConfig,
      modelReady,
      requestId,
      cameraRequestId,
      isCameraActive,
      cameraReady,
      activeModel,
      stillImageSource,
      stereoVideoSource,
    };
  });
  const drawRef = useRef<
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      objects: DetectedObject[],
      semanticMask?: HTMLCanvasElement,
      depthMap?: StereoDepthMapData
    ) => void
  >(() => {});
  const [stereoSummary, setStereoSummary] = useState<{
    colormap: StereoColormap;
    scaleMinDisparity: number;
    scaleMaxDisparity: number;
    medianDisparity: number;
  } | null>(null);

  // Initialize transport metadata when stereoVideoSource changes
  useEffect(() => {
    if (!stereoVideoSource) {
      setUploadedVideoReady(false);
      setPresetVideoSettled(true);
      setUploadedVideoError(null);
      if (uploadedVideoElRef.current) {
        uploadedVideoElRef.current.pause();
        uploadedVideoElRef.current.removeAttribute('src');
        uploadedVideoElRef.current = null;
      }
      latest.current.onStereoVideoRegions?.([]);
      return;
    }

    videoTimeRef.current = 0;
    setVideoCurrentTime(0);
    setVideoPlaying(true);
    setUploadedVideoError(null);

    const presetMeta =
      stereoVideoSource.kind === 'preset'
        ? STEREO_VIDEO_PRESETS[stereoVideoSource.presetId]
        : null;

    if (presetMeta) {
      setVideoDuration(presetMeta.durationSec);
      setVideoFps(presetMeta.fps);
      if (!presetMeta.videoUrl) {
        setUploadedVideoReady(false);
        setPresetVideoSettled(true);
        return;
      }
    } else {
      setVideoFps(30);
    }

    const mediaSrc =
      stereoVideoSource.kind === 'uploaded-file'
        ? stereoVideoSource.fileUrl
        : presetMeta?.videoUrl ?? '';
    if (!mediaSrc) {
      setUploadedVideoReady(false);
      setPresetVideoSettled(true);
      return;
    }

    // Offscreen HTMLVideoElement for real-world filmed preset WebM or uploaded video file
    setUploadedVideoReady(false);
    const isPresetWithUrl = stereoVideoSource.kind === 'preset' && Boolean(presetMeta?.videoUrl);
    setPresetVideoSettled(!isPresetWithUrl);

    const v = document.createElement('video');
    v.muted = true;
    v.playsInline = true;
    v.preload = 'auto';
    v.crossOrigin = 'anonymous';
    v.src = mediaSrc;
    uploadedVideoElRef.current = v;

    const fallbackTimer = isPresetWithUrl
      ? setTimeout(() => {
          if (uploadedVideoElRef.current === v) {
            setPresetVideoSettled(true);
          }
        }, 200)
      : null;

    const handleLoadedMetadata = () => {
      if (uploadedVideoElRef.current !== v) return;
      const dur =
        Number.isFinite(v.duration) && v.duration > 0
          ? v.duration
          : presetMeta?.durationSec ?? 10;
      setVideoDuration(dur);
    };
    const handleLoadedData = () => {
      if (uploadedVideoElRef.current !== v) return;
      if (fallbackTimer !== null) clearTimeout(fallbackTimer);
      setUploadedVideoReady(true);
      setPresetVideoSettled(true);
    };
    const handleError = () => {
      if (uploadedVideoElRef.current !== v) return;
      if (fallbackTimer !== null) clearTimeout(fallbackTimer);
      setPresetVideoSettled(true);
      if (stereoVideoSource.kind === 'uploaded-file') {
        setUploadedVideoError(
          'ステレオ動画ファイルのデコードに失敗しました。MP4 / WebM 形式をお試しください。'
        );
      }
    };

    v.addEventListener('loadedmetadata', handleLoadedMetadata);
    v.addEventListener('loadeddata', handleLoadedData);
    v.addEventListener('error', handleError);
    v.load();

    return () => {
      if (fallbackTimer !== null) clearTimeout(fallbackTimer);
      v.removeEventListener('loadedmetadata', handleLoadedMetadata);
      v.removeEventListener('loadeddata', handleLoadedData);
      v.removeEventListener('error', handleError);
      v.pause();
      v.removeAttribute('src');
      if (uploadedVideoElRef.current === v) {
        uploadedVideoElRef.current = null;
      }
    };
  }, [stereoVideoSource]);

  // Sync HTMLVideoElement playbackRate and loop without resetting inference effects
  useEffect(() => {
    const v = uploadedVideoElRef.current;
    if (!v || !uploadedVideoReady) return;
    v.playbackRate = videoPlaybackRate;
    v.loop = videoLoop;
  }, [uploadedVideoReady, videoPlaybackRate, videoLoop]);

  useLayoutEffect(() => {
    const generationOwner = schedulerGeneration;
    generationOwner.current++;
    latestFrameRef.current = null;
    drawnFrameRef.current = null;
    stereoVideoInFlightRef.current = false;
    pendingStereoVideoEvalRef.current = null;
    stereoVideoMetricsRef.current = new PerformanceMetrics();
    setCustomSourceInferring(false);
    engine.clearDetections();
    setClickRipple(null);
    setStereoSummary(null);
    latest.current.onStereoHoverProbe?.(null);
    setFault(null);
    setInferenceError(false);
    setPipelineStats(null);
    setStats({ inferenceFps: null, drawFps: null, processingMs: null, stalled: false });
    try {
      latest.current.onFrame(null);
    } catch {
      setFault('結果通知エラー');
    }
    const canvas = canvasRef.current;
    try {
      if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    } catch {
      setFault('描画エラー');
    }
    return () => {
      generationOwner.current++;
    };
  }, [
    activeModel,
    requestId,
    isCameraActive,
    facingMode,
    cameraReady,
    cameraRequestId,
    engine,
    retry,
    threshold,
    stillImageSource,
    stereoVideoSource,
  ]);

  useLayoutEffect(() => {
    if (!fault) return;
    setPipelineStats(null);
    schedulerGeneration.current++;
    latestFrameRef.current = null;
    drawnFrameRef.current = null;
    stereoVideoInFlightRef.current = false;
    pendingStereoVideoEvalRef.current = null;
    setCustomSourceInferring(false);
    engine.clearDetections();
    const canvas = canvasRef.current;
    if (canvas) {
      const width = canvas.width;
      canvas.width = width;
    }
    try {
      latest.current.onFrame(null);
    } catch {
      /* The stopped notification error is already visible. */
    }
  }, [fault, engine]);

  // Render helper for Still Image & Stereo Video frames onto canvasRef
  const renderCustomSourceFrame = useCallback(
    (sourceCanvas: HTMLCanvasElement, frame: DetectionFrame | null) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const isStereoModel = latest.current.activeModel === 'stereo-depth';
      const rawW = sourceCanvas.width || 640;
      const rawH = sourceCanvas.height || 480;
      const singleEyeW = isStereoModel && rawW >= 2 ? Math.floor(rawW / 2) : rawW;
      const w = frame?.depthMap
        ? frame.depthMap.viewMode === 'split-3pane'
          ? frame.depthMap.sourceEyeWidth * 3
          : frame.depthMap.sourceEyeWidth
        : singleEyeW;
      const h = frame?.depthMap ? frame.depthMap.sourceEyeHeight : rawH;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);
      if (frame?.depthMap) {
        drawRef.current(ctx, w, h, frame.objects, frame.semanticMask, frame.depthMap);
      } else if (isStereoModel && rawW >= 2) {
        const swap = Boolean(latest.current.inferenceParams?.swapLeftRight);
        ctx.drawImage(sourceCanvas, swap ? singleEyeW : 0, 0, singleEyeW, rawH, 0, 0, w, h);
      } else {
        ctx.drawImage(sourceCanvas, 0, 0, w, h);
        if (frame) {
          drawRef.current(ctx, w, h, frame.objects, frame.semanticMask);
        }
      }
    },
    []
  );

  // Still Image Inference Effect
  useEffect(() => {
    if (!stillImageSource || stereoVideoSource || !modelReady || fault) return;
    const generation = schedulerGeneration.current;
    let cancelled = false;

    // Draw the source image immediately while inference runs
    renderCustomSourceFrame(stillImageSource.canvas, null);
    setCustomSourceInferring(true);

    void (async () => {
      const start = performance.now();
      try {
        const effectiveInferenceParams =
          activeModel === 'stereo-depth'
            ? { ...latest.current.inferenceParams, rotate180: false }
            : latest.current.inferenceParams;
        const res = await Promise.resolve(
          engine.detectVideoFrame(
            stillImageSource.canvas,
            activeModel,
            latest.current.renderOptions.confidenceThreshold,
            undefined,
            effectiveInferenceParams
          )
        );
        const end = performance.now();
        const state = engine.getState();
        if (
          cancelled ||
          generation !== schedulerGeneration.current ||
          state.status !== 'ready' ||
          state.requestId !== requestId ||
          state.model !== activeModel ||
          res.status !== 'ready' ||
          res.requestId !== requestId ||
          res.model !== activeModel
        ) {
          return;
        }
        setInferenceError(false);
        setStats({
          inferenceFps: end > start ? 1000 / (end - start) : null,
          drawFps: null,
          processingMs: end - start,
          stalled: false,
        });
        latestFrameRef.current = res;
        drawnFrameRef.current = res;
        if (res.depthMap) {
          setStereoSummary({
            colormap: res.depthMap.colormap,
            scaleMinDisparity: res.depthMap.scaleMinDisparity,
            scaleMaxDisparity: res.depthMap.scaleMaxDisparity,
            medianDisparity: res.depthMap.medianDisparity,
          });
        }
        renderCustomSourceFrame(stillImageSource.canvas, res);
        latest.current.onFrame(res);
      } catch (cause) {
        if (cancelled || generation !== schedulerGeneration.current) return;
        setInferenceError(true);
        console.warn('Still image inference failed', { cause });
      } finally {
        if (!cancelled && generation === schedulerGeneration.current) {
          setCustomSourceInferring(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    stillImageSource,
    stereoVideoSource,
    activeModel,
    engine,
    modelReady,
    requestId,
    fault,
    retry,
    threshold,
    stereoEstimationKey,
    renderCustomSourceFrame,
  ]);

  // Populate stereoVideoCanvasRef at given timeSec
  const prepareStereoVideoFrameCanvas = useCallback(
    (timeSec: number): HTMLCanvasElement | null => {
      if (!stereoVideoSource) return null;
      stereoVideoCanvasRef.current ??= document.createElement('canvas');
      const sbsCanvas = stereoVideoCanvasRef.current;

      const v = uploadedVideoElRef.current;
      const hasDecodedVideoFrame =
        Boolean(v) && uploadedVideoReady && v!.videoWidth >= 2 && v!.videoHeight >= 1;

      if (stereoVideoSource.kind === 'preset') {
        if (hasDecodedVideoFrame && v) {
          const presetMeta = STEREO_VIDEO_PRESETS[stereoVideoSource.presetId];
          const targetW = presetMeta ? presetMeta.eyeWidth * 2 : v.videoWidth;
          const targetH = presetMeta ? presetMeta.eyeHeight : v.videoHeight;
          if (sbsCanvas.width !== targetW || sbsCanvas.height !== targetH) {
            sbsCanvas.width = targetW;
            sbsCanvas.height = targetH;
          }
          const ctx = sbsCanvas.getContext('2d');
          ctx?.drawImage(v, 0, 0, targetW, targetH);
          const refRegions = getStereoVideoReferenceRegions(
            stereoVideoSource.presetId,
            timeSec
          );
          latest.current.onStereoVideoRegions?.(refRegions);
          return sbsCanvas;
        }
        const rendered = renderStereoVideoPresetFrame(
          stereoVideoSource.presetId,
          timeSec,
          sbsCanvas
        );
        latest.current.onStereoVideoRegions?.(rendered.regions);
        return rendered.canvas;
      }

      if (!v || !hasDecodedVideoFrame) {
        return null;
      }
      if (sbsCanvas.width !== v.videoWidth || sbsCanvas.height !== v.videoHeight) {
        sbsCanvas.width = v.videoWidth;
        sbsCanvas.height = v.videoHeight;
      }
      const ctx = sbsCanvas.getContext('2d');
      ctx?.drawImage(v, 0, 0, sbsCanvas.width, sbsCanvas.height);
      latest.current.onStereoVideoRegions?.([]);
      return sbsCanvas;
    },
    [stereoVideoSource, uploadedVideoReady]
  );

  // Shared Stereo Video Frame Evaluator (persists across Play/Pause transitions so in-flight
  // Worker inferences are never discarded when pausing, and coalesces rapid seeks into 1 pending slot)
  const evaluateStereoVideoFrame = useCallback(
    async (targetTime: number) => {
      if (!stereoVideoSource || !latest.current.modelReady || fault) return;
      if (stereoVideoSource.kind === 'uploaded-file' && !uploadedVideoReady) return;
      if (stereoVideoSource.kind === 'preset' && !presetVideoSettled) {
        const previewCanvas = prepareStereoVideoFrameCanvas(targetTime);
        if (previewCanvas && !drawnFrameRef.current) {
          renderCustomSourceFrame(previewCanvas, null);
        }
        return;
      }

      if (stereoVideoInFlightRef.current) {
        pendingStereoVideoEvalRef.current = targetTime;
        return;
      }

      const generation = schedulerGeneration.current;
      const currentReqId = latest.current.requestId;
      const currentModel = latest.current.activeModel;

      if (
        !videoTransportRef.current.videoPlaying &&
        uploadedVideoReady &&
        uploadedVideoElRef.current
      ) {
        const v = uploadedVideoElRef.current;
        if (Math.abs(v.currentTime - targetTime) > 1e-3) {
          try {
            v.currentTime = targetTime;
          } catch {
            // ignore seek error on unbuffered range
          }
        }
      }

      const sbsCanvas = prepareStereoVideoFrameCanvas(targetTime);
      if (!sbsCanvas) return;

      if (!drawnFrameRef.current) {
        renderCustomSourceFrame(sbsCanvas, null);
      }

      stereoVideoInFlightRef.current = true;
      setCustomSourceInferring(true);
      const metrics = stereoVideoMetricsRef.current;
      metrics.input();
      const start = performance.now();

      try {
        const effectiveInferenceParams =
          currentModel === 'stereo-depth'
            ? { ...latest.current.inferenceParams, rotate180: false }
            : latest.current.inferenceParams;
        const res = await Promise.resolve(
          engine.detectVideoFrame(
            sbsCanvas,
            currentModel,
            latest.current.renderOptions.confidenceThreshold,
            undefined,
            effectiveInferenceParams
          )
        );
        const end = performance.now();
        const state = engine.getState();
        if (
          generation !== schedulerGeneration.current ||
          state.status !== 'ready' ||
          state.requestId !== currentReqId ||
          state.model !== currentModel ||
          res.status !== 'ready' ||
          res.requestId !== currentReqId ||
          res.model !== currentModel
        ) {
          return;
        }
        setInferenceError(false);
        metrics.inferenceCompleted(start, end);
        metrics.drawCompleted();
        const snap = metrics.snapshot();
        setStats({
          inferenceFps: snap.inferenceFps ?? (end > start ? 1000 / (end - start) : null),
          drawFps: snap.drawFps ?? (end > start ? 1000 / (end - start) : null),
          processingMs: snap.processingMs ?? end - start,
          stalled: false,
        });
        latestFrameRef.current = res;
        drawnFrameRef.current = res;
        if (res.depthMap) {
          setStereoSummary({
            colormap: res.depthMap.colormap,
            scaleMinDisparity: res.depthMap.scaleMinDisparity,
            scaleMaxDisparity: res.depthMap.scaleMaxDisparity,
            medianDisparity: res.depthMap.medianDisparity,
          });
        }
        const displayCanvas =
          prepareStereoVideoFrameCanvas(videoTimeRef.current) ?? sbsCanvas;
        renderCustomSourceFrame(displayCanvas, res);
        latest.current.onFrame(res);
        if (
          !videoTransportRef.current.videoPlaying &&
          Math.abs(videoTimeRef.current - targetTime) > 1e-3 &&
          pendingStereoVideoEvalRef.current === null
        ) {
          pendingStereoVideoEvalRef.current = videoTimeRef.current;
        }
      } catch (cause) {
        if (generation !== schedulerGeneration.current) return;
        setInferenceError(true);
        console.warn('Stereo video frame inference failed', { cause });
      } finally {
        if (generation === schedulerGeneration.current) {
          stereoVideoInFlightRef.current = false;
          const nextPendingTime = pendingStereoVideoEvalRef.current;
          pendingStereoVideoEvalRef.current = null;
          if (nextPendingTime !== null) {
            void evaluateStereoVideoFrame(nextPendingTime);
          } else {
            setCustomSourceInferring(false);
          }
        }
      }
    },
    [
      stereoVideoSource,
      uploadedVideoReady,
      presetVideoSettled,
      fault,
      engine,
      prepareStereoVideoFrameCanvas,
      renderCustomSourceFrame,
    ]
  );

  // Stereo Video Playback Loop Effect (active only while videoPlaying === true)
  useEffect(() => {
    if (!stereoVideoSource || !videoPlaying || !modelReady || fault) return;
    if (stereoVideoSource.kind === 'uploaded-file' && !uploadedVideoReady) return;
    if (stereoVideoSource.kind === 'preset' && !presetVideoSettled) {
      const preview = prepareStereoVideoFrameCanvas(videoTimeRef.current);
      if (preview && !drawnFrameRef.current) {
        renderCustomSourceFrame(preview, null);
      }
      return;
    }

    let cancelled = false;
    let rafId: number | null = null;
    let lastTickMs: number | null = null;
    let lastEvaluatedFrameIdx = -1;

    // Kick off initial frame evaluation
    void evaluateStereoVideoFrame(videoTimeRef.current);

    const uploadedVideo = uploadedVideoReady ? uploadedVideoElRef.current : null;
    if (uploadedVideo) {
      uploadedVideo.playbackRate = videoTransportRef.current.videoPlaybackRate;
      uploadedVideo.loop = videoTransportRef.current.videoLoop;
      void uploadedVideo.play().catch(() => {});
    }

    let lastUiTimeUpdateMs = -Infinity;

    const tick = (nowMs: number) => {
      if (cancelled) return;
      if (lastTickMs === null) {
        lastTickMs = nowMs;
      }
      const deltaSec = Math.max(0, Math.min(0.25, (nowMs - lastTickMs) / 1000));
      lastTickMs = nowMs;

      const {
        videoDuration: curDuration,
        videoFps: curFps,
        videoPlaybackRate: curRate,
        videoLoop: curLoop,
      } = videoTransportRef.current;

      let nextTime: number;
      if (uploadedVideo) {
        nextTime = uploadedVideo.currentTime;
        if (!curLoop && uploadedVideo.ended) {
          videoTimeRef.current = nextTime;
          setVideoCurrentTime(nextTime);
          setVideoPlaying(false);
          return;
        }
      } else {
        const rawNext = videoTimeRef.current + deltaSec * curRate;
        if (!curLoop && rawNext >= curDuration) {
          nextTime = curDuration;
          videoTimeRef.current = nextTime;
          setVideoCurrentTime(nextTime);
          void evaluateStereoVideoFrame(nextTime);
          setVideoPlaying(false);
          return;
        }
        nextTime = clampVideoTime(rawNext, curDuration, curLoop);
      }

      videoTimeRef.current = nextTime;
      if (nowMs - lastUiTimeUpdateMs >= 100) {
        lastUiTimeUpdateMs = nowMs;
        setVideoCurrentTime(nextTime);
      }
      const frameIdx = Math.floor(nextTime * curFps + 1e-4);
      if (frameIdx !== lastEvaluatedFrameIdx && !stereoVideoInFlightRef.current) {
        lastEvaluatedFrameIdx = frameIdx;
        void evaluateStereoVideoFrame(nextTime);
      } else if (stereoVideoInFlightRef.current && !drawnFrameRef.current) {
        // Keep single-eye video preview live while the very first heavy ONNX depth frame computes
        const currentSbs = prepareStereoVideoFrameCanvas(nextTime);
        if (currentSbs) {
          renderCustomSourceFrame(currentSbs, null);
        }
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      if (rafId !== null) cancelAnimationFrame(rafId);
      if (uploadedVideo) {
        uploadedVideo.pause();
      }
    };
  }, [
    stereoVideoSource,
    uploadedVideoReady,
    presetVideoSettled,
    videoPlaying,
    activeModel,
    engine,
    modelReady,
    requestId,
    fault,
    retry,
    threshold,
    stereoEstimationKey,
    evaluateStereoVideoFrame,
    prepareStereoVideoFrameCanvas,
    renderCustomSourceFrame,
  ]);

  // Paused Stereo Video Frame Evaluation Effect (active only while videoPlaying === false)
  useEffect(() => {
    if (!stereoVideoSource || videoPlaying || !modelReady || fault) return;
    if (stereoVideoSource.kind === 'uploaded-file' && !uploadedVideoReady) return;
    if (stereoVideoSource.kind === 'preset' && !presetVideoSettled) return;

    void evaluateStereoVideoFrame(videoCurrentTime);
  }, [
    stereoVideoSource,
    uploadedVideoReady,
    presetVideoSettled,
    videoPlaying,
    videoCurrentTime,
    activeModel,
    engine,
    modelReady,
    requestId,
    fault,
    retry,
    threshold,
    stereoEstimationKey,
    evaluateStereoVideoFrame,
  ]);

  // Redraw overlay on still image or stereo video when renderOptions, hover probe, or pinned probes change
  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = drawnFrameRef.current;
    const activeSourceCanvas = stereoVideoSource
      ? stereoVideoCanvasRef.current
      : stillImageSource?.canvas ?? null;
    if (!activeSourceCanvas || !canvas || !frame) return;
    renderCustomSourceFrame(activeSourceCanvas, frame);
  }, [
    stillImageSource,
    stereoVideoSource,
    renderOptions,
    selectedObjectId,
    clickRipple,
    stereoHoverProbe,
    stereoPinnedProbes,
    renderCustomSourceFrame,
  ]);

  // Immediately re-render cached depth map when stereo visualization options change (Live Camera, Still Image, or Stereo Video)
  useEffect(() => {
    const hasCustomSource = Boolean(stillImageSource || stereoVideoSource);
    if (
      activeModel !== 'stereo-depth' ||
      (!hasCustomSource && (!isCameraActive || !cameraReady)) ||
      !modelReady ||
      fault
    ) {
      return;
    }
    const frame = latestFrameRef.current;
    if (!frame?.depthMap) return;
    const updatedDepthMap = engine.rerenderStereoDepthOverlay(frame.depthMap, {
      colormap: inferenceParams?.colormap,
      viewMode: inferenceParams?.viewMode,
      autoNormalizeDisparity: inferenceParams?.autoNormalizeDisparity,
      minDisparityPx: inferenceParams?.minDisparityPx,
      maxDisparityPx: inferenceParams?.maxDisparityPx,
      swapLeftRight: frame.depthMap.swapLeftRight,
      baselineMm: inferenceParams?.baselineMm,
      focalLengthPx: inferenceParams?.focalLengthPx,
      showCalibrationDepth: inferenceParams?.showCalibrationDepth,
      confidenceThreshold: latest.current.renderOptions.confidenceThreshold,
    });
    if (!updatedDepthMap) return;
    const updatedFrame: DetectionFrame = { ...frame, depthMap: updatedDepthMap };
    latestFrameRef.current = updatedFrame;
    if (hasCustomSource) {
      drawnFrameRef.current = updatedFrame;
      const activeSourceCanvas = stereoVideoSource
        ? stereoVideoCanvasRef.current
        : stillImageSource?.canvas ?? null;
      if (activeSourceCanvas) {
        renderCustomSourceFrame(activeSourceCanvas, updatedFrame);
      }
    }
    setStereoSummary({
      colormap: updatedDepthMap.colormap,
      scaleMinDisparity: updatedDepthMap.scaleMinDisparity,
      scaleMaxDisparity: updatedDepthMap.scaleMaxDisparity,
      medianDisparity: updatedDepthMap.medianDisparity,
    });
    latest.current.onFrame(updatedFrame);
  }, [
    activeModel,
    stillImageSource,
    stereoVideoSource,
    isCameraActive,
    cameraReady,
    modelReady,
    fault,
    engine,
    renderCustomSourceFrame,
    inferenceParams?.colormap,
    inferenceParams?.viewMode,
    inferenceParams?.autoNormalizeDisparity,
    inferenceParams?.minDisparityPx,
    inferenceParams?.maxDisparityPx,
    inferenceParams?.baselineMm,
    inferenceParams?.focalLengthPx,
    inferenceParams?.showCalibrationDepth,
  ]);

  // Live Camera Scheduler Effect
  useEffect(() => {
    const video = videoRef.current;
    if (
      stillImageSource ||
      stereoVideoSource ||
      !video ||
      !isCameraActive ||
      !cameraReady ||
      !modelReady ||
      fault
    ) {
      return;
    }
    const previous = previousScheduler.current;
    // A threshold change invalidates the attempted old result, but is not a new
    // camera/model session or an explicit retry of its already consumed image.
    const bootstrap =
      !previous ||
      previous.engine !== engine ||
      previous.activeModel !== activeModel ||
      previous.requestId !== requestId ||
      previous.cameraRequestId !== cameraRequestId ||
      previous.retry !== retry ||
      previous.threshold === threshold;
    previousScheduler.current = {
      engine,
      activeModel,
      requestId,
      cameraRequestId,
      threshold,
      retry,
    };
    const generation = schedulerGeneration.current;
    let stopped = false;
    let drawId: number | null = null;
    let dirty = false;
    let reported: DetectionFrame | null = null;
    let warned = false;
    let inferenceFailed = false;
    const metrics = new PerformanceMetrics();
    const pipeline = new PipelineMetrics();
    const drawProbe = new PipelineProbe(pipeline);
    let latestMeasurement: InferenceMeasurement | undefined;
    let needsFirstDraw = false;
    const discard = (reason: 'cancelled' | 'hidden' | 'draw-failed') => {
      telemetry(() => pipeline.discardAll(reason));
      needsFirstDraw = false;
    };
    const visibility = () => {
      if (document.hidden) discard('hidden');
    };
    document.addEventListener('visibilitychange', visibility);
    const current = () => {
      const props = latest.current;
      const state = engine.getState();
      return (
        !stopped &&
        generation === schedulerGeneration.current &&
        props.isCameraActive &&
        props.cameraReady &&
        props.modelReady &&
        props.requestId === requestId &&
        props.cameraRequestId === cameraRequestId &&
        props.activeModel === activeModel &&
        state.status === 'ready' &&
        state.requestId === requestId &&
        state.model === activeModel
      );
    };
    let cancelAcquisition = () => {};
    const fail = (stage: string) => {
      if (!current()) return;
      discard(stage === '描画エラー' ? 'draw-failed' : 'cancelled');
      stopped = true;
      cancelAcquisition();
      if (drawId !== null) cancelAnimationFrame(drawId);
      latestFrameRef.current = null;
      drawnFrameRef.current = null;
      engine.clearDetections();
      setFault(stage);
    };
    cancelAcquisition = startFrameScheduler({
      video,
      current,
      bootstrap,
      instrumentation: pipeline,
      lock: lockRef.current,
      input: () => metrics.input(),
      captureError: () => fail('映像フレーム取得エラー'),
      work: (input) => {
        const start = performance.now();
        const effectiveInferenceParams =
          activeModel === 'stereo-depth'
            ? {
                ...latest.current.inferenceParams,
                rotate180: latest.current.inferenceParams?.cameraRotate180 ?? true,
              }
            : latest.current.inferenceParams;
        const res = engine.detectVideoFrame(
          input,
          activeModel,
          latest.current.renderOptions.confidenceThreshold,
          pipeline,
          effectiveInferenceParams
        );
        if (res instanceof Promise) {
          return res.then((result) => ({ result, start, end: performance.now() }));
        }
        const end = performance.now();
        return { result: res, start, end };
      },
      adopt: ({ result, start, end }, measurement) => {
        if (
          !current() ||
          result.status !== 'ready' ||
          result.requestId !== requestId ||
          result.model !== activeModel
        ) {
          return;
        }
        inferenceFailed = false;
        metrics.inferenceCompleted(start, end);
        telemetry(() => {
          if (latestMeasurement) pipeline.discard(latestMeasurement.jobId, 'overwritten');
          if (measurement) pipeline.accept(measurement);
        });
        latestMeasurement = measurement;
        needsFirstDraw = !!measurement;
        if (document.hidden) discard('hidden');
        latestFrameRef.current = result;
        dirty = true;
        const config = latest.current.ttsConfig;
        if (config.enabled && config.autoReadout) {
          try {
            for (const obj of [...result.objects].sort((a, b) => b.confidence - a.confidence)) {
              if (speechService.canSpeakLabel(obj.labelJa, config.cooldownSec)) {
                const verb = obj.category === 'テキスト (OCR)' ? 'を読み取りました' : 'を認識しました';
                speechService.speak(`${obj.labelJa}${verb}`, obj.labelJa, config, 'auto');
                break;
              }
            }
          } catch {
            fail('音声エラー');
          }
        }
      },
      error: (cause) => {
        if (!current()) return;
        // Repair only temporary scene state, retaining the ready model/task.
        discard('cancelled');
        inferenceFailed = true;
        engine.clearDetections();
        latestFrameRef.current = null;
        drawnFrameRef.current = null;
        dirty = true;
        if (!warned) {
          warned = true;
          console.warn('Vision frame failed; waiting for a new frame', { cause });
        }
      },
    });
    const draw = () => {
      if (!current()) return;
      const drawStart = drawProbe.start();
      let drawOutcome: 'success' | 'failed' | 'cancelled' = 'cancelled';
      let drawEnded = false;
      try {
        const canvas = canvasRef.current;
        const frame = latestFrameRef.current;
        if (canvas) {
          const copied = drawCanvasFrame(
            canvas,
            video,
            frame,
            offscreenCanvasRef,
            drawRef.current,
            current
          );
          if (!copied && activeModel === 'stereo-depth' && !frame && video.readyState >= 2 && video.videoWidth >= 2 && video.videoHeight >= 1) {
            const eyeW = Math.floor(video.videoWidth / 2);
            const eyeH = video.videoHeight;
            if (eyeW > 0 && eyeH > 0) {
              if (canvas.width !== eyeW) canvas.width = eyeW;
              if (canvas.height !== eyeH) canvas.height = eyeH;
              const previewCtx = canvas.getContext('2d');
              if (previewCtx) {
                const rotate180 = latest.current.inferenceParams?.cameraRotate180 ?? true;
                const swapEyes = Boolean(latest.current.inferenceParams?.swapLeftRight) !== rotate180;
                previewCtx.clearRect(0, 0, canvas.width, canvas.height);
                if (rotate180) {
                  previewCtx.save?.();
                  if (typeof previewCtx.translate === 'function' && typeof previewCtx.rotate === 'function') {
                    previewCtx.translate(eyeW, eyeH);
                    previewCtx.rotate(Math.PI);
                  }
                  previewCtx.drawImage(video, swapEyes ? eyeW : 0, 0, eyeW, eyeH, 0, 0, eyeW, eyeH);
                  previewCtx.restore?.();
                } else {
                  previewCtx.drawImage(video, swapEyes ? eyeW : 0, 0, eyeW, eyeH, 0, 0, eyeW, eyeH);
                }
              }
            }
          }
          drawnFrameRef.current = copied && current() ? frame : null;
          drawOutcome = current() ? 'success' : 'cancelled';
          drawProbe.end('draw', drawStart, drawOutcome);
          drawEnded = true;
          if (copied) {
            if (needsFirstDraw && latestMeasurement) {
              telemetry(() => pipeline.firstDraw(latestMeasurement!));
              needsFirstDraw = false;
            }
            metrics.drawCompleted();
          }
        }
      } catch {
        drawOutcome = 'failed';
        drawProbe.end('draw', drawStart, drawOutcome);
        drawEnded = true;
        fail('描画エラー');
      } finally {
        if (!drawEnded) drawProbe.end('draw', drawStart, drawOutcome);
        if (current()) drawId = requestAnimationFrame(draw);
      }
    };
    drawId = requestAnimationFrame(draw);
    const hud = setInterval(() => {
      if (!current()) return;
      setStats(metrics.snapshot());
      const snapshot = telemetry(() => readPipelineSnapshot(pipeline.snapshot()));
      setPipelineStats(snapshot ?? null);
      setInferenceError(inferenceFailed);
      if (dirty) {
        dirty = false;
        const frame = latestFrameRef.current;
        if (frame?.depthMap) {
          setStereoSummary({
            colormap: frame.depthMap.colormap,
            scaleMinDisparity: frame.depthMap.scaleMinDisparity,
            scaleMaxDisparity: frame.depthMap.scaleMaxDisparity,
            medianDisparity: frame.depthMap.medianDisparity,
          });
        } else if (activeModel !== 'stereo-depth') {
          setStereoSummary(null);
        }
        if (frame !== reported) {
          reported = frame;
          try {
            latest.current.onFrame(frame);
          } catch {
            fail('結果通知エラー');
          }
        }
      }
    }, HUD_INTERVAL_MS);
    return () => {
      discard('cancelled');
      document.removeEventListener('visibilitychange', visibility);
      stopped = true;
      drawnFrameRef.current = null;
      cancelAcquisition();
      clearInterval(hud);
      if (drawId !== null) cancelAnimationFrame(drawId);
    };
  }, [
    isCameraActive,
    cameraReady,
    cameraRequestId,
    activeModel,
    engine,
    modelReady,
    requestId,
    fault,
    retry,
    threshold,
    stillImageSource,
    stereoVideoSource,
  ]);

  const drawOverlays = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    objects: DetectedObject[],
    semanticMask?: HTMLCanvasElement,
    depthMap?: StereoDepthMapData
  ) => {
    if (depthMap) {
      ctx.save();
      ctx.clearRect(0, 0, width, height);
      if (depthMap.viewMode === 'split-3pane') {
        const paneW = width / 3;
        ctx.drawImage(depthMap.leftCanvas, 0, 0, paneW, height);
        ctx.drawImage(depthMap.rightCanvas, paneW, 0, paneW, height);
        ctx.fillStyle = '#040711';
        ctx.fillRect(paneW * 2, 0, paneW, height);
        ctx.drawImage(depthMap.overlayCanvas, paneW * 2, 0, paneW, height);

        ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(paneW, 0);
        ctx.lineTo(paneW, height);
        ctx.moveTo(paneW * 2, 0);
        ctx.lineTo(paneW * 2, height);
        ctx.stroke();

        ctx.font = 'bold 11px monospace';
        ctx.textBaseline = 'top';
        const labels: ReadonlyArray<[string, number]> = [
          ['LEFT EYE', 8],
          ['RIGHT EYE', paneW + 8],
          ['DISPARITY', paneW * 2 + 8],
        ];
        for (const [text, lx] of labels) {
          const tw = ctx.measureText(text).width + 10;
          ctx.fillStyle = 'rgba(4, 7, 17, 0.72)';
          ctx.fillRect(lx, 8, tw, 18);
          ctx.fillStyle = '#e2e8f0';
          ctx.fillText(text, lx + 5, 11);
        }
      } else if (depthMap.viewMode === 'depth-only') {
        ctx.fillStyle = '#040711';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(depthMap.overlayCanvas, 0, 0, width, height);
      } else {
        ctx.drawImage(depthMap.leftCanvas, 0, 0, width, height);
        if (renderOptions.showMasks) {
          ctx.globalAlpha = renderOptions.maskOpacity;
          ctx.drawImage(depthMap.overlayCanvas, 0, 0, width, height);
          ctx.globalAlpha = 1;
        }
      }

      const drawProbeCrosshair = (
        normX: number,
        normY: number,
        color: string,
        badgeText?: string
      ) => {
        const paneCount = depthMap.viewMode === 'split-3pane' ? 3 : 1;
        const paneW = width / paneCount;
        const paneIndices = depthMap.viewMode === 'split-3pane' ? [0, 2] : [0];
        for (const paneIndex of paneIndices) {
          const cx = paneIndex * paneW + normX * paneW;
          const cy = normY * height;
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(cx - 10, cy);
          ctx.lineTo(cx + 10, cy);
          ctx.moveTo(cx, cy - 10);
          ctx.lineTo(cx, cy + 10);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(cx, cy, 4, 0, Math.PI * 2);
          ctx.stroke();

          if (badgeText && (paneCount === 1 || paneIndex === 2)) {
            ctx.font = 'bold 11px monospace';
            ctx.textBaseline = 'middle';
            const bw = ctx.measureText(badgeText).width + 10;
            const bx = Math.min(Math.max(4, cx + 8), width - bw - 4);
            const by = Math.max(14, Math.min(height - 14, cy - 12));
            ctx.fillStyle = 'rgba(4, 7, 17, 0.85)';
            ctx.fillRect(bx, by - 9, bw, 18);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(bx, by - 9, bw, 18);
            ctx.fillStyle = '#f8fafc';
            ctx.fillText(badgeText, bx + 5, by);
          }
        }
      };

      const bMm = inferenceParams?.baselineMm ?? 60;
      const fPx = inferenceParams?.focalLengthPx ?? 500;
      for (const pin of stereoPinnedProbes) {
        const liveProbe = engine.inspectDepthAtPoint(pin.normX, pin.normY, depthMap, bMm, fPx);
        const isValid = liveProbe ? liveProbe.valid : pin.valid;
        const dispVal = liveProbe ? liveProbe.disparityPx : pin.disparityPx;
        const depthVal = liveProbe ? liveProbe.depthMeters : pin.depthMeters;
        const dispText = isValid ? `${dispVal.toFixed(1)}px` : '--';
        const depthText = depthVal !== null ? `${depthVal.toFixed(2)}m` : '--';
        drawProbeCrosshair(pin.normX, pin.normY, '#38bdf8', `${pin.label}: ${dispText} / ${depthText}`);
      }

      if (stereoHoverProbe) {
        drawProbeCrosshair(stereoHoverProbe.normX, stereoHoverProbe.normY, '#facc15');
      }

      ctx.restore();
    }

    // 1. Draw Masks
    if (renderOptions.showMasks) {
      if (semanticMask) {
        ctx.save();
        ctx.globalAlpha = renderOptions.maskOpacity;
        ctx.drawImage(semanticMask, 0, 0, width, height);
        ctx.restore();
      }
      objects.forEach((obj) => {
        const isSelected = obj.id === selectedObjectId;
        const opacity = isSelected
          ? Math.min(1, renderOptions.maskOpacity + 0.3)
          : renderOptions.maskOpacity;

        // Real pixel segmentation mask (MediaPipe Selfie Segmenter)
        if (obj.maskCanvas) {
          ctx.save();
          ctx.globalAlpha = opacity;
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = 'high';
          ctx.drawImage(obj.maskCanvas, 0, 0, width, height);
          ctx.restore();
          return;
        }

        // Polygon contour fallback
        if (obj.maskPolygon && obj.maskPolygon.length >= 3) {
          ctx.save();
          ctx.beginPath();
          const start = obj.maskPolygon[0];
          ctx.moveTo(start.x * width, start.y * height);

          for (let i = 1; i < obj.maskPolygon.length; i++) {
            const pt = obj.maskPolygon[i];
            ctx.lineTo(pt.x * width, pt.y * height);
          }
          ctx.closePath();

          ctx.globalAlpha = opacity;
          ctx.fillStyle = obj.color;
          ctx.fill();

          ctx.globalAlpha = isSelected ? 1.0 : 0.85;
          ctx.strokeStyle = isSelected ? '#ffffff' : obj.color;
          ctx.lineWidth = isSelected ? 3 : 2;
          ctx.stroke();

          ctx.restore();
        }
      });
    }

    // 2. Draw Bounding Boxes
    if (renderOptions.showBoxes) {
      objects.forEach((obj) => {
        const isSelected = obj.id === selectedObjectId;
        const bx = obj.bbox.x * width;
        const by = obj.bbox.y * height;
        const bw = obj.bbox.width * width;
        const bh = obj.bbox.height * height;

        ctx.save();
        ctx.strokeStyle = isSelected ? '#ffffff' : obj.color;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, 6);
        ctx.stroke();

        if (renderOptions.showLabels) {
          const confText = renderOptions.showConfidence
            ? ` ${obj.confidenceKind ? `${confidenceDescription(obj)} ` : ''}${Math.round(obj.confidence * 100)}%`
            : '';
          const fullLabel = `${obj.labelJa}${confText}`;

          ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
          const textMetrics = ctx.measureText(fullLabel);
          const badgeW = textMetrics.width + 14;
          const badgeH = 20;
          const badgeY = Math.max(0, by - badgeH);

          ctx.fillStyle = isSelected ? '#ffffff' : obj.color;
          ctx.beginPath();
          ctx.roundRect(bx, badgeY, badgeW, badgeH, [4, 4, 0, 0]);
          ctx.fill();

          ctx.fillStyle = isSelected ? '#0f172a' : '#ffffff';
          ctx.textBaseline = 'middle';
          ctx.fillText(fullLabel, bx + 6, badgeY + badgeH / 2);
        }

        ctx.restore();
      });
    }

    // 3. Draw Click Feedback
    if (clickRipple) {
      ctx.save();
      const rx = clickRipple.x * width;
      const ry = clickRipple.y * height;

      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(rx, ry, 24, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = 'rgba(56, 189, 248, 0.95)';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`🔊 ${clickRipple.label}`, rx, ry - 30);
      ctx.restore();
    }
  };

  useLayoutEffect(() => {
    drawRef.current = drawOverlays;
  });
  const rippleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (rippleTimer.current !== null) clearTimeout(rippleTimer.current);
    },
    []
  );

  const mapStereoClientPoint = (clientX: number, clientY: number, frame: DetectionFrame) => {
    const canvas = canvasRef.current;
    const depthMap = frame.depthMap;
    if (!canvas || !depthMap) return null;
    const raw = clientPointToVideo(
      clientX,
      clientY,
      canvas.getBoundingClientRect(),
      canvas.width,
      canvas.height
    );
    if (!raw) return null;
    if (depthMap.viewMode === 'split-3pane') {
      const scaled = raw.x * 3;
      const normX = Math.max(0, Math.min(1, scaled - Math.floor(Math.min(2, scaled))));
      return { normX, normY: raw.y, rawX: raw.x, rawY: raw.y };
    }
    return { normX: raw.x, normY: raw.y, rawX: raw.x, rawY: raw.y };
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeModel !== 'stereo-depth' || !onStereoHoverProbe) return;
    const frame = drawnFrameRef.current;
    const state = engine.getState();
    if (
      !frame?.depthMap ||
      frame !== latestFrameRef.current ||
      frame.model !== activeModel ||
      frame.requestId !== requestId ||
      state.status !== 'ready' ||
      state.model !== frame.model ||
      state.requestId !== frame.requestId
    ) {
      return;
    }
    const mapped = mapStereoClientPoint(e.clientX, e.clientY, frame);
    if (!mapped) {
      onStereoHoverProbe(null);
      return;
    }
    const baselineMm = inferenceParams?.baselineMm ?? 60;
    const focalLengthPx = inferenceParams?.focalLengthPx ?? 500;
    const probe = engine.inspectDepthAtPoint(
      mapped.normX,
      mapped.normY,
      frame.depthMap,
      baselineMm,
      focalLengthPx
    );
    onStereoHoverProbe(probe);
  };

  const handleCanvasMouseLeave = () => {
    if (activeModel === 'stereo-depth' && onStereoHoverProbe) {
      onStereoHoverProbe(null);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const frame = drawnFrameRef.current;
    const state = engine.getState();
    const customSourceReady = Boolean((stillImageSource || stereoVideoSource) && canvas);
    const liveReady = Boolean(video && video.readyState >= 2 && isCameraActive && cameraReady);
    if (
      (!customSourceReady && !liveReady) ||
      !modelReady ||
      fault ||
      !frame ||
      frame !== latestFrameRef.current ||
      frame.model !== activeModel ||
      frame.requestId !== requestId ||
      state.status !== 'ready' ||
      state.model !== frame.model ||
      state.requestId !== frame.requestId
    ) {
      return;
    }

    if (activeModel === 'stereo-depth' && frame.depthMap) {
      const mapped = mapStereoClientPoint(e.clientX, e.clientY, frame);
      if (!mapped) return;
      const baselineMm = inferenceParams?.baselineMm ?? 60;
      const focalLengthPx = inferenceParams?.focalLengthPx ?? 500;
      const probe = engine.inspectDepthAtPoint(
        mapped.normX,
        mapped.normY,
        frame.depthMap,
        baselineMm,
        focalLengthPx
      );
      if (!probe) return;
      onPinStereoProbe?.(probe);
      const label = probe.valid
        ? `${probe.disparityPx.toFixed(1)}px${probe.depthMeters !== null ? ` / ${probe.depthMeters.toFixed(2)}m` : ''}`
        : '視差未確定';
      setClickRipple({ x: mapped.rawX, y: mapped.rawY, label });
      if (rippleTimer.current !== null) clearTimeout(rippleTimer.current);
      rippleTimer.current = setTimeout(() => setClickRipple(null), 1200);
      if (ttsConfig.enabled && probe.valid) {
        const depthSpeech =
          probe.depthMeters !== null ? `、推定距離${probe.depthMeters.toFixed(2)}メートル` : '';
        try {
          speechService.speak(
            `視差${probe.disparityPx.toFixed(1)}ピクセル${depthSpeech}`,
            'stereo-probe',
            ttsConfig,
            'tap'
          );
        } catch {
          setFault('音声エラー');
        }
      }
      return;
    }

    // Read current geometry at input time, including a resize or new source size
    // that has not reached the next overlay drawing frame yet.
    const customCanvas = stereoVideoSource
      ? stereoVideoCanvasRef.current
      : stillImageSource?.canvas ?? null;
    const point =
      customCanvas && canvas
        ? clientPointToVideo(
            e.clientX,
            e.clientY,
            canvas.getBoundingClientRect(),
            customCanvas.width,
            customCanvas.height
          )
        : video
          ? clientPointToVideo(
              e.clientX,
              e.clientY,
              video.getBoundingClientRect(),
              video.videoWidth,
              video.videoHeight
            )
          : null;
    if (!point) return;
    const { x: clickX, y: clickY } = point;

    const hit = engine.findObjectAtPoint(clickX, clickY, frame.objects);

    if (hit) {
      try {
        onSelectObject(hit);
      } catch {
        setFault('結果通知エラー');
        return;
      }
      setClickRipple({ x: clickX, y: clickY, label: hit.labelJa });
      if (rippleTimer.current !== null) clearTimeout(rippleTimer.current);
      rippleTimer.current = setTimeout(() => setClickRipple(null), 1200);

      // Force speak on direct user tap
      try {
        speechService.speak(
          `「${hit.labelJa}」、${confidenceDescription(hit)}${Math.round(hit.confidence * 100)}%`,
          hit.labelJa,
          { ...ttsConfig, enabled: true },
          'tap'
        );
      } catch {
        setFault('音声エラー');
      }
    }
  };

  const handleVideoStepFrame = (deltaFrames: -1 | 1) => {
    setVideoPlaying(false);
    const nextTime = stepVideoTime(
      videoTimeRef.current,
      deltaFrames,
      videoFps,
      videoDuration,
      videoLoop
    );
    videoTimeRef.current = nextTime;
    setVideoCurrentTime(nextTime);
  };

  const handleVideoSeek = (nextTimeSec: number) => {
    const clamped = clampVideoTime(nextTimeSec, videoDuration, false);
    videoTimeRef.current = clamped;
    setVideoCurrentTime(clamped);
    if (uploadedVideoElRef.current) {
      try {
        uploadedVideoElRef.current.currentTime = clamped;
      } catch {
        // ignore seek error
      }
    }
  };

  const baselineMm = inferenceParams?.baselineMm ?? 60;
  const focalLengthPx = inferenceParams?.focalLengthPx ?? 500;
  const showCalibDepth = inferenceParams?.showCalibrationDepth ?? true;
  const minLegendDepth = stereoSummary
    ? disparityToDepthMeters(stereoSummary.scaleMaxDisparity, baselineMm, focalLengthPx)
    : null;
  const maxLegendDepth = stereoSummary
    ? disparityToDepthMeters(stereoSummary.scaleMinDisparity, baselineMm, focalLengthPx)
    : null;

  const formattedCurrent = formatVideoTimecode(videoCurrentTime, videoFps);
  const formattedTotal = formatVideoTimecode(videoDuration, videoFps);
  const hasCustomInput = Boolean(stillImageSource || stereoVideoSource);

  return (
    <div className="relative w-full h-full min-h-0 min-w-0 max-h-[45vh] md:max-h-[calc(100vh-4.5rem)] aspect-[4/3] sm:aspect-video mx-auto bg-[#040711] rounded-xl overflow-hidden border border-slate-800 shadow-xl flex items-center justify-center select-none">
      {/* Video Stream Element */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`w-full h-full object-contain ${
          isCameraActive && !hasCustomInput
            ? activeModel === 'stereo-depth'
              ? 'block opacity-0 pointer-events-none'
              : 'block'
            : 'hidden'
        }`}
      />

      {/* Interactive Overlay Canvas */}
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMouseMove}
        onMouseLeave={handleCanvasMouseLeave}
        className={`absolute inset-0 w-full h-full object-contain z-10 ${
          activeModel === 'stereo-depth' ? 'cursor-crosshair' : 'cursor-pointer'
        }`}
      />

      {/* Camera Inactive Standby Screen */}
      {!isCameraActive && !hasCustomInput && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center p-4 text-center bg-[#070b16]/95">
          <div className="w-12 h-12 rounded-xl bg-blue-950/60 border border-blue-600/30 flex items-center justify-center text-blue-400 mb-3 shadow-md">
            <Camera className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-bold text-white mb-1">Webカメラ待機中</h3>
          <p className="text-xs text-slate-400 max-w-xs mb-4 leading-relaxed">
            カメラを開始し、モデルの準備が完了すると推論結果を取得します。
          </p>
          <button
            onClick={onCameraStart}
            className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-500 rounded-lg shadow-sm transition-all flex items-center gap-1.5"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>カメラを開始する</span>
          </button>
        </div>
      )}

      {/* Still Image Active Badge & Return to Camera Button */}
      {stillImageSource && !stereoVideoSource && (
        <div className="absolute top-2.5 right-2.5 z-25 flex items-center gap-1.5">
          <span className="px-2.5 py-1.5 rounded-lg bg-cyan-950/85 border border-cyan-500/60 text-[11px] font-semibold text-cyan-200 shadow-lg">
            静止画検証: {stillImageSource.label} ({stillImageSource.canvas.width}×
            {stillImageSource.canvas.height})
          </span>
          {onClearStillImage && (
            <button
              type="button"
              onClick={onClearStillImage}
              className="px-2.5 py-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 border border-slate-700 text-[11px] font-semibold text-slate-200 transition-colors shadow-lg"
            >
              カメラへ戻す
            </button>
          )}
        </div>
      )}

      {/* Stereo Video Active Badge & Return to Camera Button */}
      {stereoVideoSource && (
        <div className="absolute top-2.5 right-2.5 z-25 flex items-center gap-1.5">
          <span className="px-2.5 py-1.5 rounded-lg bg-cyan-950/90 border border-cyan-500/60 text-[11px] font-semibold text-cyan-100 shadow-lg flex items-center gap-1.5">
            <Film className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <span>{stereoVideoSource.label}</span>
          </span>
          {onClearStereoVideo && (
            <button
              type="button"
              onClick={onClearStereoVideo}
              className="px-2.5 py-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 border border-slate-700 text-[11px] font-semibold text-slate-200 transition-colors shadow-lg whitespace-nowrap"
            >
              カメラへ戻す
            </button>
          )}
        </div>
      )}

      {/* Camera Flip Button (Top-Right Corner of video) */}
      {isCameraActive && !hasCustomInput && activeModel !== 'stereo-depth' && (
        <button
          onClick={onToggleFacingMode}
          title={
            facingMode === 'user'
              ? 'アウトカメラ（背面）に切り替え'
              : 'インカメラ（前面）に切り替え'
          }
          className="absolute top-2.5 right-2.5 z-25 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-black/70 hover:bg-black/90 backdrop-blur-md border border-slate-700/80 text-xs font-medium text-slate-200 transition-colors shadow-lg"
        >
          <SwitchCamera className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-[11px] font-semibold">
            {facingMode === 'user' ? 'インカメラ' : 'アウトカメラ'}
          </span>
        </button>
      )}

      <div className="absolute top-2.5 left-2.5 z-20 pointer-events-none px-2.5 py-1 rounded-md bg-black/60 text-[11px]">
        <span>{modelMetadata(activeModel).name}</span>
        <span aria-label="カメラ状態" className="ml-2">
          {stereoVideoSource
            ? `ステレオ動画入力 (${stereoVideoSource.label})`
            : stillImageSource
              ? `静止画入力 (${stillImageSource.label})`
              : cameraReady
                ? 'カメラ映像を取得しました'
                : isCameraActive && !cameraError
                  ? 'カメラ準備中'
                  : 'カメラ停止中'}
        </span>
        <span aria-label="性能状態" className="ml-2">
          {fault ??
            (modelState.status === 'error'
              ? 'モデルエラー'
              : stereoVideoSource
                ? !modelReady
                  ? 'モデル読込中'
                  : inferenceError
                    ? '推論エラー'
                    : videoPlaying || customSourceInferring
                      ? 'ステレオ動画推論中'
                      : 'ステレオ動画一時停止'
                : stillImageSource
                  ? !modelReady
                    ? 'モデル読込中'
                    : inferenceError
                      ? '推論エラー'
                      : customSourceInferring
                        ? '静止画推論中'
                        : '静止画推論完了'
                  : !isCameraActive
                    ? '停止中'
                    : !modelReady
                      ? 'モデル読込中'
                      : !cameraReady
                        ? 'カメラ準備中'
                        : inferenceError
                          ? '推論エラー：次の映像を待機'
                          : stats.stalled
                            ? '映像入力停滞'
                            : '計測中')}
        </span>
        <span aria-label="推論FPS" className="ml-2">
          推論:{' '}
          {stats.inferenceFps === null ||
          fault ||
          (!cameraReady && !hasCustomInput) ||
          !modelReady
            ? '未計測'
            : `${stats.inferenceFps.toFixed(1)} FPS`}
        </span>
        <span aria-label="描画FPS" className="ml-2">
          描画:{' '}
          {stats.drawFps === null ||
          fault ||
          (!cameraReady && !stereoVideoSource) ||
          !modelReady
            ? '未計測'
            : `${stats.drawFps.toFixed(1)} FPS`}
        </span>
        <span aria-label="フレーム処理時間" className="ml-2">
          フレーム処理:{' '}
          {stats.processingMs === null ||
          fault ||
          (!cameraReady && !hasCustomInput) ||
          !modelReady
            ? '未計測'
            : `${stats.processingMs.toFixed(1)} ms`}
        </span>
      </div>

      <PipelineDetails snapshot={pipelineStats} />

      {activeModel === 'stereo-depth' && customSourceInferring && (
        <div
          aria-label="深度推論進行状況"
          className="absolute top-11 left-2.5 z-25 pointer-events-none flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-950/90 border border-blue-500/60 text-xs font-semibold text-blue-100 shadow-lg"
        >
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping shrink-0" />
          <span>
            {stereoSummary
              ? '深度マップ更新中 (Web Worker)...'
              : '深度マップ推論中 (Web Worker)...'}
          </span>
        </div>
      )}

      {fault && (
        <div className="absolute bottom-10 z-30 bg-slate-900 p-2 text-xs" role="alert">
          {fault}：処理を停止しました。
          <button onClick={() => setRetry((value) => value + 1)}>処理を再試行</button>
        </div>
      )}

      {uploadedVideoError && (
        <div
          role="alert"
          className="absolute top-12 right-2.5 z-30 bg-rose-950/90 border border-rose-500/60 text-rose-100 px-3 py-1.5 rounded-lg text-xs"
        >
          {uploadedVideoError}
        </div>
      )}

      {/* Stereo Video Transport Controls Bar */}
      {stereoVideoSource && (
        <div
          aria-label="ステレオ動画トランスポート"
          className="absolute bottom-11 left-2.5 right-2.5 z-25 bg-[#090d18]/90 border border-slate-700/80 rounded-lg px-2.5 py-1.5 flex flex-wrap items-center justify-between gap-2 shadow-lg"
        >
          {/* Play/Pause & Frame Step Buttons */}
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              aria-label={videoPlaying ? '一時停止' : '再生'}
              onClick={() => {
                setVideoCurrentTime(videoTimeRef.current);
                setVideoPlaying((prev) => !prev);
              }}
              className="px-2.5 py-1 rounded bg-cyan-700 hover:bg-cyan-600 text-white text-[11px] font-semibold flex items-center gap-1 transition-colors whitespace-nowrap"
            >
              {videoPlaying ? (
                <>
                  <Pause className="w-3 h-3" />
                  <span>一時停止</span>
                </>
              ) : (
                <>
                  <Play className="w-3 h-3" />
                  <span>再生</span>
                </>
              )}
            </button>
            <button
              type="button"
              aria-label="1フレーム戻る"
              title="1フレーム戻る (-1F)"
              onClick={() => handleVideoStepFrame(-1)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-[11px] font-mono flex items-center gap-0.5 transition-colors whitespace-nowrap"
            >
              <SkipBack className="w-3 h-3 text-cyan-400" />
              <span>-1F</span>
            </button>
            <button
              type="button"
              aria-label="1フレーム進む"
              title="1フレーム進む (+1F)"
              onClick={() => handleVideoStepFrame(1)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-[11px] font-mono flex items-center gap-0.5 transition-colors whitespace-nowrap"
            >
              <span>+1F</span>
              <SkipForward className="w-3 h-3 text-cyan-400" />
            </button>
          </div>

          {/* Seek Slider & Timecode */}
          <div className="flex-1 min-w-[160px] flex items-center gap-2">
            <input
              type="range"
              aria-label="ステレオ動画シーク"
              min={0}
              max={Math.max(0.1, videoDuration)}
              step={Math.max(0.01, 1 / videoFps)}
              value={Math.min(videoCurrentTime, videoDuration)}
              onChange={(e) => handleVideoSeek(parseFloat(e.target.value))}
              className="flex-1 accent-cyan-400 cursor-pointer h-1.5"
            />
            <span
              aria-label="動画タイムコード"
              className="font-mono tabular-nums text-[10px] text-slate-200 whitespace-nowrap shrink-0"
            >
              {formattedCurrent.timeText} / {formattedTotal.timeText} · F
              {String(formattedCurrent.frameIndex).padStart(4, '0')}
            </span>
          </div>

          {/* Speed Selector & Loop Toggle */}
          <div className="flex items-center gap-1 shrink-0">
            <div className="flex items-center gap-0.5 bg-slate-900 border border-slate-800 rounded p-0.5">
              {PLAYBACK_RATES.map((rate) => (
                <button
                  key={rate}
                  type="button"
                  aria-label={`再生速度 ${rate}x`}
                  onClick={() => setVideoPlaybackRate(rate)}
                  className={`px-1.5 py-0.5 rounded font-mono text-[10px] transition-colors whitespace-nowrap ${
                    videoPlaybackRate === rate
                      ? 'bg-cyan-700 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {rate}x
                </button>
              ))}
            </div>
            <button
              type="button"
              aria-label="ループ再生切替"
              aria-pressed={videoLoop}
              onClick={() => setVideoLoop((prev) => !prev)}
              className={`px-2 py-1 rounded border text-[10px] font-semibold flex items-center gap-1 transition-colors whitespace-nowrap ${
                videoLoop
                  ? 'bg-slate-800 border-cyan-500/60 text-cyan-300'
                  : 'bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300'
              }`}
            >
              <Repeat className="w-3 h-3" />
              <span>ループ</span>
            </button>
          </div>
        </div>
      )}

      {/* Stereo Depth Colormap Legend + Hover Crosshair Readout */}
      {activeModel === 'stereo-depth' && (isCameraActive || hasCustomInput) && stereoSummary && (
        <div
          aria-label="視差カラーマップ凡例"
          className="absolute bottom-2 left-2.5 right-2.5 z-20 pointer-events-none flex flex-wrap items-center justify-between gap-2"
        >
          <div className="bg-black/75 border border-slate-700/80 rounded-md px-2.5 py-1.5 flex items-center gap-2 text-[10px] font-mono tabular-nums text-slate-200">
            <span className="text-slate-400">遠</span>
            <span className="text-slate-300">
              {stereoSummary.scaleMinDisparity.toFixed(1)}px
              {showCalibDepth && maxLegendDepth !== null ? ` (${maxLegendDepth.toFixed(2)}m)` : ''}
            </span>
            <div
              className="w-24 sm:w-32 h-2.5 rounded-sm border border-slate-600/80"
              style={{ background: buildLegendGradientCss(stereoSummary.colormap) }}
            />
            <span className="text-slate-300">
              {stereoSummary.scaleMaxDisparity.toFixed(1)}px
              {showCalibDepth && minLegendDepth !== null ? ` (${minLegendDepth.toFixed(2)}m)` : ''}
            </span>
            <span className="text-slate-400">近</span>
          </div>

          <div
            aria-label="ホバー座標深度"
            className="bg-black/80 border border-slate-700/80 rounded-md px-2.5 py-1.5 text-[10px] font-mono tabular-nums text-cyan-200"
          >
            {stereoHoverProbe ? (
              stereoHoverProbe.valid ? (
                <>
                  ({stereoHoverProbe.pixelX}, {stereoHoverProbe.pixelY}) · d=
                  {stereoHoverProbe.disparityPx.toFixed(1)}px
                  {showCalibDepth && stereoHoverProbe.depthMeters !== null
                    ? ` · Z=${stereoHoverProbe.depthMeters.toFixed(2)}m`
                    : ''}
                </>
              ) : (
                `(${stereoHoverProbe.pixelX}, ${stereoHoverProbe.pixelY}) · 視差未確定`
              )
            ) : (
              'カーソル移動で深度計測／クリックでピン固定'
            )}
          </div>
        </div>
      )}

      {/* Tap Hint (Bottom Center) */}
      {(isCameraActive || hasCustomInput) && activeModel !== 'stereo-depth' && (
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <span className="text-[10px] text-slate-400 bg-black/60 backdrop-blur-md border border-slate-800 px-2.5 py-0.5 rounded-full">
            物体をタップすると日本語で発話
          </span>
        </div>
      )}

      {/* Camera Error Screen */}
      {cameraError && !hasCustomInput && (
        <div className="absolute inset-0 z-30 flex items-center justify-center p-4 bg-black/90">
          <div className="bg-slate-900 border border-rose-600/40 rounded-xl p-4 max-w-sm text-center">
            <AlertCircle className="w-6 h-6 text-rose-500 mx-auto mb-2" />
            <h4 className="text-xs font-semibold text-white mb-1">カメラエラー</h4>
            <p className="text-[11px] text-slate-300 leading-relaxed mb-3">{cameraError}</p>
            <button
              onClick={clearError}
              className="px-3 py-1.5 text-xs text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-md"
            >
              閉じる
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
