import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Camera, AlertCircle, SwitchCamera } from 'lucide-react';
import {
  DetectionFrame,
  ModelState,
  DetectedObject,
  RunnableModelType,
  CameraFacingMode,
  RenderOptions,
  TTSConfig
} from '../types/vision';
import { speechService } from '../services/speechService';
import { VisionEngine } from '../services/visionEngine';
import { JobLock, startFrameScheduler } from '../services/frameScheduler';
import { HUD_INTERVAL_MS, PerformanceMetrics, type PerformanceSnapshot } from '../services/performanceMetrics';
import { PipelineMetrics, PipelineProbe, telemetry, type PipelineSnapshot, type InferenceMeasurement } from '../services/pipelineMetrics';
import { modelMetadata } from '../services/modelRegistry';
import { useCameraStream } from '../hooks/useCameraStream';
import { drawCanvasFrame } from '../services/canvasRenderer';
import { PipelineDetails } from './PipelineDetails';
import { clientPointToVideo } from '../services/videoCoordinates';

interface VisionCanvasProps {
  activeModel: RunnableModelType;
  engine: VisionEngine;
  modelState: ModelState;
  facingMode: CameraFacingMode;
  onToggleFacingMode: () => void;
  renderOptions: RenderOptions;
  ttsConfig: TTSConfig;
  isCameraActive: boolean;
  onCameraStart: () => void;
  onCameraStop: () => void;
  onSelectObject: (obj: DetectedObject) => void;
  selectedObjectId?: string;
  onFrame: (frame: DetectionFrame | null) => void;
}

export const VisionCanvas: React.FC<VisionCanvasProps> = ({
  activeModel,
  engine,
  modelState,
  facingMode,
  onToggleFacingMode,
  renderOptions,
  ttsConfig,
  isCameraActive,
  onCameraStart,
  onCameraStop,
  onSelectObject,
  selectedObjectId,
  onFrame
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { cameraError, clearError, cameraReady, cameraRequestId } = useCameraStream(videoRef, isCameraActive, facingMode, onCameraStop);
  const [clickRipple, setClickRipple] = useState<{ x: number; y: number; label: string } | null>(null);
  const [currentFrame, setCurrentFrame] = useState<DetectionFrame | null>(null);
  const modelReady = modelState.status === 'ready' && modelState.model === activeModel;
  const requestId = modelState.requestId;
  const threshold = renderOptions.confidenceThreshold;
  const currentObjects = cameraReady && modelReady && currentFrame?.model === activeModel && currentFrame.requestId === requestId
    ? currentFrame.objects : [];

  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestFrameRef = useRef<DetectionFrame | null>(null);
  const lockRef = useRef(new JobLock());
  const schedulerGeneration = useRef(0);
  const previousScheduler = useRef<{ engine: VisionEngine; activeModel: RunnableModelType; requestId: number; cameraRequestId: number; threshold: number; retry: number } | null>(null);
  const [inferenceError, setInferenceError] = useState(false);
  const [fault, setFault] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [stats, setStats] = useState<PerformanceSnapshot>({ inferenceFps: null, drawFps: null, processingMs: null, stalled: false });
  const [pipelineStats, setPipelineStats] = useState<PipelineSnapshot | null>(null);
  const latest = useRef({ onFrame, renderOptions, ttsConfig, modelReady, requestId, cameraRequestId, isCameraActive, cameraReady, activeModel });
  useLayoutEffect(() => {
    latest.current = { onFrame, renderOptions, ttsConfig, modelReady, requestId, cameraRequestId, isCameraActive, cameraReady, activeModel };
  });
  const drawRef = useRef<(ctx: CanvasRenderingContext2D, width: number, height: number, objects: DetectedObject[]) => void>(() => {});

  useLayoutEffect(() => {
    const generationOwner = schedulerGeneration;
    generationOwner.current++;
    latestFrameRef.current = null;
    engine.clearDetections();
    setCurrentFrame(null);
    setClickRipple(null);
    setFault(null);
    setInferenceError(false);
    setPipelineStats(null);
    setStats({ inferenceFps: null, drawFps: null, processingMs: null, stalled: false });
    try { latest.current.onFrame(null); }
    catch { setFault('結果通知エラー'); }
    const canvas = canvasRef.current;
    try { if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height); }
    catch { setFault('描画エラー'); }
    return () => { generationOwner.current++; };
  }, [activeModel, requestId, isCameraActive, facingMode, cameraReady, cameraRequestId, engine, retry, threshold]);

  useLayoutEffect(() => {
    if (!fault) return;
    setPipelineStats(null);
    schedulerGeneration.current++;
    latestFrameRef.current = null;
    engine.clearDetections();
    setCurrentFrame(null);
    const canvas = canvasRef.current;
    if (canvas) { const width = canvas.width; canvas.width = width; }
    try { latest.current.onFrame(null); } catch { /* The stopped notification error is already visible. */ }
  }, [fault, engine]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isCameraActive || !cameraReady || !modelReady || fault) return;
    const previous = previousScheduler.current;
    // A threshold change invalidates the attempted old result, but is not a new
    // camera/model session or an explicit retry of its already consumed image.
    const bootstrap = !previous || previous.engine !== engine || previous.activeModel !== activeModel || previous.requestId !== requestId || previous.cameraRequestId !== cameraRequestId
      || previous.retry !== retry || previous.threshold === threshold;
    previousScheduler.current = { engine, activeModel, requestId, cameraRequestId, threshold, retry };
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
      telemetry(() => pipeline.discardAll(reason)); needsFirstDraw = false;
    };
    const visibility = () => { if (document.hidden) discard('hidden'); };
    document.addEventListener('visibilitychange', visibility);
    const current = () => {
      const props = latest.current;
      const state = engine.getState();
      return !stopped && generation === schedulerGeneration.current && props.isCameraActive && props.cameraReady && props.modelReady
        && props.requestId === requestId && props.cameraRequestId === cameraRequestId && props.activeModel === activeModel
        && state.status === 'ready' && state.requestId === requestId && state.model === activeModel;
    };
    let cancelAcquisition = () => {};
    const fail = (stage: string) => {
      if (!current()) return;
      discard(stage === '描画エラー' ? 'draw-failed' : 'cancelled');
      stopped = true;
      cancelAcquisition();
      if (drawId !== null) cancelAnimationFrame(drawId);
      latestFrameRef.current = null;
      engine.clearDetections();
      setCurrentFrame(null);
      setFault(stage);
    };
    cancelAcquisition = startFrameScheduler({
      video, current, bootstrap, instrumentation: pipeline, lock: lockRef.current, input: () => metrics.input(),
      captureError: () => fail('映像フレーム取得エラー'),
      work: input => {
        const start = performance.now();
        const result = engine.detectVideoFrame(input, activeModel, latest.current.renderOptions.confidenceThreshold, pipeline);
        const end = performance.now();
        return { result, start, end };
      },
      adopt: ({ result, start, end }, measurement) => {
        if (!current() || result.status !== 'ready' || result.requestId !== requestId || result.model !== activeModel) return;
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
                speechService.speak(`${obj.labelJa}を認識しました`, obj.labelJa, config, 'auto');
                break;
              }
            }
          } catch { fail('音声エラー'); }
        }
      },
      error: cause => {
        if (!current()) return;
        // Repair only temporary scene state, retaining the ready model/task.
        discard('cancelled');
        inferenceFailed = true;
        engine.clearDetections();
        latestFrameRef.current = null;
        dirty = true;
        if (!warned) { warned = true; console.warn('Vision frame failed; waiting for a new frame', { cause }); }
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
          const copied = drawCanvasFrame(canvas, video, frame, offscreenCanvasRef, drawRef.current, current);
          drawOutcome = current() ? 'success' : 'cancelled';
          drawProbe.end('draw', drawStart, drawOutcome); drawEnded = true;
          if (copied) {
            if (needsFirstDraw && latestMeasurement) {
              telemetry(() => pipeline.firstDraw(latestMeasurement!));
              needsFirstDraw = false;
            }
            metrics.drawCompleted();
          }
        }
      } catch { drawOutcome = 'failed'; drawProbe.end('draw', drawStart, drawOutcome); drawEnded = true; fail('描画エラー'); }
      finally { if (!drawEnded) drawProbe.end('draw', drawStart, drawOutcome); if (current()) drawId = requestAnimationFrame(draw); }
    };
    drawId = requestAnimationFrame(draw);
    const hud = setInterval(() => {
      if (!current()) return;
      setStats(metrics.snapshot());
      const snapshot = telemetry(() => pipeline.snapshot());
      setPipelineStats(snapshot ?? null);
      setInferenceError(inferenceFailed);
      if (dirty) {
        dirty = false;
        const frame = latestFrameRef.current;
        setCurrentFrame(frame);
        if (frame !== reported) {
          reported = frame;
          try { latest.current.onFrame(frame); } catch { fail('結果通知エラー'); }
        }
      }
    }, HUD_INTERVAL_MS);
    return () => {
      discard('cancelled');
      document.removeEventListener('visibilitychange', visibility);
      stopped = true;
      cancelAcquisition();
      clearInterval(hud);
      if (drawId !== null) cancelAnimationFrame(drawId);
    };
  }, [isCameraActive, cameraReady, cameraRequestId, activeModel, engine, modelReady, requestId, fault, retry, threshold]);

  const drawOverlays = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    objects: DetectedObject[]
  ) => {
    // 1. Draw Masks
    if (renderOptions.showMasks) {
      objects.forEach((obj) => {
        const isSelected = obj.id === selectedObjectId;
        const opacity = isSelected ? Math.min(1, renderOptions.maskOpacity + 0.3) : renderOptions.maskOpacity;

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
            ? ` ${Math.round(obj.confidence * 100)}%`
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

  useLayoutEffect(() => { drawRef.current = drawOverlays; });
  const rippleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (rippleTimer.current !== null) clearTimeout(rippleTimer.current); }, []);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    // Read current geometry at input time, including a resize or new source size
    // that has not reached the next overlay drawing frame yet.
    const point = clientPointToVideo(e.clientX, e.clientY, video.getBoundingClientRect(), video.videoWidth, video.videoHeight);
    if (!point) return;
    const { x: clickX, y: clickY } = point;

    const hit = engine.findObjectAtPoint(clickX, clickY, currentObjects);

    if (hit) {
      try { onSelectObject(hit); } catch { setFault('結果通知エラー'); return; }
      setClickRipple({ x: clickX, y: clickY, label: hit.labelJa });
      if (rippleTimer.current !== null) clearTimeout(rippleTimer.current);
      rippleTimer.current = setTimeout(() => setClickRipple(null), 1200);

      // Force speak on direct user tap
      try { speechService.speak(
        `「${hit.labelJa}」、信頼度${Math.round(hit.confidence * 100)}%`,
        hit.labelJa,
        { ...ttsConfig, enabled: true },
        'tap'
      ); } catch { setFault('音声エラー'); }
    }
  };

  return (
    <div className="relative w-full h-full max-h-[calc(100vh-4.5rem)] aspect-[4/3] sm:aspect-video mx-auto bg-[#040711] rounded-xl overflow-hidden border border-slate-800 shadow-xl flex items-center justify-center select-none">
      {/* Video Stream Element */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`w-full h-full object-contain ${isCameraActive ? 'block' : 'hidden'}`}
      />

      {/* Interactive Overlay Canvas */}
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="absolute inset-0 w-full h-full object-contain z-10 cursor-pointer"
      />

      {/* Camera Inactive Standby Screen */}
      {!isCameraActive && (
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

      {/* Camera Flip Button (Top-Right Corner of video) */}
      {isCameraActive && (
        <button
          onClick={onToggleFacingMode}
          title={facingMode === 'user' ? 'アウトカメラ（背面）に切り替え' : 'インカメラ（前面）に切り替え'}
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
        <span aria-label="カメラ状態" className="ml-2">{cameraReady ? 'カメラ映像を取得しました' : isCameraActive && !cameraError ? 'カメラ準備中' : 'カメラ停止中'}</span>
        <span aria-label="性能状態" className="ml-2">{fault ?? (modelState.status === 'error' ? 'モデルエラー' : !isCameraActive ? '停止中' : !modelReady ? 'モデル読込中' : !cameraReady ? 'カメラ準備中' : inferenceError ? '推論エラー：次の映像を待機' : stats.stalled ? '映像入力停滞' : '計測中')}</span>
        <span aria-label="推論FPS" className="ml-2">推論: {stats.inferenceFps === null || fault || !cameraReady || !modelReady ? '未計測' : `${stats.inferenceFps.toFixed(1)} FPS`}</span>
        <span aria-label="描画FPS" className="ml-2">描画: {stats.drawFps === null || fault || !cameraReady || !modelReady ? '未計測' : `${stats.drawFps.toFixed(1)} FPS`}</span>
        <span aria-label="フレーム処理時間" className="ml-2">フレーム処理: {stats.processingMs === null || fault || !cameraReady || !modelReady ? '未計測' : `${stats.processingMs.toFixed(1)} ms`}</span>
      </div>

      <PipelineDetails snapshot={pipelineStats} />

      {fault && <div className="absolute bottom-10 z-30 bg-slate-900 p-2 text-xs" role="alert">{fault}：処理を停止しました。<button onClick={() => setRetry(value => value + 1)}>処理を再試行</button></div>}

      {/* Tap Hint (Bottom Center) */}
      {isCameraActive && (
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <span className="text-[10px] text-slate-400 bg-black/60 backdrop-blur-md border border-slate-800 px-2.5 py-0.5 rounded-full">
            物体をタップすると日本語で発話
          </span>
        </div>
      )}

      {/* Camera Error Screen */}
      {cameraError && (
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
