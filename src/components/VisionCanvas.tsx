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
import { modelMetadata } from '../services/modelRegistry';

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

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [clickRipple, setClickRipple] = useState<{ x: number; y: number; label: string } | null>(null);
  const [currentFrame, setCurrentFrame] = useState<DetectionFrame | null>(null);
  const modelReady = modelState.status === 'ready' && modelState.model === activeModel;
  const requestId = modelState.requestId;
  const currentObjects = modelReady && currentFrame?.model === activeModel && currentFrame.requestId === requestId
    ? currentFrame.objects : [];

  const animFrameIdRef = useRef<number | null>(null);
  const lastReportedTimeRef = useRef<number>(0);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const isInferringRef = useRef<boolean>(false);
  const latestFrameRef = useRef<DetectionFrame | null>(null);

  // Clear the visible old overlay before paint; initialization and camera state
  // remain separate. New results carry the engine generation, not just a label.
  useLayoutEffect(() => {
    latestFrameRef.current = null;
    setCurrentFrame(null);
    setClickRipple(null);
    onFrame(null);
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  }, [activeModel, requestId, isCameraActive, onFrame]);

  // Handle webcam stream start/stop and facing mode flip
  useEffect(() => {
    let stream: MediaStream | null = null;

    if (isCameraActive) {
      setCameraError(null);

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      navigator.mediaDevices
        ?.getUserMedia(constraints)
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = s;
            videoRef.current.play().catch(e => console.warn('Video play error:', e));
          }
        })
        .catch((err) => {
          console.warn('Webcam permission error:', err);
          setCameraError(
            err.name === 'NotAllowedError'
              ? 'カメラへのアクセスが拒否されました。ブラウザの設定でカメラ許可をオンにしてください。'
              : 'カメラデバイスが見つからないか使用中です。'
          );
          onCameraStop();
        });
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, [isCameraActive, facingMode, onCameraStop]);

  // Main rendering loop for video frames and overlay canvas (Double-buffered & Decoupled)
  useEffect(() => {
    let isRunning = true;
    let inferenceTimer: ReturnType<typeof setTimeout> | undefined;

    const renderLoop = () => {
      if (!isRunning) return;

      const canvas = canvasRef.current;
      const video = videoRef.current;

      if (!canvas || !video) {
        animFrameIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animFrameIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      // Sync canvas dimensions with video
      if (isCameraActive && modelReady && video.readyState >= 2) {
        const targetW = video.videoWidth || 640;
        const targetH = video.videoHeight || 480;

        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }

        // DOUBLE BUFFERING: Ensure offscreen memory buffer matches dimensions
        if (!offscreenCanvasRef.current) {
          offscreenCanvasRef.current = document.createElement('canvas');
        }
        const offscreen = offscreenCanvasRef.current;
        if (offscreen.width !== targetW || offscreen.height !== targetH) {
          offscreen.width = targetW;
          offscreen.height = targetH;
        }

        // Preserve the existing timer-based scheduling. Cancel its one pending
        // callback when readiness changes so the new generation can start.
        if (!isInferringRef.current) {
          isInferringRef.current = true;
          inferenceTimer = setTimeout(() => {
            try {
              if (!isRunning || engine.getState().requestId !== requestId) return;
              const result = engine.detectVideoFrame(video, activeModel, renderOptions.confidenceThreshold);
              latestFrameRef.current = result.status === 'ready' && result.requestId === requestId ? result : null;
            } catch (cause) {
              console.warn('Vision frame failed', { model: activeModel, cause });
              latestFrameRef.current = null;
            } finally {
              isInferringRef.current = false;
            }
          }, 0);
        }

        const frame = latestFrameRef.current;
        const currentDetections = frame?.requestId === requestId ? frame.objects : [];

        // 1. RENDER OVERLAYS ONTO IN-MEMORY OFFSCREEN BUFFER (NEVER BLANK ON SCREEN)
        const offCtx = offscreen.getContext('2d');
        if (offCtx) {
          offCtx.clearRect(0, 0, offscreen.width, offscreen.height);
          drawOverlays(offCtx, offscreen.width, offscreen.height, currentDetections);

          // Copy the overlay buffer to the visible canvas.
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(offscreen, 0, 0);
        }

        setCurrentFrame(frame);

        // Rate limit parent state updates (max once per 200ms) to prevent React DOM re-render thrashing
        const nowMs = performance.now();
        if (nowMs - lastReportedTimeRef.current >= 200) {
          lastReportedTimeRef.current = nowMs;
          onFrame(frame);
        }

        // Auto TTS if enabled
        if (ttsConfig.enabled && ttsConfig.autoReadout && currentDetections.length > 0) {
          const prioritized = [...currentDetections].sort((a, b) => b.confidence - a.confidence);
          for (const obj of prioritized) {
            if (speechService.canSpeakLabel(obj.labelJa, ttsConfig.cooldownSec)) {
              speechService.speak(
                `${obj.labelJa}を認識しました`,
                obj.labelJa,
                ttsConfig,
                'auto'
              );
              break;
            }
          }
        }
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      animFrameIdRef.current = requestAnimationFrame(renderLoop);
    };

    animFrameIdRef.current = requestAnimationFrame(renderLoop);

    return () => {
      isRunning = false;
      clearTimeout(inferenceTimer);
      isInferringRef.current = false;
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
    // Retain the existing drawOverlays closure exception (including clickRipple).
    // PR-02 adds only engine readiness/generation and pending-timer cancellation;
    // full rendering/scheduler and overlay closure repair stays in PR-04/06.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isCameraActive, activeModel, engine, modelReady, requestId, renderOptions, ttsConfig, onFrame, selectedObjectId]);

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

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) / rect.width;
    const clickY = (e.clientY - rect.top) / rect.height;

    const hit = engine.findObjectAtPoint(clickX, clickY, currentObjects);

    if (hit) {
      onSelectObject(hit);
      setClickRipple({ x: clickX, y: clickY, label: hit.labelJa });
      setTimeout(() => setClickRipple(null), 1200);

      // Force speak on direct user tap
      speechService.speak(
        `「${hit.labelJa}」、信頼度${Math.round(hit.confidence * 100)}%`,
        hit.labelJa,
        { ...ttsConfig, enabled: true },
        'tap'
      );
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
        <span className="ml-2">性能：計測未整備</span>
      </div>

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
              onClick={() => setCameraError(null)}
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
