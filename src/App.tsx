import { ImageClassificationWorkspace } from './components/ImageClassificationWorkspace';
import type { ImageClassificationEngine } from './services/imageClassificationEngine';
import type { ImageClassifierVariantId, ModelFamilyType } from './types/classification';
import React, { useState, useEffect, useCallback, useMemo, useSyncExternalStore } from 'react';
import {
  DetectionFrame,
  CameraFacingMode,
  RunnableModelType,
  RenderOptions,
  TTSConfig,
  PerModelParameterConfig,
  DEFAULT_MODEL_PARAMS,
  StereoDepthProbeResult,
  StereoPinnedProbe,
  StereoDepthVariantId,
  StereoBackendPreference,
} from './types/vision';
import { Header } from './components/Header';
import { VisionCanvas, type StillImageSource, type StereoVideoSource } from './components/VisionCanvas';
import { ControlPanel } from './components/ControlPanel';
import { ModelComparisonTable } from './components/ModelComparisonTable';
import { VisionEngine, type ExtraInferenceOptions } from './services/visionEngine';
import { modelMetadata, isRunnableModel, getVariantMetadata } from './services/modelRegistry';
import {
  loadPersistedModelSelection,
  savePersistedModelSelection,
} from './services/modelSelectionPersistence';
import { isMirageExperimentalVariant } from './services/mirageExperimentalOnnxService';
import { clearLocalStereoOnnxBytes, registerLocalStereoOnnxBytes } from './services/stereoDepthService';
import {
  STEREO_VIDEO_PRESETS,
  type StereoVideoGroundTruthRegion,
  type StereoVideoPresetId,
} from './services/stereoVideoService';
import { ModelStatus } from './components/ModelStatus';
import { DiagnosticsModal } from './components/DiagnosticsModal';
import { diagnosticLogger } from './services/diagnosticLogger';
import { speechService } from './services/speechService';

function drawSyntheticFixtureCanvas(
  preset: 'overlap_shapes' | 'text_regions'
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  if (preset === 'overlap_shapes') {
    canvas.width = 480;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'rgb(245, 245, 242)';
      ctx.fillRect(0, 0, 480, 360);
      ctx.fillStyle = 'rgb(215, 48, 39)';
      ctx.fillRect(40, 50, 180, 170);
      ctx.fillStyle = 'rgb(69, 117, 180)';
      ctx.beginPath();
      ctx.ellipse(240, 190, 90, 90, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgb(26, 152, 80)';
      ctx.beginPath();
      ctx.moveTo(270, 290);
      ctx.lineTo(430, 290);
      ctx.lineTo(360, 110);
      ctx.closePath();
      ctx.fill();
    }
  } else {
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'rgb(252, 252, 248)';
      ctx.fillRect(0, 0, 640, 360);
      ctx.strokeStyle = 'rgb(30, 30, 30)';
      ctx.lineWidth = 3;
      ctx.strokeRect(30, 28, 580, 102);
      ctx.fillStyle = 'rgb(10, 10, 10)';
      ctx.font = 'bold 28px sans-serif';
      ctx.fillText('MIRAGE OCR 123', 48, 72);
      ctx.fillStyle = 'rgb(25, 25, 90)';
      ctx.fillText(' FAST EDGE VLM 2026', 48, 112);
      ctx.fillStyle = 'rgb(230, 140, 20)';
      ctx.fillRect(70, 185, 170, 125);
      ctx.fillStyle = 'rgb(90, 50, 140)';
      ctx.beginPath();
      ctx.ellipse(445, 245, 75, 65, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  return canvas;
}

function drawTexturedPanel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  baseRgb: readonly [number, number, number],
  accentRgb: readonly [number, number, number],
  seed: number,
  label?: string
) {
  ctx.fillStyle = `rgb(${baseRgb[0]}, ${baseRgb[1]}, ${baseRgb[2]})`;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = `rgb(${accentRgb[0]}, ${accentRgb[1]}, ${accentRgb[2]})`;
  const cell = 10;
  for (let py = 0; py < h; py += cell) {
    for (let px = 0; px < w; px += cell) {
      const hash = ((px * 73 + py * 179 + seed * 31) >>> 0) % 7;
      if (hash <= 2) {
        ctx.fillRect(x + px + 1, y + py + 1, cell - 3, cell - 3);
      }
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.65)';
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  if (label) {
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
    ctx.fillRect(x + 6, y + 6, Math.min(w - 12, label.length * 8 + 10), 18);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 11px monospace';
    ctx.fillText(label, x + 10, y + 19);
  }
}

function drawSyntheticStereoSbsCanvas(
  preset: 'stereo_blocks' | 'stereo_corridor'
): HTMLCanvasElement {
  const eyeW = 480;
  const eyeH = 270;
  const canvas = document.createElement('canvas');
  canvas.width = eyeW * 2;
  canvas.height = eyeH;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;

  const drawEye = (offsetX: number, eyeSign: 0 | 1) => {
    ctx.save();
    ctx.beginPath();
    ctx.rect(offsetX, 0, eyeW, eyeH);
    ctx.clip();

    if (preset === 'stereo_blocks') {
      // Background plane: subtle textured wall at disparity = 4px
      const bgDisp = eyeSign === 1 ? -4 : 0;
      drawTexturedPanel(ctx, offsetX + bgDisp - 12, -12, eyeW + 24, eyeH + 24, [38, 46, 66], [62, 74, 102], 11);

      // Far block (disparity = 8px)
      const farDisp = eyeSign === 1 ? -8 : 0;
      drawTexturedPanel(ctx, offsetX + 295 + farDisp, 38, 130, 120, [48, 105, 148], [86, 156, 204], 23, 'FAR (8px)');

      // Mid block (disparity = 18px)
      const midDisp = eyeSign === 1 ? -18 : 0;
      drawTexturedPanel(ctx, offsetX + 165 + midDisp, 75, 130, 135, [44, 145, 112], [90, 200, 160], 37, 'MID (18px)');

      // Near foreground block (disparity = 32px)
      const nearDisp = eyeSign === 1 ? -32 : 0;
      drawTexturedPanel(ctx, offsetX + 55 + nearDisp, 110, 115, 125, [196, 86, 48], [242, 142, 84], 53, 'NEAR (32px)');
    } else {
      // Corridor / depth-gradient scene with stepped depth planes
      const backDisp = eyeSign === 1 ? -4 : 0;
      drawTexturedPanel(ctx, offsetX + backDisp - 16, -16, eyeW + 32, eyeH + 32, [28, 36, 54], [48, 60, 88], 7);

      const bands: ReadonlyArray<{ x: number; y: number; w: number; h: number; d: number; base: [number, number, number]; acc: [number, number, number]; seed: number; label: string }> = [
        { x: 185, y: 80, w: 110, h: 110, d: 6, base: [56, 88, 146], acc: [94, 134, 198], seed: 13, label: '6px' },
        { x: 105, y: 55, w: 80, h: 160, d: 14, base: [42, 132, 144], acc: [82, 186, 196], seed: 19, label: '14px' },
        { x: 295, y: 55, w: 85, h: 160, d: 14, base: [42, 132, 144], acc: [82, 186, 196], seed: 29, label: '14px' },
        { x: 24, y: 28, w: 85, h: 214, d: 26, base: [178, 126, 40], acc: [228, 178, 76], seed: 41, label: '26px' },
        { x: 375, y: 28, w: 85, h: 214, d: 26, base: [178, 126, 40], acc: [228, 178, 76], seed: 47, label: '26px' },
        { x: 205, y: 155, w: 90, h: 95, d: 34, base: [194, 62, 62], acc: [242, 114, 108], seed: 61, label: '34px' },
      ];
      for (const b of bands) {
        const shift = eyeSign === 1 ? -b.d : 0;
        drawTexturedPanel(ctx, offsetX + b.x + shift, b.y, b.w, b.h, b.base, b.acc, b.seed, b.label);
      }
    }
    ctx.restore();
  };

  drawEye(0, 0);
  drawEye(eyeW, 1);
  return canvas;
}

export default function App({ engine: suppliedEngine, classificationEngine }: { engine?: VisionEngine; classificationEngine?: ImageClassificationEngine } = {}) {
  const [initialSelection] = useState(() => loadPersistedModelSelection());
  const [classifierMode, setClassifierMode] = useState(initialSelection.classifierMode);
  const [classifierVariant, setClassifierVariant] = useState<ImageClassifierVariantId>(initialSelection.classifierVariant);
  const [engine] = useState(() => suppliedEngine ?? new VisionEngine());
  const modelState = useSyncExternalStore(engine.subscribe, engine.getState);
  const [activeModel, setActiveModel] = useState<RunnableModelType>(initialSelection.activeModel);
  const [selectedVariants, setSelectedVariants] = useState<Record<RunnableModelType, string>>(
    initialSelection.selectedVariants
  );
  const activeVariant = selectedVariants[activeModel] || modelMetadata(activeModel).defaultVariantId;

  useEffect(() => {
    savePersistedModelSelection({
      activeModel,
      selectedVariants,
      classifierMode,
      classifierVariant,
    });
  }, [activeModel, selectedVariants, classifierMode, classifierVariant]);

  const [facingMode, setFacingMode] = useState<CameraFacingMode>('user');
  const [isCameraActive, setIsCameraActive] = useState<boolean>(true);
  const [stillImageSource, setStillImageSource] = useState<StillImageSource | null>(null);
  const [stereoVideoSource, setStereoVideoSource] = useState<StereoVideoSource | null>(null);
  const [stereoVideoRegions, setStereoVideoRegions] = useState<StereoVideoGroundTruthRegion[]>([]);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isComparisonOpen, setIsComparisonOpen] = useState<boolean>(false);
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState<boolean>(false);

  const [selectedObjectId, setSelectedObjectId] = useState<string | undefined>(undefined);
  const [lastFrame, setLastFrame] = useState<DetectionFrame | null>(null);
  const [stereoHoverProbe, setStereoHoverProbe] = useState<StereoDepthProbeResult | null>(null);
  const [stereoPinnedProbes, setStereoPinnedProbes] = useState<StereoPinnedProbe[]>([]);
  const hasCurrentFrame =
    (isCameraActive || Boolean(stillImageSource) || Boolean(stereoVideoSource)) &&
    modelState.status === 'ready' &&
    modelState.model === activeModel &&
    lastFrame?.model === activeModel &&
    lastFrame.requestId === modelState.requestId;
  const detectedObjects = hasCurrentFrame ? lastFrame.objects : [];
  const activeStereoDepthMap = hasCurrentFrame ? (lastFrame.depthMap ?? null) : null;

  useEffect(() => () => engine.dispose(), [engine]);
  useEffect(() => {
    return () => {
      if (stereoVideoSource?.kind === 'uploaded-file') {
        URL.revokeObjectURL(stereoVideoSource.fileUrl);
      }
    };
  }, [stereoVideoSource]);
  useEffect(() => {
    diagnosticLogger.enableServerTelemetry(true);
  }, []);
  const [modelParams, setModelParams] = useState<PerModelParameterConfig>(() =>
    JSON.parse(JSON.stringify(DEFAULT_MODEL_PARAMS))
  );

  const activeParams = modelParams[activeModel] || DEFAULT_MODEL_PARAMS[activeModel];
  const activeStereoBackendPref: StereoBackendPreference =
    activeModel === 'stereo-depth'
      ? (modelParams['stereo-depth']?.inference?.backendPreference ?? 'auto')
      : 'auto';

  useEffect(() => {
    if (!classifierMode) {
      void engine.initialize(
        activeModel,
        activeVariant,
        activeModel === 'stereo-depth'
          ? { stereoBackendPreference: activeStereoBackendPref }
          : undefined
      );
    }
  }, [engine, activeModel, activeVariant, classifierMode, activeStereoBackendPref]);

  const isExperimentalMirage = activeModel === 'mirage' && isMirageExperimentalVariant(activeVariant);
  const activeRenderOptions: RenderOptions = useMemo(
    () =>
      isExperimentalMirage && activeParams.render.confidenceThreshold === DEFAULT_MODEL_PARAMS.mirage.render.confidenceThreshold
        ? { ...activeParams.render, confidenceThreshold: 0.30 }
        : activeParams.render,
    [isExperimentalMirage, activeParams.render]
  );
  const activeInferenceParams: ExtraInferenceOptions = useMemo(
    () =>
      isExperimentalMirage && activeParams.inference.confidenceThreshold === DEFAULT_MODEL_PARAMS.mirage.inference.confidenceThreshold
        ? { ...activeParams.inference, confidenceThreshold: 0.30, maxDetections: 64 }
        : activeParams.inference,
    [isExperimentalMirage, activeParams.inference]
  );

  const handleChangeRenderOptions = (opts: Partial<RenderOptions>) => {
    setModelParams(prev => {
      const current = prev[activeModel] || DEFAULT_MODEL_PARAMS[activeModel];
      const newThreshold = opts.confidenceThreshold ?? current.render.confidenceThreshold;
      return {
        ...prev,
        [activeModel]: {
          ...current,
          render: { ...current.render, ...opts },
          inference: { ...current.inference, confidenceThreshold: newThreshold }
        }
      };
    });
  };

  const handleChangeInferenceParams = (params: Partial<ExtraInferenceOptions>) => {
    if ('backendPreference' in params && activeModel === 'stereo-depth') {
      setLastFrame(null);
    }
    setModelParams(prev => {
      const current = prev[activeModel] || DEFAULT_MODEL_PARAMS[activeModel];
      const newThreshold = params.confidenceThreshold ?? current.inference.confidenceThreshold;
      return {
        ...prev,
        [activeModel]: {
          ...current,
          inference: { ...current.inference, ...params },
          render: { ...current.render, confidenceThreshold: newThreshold }
        }
      };
    });
  };

  const handleResetModelParams = () => {
    if (activeModel === 'stereo-depth') {
      setLastFrame(null);
    }
    setModelParams(prev => ({
      ...prev,
      [activeModel]: JSON.parse(JSON.stringify(DEFAULT_MODEL_PARAMS[activeModel]))
    }));
  };

  const handleSelectStillImagePreset = useCallback(
    async (
      preset:
        | 'camera'
        | 'overlap_shapes'
        | 'text_regions'
        | 'coco_000000458054'
        | 'coco_000000331352'
    ) => {
      setStereoVideoSource(null);
      setStereoVideoRegions([]);
      if (preset === 'camera') {
        setStillImageSource(null);
        return;
      }
      try {
        const res = await fetch(`/api/model-cache/mirage/fixtures/${preset}.png`);
        if (res.ok) {
          const blob = await res.blob();
          const bitmap = await createImageBitmap(blob);
          const canvas = document.createElement('canvas');
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(bitmap, 0, 0);
          bitmap.close?.();
          setStillImageSource({
            id: `${preset}-${Date.now()}`,
            label: preset,
            canvas,
          });
          return;
        }
      } catch {
        // Fall through to deterministic synthetic canvas when fixture endpoint is unavailable
      }
      if (preset === 'overlap_shapes' || preset === 'text_regions') {
        const canvas = drawSyntheticFixtureCanvas(preset);
        setStillImageSource({
          id: `${preset}-${Date.now()}`,
          label: preset,
          canvas,
        });
      }
    },
    []
  );

  const handleSelectStereoStillPreset = useCallback(
    (preset: 'camera' | 'stereo_blocks' | 'stereo_corridor') => {
      setStereoHoverProbe(null);
      setStereoPinnedProbes([]);
      setStereoVideoSource(null);
      setStereoVideoRegions([]);
      if (preset === 'camera') {
        setStillImageSource(null);
        return;
      }
      const canvas = drawSyntheticStereoSbsCanvas(preset);
      setStillImageSource({
        id: `${preset}-${Date.now()}`,
        label: preset,
        canvas,
      });
    },
    []
  );

  const handleSelectStereoVideoPreset = useCallback((presetId: StereoVideoPresetId) => {
    const meta = STEREO_VIDEO_PRESETS[presetId];
    if (!meta) return;
    setStereoHoverProbe(null);
    setStereoPinnedProbes([]);
    setStillImageSource(null);
    setStereoVideoSource({
      kind: 'preset',
      presetId,
      id: `${presetId}-${Date.now()}`,
      label: meta.shortLabel,
    });
  }, []);

  const handleUploadStereoVideoFile = useCallback((file: File) => {
    const fileUrl = URL.createObjectURL(file);
    setStereoHoverProbe(null);
    setStereoPinnedProbes([]);
    setStillImageSource(null);
    setStereoVideoRegions([]);
    setStereoVideoSource({
      kind: 'uploaded-file',
      id: `sbs-video-${file.name}-${Date.now()}`,
      label: `SBS動画: ${file.name}`,
      fileUrl,
      fileName: file.name,
    });
  }, []);

  const handleClearStereoVideo = useCallback(() => {
    setStereoVideoSource(null);
    setStereoVideoRegions([]);
    setStereoHoverProbe(null);
  }, []);

  const handleLoadStereoOnnxFile = useCallback(
    async (variantId: StereoDepthVariantId, file: File) => {
      try {
        const buffer = await file.arrayBuffer();
        registerLocalStereoOnnxBytes(variantId, new Uint8Array(buffer), file.name);
        if (activeModel === 'stereo-depth' && activeVariant === variantId) {
          engine.dispose();
          setLastFrame(null);
          void engine.initialize('stereo-depth', variantId, {
            stereoBackendPreference: activeStereoBackendPref,
          });
        }
      } catch (err) {
        console.warn('Failed to load stereo ONNX file', err);
      }
    },
    [activeModel, activeVariant, activeStereoBackendPref, engine]
  );

  const handleClearStereoOnnxFile = useCallback(
    (variantId: StereoDepthVariantId) => {
      clearLocalStereoOnnxBytes(variantId);
      if (activeModel === 'stereo-depth' && activeVariant === variantId) {
        engine.dispose();
        setLastFrame(null);
        void engine.initialize('stereo-depth', variantId, {
          stereoBackendPreference: activeStereoBackendPref,
        });
      }
    },
    [activeModel, activeVariant, activeStereoBackendPref, engine]
  );

  const handlePinStereoProbe = useCallback(
    (probe: StereoDepthProbeResult) => {
      setStereoPinnedProbes((prev) => {
        const nextIdx = prev.length + 1;
        const pin: StereoPinnedProbe = {
          ...probe,
          id: `pin-${Date.now()}-${nextIdx}`,
          label: `P${nextIdx}`,
          variantId: activeVariant,
        };
        return [...prev, pin];
      });
    },
    [activeVariant]
  );

  const handleUploadStillImage = useCallback(async (file: File) => {
    try {
      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(bitmap, 0, 0);
      bitmap.close?.();
      setStereoHoverProbe(null);
      setStereoPinnedProbes([]);
      setStereoVideoSource(null);
      setStereoVideoRegions([]);
      setStillImageSource({
        id: `upload-${file.name}-${Date.now()}`,
        label: file.name,
        canvas,
      });
    } catch (err) {
      console.warn('Failed to decode uploaded still image', err);
    }
  }, []);

  // Default: Voice readout is OFF (enabled: false)
  const [ttsConfig, setTTSConfig] = useState<TTSConfig>({
    enabled: false,
    autoReadout: false,
    cooldownSec: 3.0,
    rate: 1.0,
    pitch: 1.0,
    volume: 1.0,
    voiceURI: ''
  });

  useEffect(() => {
    const unsub = speechService.subscribe((speaking) => {
      setIsSpeaking(speaking);
    });
    return () => {
      unsub();
    };
  }, []);

  const handleSelectModel = (model: RunnableModelType) => {
    if (!isRunnableModel(model) || model === activeModel) return;
    engine.dispose();
    setLastFrame(null);
    setActiveModel(model);
    setSelectedObjectId(undefined);
    if (model !== 'stereo-depth') {
      setStereoVideoSource(null);
      setStereoVideoRegions([]);
    }
    const variantId = selectedVariants[model] || modelMetadata(model).defaultVariantId;
    if (ttsConfig.enabled) {
      speechService.speak(
        `${modelMetadata(model).name}（${getVariantMetadata(model, variantId).name}）に切り替えました`,
        'model-switch',
        ttsConfig,
        'manual'
      );
    }
  };

  const handleSelectVariant = (variantId: string) => {
    if (selectedVariants[activeModel] === variantId) return;
    engine.dispose();
    setLastFrame(null);
    setSelectedVariants(prev => ({ ...prev, [activeModel]: variantId }));
    setSelectedObjectId(undefined);
    if (ttsConfig.enabled) {
      speechService.speak(
        `${modelMetadata(activeModel).name}（${getVariantMetadata(activeModel, variantId).name}）に切り替えました`,
        'variant-switch',
        ttsConfig,
        'manual'
      );
    }
  };

  const handleSelectModelAndVariant = (model: RunnableModelType, variantId: string) => {
    if (!isRunnableModel(model)) return;
    engine.dispose();
    setLastFrame(null);
    setSelectedVariants(prev => ({ ...prev, [model]: variantId }));
    setActiveModel(model);
    setSelectedObjectId(undefined);
    if (model !== 'stereo-depth') {
      setStereoVideoSource(null);
      setStereoVideoRegions([]);
    }
    if (ttsConfig.enabled) {
      speechService.speak(
        `${modelMetadata(model).name}（${getVariantMetadata(model, variantId).name}）に切り替えました`,
        'model-switch',
        ttsConfig,
        'manual'
      );
    }
  };

  const handleToggleFacingMode = useCallback(() => {
    setFacingMode(prev => (prev === 'user' ? 'environment' : 'user'));
  }, []);
  const handleCameraStart = useCallback(() => {
    setStillImageSource(null);
    setStereoVideoSource(null);
    setStereoVideoRegions([]);
    setIsCameraActive(true);
  }, []);
  const handleCameraStop = useCallback(() => setIsCameraActive(false), []);
  const handleToggleCamera = useCallback(() => {
    setStillImageSource(null);
    setStereoVideoSource(null);
    setStereoVideoRegions([]);
    setIsCameraActive(previous => !previous);
  }, []);

  const handleFrame = useCallback((frame: DetectionFrame | null) => {
    setLastFrame(frame);
    if (!frame) setSelectedObjectId(undefined);
  }, []);

  const selectFamily = (model: ModelFamilyType, variant?: string) => {
    if (model === 'image-classifier') {
      engine.dispose(); speechService.cancel(); setLastFrame(null); setSelectedObjectId(undefined);
      if (variant === 'lite0' || variant === 'lite2') setClassifierVariant(variant);
      setClassifierMode(true); setIsComparisonOpen(false);
    } else {
      speechService.cancel(); setClassifierMode(false);
      if (variant) handleSelectModelAndVariant(model, variant); else handleSelectModel(model);
    }
  };
  if (classifierMode) return <ImageClassificationWorkspace engine={classificationEngine} variant={classifierVariant} onVariant={setClassifierVariant}
    onSelectDetection={selectFamily} cameraActive={isCameraActive} facingMode={facingMode} onToggleFacing={handleToggleFacingMode}
    onStart={handleCameraStart} onStop={handleCameraStop} ttsConfig={ttsConfig} />;

  const experimentalDiagnostics =
    isExperimentalMirage &&
    (hasCurrentFrame ||
      (modelState.status === 'ready' && modelState.model === activeModel))
      ? engine.getExperimentalMirageDiagnostics()
      : null;

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Compact Top Bar */}
      <Header
        activeModel={activeModel}
        onSelectModel={selectFamily}
        activeVariant={activeVariant}
        onSelectVariant={handleSelectVariant}
        modelState={modelState}
        isCameraActive={isCameraActive}
        onToggleCamera={handleToggleCamera}
        ttsEnabled={ttsConfig.enabled}
        onToggleTTS={() => setTTSConfig(prev => ({ ...prev, enabled: !prev.enabled }))}
        isSpeaking={isSpeaking}
        onOpenComparison={() => setIsComparisonOpen(true)}
        onOpenDiagnostics={() => setIsDiagnosticsOpen(true)}
      />

      <ModelStatus
        state={modelState}
        onRetry={() => {
          void engine.initialize(
            activeModel,
            activeVariant,
            activeModel === 'stereo-depth'
              ? { stereoBackendPreference: activeStereoBackendPref }
              : undefined
          );
        }}
        onOpenDiagnostics={() => setIsDiagnosticsOpen(true)}
      />

      {/* Zero-Scroll Main Stage (Side-by-Side) */}
      <main className="flex-1 min-h-0 overflow-hidden p-2.5 sm:p-4 max-w-7xl w-full mx-auto grid grid-cols-1 grid-rows-[auto_1fr] md:grid-rows-1 md:grid-cols-12 gap-3 items-stretch">
        {/* Left Column: Compact Camera Preview with In-Preview Flip Button */}
        <div className="md:col-span-8 h-full min-h-0 min-w-0 flex flex-col justify-center overflow-hidden">
          <VisionCanvas
            activeModel={activeModel}
            engine={engine}
            modelState={modelState}
            facingMode={facingMode}
            onToggleFacingMode={handleToggleFacingMode}
            renderOptions={activeRenderOptions}
            inferenceParams={activeInferenceParams}
            ttsConfig={ttsConfig}
            isCameraActive={isCameraActive}
            onCameraStart={handleCameraStart}
            onCameraStop={handleCameraStop}
            onSelectObject={(obj) => setSelectedObjectId(obj.id)}
            selectedObjectId={selectedObjectId}
            onFrame={handleFrame}
            stillImageSource={stillImageSource}
            onClearStillImage={() => setStillImageSource(null)}
            stereoVideoSource={stereoVideoSource}
            onClearStereoVideo={handleClearStereoVideo}
            onStereoVideoRegions={setStereoVideoRegions}
            stereoHoverProbe={stereoHoverProbe}
            onStereoHoverProbe={setStereoHoverProbe}
            stereoPinnedProbes={stereoPinnedProbes}
            onPinStereoProbe={handlePinStereoProbe}
          />
        </div>

        {/* Right Column: Compact Control & Detection Dock */}
        <div className="md:col-span-4 h-full min-h-0 min-w-0 flex flex-col overflow-hidden">
          <ControlPanel
            activeModel={activeModel}
            activeVariant={activeVariant}
            modelState={modelState}
            renderOptions={activeRenderOptions}
            onChangeRenderOptions={handleChangeRenderOptions}
            inferenceParams={activeInferenceParams}
            onChangeInferenceParams={handleChangeInferenceParams}
            onResetModelParams={handleResetModelParams}
            ttsConfig={ttsConfig}
            onChangeTTSConfig={(cfg) => setTTSConfig(prev => ({ ...prev, ...cfg }))}
            detectedObjects={detectedObjects}
            hasCurrentFrame={hasCurrentFrame}
            onSelectObject={(obj) => setSelectedObjectId(obj.id)}
            selectedObjectId={selectedObjectId}
            isSpeaking={isSpeaking}
            experimentalDiagnostics={experimentalDiagnostics}
            stillImageLabel={stillImageSource?.label ?? null}
            onSelectStillImagePreset={handleSelectStillImagePreset}
            onUploadStillImage={handleUploadStillImage}
            stereoDepthMap={activeStereoDepthMap}
            stereoHoverProbe={stereoHoverProbe}
            stereoPinnedProbes={stereoPinnedProbes}
            onClearStereoPins={() => setStereoPinnedProbes([])}
            onRemoveStereoPin={(id) => setStereoPinnedProbes((prev) => prev.filter((p) => p.id !== id))}
            onPinStereoProbe={handlePinStereoProbe}
            onSelectStereoStillPreset={handleSelectStereoStillPreset}
            stereoVideoSource={stereoVideoSource}
            stereoVideoRegions={stereoVideoRegions}
            onSelectStereoVideoPreset={handleSelectStereoVideoPreset}
            onUploadStereoVideoFile={handleUploadStereoVideoFile}
            onLoadStereoOnnxFile={handleLoadStereoOnnxFile}
            onClearStereoOnnxFile={handleClearStereoOnnxFile}
          />
        </div>
      </main>

      {/* Model Specs Modal */}
      <ModelComparisonTable
        isOpen={isComparisonOpen}
        onClose={() => setIsComparisonOpen(false)}
        activeModel={activeModel}
        activeVariant={activeVariant}
        onSelectModelAndVariant={selectFamily}
      />

      {/* Model Diagnostics & Log Inspector Modal */}
      <DiagnosticsModal
        isOpen={isDiagnosticsOpen}
        onClose={() => setIsDiagnosticsOpen(false)}
        activeModel={activeModel}
        activeVariant={activeVariant}
        onRetryModel={() => {
          void engine.initialize(
            activeModel,
            activeVariant,
            activeModel === 'stereo-depth'
              ? { stereoBackendPreference: activeStereoBackendPref }
              : undefined
          );
        }}
      />
    </div>
  );
}
