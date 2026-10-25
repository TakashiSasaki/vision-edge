import React, { useState, useEffect, useCallback, useSyncExternalStore } from 'react';
import {
  DetectionFrame,
  CameraFacingMode,
  RunnableModelType,
  RenderOptions,
  TTSConfig
} from './types/vision';
import { Header } from './components/Header';
import { VisionCanvas } from './components/VisionCanvas';
import { ControlPanel } from './components/ControlPanel';
import { ModelComparisonTable } from './components/ModelComparisonTable';
import { VisionEngine } from './services/visionEngine';
import { modelMetadata, isRunnableModel, getVariantMetadata } from './services/modelRegistry';
import { ModelStatus } from './components/ModelStatus';
import { speechService } from './services/speechService';

export default function App({ engine: suppliedEngine }: { engine?: VisionEngine } = {}) {
  const [engine] = useState(() => suppliedEngine ?? new VisionEngine());
  const modelState = useSyncExternalStore(engine.subscribe, engine.getState);
  const [activeModel, setActiveModel] = useState<RunnableModelType>('selfie-segmenter');
  const [selectedVariants, setSelectedVariants] = useState<Record<RunnableModelType, string>>({
    'selfie-segmenter': 'general',
    'mediapipe': 'lite0',
    'yolo-seg': 'yolov8n-seg',
    'segformer': 'b0',
  });
  const activeVariant = selectedVariants[activeModel] || modelMetadata(activeModel).defaultVariantId;

  const [facingMode, setFacingMode] = useState<CameraFacingMode>('user');
  const [isCameraActive, setIsCameraActive] = useState<boolean>(true);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isComparisonOpen, setIsComparisonOpen] = useState<boolean>(false);

  const [selectedObjectId, setSelectedObjectId] = useState<string | undefined>(undefined);
  const [lastFrame, setLastFrame] = useState<DetectionFrame | null>(null);
  const hasCurrentFrame = isCameraActive && modelState.status === 'ready'
    && modelState.model === activeModel && lastFrame?.model === activeModel
    && lastFrame.requestId === modelState.requestId;
  const detectedObjects = hasCurrentFrame ? lastFrame.objects : [];

  useEffect(() => () => engine.dispose(), [engine]);
  useEffect(() => { void engine.initialize(activeModel, activeVariant); }, [engine, activeModel, activeVariant]);

  const [renderOptions, setRenderOptions] = useState<RenderOptions>({
    showBoxes: true,
    showMasks: true,
    showLabels: true,
    showConfidence: true,
    maskOpacity: 0.35,
    confidenceThreshold: 0.35
  });

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
  const handleCameraStart = useCallback(() => setIsCameraActive(true), []);
  const handleCameraStop = useCallback(() => setIsCameraActive(false), []);
  const handleToggleCamera = useCallback(() => setIsCameraActive(previous => !previous), []);

  const handleFrame = useCallback((frame: DetectionFrame | null) => {
    setLastFrame(frame);
    if (!frame) setSelectedObjectId(undefined);
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Compact Top Bar */}
      <Header
        activeModel={activeModel}
        onSelectModel={handleSelectModel}
        activeVariant={activeVariant}
        onSelectVariant={handleSelectVariant}
        modelState={modelState}
        isCameraActive={isCameraActive}
        onToggleCamera={handleToggleCamera}
        ttsEnabled={ttsConfig.enabled}
        onToggleTTS={() => setTTSConfig(prev => ({ ...prev, enabled: !prev.enabled }))}
        isSpeaking={isSpeaking}
        onOpenComparison={() => setIsComparisonOpen(true)}
      />

      <ModelStatus state={modelState} onRetry={() => { void engine.initialize(activeModel, activeVariant); }} />

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
            renderOptions={renderOptions}
            ttsConfig={ttsConfig}
            isCameraActive={isCameraActive}
            onCameraStart={handleCameraStart}
            onCameraStop={handleCameraStop}
            onSelectObject={(obj) => setSelectedObjectId(obj.id)}
            selectedObjectId={selectedObjectId}
            onFrame={handleFrame}
          />
        </div>

        {/* Right Column: Compact Control & Detection Dock */}
        <div className="md:col-span-4 h-full min-h-0 min-w-0 flex flex-col overflow-hidden">
          <ControlPanel
            activeModel={activeModel}
            renderOptions={renderOptions}
            onChangeRenderOptions={(opts) => setRenderOptions(prev => ({ ...prev, ...opts }))}
            ttsConfig={ttsConfig}
            onChangeTTSConfig={(cfg) => setTTSConfig(prev => ({ ...prev, ...cfg }))}
            detectedObjects={detectedObjects}
            hasCurrentFrame={hasCurrentFrame}
            onSelectObject={(obj) => setSelectedObjectId(obj.id)}
            selectedObjectId={selectedObjectId}
            isSpeaking={isSpeaking}
          />
        </div>
      </main>

      {/* Model Specs Modal */}
      <ModelComparisonTable
        isOpen={isComparisonOpen}
        onClose={() => setIsComparisonOpen(false)}
        activeModel={activeModel}
        activeVariant={activeVariant}
        onSelectModelAndVariant={handleSelectModelAndVariant}
      />
    </div>
  );
}
