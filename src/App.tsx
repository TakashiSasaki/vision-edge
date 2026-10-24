import React, { useState, useEffect, useCallback } from 'react';
import { 
  BackendOption,
  BenchmarkStats, 
  CameraFacingMode,
  DetectedObject, 
  ModelType, 
  RenderOptions, 
  TTSConfig 
} from './types/vision';
import { Header } from './components/Header';
import { VisionCanvas } from './components/VisionCanvas';
import { ControlPanel } from './components/ControlPanel';
import { ModelComparisonTable } from './components/ModelComparisonTable';
import { MODELS_METADATA } from './services/visionEngine';
import { speechService } from './services/speechService';

export default function App() {
  const [activeModel, setActiveModel] = useState<ModelType>('selfie-segmenter');
  const [selectedBackend, setSelectedBackend] = useState<BackendOption>('auto');
  const [facingMode, setFacingMode] = useState<CameraFacingMode>('user');
  const [isCameraActive, setIsCameraActive] = useState<boolean>(true);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [isComparisonOpen, setIsComparisonOpen] = useState<boolean>(false);

  const [selectedObjectId, setSelectedObjectId] = useState<string | undefined>(undefined);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [, setStats] = useState<BenchmarkStats>({
    inferenceTimeMs: 12,
    fps: 60,
    objectCount: 0,
    backend: 'MediaPipe GPU (WebGL)',
    resolution: '640x480',
    processedFrames: 0
  });

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

  const handleSelectModel = (model: ModelType) => {
    setActiveModel(model);
    setSelectedObjectId(undefined);
    if (ttsConfig.enabled) {
      speechService.speak(`${MODELS_METADATA[model].name}に切り替えました`, 'model-switch', ttsConfig, 'manual');
    }
  };

  const handleToggleFacingMode = () => {
    setFacingMode(prev => (prev === 'user' ? 'environment' : 'user'));
  };

  const handleStatsUpdate = useCallback((newStats: BenchmarkStats, newObjects: DetectedObject[]) => {
    setStats(newStats);
    setDetectedObjects(newObjects);
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* Compact Top Bar */}
      <Header
        activeModel={activeModel}
        onSelectModel={handleSelectModel}
        selectedBackend={selectedBackend}
        onSelectBackend={setSelectedBackend}
        isCameraActive={isCameraActive}
        onToggleCamera={() => setIsCameraActive(prev => !prev)}
        ttsEnabled={ttsConfig.enabled}
        onToggleTTS={() => setTTSConfig(prev => ({ ...prev, enabled: !prev.enabled }))}
        isSpeaking={isSpeaking}
        onOpenComparison={() => setIsComparisonOpen(true)}
      />

      {/* Zero-Scroll Main Stage (Side-by-Side) */}
      <main className="flex-1 overflow-hidden p-2.5 sm:p-4 max-w-7xl w-full mx-auto grid grid-cols-1 md:grid-cols-12 gap-3 items-stretch">
        {/* Left Column: Compact Camera Preview with In-Preview Flip Button */}
        <div className="md:col-span-8 h-full flex flex-col justify-center overflow-hidden">
          <VisionCanvas
            activeModel={activeModel}
            selectedBackend={selectedBackend}
            facingMode={facingMode}
            onToggleFacingMode={handleToggleFacingMode}
            renderOptions={renderOptions}
            ttsConfig={ttsConfig}
            isCameraActive={isCameraActive}
            onCameraStart={() => setIsCameraActive(true)}
            onCameraStop={() => setIsCameraActive(false)}
            onSelectObject={(obj) => setSelectedObjectId(obj.id)}
            selectedObjectId={selectedObjectId}
            onStatsUpdate={handleStatsUpdate}
          />
        </div>

        {/* Right Column: Compact Control & Detection Dock */}
        <div className="md:col-span-4 h-full flex flex-col overflow-hidden">
          <ControlPanel
            activeModel={activeModel}
            renderOptions={renderOptions}
            onChangeRenderOptions={(opts) => setRenderOptions(prev => ({ ...prev, ...opts }))}
            ttsConfig={ttsConfig}
            onChangeTTSConfig={(cfg) => setTTSConfig(prev => ({ ...prev, ...cfg }))}
            detectedObjects={detectedObjects}
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
        onSelectModel={handleSelectModel}
      />
    </div>
  );
}
