import { describe, expect, it, vi } from 'vitest';
import {
  buildDownstreamLlmPayload,
  convertMirageBBoxToNormalized,
  convertMirageMaskToPolygon,
  createVisionPipelineProvider,
  evaluateHybridTriage,
  mapMirageFormToClassifications,
  mapMirageObservationToDetectedObjects,
  MIRAGE_DEFAULT_TRIAGE_FORM,
  MIRAGE_MODEL_METADATA,
  MirageVisionAdapter,
  summarizeMirageObservation,
  type MirageInferenceRunner,
  type MirageRawObservation,
} from './mirageVisionAdapter';

const SAMPLE_MIRAGE_OBSERVATION: MirageRawObservation = {
  scene: {
    background: 'application interface',
    style: 'digital screenshot',
  },
  objects: [
    {
      id: 0,
      name: 'person',
      bbox: [100, 200, 500, 900],
      score: 0.91,
      mask: [100, 200, 500, 200, 500, 900, 100, 900],
      position: 'center',
      description: { appearance: 'astronaut portrait' },
    },
    {
      id: 1,
      name: 'chair',
      bbox: [600, 400, 850, 950],
      score: 0.72,
      mask: [600, 400, 850, 400, 850, 950],
      position: 'right',
    },
    {
      id: 2,
      name: 'bottle',
      bbox: [50, 50, 120, 180],
      score: 0.18, // Below default 0.25 threshold
    },
  ],
  regions: [
    {
      id: 3,
      name: 'wall',
      bbox: [0, 0, 1000, 1000],
      score: 0.99,
    },
  ],
  text: [
    {
      string: 'VisionStudio AI',
      bbox: [20, 15, 240, 55],
      score: 0.99,
    },
    {
      string: 'YOLOv8n-seg (ONNX)',
      bbox: [410, 18, 630, 52],
      score: 0.96,
    },
  ],
  relations: [
    {
      subject: 0,
      predicate: 'next to',
      object: 1,
    },
  ],
  reasoning: 'Internal perception trace that should be omitted by default.',
  form: {
    scene_domain: {
      value: 'application_ui_screenshot',
      probabilities: {
        application_ui_screenshot: 0.89,
        natural_photography: 0.05,
        document_or_form: 0.03,
        industrial_inspection: 0.01,
        '-': 0.02,
      },
    },
    contains_readable_text: {
      value: 'yes',
      probabilities: {
        yes: 0.98,
        no: 0.01,
        '-': 0.01,
      },
    },
    person_count: {
      value: 1,
      probabilities: {
        '0': 0.02,
        '1': 0.94,
        '2': 0.03,
        '-': 0.01,
      },
    },
  },
};

describe('MirageVisionAdapter & Mirage Perception Utilities', () => {
  it('exposes accurate Cactus Mirage model metadata', () => {
    expect(MIRAGE_MODEL_METADATA.id).toBe('mirage');
    expect(MIRAGE_MODEL_METADATA.parametersCount).toBe(51_300_000);
    expect(MIRAGE_MODEL_METADATA.quantizedSizeBytes).toBe(21_611_043);
    expect(MIRAGE_MODEL_METADATA.coordinateScale).toBe(1000);
  });

  it('converts 0..1000 Mirage bounding boxes and polygons into normalized [0, 1] coordinates', () => {
    const bbox = convertMirageBBoxToNormalized([100, 200, 500, 900]);
    expect(bbox).toEqual({
      x: 0.1,
      y: 0.2,
      width: 0.4,
      height: 0.7,
    });

    const clamped = convertMirageBBoxToNormalized([-50, 1200, 500, 200]);
    expect(clamped).toEqual({
      x: 0,
      y: 0.2,
      width: 0.5,
      height: 0.8,
    });

    const polygon = convertMirageMaskToPolygon([100, 200, 500, 200, 500, 900, 100, 900]);
    expect(polygon).toEqual([
      { x: 0.1, y: 0.2 },
      { x: 0.5, y: 0.2 },
      { x: 0.5, y: 0.9 },
      { x: 0.1, y: 0.9 },
    ]);

    expect(convertMirageMaskToPolygon([100, 200, 300, 400])).toBeUndefined();
  });

  it('maps Mirage objects and OCR lines to VisionStudio DetectedObject[] with Japanese dictionary labels', () => {
    const { objects, ocrRegions } = mapMirageObservationToDetectedObjects(
      SAMPLE_MIRAGE_OBSERVATION,
      {
        confidenceThreshold: 0.25,
        includeOcrAsDetections: false,
      }
    );

    expect(objects).toHaveLength(2);
    expect(objects[0]).toMatchObject({
      label: 'person',
      labelJa: '人物',
      category: '人物',
      confidence: 0.91,
      bbox: { x: 0.1, y: 0.2, width: 0.4, height: 0.7 },
    });
    expect(objects[0].maskPolygon).toHaveLength(4);

    expect(objects[1]).toMatchObject({
      label: 'chair',
      labelJa: '椅子',
      confidence: 0.72,
    });

    expect(ocrRegions).toHaveLength(2);
    expect(ocrRegions[0]).toMatchObject({
      label: 'ocr:VisionStudio AI',
      category: 'テキスト (OCR)',
      confidence: 0.99,
    });

    const withOcrAndRegions = mapMirageObservationToDetectedObjects(
      SAMPLE_MIRAGE_OBSERVATION,
      {
        confidenceThreshold: 0.25,
        includeOcrAsDetections: true,
        includeRegionsAsDetections: true,
      }
    );
    expect(withOcrAndRegions.objects).toHaveLength(5);

    // Ensure multiple instances sharing the same Mirage class id and name get unique DetectedObject IDs
    const duplicateClassIdObs: MirageRawObservation = {
      objects: [
        { id: 1, name: 'box', bbox: [10, 10, 200, 200], score: 0.6 },
        { id: 1, name: 'box', bbox: [300, 300, 500, 500], score: 0.55 },
        { id: 1, name: 'bottle', bbox: [600, 100, 700, 400], score: 0.5 },
        { id: 1, name: 'bottle', bbox: [750, 100, 850, 400], score: 0.48 },
      ],
    };
    const mappedDup = mapMirageObservationToDetectedObjects(duplicateClassIdObs, {
      confidenceThreshold: 0.25,
    });
    const ids = mappedDup.objects.map((o) => o.id);
    expect(new Set(ids).size).toBe(4);
  });

  it('maps Mirage Decisions API form results into ImageClassifications[] including abstention handling', () => {
    const classifications = mapMirageFormToClassifications(
      SAMPLE_MIRAGE_OBSERVATION.form
    );
    expect(classifications).toHaveLength(3);
    expect(classifications[0].headName).toBe('scene_domain');
    expect(classifications[0].categories[0]).toMatchObject({
      categoryName: 'scene_domain:application_ui_screenshot',
      score: 0.89,
    });

    // Test abstention ('-') when value is null
    const abstainedClassifications = mapMirageFormToClassifications({
      hazard_level: {
        value: null,
        probabilities: {
          safe: 0.15,
          warning: 0.1,
          '-': 0.75,
        },
      },
    });
    expect(abstainedClassifications[0].categories[0]).toMatchObject({
      index: -1,
      categoryName: 'hazard_level:abstain',
      score: 0.75,
    });
  });

  it('generates natural language summaries in both Japanese and English', () => {
    const jaSummary = summarizeMirageObservation(SAMPLE_MIRAGE_OBSERVATION, 'ja');
    expect(jaSummary.language).toBe('ja');
    expect(jaSummary.fullText).toContain('人物×1');
    expect(jaSummary.fullText).toContain('VisionStudio AI');
    expect(jaSummary.fullText).toContain('next to');
    expect(jaSummary.fullText).toContain('scene_domain=application_ui_screenshot');

    const enSummary = summarizeMirageObservation(SAMPLE_MIRAGE_OBSERVATION, 'en');
    expect(enSummary.language).toBe('en');
    expect(enSummary.fullText).toContain('person x1 (91%)');
    expect(enSummary.fullText).toContain('"VisionStudio AI"');
    expect(enSummary.fullText).toContain('person next to chair');
  });

  it('builds downstream text-LLM payloads omitting raw reasoning trace by default', () => {
    const defaultPayload = buildDownstreamLlmPayload(
      SAMPLE_MIRAGE_OBSERVATION,
      'What application is shown in the screenshot?'
    );
    expect(defaultPayload.structuredContext.reasoning).toBeUndefined();
    expect(defaultPayload.userPrompt).toContain('VisionStudio AI');
    expect(defaultPayload.userPrompt).toContain(
      'What application is shown in the screenshot?'
    );

    const withReasoningPayload = buildDownstreamLlmPayload(
      SAMPLE_MIRAGE_OBSERVATION,
      'Explain',
      { includeReasoningTrace: true }
    );
    expect(withReasoningPayload.structuredContext.reasoning).toBe(
      'Internal perception trace that should be omitted by default.'
    );
  });

  it('adapts describeFrame, extractDecisions, and detectForVideo through a custom MirageInferenceRunner', async () => {
    const mockRunner: MirageInferenceRunner = {
      describe: vi.fn().mockResolvedValue(SAMPLE_MIRAGE_OBSERVATION),
      extract: vi.fn().mockResolvedValue({
        form: SAMPLE_MIRAGE_OBSERVATION.form!,
      }),
      close: vi.fn(),
    };

    const adapter = new MirageVisionAdapter({
      runner: mockRunner,
      confidenceThreshold: 0.3,
    });

    const detected = await adapter.detectForVideo('sample.png', 0.3, {
      maxDetections: 5,
    });
    expect(mockRunner.describe).toHaveBeenCalledTimes(1);
    expect(detected).toHaveLength(2);
    expect(detected[0].label).toBe('person');

    const decisions = await adapter.extractDecisions(
      'sample.png',
      MIRAGE_DEFAULT_TRIAGE_FORM
    );
    expect(mockRunner.extract).toHaveBeenCalledTimes(1);
    expect(decisions.classificationFrame.model).toBe('image-classifier');
    expect(decisions.classifications[0].headName).toBe('scene_domain');

    adapter.close();
    expect(mockRunner.close).toHaveBeenCalledTimes(1);
  });

  it('switches between existing, mirage, and hybrid modes via ConfigurableVisionPipeline', async () => {
    const mockRunner: MirageInferenceRunner = {
      describe: vi.fn().mockResolvedValue(SAMPLE_MIRAGE_OBSERVATION),
      extract: vi.fn().mockResolvedValue({
        form: SAMPLE_MIRAGE_OBSERVATION.form!,
      }),
    };

    const pipeline = createVisionPipelineProvider({
      provider: 'mirage',
      mirageOptions: { runner: mockRunner },
    });

    expect(pipeline.getProviderMode()).toBe('mirage');

    const initResult = await pipeline.initialize('yolo-seg', 'yolov8n-seg');
    expect(initResult.status).toBe('ready');

    const fakeCanvas = { width: 640, height: 480 } as HTMLCanvasElement;
    const frameResult = await pipeline.detectVideoFrame(
      fakeCanvas,
      'yolo-seg',
      0.35
    );
    expect(frameResult.status).toBe('ready');
    if (frameResult.status === 'ready') {
      expect(frameResult.objects).toHaveLength(2);
      expect(frameResult.objects[0].label).toBe('person');
    }

    // Verify Hybrid Edge-Triage routing: high confidence stays on edge
    pipeline.setProviderMode('hybrid');
    const edgeOutcome = await pipeline.runHybridPerception(
      SAMPLE_MIRAGE_OBSERVATION
    );
    expect(edgeOutcome.providerUsed).toBe('mirage-edge');
    expect(edgeOutcome.triage.route).toBe('local-mirage-only');
    expect(edgeOutcome.triage.shouldEscalateToCloudVlm).toBe(false);

    // Verify Hybrid routing with a user query routes structured JSON to lightweight text-LLM
    const textLlmOutcome = await pipeline.runHybridPerception(
      SAMPLE_MIRAGE_OBSERVATION,
      {
        policy: { userQuery: 'Summarize the UI layout and detected objects.' },
      }
    );
    expect(textLlmOutcome.providerUsed).toBe('mirage-plus-text-llm');
    expect(textLlmOutcome.triage.route).toBe('mirage-json-to-text-llm');

    // Verify Hybrid routing escalates to Cloud VLM when Decisions abstention "-" is high
    const lowConfidenceObservation: MirageRawObservation = {
      objects: [{ id: 0, name: 'unknown_part', bbox: [10, 10, 90, 90], score: 0.28 }],
      text: [],
      form: {
        defect_class: {
          value: null,
          probabilities: {
            scratch: 0.12,
            dent: 0.1,
            '-': 0.78,
          },
        },
      },
    };

    const cloudFallbackSpy = vi.fn().mockResolvedValue({
      answer: 'Cloud VLM identified subtle surface micro-fracture.',
    });

    const escalatedOutcome = await pipeline.runHybridPerception(
      lowConfidenceObservation,
      {
        cloudVlmFallback: cloudFallbackSpy,
      }
    );
    expect(escalatedOutcome.providerUsed).toBe('cloud-vlm-fallback');
    expect(escalatedOutcome.triage.shouldEscalateToCloudVlm).toBe(true);
    expect(cloudFallbackSpy).toHaveBeenCalledTimes(1);
    expect(escalatedOutcome.finalAnswer).toContain('Cloud VLM identified');

    pipeline.dispose();
  });

  it('evaluates hybrid triage escalation reasons accurately', () => {
    const emptyDecision = evaluateHybridTriage({ objects: [], text: [] });
    expect(emptyDecision.shouldEscalateToCloudVlm).toBe(true);
    expect(emptyDecision.route).toBe('cloud-vlm-fallback');
    expect(emptyDecision.reasons[0]).toContain('No foreground objects or OCR text');
  });

  it('builds Fast and Full Mirage WASM describe options and executes MirageWasmEngine via Web Worker interface', async () => {
    const { buildMirageDescribeOptions, MirageWasmEngine } = await import('./mirageWasmService');
    const { sanitizeCactusJson, runCactusDescribe } = await import('./mirageWasmWorker');
    const { VisionEngine } = await import('./visionEngine');

    // Verify NaN sanitization from C WASM buffers
    const sanitized = sanitizeCactusJson('{"objects":[{"name":"cup","score":-nan,"bbox":[10,10,50,50]}]}');
    expect(JSON.parse(sanitized)).toEqual({
      objects: [{ name: 'cup', score: 0, bbox: [10, 10, 50, 50] }],
    });

    // Verify nested 2D polygon masks returned by mask_format: "polygons" are converted properly
    const nestedPoly = convertMirageMaskToPolygon([
      [100, 200, 300, 200],
      [100, 200, 400, 200, 400, 600, 100, 600],
    ]);
    expect(nestedPoly).toEqual([
      { x: 0.1, y: 0.2 },
      { x: 0.4, y: 0.2 },
      { x: 0.4, y: 0.6 },
      { x: 0.1, y: 0.6 },
    ]);

    // Verify runCactusDescribe works with an Emscripten Module that only exports HEAPU8 and UTF8ToString
    // (without lengthBytesUTF8 or stringToUTF8)
    const heap = new Uint8Array(512 * 1024);
    let nextPtr = 64;
    const minimalEmscriptenMod = {
      HEAPU8: heap,
      _malloc: (size: number) => {
        const ptr = nextPtr;
        nextPtr += size + 8;
        return ptr;
      },
      _free: vi.fn(),
      _cactus_load: vi.fn(() => 0),
      _cactus_unload: vi.fn(),
      _cactus_models: vi.fn(() => 0),
      _cactus_error: vi.fn(() => 0),
      _cactus_describe: (
        _rgbPtr: number,
        _w: number,
        _h: number,
        optsPtr: number,
        _langPtr: number,
        _kwPtr: number,
        outPtr: number
      ) => {
        let end = optsPtr;
        while (heap[end] !== 0 && end < heap.length) end++;
        const decodedOpts = new TextDecoder().decode(heap.subarray(optsPtr, end));
        const parsedOpts = JSON.parse(decodedOpts);
        expect(parsedOpts.mask_format).toBe('polygons');

        const resultJson = JSON.stringify({
          objects: [
            {
              id: 1,
              name: 'laptop',
              bbox: [100, 150, 600, 700],
              score: 0.84,
              mask: [[100, 150, 600, 150, 600, 700, 100, 700]],
            },
          ],
          text: [{ string: 'OK', bbox: [200, 200, 280, 240], score: 0.91 }],
        });
        const encodedOut = new TextEncoder().encode(resultJson);
        heap.set(encodedOut, outPtr);
        heap[outPtr + encodedOut.byteLength] = 0;
        return encodedOut.byteLength;
      },
      UTF8ToString: (ptr: number) => {
        let end = ptr;
        while (heap[end] !== 0 && end < heap.length) end++;
        return new TextDecoder().decode(heap.subarray(ptr, end));
      },
    };

    const describeResult = runCactusDescribe(
      minimalEmscriptenMod,
      new Uint8Array(32 * 32 * 3),
      32,
      32,
      JSON.stringify({ sections: ['objects', 'text'], mask_format: 'polygons' })
    );
    expect(describeResult.objects?.[0]?.name).toBe('laptop');
    expect(describeResult.text?.[0]?.string).toBe('OK');

    // Fast mode vs Full mode options
    const fastCfg = buildMirageDescribeOptions(0.3, {
      analysisMode: 'fast',
      includeOcr: true,
      maxDetections: 12,
    });
    expect(fastCfg.maxDim).toBe(320);
    expect(fastCfg.describeOptions.sections).toEqual(['objects', 'text']);
    expect(fastCfg.describeOptions.fields).toEqual(['bbox', 'score', 'mask', 'position']);

    const fullCfg = buildMirageDescribeOptions(0.35, {
      analysisMode: 'full',
      includeOcr: false,
      maxDetections: 20,
    });
    expect(fullCfg.maxDim).toBe(512);
    expect(fullCfg.describeOptions.sections).toEqual(['scene', 'objects', 'regions', 'relations']);
    expect(fullCfg.describeOptions.fields).toContain('description');

    // Mock Worker communication for MirageWasmEngine
    let onMessageHandler: ((ev: MessageEvent) => void) | null = null;
    const mockWorker = {
      get onmessage() {
        return onMessageHandler;
      },
      set onmessage(fn: ((ev: MessageEvent) => void) | null) {
        onMessageHandler = fn;
      },
      onerror: null,
      terminate: vi.fn(),
      postMessage: vi.fn((msg: { type: string; id: number }) => {
        queueMicrotask(() => {
          if (msg.type === 'init') {
            onMessageHandler?.({
              data: { type: 'init-result', id: msg.id, ok: true, sourceUsed: 'primary' },
            } as MessageEvent);
          } else if (msg.type === 'describe') {
            onMessageHandler?.({
              data: {
                type: 'describe-result',
                id: msg.id,
                ok: true,
                observation: SAMPLE_MIRAGE_OBSERVATION,
              },
            } as MessageEvent);
          }
        });
      }),
    };

    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage: vi.fn(),
      getImageData: (_x: number, _y: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
      }),
    } as unknown as CanvasRenderingContext2D);

    const wasmEngine = new MirageWasmEngine(mockWorker);
    const sourceUsed = await wasmEngine.initializeWorker(
      '/wasm/cactus/cactus.js',
      '/wasm/cactus/cactus.wasm',
      '/api/model-cache/mirage/mirage.cact',
      null
    );
    expect(sourceUsed).toBe('primary');

    const engine = new VisionEngine({
      load: vi.fn(),
      detector: vi.fn(),
      segmenter: vi.fn(),
      mirage: vi.fn(async () => wasmEngine),
    });

    const initState = await engine.initialize('mirage', 'mirage-51m');
    expect(initState).toMatchObject({
      status: 'ready',
      model: 'mirage',
      variantId: 'mirage-51m',
    });

    const fakeCanvas = document.createElement('canvas');
    fakeCanvas.width = 640;
    fakeCanvas.height = 480;

    const frame = await engine.detectVideoFrame(fakeCanvas, 'mirage', 0.3, undefined, {
      analysisMode: 'fast',
      includeOcr: true,
      maxDetections: 10,
    });
    expect(frame.status).toBe('ready');
    if (frame.status === 'ready') {
      // Includes 2 objects + 1 region + 2 OCR lines = 5 items with directional Japanese labels
      expect(frame.objects.length).toBeGreaterThanOrEqual(4);
      expect(frame.objects.some((o) => o.label === 'person' && o.labelJa.includes('人物'))).toBe(true);
      expect(frame.objects.some((o) => o.category === 'テキスト (OCR)' && o.labelJa.includes('VisionStudio AI'))).toBe(true);
    }

    engine.dispose();
    expect(mockWorker.terminate).toHaveBeenCalledTimes(1);
    vi.restoreAllMocks();
  });
});
