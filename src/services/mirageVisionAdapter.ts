import type {
  BoundingBox,
  DetectedObject,
  DetectionFrame,
  FrameResult,
  InferenceInput,
  ModelState,
  OnnxBackendInfo,
  RunnableModelType,
} from '../types/vision';
import type {
  ImageCategory,
  ImageClassifications,
  WholeImageClassificationFrame,
} from '../types/classification';
import { getCategoryColor, getLabelInfo } from './dictionary';
import {
  VisionEngine,
  type ExtraInferenceOptions,
  type VisionTaskFactory,
} from './visionEngine';
import type { PipelineObserver } from './pipelineMetrics';

/**
 * Mirage (Cactus Compute, Inc.) structured output types.
 * Coordinates (`bbox` and `mask` polygon vertices) are emitted by Mirage on a 0..1000 scale.
 */
export type MirageBBox1000 = [number, number, number, number];

export interface MirageObject {
  id: number;
  name: string;
  bbox: MirageBBox1000;
  score: number;
  /** Flat polygon vertex list [x1, y1, ...] or nested polygon list [[x1, y1, ...], ...] on a 0..1000 scale */
  mask?: number[] | number[][];
  position?: string;
  count?: number;
  description?: Record<string, string>;
}

export interface MirageRegion {
  id?: number;
  name: string;
  bbox?: MirageBBox1000;
  score: number;
  mask?: number[] | number[][];
  position?: string;
  description?: Record<string, string>;
}

export interface MirageTextRegion {
  string: string;
  bbox: MirageBBox1000;
  score: number;
}

export interface MirageRelation {
  subject: number;
  predicate: string;
  object: number;
}

export interface MirageSceneInfo {
  background?: string;
  style?: string;
  lighting?: string;
  environment?: string;
  [key: string]: unknown;
}

export type MirageChoiceFieldSchema = {
  enum: readonly string[];
};

export type MirageCountFieldSchema = {
  type: 'integer';
  minimum: number;
  maximum: number;
};

export type MirageTextFieldSchema = {
  type: 'string';
};

export type MirageFormFieldSchema =
  | MirageChoiceFieldSchema
  | MirageCountFieldSchema
  | MirageTextFieldSchema;

export type MirageFormSchema = Record<string, MirageFormFieldSchema>;

export interface MirageChoiceOrCountFieldResult {
  value: string | number | null;
  probabilities: Record<string, number>;
}

export interface MirageTextFieldResult {
  value: string | null;
  probability?: number;
}

export type MirageFormFieldResult =
  | MirageChoiceOrCountFieldResult
  | MirageTextFieldResult;

export type MirageFormResults = Record<string, MirageFormFieldResult>;

export interface MirageRawObservation {
  scene?: MirageSceneInfo;
  objects?: MirageObject[];
  regions?: MirageRegion[];
  text?: MirageTextRegion[];
  relations?: MirageRelation[];
  events?: Array<Record<string, unknown>>;
  reasoning?: string | Record<string, unknown> | Array<unknown>;
  form?: MirageFormResults;
}

export interface MirageDescribeOptions {
  sections?: Array<
    'scene' | 'objects' | 'regions' | 'text' | 'relations' | 'events' | 'reasoning'
  >;
  fields?: Array<'bbox' | 'score' | 'mask' | 'position' | 'description' | 'count'>;
  mask_format?: 'polygons' | 'largest' | 'merged' | 'bitmap';
  conf?: number;
  text_conf?: number;
  iou?: number;
  agnostic_nms?: boolean;
  max_objects?: number;
  max_text_lines?: number;
  max_tokens?: number;
}

export interface MirageAdapterOptions {
  confidenceThreshold?: number;
  textConfidenceThreshold?: number;
  maxDetections?: number;
  includeOcrAsDetections?: boolean;
  includeRegionsAsDetections?: boolean;
  language?: 'ja' | 'en';
  /**
   * Cactus VQAv2 ablation shows passing structured JSON *without* Mirage's raw `reasoning` trace
   * improves downstream LLM accuracy (71.1% vs 67.3%). Defaults to `false`.
   */
  includeReasoningTrace?: boolean;
  defaultFormSchema?: MirageFormSchema;
  runner?: MirageInferenceRunner;
}

export type MirageInputSource =
  | string
  | InferenceInput
  | MirageRawObservation;

export interface MirageInferenceRunner {
  describe(
    source: string | InferenceInput,
    options?: MirageDescribeOptions
  ): Promise<MirageRawObservation>;
  extract(
    source: string | InferenceInput,
    form: MirageFormSchema,
    options?: MirageDescribeOptions
  ): Promise<{ form: MirageFormResults }>;
  close?(): Promise<void> | void;
}

export interface MirageNaturalLanguageSummary {
  language: 'ja' | 'en';
  headline: string;
  sceneSummary: string;
  objectsSummary: string;
  ocrSummary: string;
  relationsSummary: string;
  decisionsSummary: string;
  fullText: string;
}

export interface DownstreamLlmPayload {
  systemInstruction: string;
  userPrompt: string;
  structuredContextJson: string;
  structuredContext: {
    scene: MirageSceneInfo;
    objects: Array<{
      id: number;
      name: string;
      labelJa: string;
      score: number;
      bboxNormalized: BoundingBox;
      position?: string;
      description?: Record<string, string>;
    }>;
    text: Array<{
      text: string;
      score: number;
      bboxNormalized: BoundingBox;
    }>;
    relations: Array<{
      subjectId: number;
      subjectName: string;
      predicate: string;
      objectId: number;
      objectName: string;
    }>;
    decisions?: MirageFormResults;
    reasoning?: unknown;
  };
}

export interface MirageAdaptedOutput {
  raw: MirageRawObservation;
  objects: DetectedObject[];
  ocrRegions: DetectedObject[];
  classifications: ImageClassifications[];
  summary: MirageNaturalLanguageSummary;
  llmPayload: DownstreamLlmPayload;
}

export type VisionProviderMode = 'existing' | 'mirage' | 'hybrid';

export interface HybridTriagePolicy {
  /** Minimum top object/decision confidence required to stay on edge without escalation (default: 0.45) */
  minConfidenceThreshold?: number;
  /** Maximum tolerable abstention ("-") probability in Decisions API before escalating (default: 0.40) */
  maxAbstentionProbability?: number;
  /** Minimum average OCR line confidence before escalating text-heavy documents (default: 0.55) */
  minOcrConfidenceThreshold?: number;
  /** Whether the user task requires open-ended multi-turn reasoning or external world knowledge */
  requiresOpenEndedReasoning?: boolean;
  /** User query or task instruction, if any */
  userQuery?: string;
}

export type HybridRouteTarget =
  | 'local-mirage-only'
  | 'mirage-json-to-text-llm'
  | 'cloud-vlm-fallback';

export interface HybridTriageDecision {
  route: HybridRouteTarget;
  shouldEscalateToCloudVlm: boolean;
  shouldInvokeDownstreamTextLlm: boolean;
  reasons: string[];
  metrics: {
    topObjectConfidence: number;
    meanOcrConfidence: number;
    maxDecisionAbstentionProbability: number;
    detectedObjectCount: number;
    ocrLineCount: number;
  };
}

export const MIRAGE_MODEL_METADATA = {
  id: 'mirage',
  name: 'Cactus Mirage (51.3M Perception Model)',
  repositoryUrl: 'https://huggingface.co/Cactus-Compute/mirage',
  library: 'cactus-compute',
  version: '4.1.0',
  license: 'Apache-2.0',
  parametersCount: 51_300_000,
  quantizedSizeBytes: 21_611_043,
  quantizedSizeMB: 20.61,
  coordinateScale: 1000,
} as const;

export const MIRAGE_DEFAULT_TRIAGE_FORM: MirageFormSchema = {
  scene_domain: {
    enum: [
      'application_ui_screenshot',
      'natural_photography',
      'document_or_form',
      'industrial_inspection',
    ],
  },
  primary_subject: {
    enum: [
      'person',
      'vehicle',
      'animal',
      'software_interface',
      'food_or_beverage',
    ],
  },
  contains_readable_text: {
    enum: ['yes', 'no'],
  },
  person_count: {
    type: 'integer',
    minimum: 0,
    maximum: 10,
  },
};

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * Converts Mirage's 0..1000 `[x1, y1, x2, y2]` bounding box to VisionStudio's
 * normalized `[0, 1]` `{ x, y, width, height }` `BoundingBox`.
 */
export function convertMirageBBoxToNormalized(bbox: MirageBBox1000): BoundingBox {
  const [rawX1, rawY1, rawX2, rawY2] = bbox;
  const x1 = clamp01(Math.min(rawX1, rawX2) / 1000);
  const y1 = clamp01(Math.min(rawY1, rawY2) / 1000);
  const x2 = clamp01(Math.max(rawX1, rawX2) / 1000);
  const y2 = clamp01(Math.max(rawY1, rawY2) / 1000);

  return {
    x: Number(x1.toFixed(4)),
    y: Number(y1.toFixed(4)),
    width: Number(Math.max(0, x2 - x1).toFixed(4)),
    height: Number(Math.max(0, y2 - y1).toFixed(4)),
  };
}

/**
 * Converts Mirage's 0..1000 polygon vertex list (`[x1, y1, x2, y2, ...]` for `mask_format: 'largest'`
 * or `[[x1, y1, x2, y2, ...], ...]` for `mask_format: 'polygons'`)
 * into VisionStudio's normalized `Array<{ x: number; y: number }>` polygon.
 */
export function convertMirageMaskToPolygon(
  mask?: number[] | number[][]
): Array<{ x: number; y: number }> | undefined {
  if (!Array.isArray(mask) || mask.length === 0) {
    return undefined;
  }
  let flatCoords: number[];
  if (Array.isArray(mask[0])) {
    const nested = mask as number[][];
    let longest: number[] = [];
    for (const poly of nested) {
      if (Array.isArray(poly) && poly.length > longest.length) {
        longest = poly;
      }
    }
    flatCoords = longest;
  } else {
    flatCoords = mask as number[];
  }
  if (flatCoords.length < 6) {
    return undefined;
  }
  const pointCount = Math.floor(flatCoords.length / 2);
  const polygon: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < pointCount * 2; i += 2) {
    polygon.push({
      x: Number(clamp01(flatCoords[i] / 1000).toFixed(4)),
      y: Number(clamp01(flatCoords[i + 1] / 1000).toFixed(4)),
    });
  }
  return polygon.length >= 3 ? polygon : undefined;
}

/**
 * Maps Mirage objects, regions, and optional OCR text lines into `DetectedObject[]`
 * compatible with VisionEngine and canvas overlay renderers.
 */
export function mapMirageObservationToDetectedObjects(
  observation: MirageRawObservation,
  options: MirageAdapterOptions = {}
): {
  objects: DetectedObject[];
  ocrRegions: DetectedObject[];
} {
  const confidenceThreshold = options.confidenceThreshold ?? 0.25;
  const textConfidenceThreshold = options.textConfidenceThreshold ?? 0.35;
  const maxDetections = options.maxDetections ?? 25;

  const mappedObjects: DetectedObject[] = [];
  const rawObjects = observation.objects ?? [];

  for (let idx = 0; idx < rawObjects.length; idx++) {
    const item = rawObjects[idx];
    if (!item || typeof item.score !== 'number' || item.score < confidenceThreshold) {
      continue;
    }
    const normalizedLabel = (item.name || 'object').trim().toLowerCase();
    const labelInfo = getLabelInfo(normalizedLabel);
    const bbox = convertMirageBBoxToNormalized(item.bbox ?? [0, 0, 0, 0]);
    const maskPolygon = convertMirageMaskToPolygon(item.mask);

    mappedObjects.push({
      id: `mirage-obj-${idx}-${item.id ?? idx}-${normalizedLabel.replace(/\s+/g, '-')}`,
      label: normalizedLabel,
      labelJa: labelInfo.ja,
      category: labelInfo.category,
      confidence: clamp01(item.score),
      bbox,
      color: getCategoryColor(normalizedLabel),
      ...(maskPolygon ? { maskPolygon } : {}),
      areaFraction: Number((bbox.width * bbox.height).toFixed(4)),
    });
  }

  if (options.includeRegionsAsDetections && Array.isArray(observation.regions)) {
    for (let idx = 0; idx < observation.regions.length; idx++) {
      const region = observation.regions[idx];
      if (!region || !region.bbox || region.score < confidenceThreshold) {
        continue;
      }
      const normalizedLabel = (region.name || 'region').trim().toLowerCase();
      const labelInfo = getLabelInfo(normalizedLabel);
      const bbox = convertMirageBBoxToNormalized(region.bbox);
      const maskPolygon = convertMirageMaskToPolygon(region.mask);

      mappedObjects.push({
        id: `mirage-reg-${idx}-${region.id ?? idx}-${normalizedLabel.replace(/\s+/g, '-')}`,
        label: normalizedLabel,
        labelJa: labelInfo.ja,
        category: labelInfo.category,
        confidence: clamp01(region.score),
        bbox,
        color: getCategoryColor(normalizedLabel),
        ...(maskPolygon ? { maskPolygon } : {}),
        areaFraction: Number((bbox.width * bbox.height).toFixed(4)),
      });
    }
  }

  const ocrRegions: DetectedObject[] = [];
  const rawText = observation.text ?? [];
  for (let idx = 0; idx < rawText.length; idx++) {
    const line = rawText[idx];
    if (!line || !line.string || line.score < textConfidenceThreshold) {
      continue;
    }
    const bbox = convertMirageBBoxToNormalized(line.bbox ?? [0, 0, 0, 0]);
    ocrRegions.push({
      id: `mirage-ocr-${idx}`,
      label: `ocr:${line.string}`,
      labelJa: `文字: ${line.string}`,
      category: 'テキスト (OCR)',
      confidence: clamp01(line.score),
      bbox,
      color: '#38bdf8',
      areaFraction: Number((bbox.width * bbox.height).toFixed(4)),
    });
  }

  const combined = options.includeOcrAsDetections
    ? [...mappedObjects, ...ocrRegions]
    : mappedObjects;

  return {
    objects: combined.slice(0, maxDetections),
    ocrRegions,
  };
}

/**
 * Maps Mirage's Decisions API (`form` output) into `ImageClassifications[]`
 * compatible with `ImageClassificationEngine` and `WholeImageClassificationFrame`.
 */
export function mapMirageFormToClassifications(
  formResults?: MirageFormResults,
  topK = 5
): ImageClassifications[] {
  if (!formResults) return [];

  const groups: ImageClassifications[] = [];
  const entries = Object.entries(formResults);

  for (let headIndex = 0; headIndex < entries.length; headIndex++) {
    const [fieldName, fieldResult] = entries[headIndex];
    const categories: ImageCategory[] = [];

    if ('probabilities' in fieldResult && fieldResult.probabilities) {
      const sorted = Object.entries(fieldResult.probabilities)
        .filter(([optionKey]) => optionKey !== '-')
        .sort((a, b) => b[1] - a[1])
        .slice(0, topK);

      for (let idx = 0; idx < sorted.length; idx++) {
        const [optionName, score] = sorted[idx];
        categories.push({
          index: idx,
          score: clamp01(score),
          categoryName: `${fieldName}:${optionName}`,
          displayName: `${fieldName} = ${optionName}`,
        });
      }

      const abstainProb = fieldResult.probabilities['-'];
      if (typeof abstainProb === 'number' && fieldResult.value === null) {
        categories.unshift({
          index: -1,
          score: clamp01(abstainProb),
          categoryName: `${fieldName}:abstain`,
          displayName: `${fieldName} = 判定保留 (abstain)`,
        });
      }
    } else if ('value' in fieldResult && fieldResult.value !== null) {
      const prob =
        'probability' in fieldResult && typeof fieldResult.probability === 'number'
          ? fieldResult.probability
          : 1.0;
      categories.push({
        index: 0,
        score: clamp01(prob),
        categoryName: `${fieldName}:${String(fieldResult.value)}`,
        displayName: `${fieldName} = ${String(fieldResult.value)}`,
      });
    }

    groups.push({
      headIndex,
      headName: fieldName,
      categories,
    });
  }

  return groups;
}

/**
 * Synthesizes a concise, deterministic natural language summary (Japanese or English)
 * from Mirage's structured perception JSON for callers that expect natural-language text.
 */
export function summarizeMirageObservation(
  observation: MirageRawObservation,
  language: 'ja' | 'en' = 'ja'
): MirageNaturalLanguageSummary {
  const objects = observation.objects ?? [];
  const textLines = observation.text ?? [];
  const relations = observation.relations ?? [];
  const scene = observation.scene ?? {};
  const form = observation.form ?? {};

  const idToName = new Map<number, string>();
  for (const obj of objects) {
    if (typeof obj.id === 'number') {
      idToName.set(obj.id, obj.name);
    }
  }
  for (const reg of observation.regions ?? []) {
    if (typeof reg.id === 'number') {
      idToName.set(reg.id, reg.name);
    }
  }

  if (language === 'en') {
    const bg = scene.background ? `${scene.background}` : 'unspecified background';
    const style = scene.style ? ` (${scene.style})` : '';
    const sceneSummary = `Scene: ${bg}${style}.`;

    const objectCounts = new Map<string, { count: number; maxScore: number }>();
    for (const obj of objects) {
      const prev = objectCounts.get(obj.name) ?? { count: 0, maxScore: 0 };
      objectCounts.set(obj.name, {
        count: prev.count + 1,
        maxScore: Math.max(prev.maxScore, obj.score),
      });
    }
    const objectsSummary =
      objectCounts.size > 0
        ? `Detected objects (${objects.length}): ` +
          Array.from(objectCounts.entries())
            .map(
              ([name, info]) =>
                `${name} x${info.count} (${Math.round(info.maxScore * 100)}%)`
            )
            .join(', ') +
          '.'
        : 'No foreground objects detected.';

    const ocrSnippets = textLines.slice(0, 5).map((t) => `"${t.string}"`);
    const ocrSummary =
      textLines.length > 0
        ? `OCR text (${textLines.length} lines): ${ocrSnippets.join(', ')}${
            textLines.length > 5 ? ', ...' : ''
          }.`
        : 'No readable text detected.';

    const relationPhrases = relations.slice(0, 4).map((rel) => {
      const subj = idToName.get(rel.subject) ?? `#${rel.subject}`;
      const obj = idToName.get(rel.object) ?? `#${rel.object}`;
      return `${subj} ${rel.predicate} ${obj}`;
    });
    const relationsSummary =
      relationPhrases.length > 0
        ? `Spatial relations: ${relationPhrases.join('; ')}.`
        : 'No explicit spatial relations.';

    const decisionPairs = Object.entries(form).map(
      ([k, v]) => `${k}=${v.value === null ? 'abstain' : String(v.value)}`
    );
    const decisionsSummary =
      decisionPairs.length > 0
        ? `Decisions: ${decisionPairs.join(', ')}.`
        : '';

    const headline = `${objects.length} objects, ${textLines.length} OCR lines (${bg})`;
    const fullText = [
      sceneSummary,
      objectsSummary,
      ocrSummary,
      relationsSummary,
      decisionsSummary,
    ]
      .filter(Boolean)
      .join(' ');

    return {
      language: 'en',
      headline,
      sceneSummary,
      objectsSummary,
      ocrSummary,
      relationsSummary,
      decisionsSummary,
      fullText,
    };
  }

  // Japanese summary
  const bg = scene.background ? String(scene.background) : '不明';
  const style = scene.style ? `（スタイル: ${String(scene.style)}）` : '';
  const sceneSummary = `シーン背景: ${bg}${style}。`;

  const objectCountsJa = new Map<string, { labelJa: string; count: number; maxScore: number }>();
  for (const obj of objects) {
    const labelJa = getLabelInfo(obj.name).ja;
    const prev = objectCountsJa.get(obj.name) ?? { labelJa, count: 0, maxScore: 0 };
    objectCountsJa.set(obj.name, {
      labelJa,
      count: prev.count + 1,
      maxScore: Math.max(prev.maxScore, obj.score),
    });
  }

  const objectsSummary =
    objectCountsJa.size > 0
      ? `検出物体（計${objects.length}件）: ` +
        Array.from(objectCountsJa.values())
          .map(
            (info) =>
              `${info.labelJa}×${info.count}（最大信頼度${Math.round(info.maxScore * 100)}%）`
          )
          .join('、') +
        '。'
      : '検出された前景物体はありません。';

  const ocrSnippets = textLines.slice(0, 5).map((t) => `「${t.string}」`);
  const ocrSummary =
    textLines.length > 0
      ? `OCR認識テキスト（計${textLines.length}行）: ${ocrSnippets.join('、')}${
          textLines.length > 5 ? ' 他' : ''
        }。`
      : '読み取り可能なテキストは検出されませんでした。';

  const relationPhrases = relations.slice(0, 4).map((rel) => {
    const subj = idToName.get(rel.subject) ?? `ID:${rel.subject}`;
    const obj = idToName.get(rel.object) ?? `ID:${rel.object}`;
    return `${subj} → [${rel.predicate}] → ${obj}`;
  });
  const relationsSummary =
    relationPhrases.length > 0
      ? `空間関係（計${relations.length}件）: ${relationPhrases.join(' / ')}。`
      : '特筆すべき空間関係はありません。';

  const decisionPairs = Object.entries(form).map(
    ([k, v]) => `${k}=${v.value === null ? '判定保留(abstain)' : String(v.value)}`
  );
  const decisionsSummary =
    decisionPairs.length > 0 ? `スキーマ判定(Decisions): ${decisionPairs.join('、')}。` : '';

  const headline = `検出物体${objects.length}件・OCRテキスト${textLines.length}行（背景: ${bg}）`;
  const fullText = [
    sceneSummary,
    objectsSummary,
    ocrSummary,
    relationsSummary,
    decisionsSummary,
  ]
    .filter(Boolean)
    .join(' ');

  return {
    language: 'ja',
    headline,
    sceneSummary,
    objectsSummary,
    ocrSummary,
    relationsSummary,
    decisionsSummary,
    fullText,
  };
}

/**
 * Formats Mirage's structured perception output into a compact, deterministic prompt payload
 * for downstream text-only LLMs (e.g., Qwen / Gemini Flash / local LLM).
 *
 * By default `includeReasoningTrace` is `false` because Cactus Compute's VQAv2 ablation
 * demonstrated higher accuracy when omitting Mirage's raw `reasoning` trace (71.1% vs 67.3%).
 */
export function buildDownstreamLlmPayload(
  observation: MirageRawObservation,
  userQuestion = '画像の内容を構造化データに基づいて要約・回答してください。',
  options: { includeReasoningTrace?: boolean } = {}
): DownstreamLlmPayload {
  const objects = observation.objects ?? [];
  const textLines = observation.text ?? [];
  const relations = observation.relations ?? [];

  const idToName = new Map<number, string>();
  for (const obj of objects) {
    idToName.set(obj.id, obj.name);
  }
  for (const reg of observation.regions ?? []) {
    if (typeof reg.id === 'number') {
      idToName.set(reg.id, reg.name);
    }
  }

  const structuredContext: DownstreamLlmPayload['structuredContext'] = {
    scene: observation.scene ?? {},
    objects: objects.map((obj) => ({
      id: obj.id,
      name: obj.name,
      labelJa: getLabelInfo(obj.name).ja,
      score: obj.score,
      bboxNormalized: convertMirageBBoxToNormalized(obj.bbox),
      ...(obj.position ? { position: obj.position } : {}),
      ...(obj.description ? { description: obj.description } : {}),
    })),
    text: textLines.map((line) => ({
      text: line.string,
      score: line.score,
      bboxNormalized: convertMirageBBoxToNormalized(line.bbox),
    })),
    relations: relations.map((rel) => ({
      subjectId: rel.subject,
      subjectName: idToName.get(rel.subject) ?? `entity_${rel.subject}`,
      predicate: rel.predicate,
      objectId: rel.object,
      objectName: idToName.get(rel.object) ?? `entity_${rel.object}`,
    })),
    ...(observation.form ? { decisions: observation.form } : {}),
    ...(options.includeReasoningTrace && observation.reasoning !== undefined
      ? { reasoning: observation.reasoning }
      : {}),
  };

  const structuredContextJson = JSON.stringify(structuredContext, null, 2);

  const systemInstruction =
    'You are a downstream reasoning engine operating on structured visual perception JSON generated by Cactus Mirage (51.3M). ' +
    'Ground all answers strictly in the provided objects, normalized bounding boxes, OCR text lines, spatial relations, and schema decisions.';

  const userPrompt = [
    '### Structured Visual Perception (Mirage JSON)',
    '```json',
    structuredContextJson,
    '```',
    '',
    `### Task / Query`,
    userQuestion,
  ].join('\n');

  return {
    systemInstruction,
    userPrompt,
    structuredContextJson,
    structuredContext,
  };
}

/**
 * Evaluates whether local Mirage perception + Decisions output is sufficient (`local-mirage-only`),
 * should be handed to a fast text-only LLM (`mirage-json-to-text-llm`), or requires fallback
 * to a cloud multimodal VLM (`cloud-vlm-fallback`).
 */
export function evaluateHybridTriage(
  observation: MirageRawObservation,
  policy: HybridTriagePolicy = {}
): HybridTriageDecision {
  const minConfidenceThreshold = policy.minConfidenceThreshold ?? 0.45;
  const maxAbstentionProbability = policy.maxAbstentionProbability ?? 0.40;
  const minOcrConfidenceThreshold = policy.minOcrConfidenceThreshold ?? 0.55;

  const objects = observation.objects ?? [];
  const textLines = observation.text ?? [];
  const form = observation.form ?? {};

  const topObjectConfidence =
    objects.length > 0 ? Math.max(...objects.map((o) => o.score)) : 0;

  const meanOcrConfidence =
    textLines.length > 0
      ? textLines.reduce((acc, item) => acc + item.score, 0) / textLines.length
      : 1.0;

  let maxDecisionAbstentionProbability = 0;
  let hasAbstainedField = false;
  for (const fieldResult of Object.values(form)) {
    if ('probabilities' in fieldResult && fieldResult.probabilities) {
      const abstainProb = fieldResult.probabilities['-'] ?? 0;
      if (abstainProb > maxDecisionAbstentionProbability) {
        maxDecisionAbstentionProbability = abstainProb;
      }
      if (fieldResult.value === null) {
        hasAbstainedField = true;
      }
    }
  }

  const reasons: string[] = [];
  let shouldEscalateToCloudVlm = false;

  if (hasAbstainedField || maxDecisionAbstentionProbability >= maxAbstentionProbability) {
    shouldEscalateToCloudVlm = true;
    reasons.push(
      `Decisions API abstention probability (${maxDecisionAbstentionProbability.toFixed(
        2
      )}) exceeded threshold (${maxAbstentionProbability.toFixed(2)})`
    );
  }

  if (objects.length === 0 && textLines.length === 0) {
    shouldEscalateToCloudVlm = true;
    reasons.push('No foreground objects or OCR text lines were detected by Mirage');
  } else if (objects.length > 0 && topObjectConfidence < minConfidenceThreshold) {
    shouldEscalateToCloudVlm = true;
    reasons.push(
      `Top object confidence (${topObjectConfidence.toFixed(
        2
      )}) is below threshold (${minConfidenceThreshold.toFixed(2)})`
    );
  }

  if (textLines.length > 0 && meanOcrConfidence < minOcrConfidenceThreshold) {
    shouldEscalateToCloudVlm = true;
    reasons.push(
      `Mean OCR confidence (${meanOcrConfidence.toFixed(
        2
      )}) is below threshold (${minOcrConfidenceThreshold.toFixed(2)})`
    );
  }

  if (policy.requiresOpenEndedReasoning) {
    shouldEscalateToCloudVlm = true;
    reasons.push('Task explicitly requires open-ended multimodal reasoning with raw pixels');
  }

  const hasUserQuery = Boolean(policy.userQuery && policy.userQuery.trim().length > 0);
  const shouldInvokeDownstreamTextLlm = !shouldEscalateToCloudVlm && hasUserQuery;

  let route: HybridRouteTarget = 'local-mirage-only';
  if (shouldEscalateToCloudVlm) {
    route = 'cloud-vlm-fallback';
  } else if (shouldInvokeDownstreamTextLlm) {
    route = 'mirage-json-to-text-llm';
    reasons.push(
      'Local Mirage perception confidence is high; routing structured JSON to lightweight text LLM'
    );
  } else {
    reasons.push(
      'Local Mirage structured perception & Decisions satisfied confidence thresholds on edge'
    );
  }

  return {
    route,
    shouldEscalateToCloudVlm,
    shouldInvokeDownstreamTextLlm,
    reasons,
    metrics: {
      topObjectConfidence: Number(topObjectConfidence.toFixed(4)),
      meanOcrConfidence: Number(meanOcrConfidence.toFixed(4)),
      maxDecisionAbstentionProbability: Number(
        maxDecisionAbstentionProbability.toFixed(4)
      ),
      detectedObjectCount: objects.length,
      ocrLineCount: textLines.length,
    },
  };
}

/**
 * Default Node/CLI runner that invokes `scripts/mirage/test_mirage_smoke.py --json-only`
 * when executed in a Node.js environment, or throws a clear error in browser-only contexts
 * unless a custom runner / proxy is supplied.
 */
export function createNodePythonMirageRunner(
  scriptPath = 'scripts/mirage/test_mirage_smoke.py'
): MirageInferenceRunner {
  return {
    async describe(source, options) {
      if (typeof source !== 'string') {
        throw new Error(
          'NodePythonMirageRunner requires a file path string for image input'
        );
      }
      const { execFile } = await import('node:child_process');
      const { promisify } = await import('node:util');
      const execFileAsync = promisify(execFile);

      const args = [
        scriptPath,
        '--image',
        source,
        '--mode',
        'describe',
        '--json-only',
      ];
      if (options) {
        args.push('--options-json', JSON.stringify(options));
      }
      const { stdout } = await execFileAsync('python3', args, {
        maxBuffer: 16 * 1024 * 1024,
      });
      const parsed = JSON.parse(stdout.trim()) as {
        describe?: MirageRawObservation;
      };
      if (!parsed.describe) {
        throw new Error('Mirage CLI runner did not return a describe payload');
      }
      return parsed.describe;
    },

    async extract(source, form, options) {
      if (typeof source !== 'string') {
        throw new Error(
          'NodePythonMirageRunner requires a file path string for image input'
        );
      }
      const { execFile } = await import('node:child_process');
      const { promisify } = await import('node:util');
      const execFileAsync = promisify(execFile);

      const args = [
        scriptPath,
        '--image',
        source,
        '--mode',
        'extract',
        '--form-json',
        JSON.stringify(form),
        '--json-only',
      ];
      if (options) {
        args.push('--options-json', JSON.stringify(options));
      }
      const { stdout } = await execFileAsync('python3', args, {
        maxBuffer: 16 * 1024 * 1024,
      });
      const parsed = JSON.parse(stdout.trim()) as {
        extract?: { form: MirageFormResults };
      };
      if (!parsed.extract || !parsed.extract.form) {
        throw new Error('Mirage CLI runner did not return an extract form payload');
      }
      return parsed.extract;
    },
  };
}

function isRawObservation(input: MirageInputSource): input is MirageRawObservation {
  return (
    typeof input === 'object' &&
    input !== null &&
    !('readyState' in input) &&
    !('displayWidth' in input) &&
    !('getContext' in input) &&
    ('objects' in input ||
      'scene' in input ||
      'text' in input ||
      'relations' in input ||
      'form' in input)
  );
}

/**
 * Adapter class integrating Cactus Compute's Mirage (51.3M) into the existing
 * VisionStudio AI detection, segmentation, and classification interfaces.
 */
export class MirageVisionAdapter {
  private readonly options: MirageAdapterOptions;
  private runner: MirageInferenceRunner;
  private lastObservation: MirageRawObservation | null = null;
  private requestId = 0;

  public readonly backend: OnnxBackendInfo = {
    runtime: 'onnxruntime-web',
    requestedProvider: 'unreported',
    configuredProviders: ['cactus-native-cpu'],
    fallbackReason: null,
    actualBackend: 'unmeasured',
  };

  constructor(options: MirageAdapterOptions = {}) {
    this.options = options;
    this.runner = options.runner ?? createNodePythonMirageRunner();
  }

  public setRunner(runner: MirageInferenceRunner): void {
    this.runner = runner;
  }

  public getLastObservation(): MirageRawObservation | null {
    return this.lastObservation;
  }

  /**
   * Executes full perception (or adapts an already-fetched Mirage JSON observation)
   * and returns normalized objects, OCR regions, classifications, natural-language summary,
   * and downstream LLM prompt payload.
   */
  public async describeFrame(
    source: MirageInputSource,
    overrideOptions: Partial<MirageAdapterOptions> = {},
    describeOptions?: MirageDescribeOptions
  ): Promise<MirageAdaptedOutput> {
    const mergedOptions: MirageAdapterOptions = {
      ...this.options,
      ...overrideOptions,
    };

    const raw: MirageRawObservation = isRawObservation(source)
      ? source
      : await this.runner.describe(source, describeOptions);

    this.lastObservation = raw;

    const { objects, ocrRegions } = mapMirageObservationToDetectedObjects(
      raw,
      mergedOptions
    );
    const classifications = mapMirageFormToClassifications(raw.form);
    const summary = summarizeMirageObservation(
      raw,
      mergedOptions.language ?? 'ja'
    );
    const llmPayload = buildDownstreamLlmPayload(raw, undefined, {
      includeReasoningTrace: mergedOptions.includeReasoningTrace ?? false,
    });

    return {
      raw,
      objects,
      ocrRegions,
      classifications,
      summary,
      llmPayload,
    };
  }

  /**
   * Executes Mirage's Decisions API (`vision.extract(image, form)`) for zero-shot structured
   * classification, form extraction, and bounded object counting.
   */
  public async extractDecisions(
    source: MirageInputSource,
    formSchema: MirageFormSchema = this.options.defaultFormSchema ??
      MIRAGE_DEFAULT_TRIAGE_FORM
  ): Promise<{
    form: MirageFormResults;
    classifications: ImageClassifications[];
    classificationFrame: WholeImageClassificationFrame;
  }> {
    const requestId = ++this.requestId;
    const t0 = performance.now();

    let form: MirageFormResults;
    if (isRawObservation(source) && source.form) {
      form = source.form;
    } else if (isRawObservation(source)) {
      throw new Error(
        'Provided MirageRawObservation does not contain a "form" section'
      );
    } else {
      const response = await this.runner.extract(source, formSchema, {
        sections: [],
      });
      form = response.form;
      this.lastObservation = {
        ...(this.lastObservation ?? {}),
        form,
      };
    }

    const timestampMs = Math.max(0.01, performance.now() - t0);
    const classifications = mapMirageFormToClassifications(form);

    const classificationFrame: WholeImageClassificationFrame = {
      status: 'ready',
      kind: 'whole-image-classification',
      model: 'image-classifier',
      variantId: 'lite0',
      requestId,
      frameId: requestId,
      outputGeneration: 1,
      timestampMs,
      classifications,
    };

    return {
      form,
      classifications,
      classificationFrame,
    };
  }

  /**
   * Drop-in compatibility method matching `YoloSegTask.detectForVideo` and `VisionEngine` task contracts.
   */
  public async detectForVideo(
    input: InferenceInput | string | MirageRawObservation,
    confidenceThreshold = this.options.confidenceThreshold ?? 0.35,
    extraOptions?: ExtraInferenceOptions
  ): Promise<DetectedObject[]> {
    const adapted = await this.describeFrame(
      input,
      {
        confidenceThreshold,
        maxDetections: extraOptions?.maxDetections ?? this.options.maxDetections,
      },
      {
        conf: confidenceThreshold,
        iou: extraOptions?.iouThreshold,
        max_objects: extraOptions?.maxDetections,
      }
    );
    return adapted.objects;
  }

  /**
   * Converts an input frame or Mirage observation directly into a `DetectionFrame` / `FrameResult`
   * matching `VisionEngine.detectVideoFrame`.
   */
  public async toDetectionFrame(
    source: MirageInputSource,
    modelSlot: RunnableModelType = 'yolo-seg',
    confidenceThreshold = this.options.confidenceThreshold ?? 0.35,
    extraOptions?: ExtraInferenceOptions
  ): Promise<FrameResult> {
    const requestId = ++this.requestId;
    const objects = await this.detectForVideo(
      source,
      confidenceThreshold,
      extraOptions
    );
    const frame: DetectionFrame = {
      model: modelSlot,
      requestId,
      objects,
    };
    return {
      status: 'ready',
      ...frame,
    };
  }

  public close(): void {
    this.lastObservation = null;
    if (this.runner.close) {
      void Promise.resolve(this.runner.close());
    }
  }
}

export interface VisionPipelineSwitchConfig {
  provider: VisionProviderMode;
  mirageOptions?: MirageAdapterOptions;
  hybridPolicy?: HybridTriagePolicy;
  visionTaskFactory?: VisionTaskFactory;
}

/**
 * Unified Vision Pipeline Coordinator that switches between:
 * - `'existing'`: standard `VisionEngine` (MediaPipe / ONNX Runtime Web)
 * - `'mirage'`: `MirageVisionAdapter` wired into `VisionEngine` task factory & direct perception APIs
 * - `'hybrid'`: runs `MirageVisionAdapter` as an ultra-fast edge perception/triage filter first,
 *   and falls back to existing heavy/cloud VLM or text-LLM based on `HybridTriagePolicy`.
 */
export class ConfigurableVisionPipeline {
  private providerMode: VisionProviderMode;
  private readonly mirageAdapter: MirageVisionAdapter;
  private readonly visionEngine: VisionEngine;
  private hybridPolicy: HybridTriagePolicy;

  constructor(config: VisionPipelineSwitchConfig) {
    this.providerMode = config.provider;
    this.mirageAdapter = new MirageVisionAdapter(config.mirageOptions);
    this.hybridPolicy = config.hybridPolicy ?? {};

    // When provider is 'mirage', wire MirageVisionAdapter into the yoloSeg task slot
    // if no custom factory overrides it, preserving full VisionEngine compatibility.
    const baseFactory = config.visionTaskFactory;
    const wrappedFactory: VisionTaskFactory | undefined =
      baseFactory || config.provider === 'mirage'
        ? {
            load:
              baseFactory?.load ??
              (async () => ({} as Awaited<ReturnType<VisionTaskFactory['load']>>)),
            detector:
              baseFactory?.detector ??
              (async () => ({
                detectForVideo: () => ({ detections: [] }),
                close: () => undefined,
              })),
            segmenter:
              baseFactory?.segmenter ??
              (async () => ({
                segmentForVideo: () => ({ close: () => undefined }),
                close: () => undefined,
              })),
            yoloSeg:
              config.provider === 'mirage'
                ? async () => ({
                    backend: this.mirageAdapter.backend,
                    detectForVideo: (input, confidenceThreshold, extraOptions) =>
                      this.mirageAdapter.detectForVideo(
                        input,
                        confidenceThreshold,
                        extraOptions
                      ),
                    close: () => this.mirageAdapter.close(),
                  })
                : baseFactory?.yoloSeg,
            segformer: baseFactory?.segformer,
          }
        : undefined;

    this.visionEngine = new VisionEngine(wrappedFactory);
  }

  public getProviderMode(): VisionProviderMode {
    return this.providerMode;
  }

  public setProviderMode(mode: VisionProviderMode): void {
    this.providerMode = mode;
  }

  public getMirageAdapter(): MirageVisionAdapter {
    return this.mirageAdapter;
  }

  public getVisionEngine(): VisionEngine {
    return this.visionEngine;
  }

  public getState(): ModelState {
    return this.visionEngine.getState();
  }

  public initialize(model: RunnableModelType, variantId?: string) {
    return this.visionEngine.initialize(model, variantId);
  }

  public detectVideoFrame(
    videoElement: InferenceInput,
    model: RunnableModelType,
    confidenceThreshold: number,
    instrumentation?: PipelineObserver,
    extraOptions?: ExtraInferenceOptions
  ): FrameResult | Promise<FrameResult> {
    if (this.providerMode === 'mirage') {
      return this.mirageAdapter.toDetectionFrame(
        videoElement,
        model,
        confidenceThreshold,
        extraOptions
      );
    }
    return this.visionEngine.detectVideoFrame(
      videoElement,
      model,
      confidenceThreshold,
      instrumentation,
      extraOptions
    );
  }

  /**
   * Executes the Phase 4 Hybrid Edge-Triage workflow:
   * 1. Runs Mirage locally to obtain structured objects, OCR, relations, and Decisions probabilities.
   * 2. Evaluates `HybridTriagePolicy` (confidence, abstention `-`, OCR quality, open-ended flag).
   * 3. If Mirage is confident, returns the local result (and optional downstream text-LLM prompt)
   *    without invoking the heavy cloud/fallback VLM.
   * 4. If escalation is triggered, invokes `cloudVlmFallback` when provided.
   */
  public async runHybridPerception(
    source: MirageInputSource,
    options: {
      policy?: HybridTriagePolicy;
      cloudVlmFallback?: (
        source: MirageInputSource,
        miragePrecheck: MirageAdaptedOutput,
        decision: HybridTriageDecision
      ) => Promise<{ answer: string; objects?: DetectedObject[] }>;
    } = {}
  ): Promise<{
    providerUsed: 'mirage-edge' | 'mirage-plus-text-llm' | 'cloud-vlm-fallback';
    triage: HybridTriageDecision;
    mirageOutput: MirageAdaptedOutput;
    finalAnswer: string;
    finalObjects: DetectedObject[];
  }> {
    const activePolicy: HybridTriagePolicy = {
      ...this.hybridPolicy,
      ...options.policy,
    };

    const mirageOutput = await this.mirageAdapter.describeFrame(source);
    const triage = evaluateHybridTriage(mirageOutput.raw, activePolicy);

    if (triage.shouldEscalateToCloudVlm && options.cloudVlmFallback) {
      const fallbackResult = await options.cloudVlmFallback(
        source,
        mirageOutput,
        triage
      );
      return {
        providerUsed: 'cloud-vlm-fallback',
        triage,
        mirageOutput,
        finalAnswer: fallbackResult.answer,
        finalObjects: fallbackResult.objects ?? mirageOutput.objects,
      };
    }

    if (triage.shouldInvokeDownstreamTextLlm) {
      return {
        providerUsed: 'mirage-plus-text-llm',
        triage,
        mirageOutput,
        finalAnswer: mirageOutput.llmPayload.userPrompt,
        finalObjects: mirageOutput.objects,
      };
    }

    return {
      providerUsed: 'mirage-edge',
      triage,
      mirageOutput,
      finalAnswer: mirageOutput.summary.fullText,
      finalObjects: mirageOutput.objects,
    };
  }

  public dispose(): void {
    this.mirageAdapter.close();
    this.visionEngine.dispose();
  }
}

export function createVisionPipelineProvider(
  config: VisionPipelineSwitchConfig
): ConfigurableVisionPipeline {
  return new ConfigurableVisionPipeline(config);
}
