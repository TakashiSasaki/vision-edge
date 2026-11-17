import type { ImageClassifierVariantId } from '../types/classification';
import type { RunnableModelType } from '../types/vision';
import { isRunnableModel, modelMetadata } from './modelRegistry';

export const MODEL_SELECTION_STORAGE_KEY = 'visionstudio.lastModelSelection.v1';

export interface PersistedModelSelection {
  activeModel: RunnableModelType;
  selectedVariants: Record<RunnableModelType, string>;
  classifierMode: boolean;
  classifierVariant: ImageClassifierVariantId;
}

const DEFAULT_SELECTED_VARIANTS: Record<RunnableModelType, string> = {
  'selfie-segmenter': 'general',
  mediapipe: 'lite0',
  'yolo-seg': 'yolov8n-seg',
  segformer: 'b0',
  mirage: 'mirage-51m',
  'stereo-depth': 'stereo-sad-baseline',
};

export function getDefaultModelSelection(): PersistedModelSelection {
  return {
    activeModel: 'selfie-segmenter',
    selectedVariants: { ...DEFAULT_SELECTED_VARIANTS },
    classifierMode: false,
    classifierVariant: 'lite0',
  };
}

function getSafeBrowserStorage(): Storage | null {
  try {
    if (typeof window === 'undefined' || !window.localStorage) {
      return null;
    }
    return window.localStorage;
  } catch {
    return null;
  }
}

export function loadPersistedModelSelection(
  storage: Pick<Storage, 'getItem'> | null = getSafeBrowserStorage()
): PersistedModelSelection {
  const defaults = getDefaultModelSelection();
  if (!storage) return defaults;

  try {
    const raw = storage.getItem(MODEL_SELECTION_STORAGE_KEY);
    if (!raw) return defaults;

    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return defaults;
    }

    const record = parsed as Record<string, unknown>;
    const nextVariants: Record<RunnableModelType, string> = { ...defaults.selectedVariants };
    const storedVariants =
      record.selectedVariants &&
      typeof record.selectedVariants === 'object' &&
      !Array.isArray(record.selectedVariants)
        ? (record.selectedVariants as Record<string, unknown>)
        : null;

    if (storedVariants) {
      for (const modelKey of Object.keys(defaults.selectedVariants) as RunnableModelType[]) {
        const candidate = storedVariants[modelKey];
        if (typeof candidate === 'string') {
          const isValidVariant = modelMetadata(modelKey).variants.some(
            (v) => v.id === candidate
          );
          if (isValidVariant) {
            nextVariants[modelKey] = candidate;
          }
        }
      }
    }

    const activeModel: RunnableModelType = isRunnableModel(record.activeModel)
      ? record.activeModel
      : defaults.activeModel;

    const classifierMode = record.classifierMode === true;
    const classifierVariant: ImageClassifierVariantId =
      record.classifierVariant === 'lite2' ? 'lite2' : 'lite0';

    return {
      activeModel,
      selectedVariants: nextVariants,
      classifierMode,
      classifierVariant,
    };
  } catch {
    return defaults;
  }
}

export function savePersistedModelSelection(
  selection: PersistedModelSelection,
  storage: Pick<Storage, 'setItem'> | null = getSafeBrowserStorage()
): void {
  if (!storage) return;
  try {
    storage.setItem(MODEL_SELECTION_STORAGE_KEY, JSON.stringify(selection));
  } catch {
    // Ignore storage quota / private mode errors cleanly
  }
}
