import type { ClassificationOptions, WholeImageClassificationFrame } from '../types/classification';
import type { InferenceMeasurement, PipelineObserver } from './pipelineMetrics';
import { ImageClassificationEngine } from './imageClassificationEngine';
import { JobLock, startFrameScheduler } from './frameScheduler';

export interface ClassificationSchedulerOptions {
  engine: ImageClassificationEngine;
  video: HTMLVideoElement;
  current: () => boolean;
  adopt: (frame: WholeImageClassificationFrame, measurement?: InferenceMeasurement) => void;
  error: (cause: unknown) => void;
  captureError?: (cause: unknown) => void;
  input: () => void;
  lock: JobLock;
  options?: ClassificationOptions;
  instrumentation?: PipelineObserver;
}
// Reuse the actual capture/latest-frame/backpressure/resource owner. Classification
// never enters detection drawing, object picking, mask or per-object TTS paths.
export function startImageClassificationScheduler(options: ClassificationSchedulerOptions): () => void {
  const { engine } = options, owner = engine.getState();
  if (owner.status !== 'ready') throw new Error('Classification scheduler requires a ready model');
  const sameOwner = () => engine.getState().status === 'ready' && engine.getState().requestId === owner.requestId;
  const owns = () => options.current() && sameOwner();
  let stopped = false;
  const stop = startFrameScheduler({
    ...options, current: owns,
    work: input => engine.classifyVideoFrame(input, options.options, options.instrumentation),
    adopt: (result, measurement) => { if (result.status === 'ready' && owns() && engine.isCurrentFrame(result)) options.adopt(result, measurement); },
  });
  return () => { if (stopped) return; stopped = true; stop(); if (sameOwner()) engine.clearResults(); };
}
