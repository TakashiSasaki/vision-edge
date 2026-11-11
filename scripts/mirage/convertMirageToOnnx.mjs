import process from 'node:process';
import { pathToFileURL } from 'node:url';
import {
  MIRAGE_WEBGPU_UNAVAILABLE_CODE,
  MIRAGE_WEBGPU_UNAVAILABLE_REASON,
  mirageWebGpuUnavailableError,
} from '../../src/services/mirageAvailability.ts';

/** Fail before any network or filesystem operation; no faithful exporter exists. */
export async function generateMirageWebGpuOnnxFile() {
  throw mirageWebGpuUnavailableError();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.error(JSON.stringify({
    status: 'unavailable',
    code: MIRAGE_WEBGPU_UNAVAILABLE_CODE,
    error: MIRAGE_WEBGPU_UNAVAILABLE_REASON,
  }));
  process.exitCode = 1;
}
