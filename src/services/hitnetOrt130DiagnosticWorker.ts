import {
  executeOrtDiagnosticWorkerRun,
  HITNET_ORT_130_FAMILY_PROFILES,
  loadOrtEntryModuleWithBoundarySha256,
  type OrtVersionDiagnosticRequest,
  type OrtVersionDiagnosticResponse,
} from './hitnetOrtComparisonHelper';

const PINNED_ORT_VERSION = '1.30.0' as const;

const workerScope =
  typeof self !== 'undefined' &&
  typeof (self as unknown as { postMessage?: unknown }).postMessage === 'function' &&
  typeof window === 'undefined'
    ? (self as unknown as {
        onmessage: ((ev: MessageEvent<OrtVersionDiagnosticRequest>) => void) | null;
        postMessage: (msg: OrtVersionDiagnosticResponse, transfer?: Transferable[]) => void;
      })
    : null;

if (workerScope) {
  workerScope.onmessage = async (event: MessageEvent<OrtVersionDiagnosticRequest>) => {
    const req = event.data;
    if (!req) return;
    const { response, transfers } = await executeOrtDiagnosticWorkerRun(
      PINNED_ORT_VERSION,
      req,
      {
        loadOrtModule: (entryModuleUrl, family) =>
          loadOrtEntryModuleWithBoundarySha256(entryModuleUrl, {
            referenceLocalDistSha256:
              HITNET_ORT_130_FAMILY_PROFILES[family].referenceLocalDistSha256.entry,
          }),
      }
    );
    workerScope.postMessage(response, transfers);
  };
}
