import {
  executeOrtDiagnosticWorkerRun,
  HITNET_ORT_VERSION_PROFILES,
  type OrtLikeModule,
  type OrtVersionDiagnosticRequest,
  type OrtVersionDiagnosticResponse,
} from './hitnetOrtComparisonHelper';

const PINNED_ORT_VERSION = '1.31.1' as const;
const PROFILE_131 = HITNET_ORT_VERSION_PROFILES[PINNED_ORT_VERSION];
const ENTRY_URL_131 = `${PROFILE_131.distBaseUrl}${PROFILE_131.entryFile}`;

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
        loadOrtModule: () =>
          import(/* @vite-ignore */ ENTRY_URL_131) as Promise<OrtLikeModule>,
      }
    );
    workerScope.postMessage(response, transfers);
  };
}
