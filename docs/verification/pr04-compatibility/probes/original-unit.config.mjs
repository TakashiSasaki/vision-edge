import original from '../../pr04-independent/vectors/vitest.config.mjs';
export default { ...original, test: { ...original.test, setupFiles: ['docs/verification/pr04-compatibility/probes/original-dom-adapter.mjs'] } };
