import { build } from 'vite';
import { delegatePaths, requestedDelegate } from './delegate.mjs';
await build({ configFile: false, build: { lib: { entry: 'e2e-real-sdk/native-probe.ts', formats: ['es'], fileName: () => 'native-probe.js' }, outDir: delegatePaths(requestedDelegate()).probe } });
