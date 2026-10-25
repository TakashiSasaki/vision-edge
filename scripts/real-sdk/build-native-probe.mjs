import { build } from 'vite';
await build({ configFile: false, build: { lib: { entry: 'e2e-real-sdk/native-probe.ts', formats: ['es'], fileName: () => 'native-probe.js' }, outDir: '.cache/pr07-native-probe' } });
