import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';
import { modelCorsProxyPlugin } from './src/services/stereoCorsProxyServer.ts';

function excludeOrtWasmPlugin() {
  return {
    name: 'exclude-ort-wasm',
    generateBundle(_options: unknown, bundle: Record<string, unknown>) {
      for (const fileName of Object.keys(bundle)) {
        if (fileName.includes('ort-wasm') && fileName.endsWith('.wasm')) {
          delete bundle[fileName];
        }
      }
    },
  };
}

export default defineConfig(({ mode }) => {
  return {
    plugins: [
      react(),
      tailwindcss(),
      excludeOrtWasmPlugin(),
      modelCorsProxyPlugin(),
    ],
    resolve: {
      alias: [
        ...(mode === 'e2e' ? [{ find: /^\.\/imageClassifierAssets$/, replacement: path.resolve(import.meta.dirname, 'e2e/fixtures/classifierAssets.ts') }] : []),
        ...Object.entries({
        ...(mode === 'e2e' ? { '@mediapipe/tasks-vision': path.resolve(import.meta.dirname, 'e2e/fixtures/visionSdk.ts') } : {}),
        '@': path.resolve(import.meta.dirname, '.'),
        }).map(([find, replacement]) => ({ find, replacement })),
      ],
    },
    server: {
      port: 3000,
      host: '0.0.0.0',
      allowedHosts: true,
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
    preview: {
      port: Number(process.env.PORT) || 3000,
      host: '0.0.0.0',
      allowedHosts: true,
    },
  };
});
