import { defineConfig, mergeConfig } from 'vite';
import base from '../../../../vite.config';
import path from 'node:path';
export default defineConfig(environment => mergeConfig(
  typeof base === 'function' ? base(environment) : base,
  { build: { outDir: '.cache/pr06-overhead', rollupOptions: { input: path.resolve(import.meta.dirname, 'index.html') } } },
));
