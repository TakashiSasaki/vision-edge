import path from 'node:path';
import { defineConfig, mergeConfig } from 'vite';
import base from './vite.config';

// Independent build only. The ordinary production config has no observer imports.
export default defineConfig(async env => mergeConfig(
  typeof base === 'function' ? await base({ ...env, mode: 'production' }) : base,
  {
    build: { outDir: '.cache/vision-edge-real-sdk', emptyOutDir: true },
    plugins: [{
      name: 'real-sdk-observation-only',
      transform(code: string, id: string) {
        if (id === path.resolve('src/main.tsx')) return `import '../e2e-real-sdk/observer';\n${code}`;
        if (id === path.resolve('src/services/canvasRenderer.ts')) {
          const anchor = 'ctx.drawImage(offscreen, 0, 0);';
          if (!code.includes(anchor)) throw new Error('Renderer observation anchor changed; inspect before running');
          return `import { observeDraw } from '../../e2e-real-sdk/observer';\n${code.replace(anchor, `${anchor}\n    observeDraw(frame, canvas);`)}`;
        }
      },
    }],
  },
));
