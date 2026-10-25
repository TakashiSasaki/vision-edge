import path from 'node:path';
import { defineConfig, mergeConfig } from 'vite';
import base from './vite.config';
import { requestedDelegate, delegatePaths, cpuEngineVariant } from './scripts/real-sdk/delegate.mjs';
const delegate=requestedDelegate();
const paths=delegatePaths(delegate);

// Independent build only. The ordinary production config has no observer imports.
export default defineConfig(async env => mergeConfig(
  typeof base === 'function' ? await base({ ...env, mode: 'production' }) : base,
  {
    build: { outDir: paths.build, emptyOutDir: true },
    plugins: [{
      name: 'real-sdk-observation-only',
      enforce: 'pre',
      transform(code: string, id: string) {
        if(delegate==='CPU'&&id===path.resolve('src/services/visionEngine.ts'))return cpuEngineVariant(code,id);
        if (id === path.resolve('src/main.tsx')) return `import '../e2e-real-sdk/observer';\n${code}`;
        if (id === path.resolve('src/components/VisionCanvas.tsx')) {
          const anchor = 'latestFrameRef.current = result;';
          if (code.split(anchor).length !== 2) throw new Error('Adoption observation anchor changed; inspect before running');
          return `import { observeAdoption } from '../../e2e-real-sdk/observer';\n${code.replace(anchor, `${anchor}\n        observeAdoption(result);`)}`;
        }
        if (id === path.resolve('src/services/canvasRenderer.ts')) {
          const anchor = 'ctx.drawImage(offscreen, 0, 0);';
          if (!code.includes(anchor)) throw new Error('Renderer observation anchor changed; inspect before running');
          return `import { observeDraw } from '../../e2e-real-sdk/observer';\n${code.replace(anchor, `${anchor}\n    observeDraw(frame, canvas);`)}`;
        }
      },
    }],
  },
));
