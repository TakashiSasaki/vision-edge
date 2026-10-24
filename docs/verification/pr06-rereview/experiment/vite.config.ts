import {defineConfig,mergeConfig} from 'vite';
import base from '../../../../vite.config';
import path from 'node:path';
export default defineConfig(env=>mergeConfig(base(env),{build:{outDir:'.cache/pr06-rereview-hud',rollupOptions:{input:path.resolve(import.meta.dirname,'index.html')}}}));
