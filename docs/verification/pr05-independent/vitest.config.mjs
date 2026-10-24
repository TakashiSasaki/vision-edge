import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()],test:{environment:'jsdom',include:['docs/verification/pr05-independent/component-oracle.test.tsx'],clearMocks:true,restoreMocks:true}});
