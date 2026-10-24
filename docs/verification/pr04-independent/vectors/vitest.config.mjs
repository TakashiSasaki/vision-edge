import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    include: ['docs/verification/pr04-independent/vectors/*.test.{js,jsx}'],
    restoreMocks: true,
  },
});
