import { defineConfig } from 'vitest/config';
export default defineConfig({test:{environment:'jsdom',include:['docs/verification/pr04-rereview/vectors/*.test.js'],restoreMocks:true}});
