#!/usr/bin/env tsx
import { ContainerModelCacheManager } from '../src/services/containerModelCacheServer.ts';

export async function prefetchModels(manager = new ContainerModelCacheManager()): Promise<{
  cachedCount: number;
  unavailableCount: number;
  totalCount: number;
}> {
  console.log('[container-model-cache] Initializing container model cache manager...');
  console.log(`[container-model-cache] Cache directory: ${manager.cacheDir}`);

  const startTime = Date.now();
  console.log('[container-model-cache] Starting download for all registered model variants (11 variants)...');

  await manager.prefetchAllModels({ concurrency: 1, prioritizeDefaults: true });
  const variants = manager.getVariantsSnapshot();
  const keys = Object.keys(variants);

  let cachedCount = 0;
  let unavailableCount = 0;

  for (const key of keys) {
    const entry = variants[key];
    if (entry.status === 'cached') {
      cachedCount++;
      console.log(`  ✓ ${key}: cached (${entry.size?.toLocaleString()} bytes, sha256: ${entry.sha256?.slice(0, 16)}...)`);
    } else {
      unavailableCount++;
      console.warn(`  ✗ ${key}: unavailable (${entry.error ?? 'unknown reason'})`);
    }
  }

  const durationSec = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[container-model-cache] Download process completed in ${durationSec}s: ${cachedCount}/${keys.length} cached, ${unavailableCount} unavailable.`);

  return {
    cachedCount,
    unavailableCount,
    totalCount: keys.length,
  };
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/^[./\\]+/, ''))) {
  prefetchModels().catch(err => {
    console.error('[container-model-cache] Fatal prefetch error:', err);
    process.exitCode = 1;
  });
}
