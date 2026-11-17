#!/usr/bin/env node
/**
 * Reproducible runner for the Mirage ONNX (WebGPU & WASM) vs Official Cactus
 * output parity verification and processing speed benchmark.
 *
 * Usage:
 *   node scripts/mirage/run_onnx_parity_and_speed_benchmark.mjs
 *   npm run mirage:parity-benchmark
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

const WORKSPACE_ROOT = process.cwd();
const OUTPUT_JSON_PATH = path.resolve(
  WORKSPACE_ROOT,
  'docs/mirage-onnx-parity-and-speed.json'
);

function resolvePlaywrightChromiumExecutable() {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  const msPlaywrightDir = '/ms-playwright';
  if (existsSync(msPlaywrightDir)) {
    const entries = readdirSync(msPlaywrightDir).sort().reverse();
    for (const entry of entries) {
      const candidate = path.join(
        msPlaywrightDir,
        entry,
        'chrome-headless-shell-linux64',
        'chrome-headless-shell'
      );
      if (existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return undefined;
}

function runStep(label, command, args, extraEnv = {}) {
  console.log(`\n=== ${label} ===`);
  const res = spawnSync(command, args, {
    cwd: WORKSPACE_ROOT,
    stdio: 'inherit',
    env: {
      ...process.env,
      ...extraEnv,
    },
  });
  if (res.status !== 0) {
    console.error(`Step "${label}" failed with exit code ${res.status ?? 1}`);
    process.exit(res.status ?? 1);
  }
}

const chromiumExecutable = resolvePlaywrightChromiumExecutable();
if (chromiumExecutable) {
  console.log(`Using Chromium executable: ${chromiumExecutable}`);
}

runStep('1. Build production bundle', 'node', ['scripts/testing/run.mjs', 'build']);
runStep(
  '2. Run Playwright Mirage ONNX Parity & Speed Benchmark',
  'npm',
  ['run', 'test:e2e', '--', 'e2e/mirage-parity-and-speed.spec.ts'],
  {
    REQUIRE_MIRAGE_REAL_MODEL: '1',
    WRITE_MIRAGE_PARITY_REPORT: '1',
    ...(chromiumExecutable
      ? { PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH: chromiumExecutable }
      : {}),
  }
);

if (!existsSync(OUTPUT_JSON_PATH)) {
  console.error(`Expected benchmark JSON at ${OUTPUT_JSON_PATH} was not written.`);
  process.exit(1);
}

const report = JSON.parse(readFileSync(OUTPUT_JSON_PATH, 'utf8'));
console.log(`\n=== Benchmark Complete ===`);
console.log(`Recorded JSON report: ${OUTPUT_JSON_PATH}`);
console.log(
  `WebGPU Adapter: ${report.environment.webgpuAdapter.vendor} / ${report.environment.webgpuAdapter.architecture} (${report.environment.webgpuAdapter.description}) [software=${report.environment.webgpuAdapter.isSoftwareAdapter}]`
);
for (const [name, c] of Object.entries(report.fastProposalMode.cases)) {
  console.log(
    `[Fast Mode] ${name}: ONNX-WASM=${c.speed.onnxWasmTotalMs.medianMs}ms | ONNX-WebGPU=${c.speed.onnxWebgpuTotalMs.medianMs}ms | Official-Cactus-WASM=${c.speed.officialCactusWasmMs.medianMs}ms | Faster ONNX=${c.speed.fasterOnnxProvider} (${c.speed.wasmVsWebgpuSpeedupRatio}x)`
  );
}
for (const [name, c] of Object.entries(report.full4GraphPipelineMode.cases)) {
  console.log(
    `[Full 4-Graph Mode] ${name}: ONNX-WASM=${c.speedBreakdownMs.onnxWasm.totalMs}ms | ONNX-WebGPU=${c.speedBreakdownMs.onnxWebgpu.totalMs}ms | Official-Cactus-WASM=${c.speedBreakdownMs.officialCactusWasm.totalMs}ms | Faster ONNX=${c.speedBreakdownMs.fasterOnnxProvider} (${c.speedBreakdownMs.wasmVsWebgpuSpeedupRatio}x)`
  );
}
