import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { resolveConfig } from 'vite';
import { manifest, root, hash } from './assets.mjs';
const config = await resolveConfig({}, 'build', 'production');
assert(!config.resolve.alias.some(row => String(row.find) === '@mediapipe/tasks-vision'), 'Production must import actual SDK');
const files = [];
async function walk(dir) { for (const entry of await readdir(dir, { withFileTypes: true })) { const file = path.join(dir, entry.name); if (entry.isDirectory()) await walk(file); else files.push(file); } }
await walk(path.join(root, 'dist'));
const js = (await Promise.all(files.filter(f => f.endsWith('.js')).map(f => readFile(f, 'utf8')))).join('\n');
for (const marker of ['visionSdkProbe', 'visionSdkGate', 'Test-only SDK fault', '__realSdkProbe', '__realCamera', 'PR07_REAL_SDK_OBSERVATION', 'e2e-real-sdk']) assert(!js.includes(marker), `Production contains test marker ${marker}`);
for (const asset of manifest.assets.filter(a => a.file.endsWith('.tflite'))) assert(js.includes(asset.url), 'Product model reference changed');
assert(js.includes('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'), 'WASM reference changed');
assert(js.includes('wasm_nosimd_internal') && js.includes('mediapipe.tasks.vision.ObjectDetectorGraph'), 'Expected real SDK graph/loader signatures absent');
const lock = JSON.parse(await readFile(path.join(root, 'package-lock.json'), 'utf8'));
assert.equal(lock.packages['node_modules/@mediapipe/tasks-vision'].version, '0.10.14');
console.log(JSON.stringify({ productionSdkAlias: false, forbiddenMarkers: 0, sdkVersion: '0.10.14', files: await Promise.all(files.map(async file => ({ file: path.relative(root, file), sha256: hash(await readFile(file)) }))) }, null, 2));
