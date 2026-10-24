import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = fileURLToPath(new URL('../../', import.meta.url));
export const cache = path.join(root, '.cache/real-sdk-assets');
/** @typedef {{id: string, file: string, url: string, sha256: string | null, size: number | null, packageFile?: string, acquisitionStatus?: string}} Asset */
// Fixed inventory mirrors the unchanged production references and licensed inputs.
const cdn = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm/';
const images = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/';
export const requiredAssets = {
  'vision_bundle.mjs': { url: 'https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-0.10.14.tgz', version: '0.10.14', packageFile: 'node_modules/@mediapipe/tasks-vision/vision_bundle.mjs' },
  ...Object.fromEntries(['vision_wasm_internal.js', 'vision_wasm_internal.wasm', 'vision_wasm_nosimd_internal.js', 'vision_wasm_nosimd_internal.wasm'].map(file => [file, { url: cdn + file, version: '0.10.14' }])),
  'astronaut.png': { url: images + 'astronaut.png', version: 'scikit-image v0.24.0' },
  'coffee.png': { url: images + 'coffee.png', version: 'scikit-image v0.24.0' },
  'efficientdet_lite0.tflite': { url: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite', version: 'float16/1', unpinned: true },
  'selfie_segmenter.tflite': { url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite', version: 'float16/latest (mutable)', unpinned: true },
};
/** Validate structure before any cache read or network request. */
export function validateManifest(value) {
  const invalid = message => { throw new Error(`INVALID_ASSET_MANIFEST: ${message}`); };
  if (!value || typeof value !== 'object' || Array.isArray(value) || value.schema !== 1) invalid('schema must be 1');
  if (!Array.isArray(value.assets)) invalid('assets must be an array');
  const seen = new Set(); const urls = new Set(); const files = new Set();
  for (const asset of value.assets) {
    if (!asset || typeof asset !== 'object' || Array.isArray(asset)) invalid('asset must be an object');
    if (typeof asset.id !== 'string' || !Object.hasOwn(requiredAssets, asset.id)) invalid('unknown asset id');
    if (seen.has(asset.id) || files.has(asset.file) || urls.has(asset.url)) invalid(`duplicate asset identity: ${asset.id}`);
    const required = requiredAssets[asset.id];
    if (asset.file !== asset.id || asset.url !== required.url || asset.version !== required.version || asset.packageFile !== required.packageFile || asset.required !== true) invalid(`inventory identity/reference mismatch: ${asset.id}`);
    for (const field of ['license', 'production']) if (typeof asset[field] !== 'string' || !asset[field].trim()) invalid(`${asset.id}: missing ${field}`);
    const unpinned = asset.sha256 === null && asset.size === null;
    if (unpinned) {
      if (!required.unpinned || typeof asset.acquisitionStatus !== 'string' || !asset.acquisitionStatus.startsWith('blocked:')) invalid(`${asset.id}: unpinned asset needs an explicit allowed blocked prerequisite`);
    } else if (typeof asset.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(asset.sha256) || !Number.isSafeInteger(asset.size) || asset.size <= 0 || asset.size > 15_000_000) invalid(`${asset.id}: invalid size/SHA256 pin`);
    seen.add(asset.id); files.add(asset.file); urls.add(asset.url);
  }
  const missing = Object.keys(requiredAssets).filter(id => !seen.has(id));
  if (missing.length) invalid(`missing required assets: ${missing.join(', ')}`);
  return value;
}
let parsed;
try { parsed = JSON.parse(await readFile(path.join(root, 'e2e-real-sdk/assets.lock.json'), 'utf8')); }
catch (error) { throw new Error(`INVALID_ASSET_MANIFEST: cannot read/parse JSON: ${error.message}`, { cause: error }); }
/** @type {{schema: number, assets: Asset[]}} */
export const manifest = validateManifest(parsed);
export const hash = bytes => createHash('sha256').update(bytes).digest('hex');
export function assetPath(asset) { return path.join(root, asset.packageFile ?? `.cache/real-sdk-assets/${asset.file}`); }
export async function verifyAsset(asset) {
  if (!asset.sha256 || !asset.size) return { id: asset.id, status: 'blocked', reason: asset.acquisitionStatus ?? 'No reviewed byte pin; acquisition/enrollment required' };
  try {
    const info = await stat(assetPath(asset));
    if (info.size !== asset.size || info.size > 15_000_000) return { id: asset.id, status: 'fail', reason: 'Cached size differs from reviewed pin (not read)' };
    const bytes = await readFile(assetPath(asset));
    if (bytes.length !== asset.size || hash(bytes) !== asset.sha256) return { id: asset.id, status: 'fail', reason: 'Cached size/SHA256 differs from reviewed pin' };
    return { id: asset.id, status: 'pass', size: bytes.length, sha256: hash(bytes) };
  } catch (error) { return { id: asset.id, status: 'blocked', reason: String(error) }; }
}
// No retries, redirects, unbounded bodies, implicit enrollment or URL alternatives.
export async function acquireAsset(asset) {
  if (!asset.sha256) return verifyAsset(asset);
  if (asset.packageFile) return verifyAsset(asset);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  const started = new Date().toISOString();
  try {
    const response = await fetch(asset.url, { signal: controller.signal, redirect: 'error' });
    if (!response.ok) throw new Error(`HTTP ${response.status}; no retry/bypass`);
    const length = Number(response.headers.get('content-length'));
    if (length > asset.size) throw new Error('Content-Length exceeds byte pin');
    const chunks = []; let size = 0;
    for await (const chunk of response.body) {
      size += chunk.length;
      if (size > asset.size || size > 15_000_000) { controller.abort(); throw new Error('Download size exceeded'); }
      chunks.push(chunk);
    }
    const bytes = Buffer.concat(chunks);
    if (size !== asset.size || hash(bytes) !== asset.sha256) throw new Error('Downloaded bytes differ from pin; review a new candidate manifest, including mutable latest');
    await mkdir(cache, { recursive: true });
    const temporary = `${assetPath(asset)}.partial`;
    try { await writeFile(temporary, bytes); await rename(temporary, assetPath(asset)); }
    finally { await rm(temporary, { force: true }); }
    return { id: asset.id, status: 'pass', url: asset.url, started, completed: new Date().toISOString(), size, sha256: hash(bytes) };
  } catch (error) { return { id: asset.id, status: /differ from pin|size exceeded|exceeds byte pin/.test(String(error)) ? 'fail' : 'blocked', url: asset.url, started, reason: String(error) }; }
  finally { clearTimeout(timer); }
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const acquisition = process.argv.includes('--acquire');
  const rows = [];
  for (const asset of manifest.assets) rows.push(await (acquisition ? acquireAsset(asset) : verifyAsset(asset)));
  const output = { mode: acquisition ? 'direct-production-URL-acquisition' : 'cached-byte-verification', at: new Date().toISOString(), rows };
  if (process.env.REAL_SDK_ASSET_REPORT) await writeFile(process.env.REAL_SDK_ASSET_REPORT, JSON.stringify(output, null, 2) + '\n');
  console.log(JSON.stringify(output, null, 2));
  process.exitCode = rows.some(row => row.status === 'fail') ? 1 : rows.some(row => row.status === 'blocked') ? 2 : 0;
}
