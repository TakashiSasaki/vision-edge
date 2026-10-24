import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = fileURLToPath(new URL('../../', import.meta.url));
export const cache = path.join(root, '.cache/real-sdk-assets');
/** @typedef {{id: string, file: string, url: string, sha256: string | null, size: number | null, packageFile?: string, acquisitionStatus?: string}} Asset */
/** @type {{schema: number, assets: Asset[]}} */
export const manifest = JSON.parse(await readFile(path.join(root, 'e2e-real-sdk/assets.lock.json'), 'utf8'));
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
