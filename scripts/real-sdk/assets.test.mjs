import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { writeFile, readFile, mkdir, rm } from 'node:fs/promises';
import { cache, verifyAsset, acquireAsset, assetPath, hash } from './assets.mjs';

for (const scenario of ['verified', 'tampered', 'missing', 'unpinned']) {
  test(`asset pin ${scenario}`, async () => {
    await mkdir(cache, { recursive: true });
    const asset = { id: 'control', file: `control-${randomUUID()}.tflite`, url: 'https://example.invalid/model', size: 4, sha256: hash(Buffer.from('real')) };
    try {
      if (scenario !== 'missing') await writeFile(assetPath(asset), scenario === 'tampered' ? 'fake' : 'real');
      if (scenario === 'unpinned') asset.sha256 = null;
      const result = await verifyAsset(asset);
      assert.equal(result.status, scenario === 'verified' ? 'pass' : scenario === 'tampered' ? 'fail' : 'blocked');
    } finally { await rm(assetPath(asset), { force: true }); }
  });
}
test('changed latest bytes cannot replace an existing verified cache', async t => {
  const asset = { id: 'latest-control', file: `control-${randomUUID()}.tflite`, url: 'https://example.invalid/latest', size: 4, sha256: hash(Buffer.from('real')) };
  await writeFile(assetPath(asset), 'real');
  t.mock.method(globalThis, 'fetch', async (_url, options) => { assert.equal(options.redirect, 'error'); return new Response('fake', { status: 200 }); });
  try {
    const result = await acquireAsset(asset);
    assert.equal(result.status, 'fail'); assert.match(result.reason, /differ from pin/);
    assert.equal(await readFile(assetPath(asset), 'utf8'), 'real');
  } finally { await rm(assetPath(asset), { force: true }); }
});
test('HTTP403 is one request, with no cache acceptance or retry', async t => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => { requests++; return new Response('denied', { status: 403 }); });
  const asset = { id: 'denied-control', file: `control-${randomUUID()}.tflite`, url: 'https://example.invalid/model', size: 4, sha256: hash(Buffer.from('real')) };
  const result = await acquireAsset(asset);
  assert.equal(result.status, 'blocked'); assert.match(result.reason, /403/); assert.equal(requests, 1);
  assert.equal((await verifyAsset(asset)).status, 'blocked');
});
test('oversized body is stopped before writing', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('oversized', { status: 200 }));
  const asset = { id: 'size-control', file: `control-${randomUUID()}.tflite`, url: 'https://example.invalid/model', size: 4, sha256: hash(Buffer.from('real')) };
  const result = await acquireAsset(asset);
  assert.equal(result.status, 'fail'); assert.match(result.reason, /size exceeded/);
  assert.equal((await verifyAsset(asset)).status, 'blocked');
});
