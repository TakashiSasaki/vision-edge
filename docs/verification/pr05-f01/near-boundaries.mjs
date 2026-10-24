import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { clientPointToVideo } from '../../../src/services/videoCoordinates.ts';

const rows = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const failures = [];
const counts = {};
for (const row of rows) {
  globalThis.devicePixelRatio = row.dpr;
  const actual = clientPointToVideo(...row.point, row.rect, ...row.video);
  const pass = row.expected === null ? actual === null : row.expected === 'inside'
    ? actual && actual.x >= 0 && actual.x <= 1 && actual.y >= 0 && actual.y <= 1
    : actual && Math.abs(actual.x - row.expected[0]) < 1e-12 && Math.abs(actual.y - row.expected[1]) < 1e-12;
  counts[row.kind] = (counts[row.kind] ?? 0) + 1;
  if (!pass) failures.push({ ...row, actual });
}
writeFileSync(process.argv[3], JSON.stringify({ count: rows.length, counts, failures }, null, 2) + '\n');
console.log(`${rows.length} checks; ${failures.length} failures; ${JSON.stringify(counts)}`);
assert.equal(failures.length, 0, JSON.stringify(failures.slice(0, 2)));
