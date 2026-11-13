import { Buffer } from 'node:buffer';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ctcCollapse,
  decodeCqMatrix,
  decodeCqRowInto,
  engramIndex,
  getFp16ToFp32Table,
  MIRAGE_CACT_CODEBOOK_LEN,
  MIRAGE_DECODER_TABLES_EXPECTED_SHA256,
  MIRAGE_DECODER_TABLES_EXPECTED_SIZE,
  MIRAGE_DECODER_TABLES_FILE,
  MIRAGE_ENGRAM_PRIME,
  MIRAGE_ENGRAM_SEED,
  MIRAGE_ENGRAM_SLOTS,
  MirageDecoderTableReader,
  parseVocabRaw,
} from './mirageCactCodec';
import {
  MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256,
  MIRAGE_CROSS_KV_ONNX_EXPECTED_SIZE,
  MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256,
  MIRAGE_DECODER_STEP_ONNX_EXPECTED_SIZE,
  MIRAGE_DECODER_TABLES_LOCAL_URL,
  MIRAGE_READER_ONNX_EXPECTED_SHA256,
  MIRAGE_READER_ONNX_EXPECTED_SIZE,
} from './modelAssetResolver';
import {
  extractVerifiedMirageDecoderTablesBytes,
  reconstructVerifiedMirageComponentBytes,
} from './mirageOnnxReconstructor';
import { ContainerModelCacheManager } from './containerModelCacheServer';

function floatToHalfBits(val: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = (x >>> 13) & 0x03ff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x0400) >>> (1 - exp);
    return sign | mant;
  }
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

function packSyntheticCq(
  mat: Float64Array,
  rows: number,
  cols: number,
  codebook: Float64Array,
  bits: 2 | 4,
  group: number = 128
): Uint8Array {
  const ng = Math.ceil(cols / group);
  const padded = ng * group;
  const cbOffset = bits === 4 ? 12 : 0;
  const cbLen = bits === 4 ? 16 : 4;
  const per = 8 / bits;
  const mask = (1 << bits) - 1;
  const rowIndexBytes = (padded * bits) / 8;
  const totalIndexBytes = rows * rowIndexBytes;
  const out = new Uint8Array(totalIndexBytes + rows * ng * 2);
  const fp16Table = getFp16ToFp32Table();

  let cbMax = 0;
  for (let i = 0; i < cbLen; i++) {
    cbMax = Math.max(cbMax, Math.abs(codebook[cbOffset + i]));
  }

  const block = new Float64Array(group);
  const temp = new Float64Array(group);
  const invSqrt = 1 / Math.sqrt(group);

  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < ng; g++) {
      block.fill(0);
      const colBase = g * group;
      const valid = Math.min(group, cols - colBase);
      for (let k = 0; k < valid; k++) {
        block[k] = mat[r * cols + colBase + k];
      }
      for (let width = 1; width < group; width <<= 1) {
        const stride = width << 1;
        for (let start = 0; start < group; start += stride) {
          for (let k = 0; k < width; k++) {
            const a = block[start + k];
            const c = block[start + width + k];
            temp[start + k] = a + c;
            temp[start + width + k] = a - c;
          }
        }
        block.set(temp);
      }
      for (let k = 0; k < group; k++) {
        block[k] *= invSqrt;
      }

      let maxAbs = 0;
      for (let k = 0; k < group; k++) {
        maxAbs = Math.max(maxAbs, Math.abs(block[k]));
      }
      const normBits = floatToHalfBits((maxAbs || 1) / (cbMax || 1));
      const norm = fp16Table[normBits] || 1;
      const normOff = totalIndexBytes + (r * ng + g) * 2;
      out[normOff] = normBits & 0xff;
      out[normOff + 1] = (normBits >>> 8) & 0xff;

      const bytesPerGroup = (group * bits) / 8;
      const packedStart = r * rowIndexBytes + g * bytesPerGroup;
      for (let k = 0; k < group; k += per) {
        let byteVal = 0;
        for (let j = 0; j < per; j++) {
          let bestCode = 0;
          let bestErr = Infinity;
          for (let c = 0; c < cbLen; c++) {
            const err = Math.abs(codebook[cbOffset + c] * norm - block[k + j]);
            if (err < bestErr) {
              bestErr = err;
              bestCode = c;
            }
          }
          byteVal |= (bestCode & mask) << (j * bits);
        }
        out[packedStart + k / per] = byteVal;
      }
    }
  }
  return out;
}

describe('mirageCactCodec pure TypeScript primitives', () => {
  it('converts binary16 special values identically to IEEE-754 float32', () => {
    const tbl = getFp16ToFp32Table();
    expect(tbl[0x0000]).toBe(0);
    expect(Object.is(tbl[0x8000], -0)).toBe(true);
    expect(tbl[0x3c00]).toBe(1);
    expect(tbl[0xbc00]).toBe(-1);
    expect(tbl[0x7bff]).toBe(65504);
    expect(tbl[0x0001]).toBeCloseTo(2 ** -24, 12);
    expect(tbl[0x7c00]).toBe(Infinity);
    expect(tbl[0xfc00]).toBe(-Infinity);
    expect(Number.isNaN(tbl[0x7e00])).toBe(true);
  });

  it('decodes 4-bit CQ and sub-group 2-bit CQ matrices and lazy rows accurately', () => {
    const cb = new Float64Array(MIRAGE_CACT_CODEBOOK_LEN);
    cb.set([-1.5, -0.5, 0.5, 1.5], 0);
    for (let i = 0; i < 16; i++) {
      cb[12 + i] = -3.75 + i * 0.5;
    }
    const fp16 = getFp16ToFp32Table();

    // 4-bit CQ [3, 128] with deterministic zero-mean normal-like values
    const mat4 = new Float64Array(3 * 128);
    let seed = 1234567;
    for (let i = 0; i < mat4.length; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      const u1 = Math.max(1e-6, seed / 0x1_0000_0000);
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      const u2 = seed / 0x1_0000_0000;
      mat4[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.35;
    }
    const packed4 = packSyntheticCq(mat4, 3, 128, cb, 4);
    const entry4 = {
      dtype: 3,
      shape: [3, 128],
      offset: 0,
      nbytes: packed4.byteLength,
      group: 128,
      bits: 4,
    };
    const decoded4 = decodeCqMatrix(packed4, entry4, cb, fp16);
    const decoded4T = decodeCqMatrix(packed4, entry4, cb, fp16, { transpose: true });
    for (let r = 0; r < 3; r++) {
      const rowOut = new Float32Array(128);
      decodeCqRowInto(packed4, entry4, cb, fp16, r, rowOut);
      for (let c = 0; c < 128; c++) {
        expect(rowOut[c]).toBe(decoded4[r * 128 + c]);
        expect(decoded4T[c * 3 + r]).toBe(decoded4[r * 128 + c]);
        expect(Math.abs(decoded4[r * 128 + c] - mat4[r * 128 + c])).toBeLessThan(0.25);
      }
    }

    // 2-bit sub-group CQ [4, 32] (like reader/pw_0/kernel!)
    const mat2 = new Float64Array(4 * 32);
    for (let i = 0; i < mat2.length; i++) {
      mat2[i] = Math.cos(i * 0.21) * 0.5;
    }
    const packed2 = packSyntheticCq(mat2, 4, 32, cb, 2);
    const entry2 = {
      dtype: 3,
      shape: [4, 32],
      offset: 0,
      nbytes: packed2.byteLength,
      group: 128,
      bits: 2,
    };
    const decoded2T = decodeCqMatrix(packed2, entry2, cb, fp16, { transpose: true });
    expect(decoded2T.length).toBe(32 * 4);
    const decoded2 = decodeCqMatrix(packed2, entry2, cb, fp16);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 32; c++) {
        expect(decoded2T[c * 4 + r]).toBe(decoded2[r * 32 + c]);
      }
    }
  });

  it('decodes and encodes SentencePiece RAW vocab with UTF-8 byte fallback and refs', () => {
    const items: Array<[string, number, number]> = [
      ['<pad>', 2, 0],
      ['</s>', 2, 0],
      ['<s>', 2, 0],
      ['<unk>', 1, 0],
      ['\u2581hello', 0, -1.2],
      ['\u2581world', 0, -1.5],
      ['<0xF0>', 4, 0],
      ['<0x9F>', 4, 0],
      ['<0x98>', 4, 0],
      ['<0x80>', 4, 0],
      ['<|en|>', 3, 0],
      ['<|fr|>', 3, 0],
    ];
    const utf8 = new TextEncoder();
    const chunks: Uint8Array[] = [];
    const header = new Uint8Array(24);
    const hv = new DataView(header.buffer);
    hv.setUint32(0, items.length, true);
    hv.setUint32(4, 0, true);
    hv.setUint32(8, 1, true);
    hv.setUint32(12, 2, true);
    hv.setUint32(16, 3, true);
    header[20] = 0;
    header[21] = 1;
    chunks.push(header);
    for (const [text, typ, score] of items) {
      const enc = utf8.encode(text);
      const rec = new Uint8Array(7 + enc.byteLength);
      const rv = new DataView(rec.buffer);
      rv.setFloat32(0, score, true);
      rec[4] = typ;
      rv.setUint16(5, enc.byteLength, true);
      rec.set(enc, 7);
      chunks.push(rec);
    }
    const raw = Buffer.concat(chunks);
    const vocab = parseVocabRaw(raw, items.length + 96);

    expect(vocab.langIds).toEqual({ en: 10, fr: 11 });
    expect(vocab.refId('object', 3)).toBe(items.length + 3);
    expect(vocab.refId('text', 2)).toBe(items.length + 64 + 2);
    expect(
      vocab.decode([2, 10, 4, 6, 7, 8, 9, vocab.refId('object', 0), 1], false)
    ).toBe(' hello😀<obj0>');
    expect(vocab.encode(' hello world😀')).toEqual([4, 5, 6, 7, 8, 9]);
  });

  it('matches reference CTC collapse and unsigned 32-bit engram hashing', () => {
    expect(ctcCollapse([0, 1, 1, 0, 1, 2, 2, 3, 0], 'ABC')).toBe('AABC');
    expect(ctcCollapse([0, 99, 0], 'ABC')).toBe('');

    const tokens = [2, 8192, 42, 17, 999];
    const got = engramIndex(tokens, 4, 1, 3, 1);
    const seed = BigInt(MIRAGE_ENGRAM_SEED);
    const prime = BigInt(MIRAGE_ENGRAM_PRIME);
    let acc = BigInt.asUintN(32, seed * BigInt(1 * 2 + 1 + 1));
    for (let j = 0; j < 3; j++) {
      acc = BigInt.asUintN(32, (acc ^ BigInt(tokens[4 - j])) * prime);
    }
    const expected = Number((acc ^ (acc >> 15n)) % BigInt(MIRAGE_ENGRAM_SLOTS));
    expect(got).toBe(expected);
  });
});

const repoRoot = path.resolve(process.cwd());
const localCactPath = path.join(repoRoot, '.cache', 'container-models', 'mirage.cact');
const hasLocalCact = existsSync(localCactPath);

describe.skipIf(!hasLocalCact)('Mirage full-pipeline TS archive reconstruction & cache delivery', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(tmpdir(), 'mirage-full-delivery-'));
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await rm(tempDir, { recursive: true, force: true });
  });

  it('reconstructs reader, crossKv, decoderStep ONNX and sparse decoder-tables.cact with exact SHA-256', async () => {
    const cactBytes = await readFile(localCactPath);

    const tablesBytes = extractVerifiedMirageDecoderTablesBytes(cactBytes);
    expect(tablesBytes.byteLength).toBe(MIRAGE_DECODER_TABLES_EXPECTED_SIZE);
    expect(createHash('sha256').update(tablesBytes).digest('hex')).toBe(
      MIRAGE_DECODER_TABLES_EXPECTED_SHA256
    );

    const fullReader = new MirageDecoderTableReader(cactBytes);
    const compactReader = new MirageDecoderTableReader(tablesBytes);
    expect(compactReader.vocab.count).toBe(fullReader.vocab.count);
    expect(compactReader.vocab.langIds).toEqual(fullReader.vocab.langIds);
    expect(compactReader.charset).toBe(fullReader.charset);
    expect(compactReader.readTokenEmbedding(2)).toEqual(fullReader.readTokenEmbedding(2));
    expect(
      compactReader.readEngramVector(0, [2, 8192, 42, 17], 3)
    ).toEqual(fullReader.readEngramVector(0, [2, 8192, 42, 17], 3));
    expect(
      compactReader.readEngramVector(1, [2, 8192, 42, 17], 3)
    ).toEqual(fullReader.readEngramVector(1, [2, 8192, 42, 17], 3));

    const components = [
      {
        key: 'reader' as const,
        size: MIRAGE_READER_ONNX_EXPECTED_SIZE,
        sha256: MIRAGE_READER_ONNX_EXPECTED_SHA256,
      },
      {
        key: 'crossKv' as const,
        size: MIRAGE_CROSS_KV_ONNX_EXPECTED_SIZE,
        sha256: MIRAGE_CROSS_KV_ONNX_EXPECTED_SHA256,
      },
      {
        key: 'decoderStep' as const,
        size: MIRAGE_DECODER_STEP_ONNX_EXPECTED_SIZE,
        sha256: MIRAGE_DECODER_STEP_ONNX_EXPECTED_SHA256,
      },
    ];

    for (const c of components) {
      const rebuilt = reconstructVerifiedMirageComponentBytes(c.key, cactBytes, repoRoot);
      expect(rebuilt.byteLength).toBe(c.size);
      expect(createHash('sha256').update(rebuilt).digest('hex')).toBe(c.sha256);
    }
  });

  it('serves verified mirage-decoder-tables.cact on HTTP startup request with query pin check', async () => {
    const cactBytes = await readFile(localCactPath);
    await writeFile(path.join(tempDir, 'mirage.cact'), cactBytes);
    const manager = new ContainerModelCacheManager({
      cacheDir: tempDir,
      rootDir: repoRoot,
    });
    const mw = manager.createMiddleware();

    const invoke = async (
      url: string,
      method: 'GET' | 'HEAD' = 'GET',
      headers: Record<string, string> = {}
    ) => {
      const resHeaders: Record<string, string> = {};
      const chunks: Buffer[] = [];
      let finishResolve!: () => void;
      const done = new Promise<void>(r => {
        finishResolve = r;
      });
      const res = {
        statusCode: 200,
        setHeader(name: string, value: string) {
          resHeaders[name.toLowerCase()] = String(value);
        },
        write(chunk: Uint8Array | string) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
          return true;
        },
        once() {},
        end(chunk?: Uint8Array | string) {
          if (chunk) {
            chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
          }
          finishResolve();
        },
        destroy() {
          finishResolve();
        },
      };
      await mw(
        { url, method, headers } as unknown as IncomingMessage,
        res as unknown as ServerResponse,
        (err?: unknown) => {
          throw err ?? new Error('Unhandled route');
        }
      );
      await done;
      return {
        status: res.statusCode,
        headers: resHeaders,
        body: Buffer.concat(chunks),
      };
    };

    const badQuery = await invoke(`/api/model-cache/mirage/${MIRAGE_DECODER_TABLES_FILE}`);
    expect(badQuery.status).toBe(400);

    const headOk = await invoke(MIRAGE_DECODER_TABLES_LOCAL_URL, 'HEAD');
    expect(headOk.status).toBe(200);
    expect(headOk.headers['x-model-sha256']).toBe(MIRAGE_DECODER_TABLES_EXPECTED_SHA256);

    const rangeOk = await invoke(MIRAGE_DECODER_TABLES_LOCAL_URL, 'GET', {
      range: 'bytes=0-15',
    });
    expect(rangeOk.status).toBe(206);
    expect(rangeOk.body.byteLength).toBe(16);
    expect(rangeOk.body.readUInt32LE(0)).toBe(0x05e12a84);
  });
});
