import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import {
  engramIndex,
  MIRAGE_CACT_CODEBOOK_LEN,
  MIRAGE_CACT_ENTRY_COUNT,
  MIRAGE_CACT_HEADER_BYTES,
  MIRAGE_CACT_MAGIC,
  MIRAGE_CACT_MIN_DATA_OFFSET,
  MIRAGE_CHARSET_ENTRY,
  MIRAGE_EMBEDDING_DIM,
  MIRAGE_ENGRAM0_EMBEDDING_ENTRY,
  MIRAGE_ENGRAM1_EMBEDDING_ENTRY,
  MIRAGE_ENGRAM_SLOTS,
  MIRAGE_TOKEN_EMBEDDING_ENTRY,
  MIRAGE_VOCAB_ENTRY,
  MirageDecoderTableReader,
} from './mirageCactCodec';
import {
  assembleMirageObservation,
  attachDiagnosticMasksToSemanticObjects,
  buildCrossKvInputArrays,
  buildFourierBox,
  extractProposalMaskPolygon1000,
  extractReaderCropPlan,
  getMirageRopeTables,
  MIRAGE_CROSS_HEAD_DIM,
  MIRAGE_CROSS_KV_HEADS,
  MIRAGE_CROSS_V_DIM,
  MIRAGE_DECODER_KV_HEADS,
  MIRAGE_DECODER_LAYERS,
  MIRAGE_MAX_M,
  MIRAGE_MAX_SEQ,
  MIRAGE_MAX_SLOTS,
  MIRAGE_READER_STRIP_H,
  MIRAGE_READER_STRIP_W,
  parseMirageDecodedText,
  readMirageTextLines,
  runMirageCrossKv,
  runMirageGreedyDecode,
  toMirageRawObservation,
  type MirageOrtSessionRunner,
  type MirageProposalLike,
} from './mirageDecoderAssembly';
import { mapMirageObservationToDetectedObjects } from './mirageVisionAdapter';

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

function buildMinimalVocabRaw(): { raw: Uint8Array; embRows: number } {
  // Indices 0..95 total? Keep small: 11 base pieces + 96 refs = 107 total rows.
  const basePieces: Array<[string, number, number]> = [
    ['<pad>', 2, 0],
    ['</s>', 2, 0],
    ['<s>', 2, 0],
    ['<unk>', 1, 0],
    ['<|en|>', 3, 0],
    ['<|fr|>', 3, 0],
    ['S', 0, -1],
    [' ', 0, -1],
    ['cat', 0, -1],
    ['E1', 0, -1],
    ['\n', 0, -1],
  ];
  const vocabCount = basePieces.length; // 11, refs occupy 11..106 -> embRows = 107
  const utf8 = new TextEncoder();
  const chunks: Uint8Array[] = [];

  const header = new Uint8Array(24);
  const hv = new DataView(header.buffer);
  hv.setUint32(0, vocabCount, true);
  hv.setUint32(4, 0, true); // pad
  hv.setUint32(8, 1, true); // eos
  hv.setUint32(12, 2, true); // bos
  hv.setUint32(16, 3, true); // unk
  header[20] = 0; // add_dummy_prefix
  header[21] = 1; // byte_fallback
  chunks.push(header);

  for (let id = 0; id < vocabCount; id++) {
    const [text, typ, score] = basePieces[id];
    const enc = utf8.encode(text);
    const rec = new Uint8Array(7 + enc.byteLength);
    const rv = new DataView(rec.buffer);
    rv.setFloat32(0, score, true);
    rec[4] = typ;
    rv.setUint16(5, enc.byteLength, true);
    rec.set(enc, 7);
    chunks.push(rec);
  }

  let total = 0;
  for (const c of chunks) total += c.length;
  const merged = new Uint8Array(total);
  let pos = 0;
  for (const c of chunks) {
    merged.set(c, pos);
    pos += c.length;
  }
  return { raw: merged, embRows: vocabCount + 96 };
}

function buildMinimalDecoderTablesCact(): Uint8Array {
  const { raw: vocabRaw, embRows } = buildMinimalVocabRaw();

  // Token embedding [embRows, 512], 4-bit CQ (dtype 3, bits 4, group 128)
  const embCols = MIRAGE_EMBEDDING_DIM;
  const embNg = embCols / 128; // 4
  const embRowIdxBytes = (embCols * 4) / 8; // 256
  const embIdxBytes = embRows * embRowIdxBytes;
  const embData = new Uint8Array(embIdxBytes + embRows * embNg * 2);
  const embDv = new DataView(embData.buffer);
  for (let r = 0; r < embRows; r++) {
    // Non-zero code index in byte 0 (code=15 -> positive codebook entry)
    embData[r * embRowIdxBytes] = 0xff;
    for (let g = 0; g < embNg; g++) {
      embDv.setUint16(
        embIdxBytes + (r * embNg + g) * 2,
        floatToHalfBits(0.1 * ((r % 7) + 1)),
        true
      );
    }
  }

  // Engram tables [4 * MIRAGE_ENGRAM_SLOTS, 128], 2-bit CQ (dtype 3, bits 2, group 128)
  const engRows = 4 * MIRAGE_ENGRAM_SLOTS;
  const engRowIdxBytes = (128 * 2) / 8; // 32
  const engIdxBytes = engRows * engRowIdxBytes;
  const eng0Data = new Uint8Array(engIdxBytes + engRows * 2);
  const eng1Data = new Uint8Array(engIdxBytes + engRows * 2);
  const eng0Dv = new DataView(eng0Data.buffer);
  const eng1Dv = new DataView(eng1Data.buffer);
  const scaleHalfPos = floatToHalfBits(0.25);
  const scaleHalfNeg = floatToHalfBits(-0.25);
  for (let r = 0; r < engRows; r++) {
    eng0Data[r * engRowIdxBytes] = 0b11;
    eng1Data[r * engRowIdxBytes] = 0b10;
    eng0Dv.setUint16(engIdxBytes + r * 2, scaleHalfPos, true);
    eng1Dv.setUint16(engIdxBytes + r * 2, scaleHalfNeg, true);
  }

  // Charset float32 [4] -> blank(0), 'A', 'B', 'C'
  const charsetFloats = new Float32Array([0, 65, 66, 67]);
  const charsetData = new Uint8Array(charsetFloats.buffer.slice(0));

  interface EntrySpec {
    index: number;
    dtype: number;
    shape: number[];
    group: number;
    bits: number;
    data: Uint8Array;
  }

  const specs: EntrySpec[] = [
    {
      index: MIRAGE_TOKEN_EMBEDDING_ENTRY,
      dtype: 3, // CQ
      shape: [embRows, embCols],
      group: 128,
      bits: 4,
      data: embData,
    },
    {
      index: MIRAGE_ENGRAM0_EMBEDDING_ENTRY,
      dtype: 3,
      shape: [engRows, 128],
      group: 128,
      bits: 2,
      data: eng0Data,
    },
    {
      index: MIRAGE_ENGRAM1_EMBEDDING_ENTRY,
      dtype: 3,
      shape: [engRows, 128],
      group: 128,
      bits: 2,
      data: eng1Data,
    },
    {
      index: MIRAGE_CHARSET_ENTRY,
      dtype: 2, // FP32
      shape: [charsetFloats.length],
      group: 0,
      bits: 32,
      data: charsetData,
    },
    {
      index: MIRAGE_VOCAB_ENTRY,
      dtype: 4, // RAW / U8 (check parseCharset/parseVocabRaw accepted dtype)
      shape: [vocabRaw.byteLength],
      group: 0,
      bits: 8,
      data: vocabRaw,
    },
  ];

  let payloadLen = 0;
  for (const s of specs) payloadLen += s.data.byteLength;
  const archive = new Uint8Array(MIRAGE_CACT_MIN_DATA_OFFSET + payloadLen);
  const view = new DataView(archive.buffer);
  view.setUint32(0, MIRAGE_CACT_MAGIC, true);
  view.setUint32(4, MIRAGE_CACT_ENTRY_COUNT, true);
  view.setUint32(8, MIRAGE_CACT_CODEBOOK_LEN, true);
  view.setFloat32(192, 100_000, true);

  // Codebook: 4 entries for 2-bit, 16 entries at offset 12 for 4-bit
  view.setFloat32(MIRAGE_CACT_HEADER_BYTES + 0 * 4, -1.5, true);
  view.setFloat32(MIRAGE_CACT_HEADER_BYTES + 1 * 4, -0.5, true);
  view.setFloat32(MIRAGE_CACT_HEADER_BYTES + 2 * 4, 0.5, true);
  view.setFloat32(MIRAGE_CACT_HEADER_BYTES + 3 * 4, 1.5, true);
  for (let i = 0; i < 16; i++) {
    view.setFloat32(MIRAGE_CACT_HEADER_BYTES + (12 + i) * 4, (i - 7.5) * 0.25, true);
  }

  const byIdx = new Map(specs.map((s) => [s.index, s]));
  let cursor = MIRAGE_CACT_MIN_DATA_OFFSET;
  let dirOffset = MIRAGE_CACT_HEADER_BYTES + 4 * MIRAGE_CACT_CODEBOOK_LEN;

  for (let i = 0; i < MIRAGE_CACT_ENTRY_COUNT; i++) {
    const spec = byIdx.get(i);
    if (!spec) {
      // Sparse omitted placeholder
      archive[dirOffset] = 1;
      archive[dirOffset + 1] = 1;
    } else {
      archive[dirOffset] = spec.dtype;
      archive[dirOffset + 1] = spec.shape.length;
      for (let d = 0; d < spec.shape.length; d++) {
        view.setUint32(dirOffset + 4 + d * 4, spec.shape[d], true);
      }
      view.setUint32(dirOffset + 20, cursor >>> 0, true);
      view.setUint32(dirOffset + 24, 0, true);
      view.setUint32(dirOffset + 28, spec.data.byteLength >>> 0, true);
      view.setUint32(dirOffset + 32, 0, true);
      view.setUint32(dirOffset + 36, spec.group, true);
      view.setUint32(dirOffset + 40, spec.bits, true);
      archive.set(spec.data, cursor);
      cursor += spec.data.byteLength;
    }
    dirOffset += 44;
  }

  return archive;
}

describe('mirageDecoderAssembly full-pipeline helpers', () => {
  it('computes RoPE tables and 64-dim Fourier box encodings faithfully', () => {
    const { cosTable, sinTable } = getMirageRopeTables();
    const halfDim = MIRAGE_CROSS_HEAD_DIM / 2;
    expect(cosTable.length).toBe(MIRAGE_MAX_SEQ * halfDim);
    expect(sinTable.length).toBe(MIRAGE_MAX_SEQ * halfDim);
    for (let k = 0; k < halfDim; k++) {
      expect(cosTable[k]).toBe(1);
      expect(sinTable[k]).toBe(0);
    }
    const freq1 = 1 / Math.pow(100_000, 2 / MIRAGE_CROSS_HEAD_DIM);
    expect(cosTable[halfDim + 1]).toBeCloseTo(Math.cos(freq1), 6);
    expect(sinTable[halfDim + 1]).toBeCloseTo(Math.sin(freq1), 6);

    const fourier = buildFourierBox([0.25, 0.5, 0.75, 0.125]);
    expect(fourier.length).toBe(64);
    expect(fourier[0]).toBeCloseTo(Math.sin(Math.PI * 0.25), 6);
    expect(fourier[8]).toBeCloseTo(Math.cos(Math.PI * 0.25), 6);
    expect(fourier[16 + 1]).toBeCloseTo(Math.sin(2 * Math.PI * 0.5), 5);
    expect(fourier[48 + 2]).toBeCloseTo(Math.sin(4 * Math.PI * 0.125), 5);
  });

  it('builds single-window padded and multi-window sliding reader crops matching stride 224', async () => {
    const emptyPlan = extractReaderCropPlan(new Uint8ClampedArray(64 * 64 * 4), 64, 64, [
      500, 500, 500, 500,
    ]);
    expect(emptyPlan.windows).toHaveLength(0);

    // 100x20 crop -> aspect width U = round(48 * 100 / 20) = 240 <= 320 -> 1 window
    const srcW = 200;
    const srcH = 100;
    const rgba = new Uint8ClampedArray(srcW * srcH * 4).fill(255);
    const singlePlan = extractReaderCropPlan(rgba, srcW, srcH, [0, 0, 500, 200]);
    expect(singlePlan.stripWidth).toBe(240);
    expect(singlePlan.nFrames).toBe(60);
    expect(singlePlan.windows).toHaveLength(1);
    expect(singlePlan.windows[0].lo).toBe(0);
    expect(singlePlan.windows[0].hi).toBe(60);
    expect(singlePlan.windows[0].xs).toBe(0);
    expect(singlePlan.windows[0].tensorData.length).toBe(
      3 * MIRAGE_READER_STRIP_H * MIRAGE_READER_STRIP_W
    );
    // White pixels map to +1.0, right padding pixels map to 127/127.5 - 1
    expect(singlePlan.windows[0].tensorData[0]).toBeCloseTo(1.0, 5);
    expect(singlePlan.windows[0].tensorData[250]).toBeCloseTo(127 / 127.5 - 1, 5);

    // Wide crop: 800x20 -> U = 48 * 800 / 20 = 1920 -> overlapping stride-224 windows
    const wideW = 800;
    const wideH = 40;
    const wideRgba = new Uint8ClampedArray(wideW * wideH * 4).fill(0);
    const widePlan = extractReaderCropPlan(wideRgba, wideW, wideH, [0, 0, 1000, 500]);
    expect(widePlan.stripWidth).toBe(1920);
    expect(widePlan.windows.length).toBeGreaterThan(2);
    expect(widePlan.windows[0].lo).toBe(0);
    expect(widePlan.windows[0].hi).toBe(68);
    expect(widePlan.windows[1].xs).toBe(224);
    expect(widePlan.windows[1].lo).toBe(12);

    // Run readMirageTextLines across mock reader session
    let winCall = 0;
    const mockReader: MirageOrtSessionRunner = {
      async run(feeds) {
        expect(feeds.crop.dims).toEqual([1, 3, 48, 320]);
        const T = 80;
        const C = 307;
        const data = new Float32Array(T * C);
        data.fill(-10);
        // Put class 0 (blank) everywhere except one distinct char per window
        for (let t = 0; t < T; t++) {
          data[t * C] = 10;
        }
        const activeT = winCall === 0 ? 5 : 20;
        const charCls = winCall === 0 ? 1 : 2; // 'A' then 'B'
        data[activeT * C] = -10;
        data[activeT * C + charCls] = 12;
        winCall++;
        return { probs: new ort.Tensor('float32', data, [1, T, C]) };
      },
    };

    const textProposals: MirageProposalLike[] = [
      {
        kind: 'text_proposal',
        index: 0,
        cell: [0, 0],
        score: 0.9,
        box: [0, 0, 0.5, 0.2],
        bbox: [0, 0, 500, 200],
      },
    ];
    const twoWindowPlanProps: MirageProposalLike[] = [
      {
        kind: 'text_proposal',
        index: 0,
        cell: [0, 0],
        score: 0.95,
        box: [0, 0, 1, 1],
        bbox: [0, 0, 1000, 1000],
      },
    ];
    const lines = await readMirageTextLines(mockReader, wideRgba, wideW, wideH, textProposals, 'ABC');
    expect(lines[0].startsWith('A')).toBe(true);
    expect(lines[0]).toContain('B');

    let staleHits = 0;
    const canceledLines = await readMirageTextLines(
      {
        async run() {
          staleHits++;
          return { probs: new ort.Tensor('float32', new Float32Array(80 * 307), [1, 80, 307]) };
        },
      },
      wideRgba,
      wideW,
      wideH,
      twoWindowPlanProps,
      'ABC',
      () => true
    );
    expect(canceledLines).toEqual([]);
    expect(staleHits).toBe(0);
  });

  it('builds cross-KV arrays, runs cross-KV and greedy decode with banning, real or synthetic .cact', async () => {
    const repoRoot = path.resolve(import.meta.dirname, '../..');
    const cactPath = path.join(repoRoot, '.cache/container-models/mirage.cact');
    const archiveBytes = existsSync(cactPath)
      ? new Uint8Array(await readFile(cactPath))
      : buildMinimalDecoderTablesCact();

    const tableReader = new MirageDecoderTableReader(archiveBytes);
    expect(tableReader.vocab.langIds.en).toBeGreaterThanOrEqual(0);
    expect(tableReader.vocab.langIds.fr).toBeGreaterThanOrEqual(0);

    const E = new Float32Array(256 * MIRAGE_EMBEDDING_DIM);
    for (let cell = 0; cell < 256; cell++) {
      E[cell * MIRAGE_EMBEDDING_DIM] = cell + 1;
    }
    const D = new Float32Array(128 * 64 * 64);
    D[2 * 4096 + (3 * 64 + 4)] = 0.75;

    const meta = {
      W: 640,
      H: 480,
      w: 512,
      h: 384,
      r: 0.8,
      coordinate_scale: [1, 512 / 384, 1, 512 / 384] as const,
    };
    const objProps: MirageProposalLike[] = [
      {
        kind: 'object_proposal',
        index: 0,
        cell: [3, 4],
        score: 0.88,
        box: [0.1, 0.2, 0.4, 0.6],
        bbox: [100, 200, 400, 600],
        canvasBox: [0.1, 0.15, 0.4, 0.45],
      },
    ];
    const txtProps: MirageProposalLike[] = [
      {
        kind: 'text_proposal',
        index: 0,
        cell: [1, 2],
        score: 0.77,
        box: [0.5, 0.5, 0.8, 0.6],
        bbox: [500, 500, 800, 600],
        canvasBox: [0.5, 0.375, 0.8, 0.45],
      },
    ];

    const kvInputs = buildCrossKvInputArrays(E, D, meta, objProps, txtProps, tableReader);
    expect(kvInputs.refs).toEqual([
      tableReader.vocab.refId('object', 0),
      tableReader.vocab.refId('text', 0),
    ]);
    // slot 0: D[2, 3, 4] copied to index 2, fourier matches canvasBox, validCells padded below h=384
    expect(kvInputs.slotDFourier[2]).toBeCloseTo(0.75, 6);
    expect(kvInputs.slotDFourier.slice(128, 192)).toEqual(
      buildFourierBox([0.1, 0.15, 0.4, 0.45])
    );
    expect(kvInputs.validCells[0]).toBe(1);
    expect(kvInputs.validCells[12 * 16]).toBe(0); // 12 * 32 = 384 >= h=384
    expect(kvInputs.slotValid[0]).toBe(1);
    expect(kvInputs.slotValid[1]).toBe(1);
    expect(kvInputs.slotValid[2]).toBe(0);

    // Run runMirageCrossKv
    const ckSize = MIRAGE_DECODER_LAYERS * MIRAGE_CROSS_KV_HEADS * MIRAGE_MAX_M * MIRAGE_CROSS_HEAD_DIM;
    const cvSize = MIRAGE_DECODER_LAYERS * MIRAGE_CROSS_KV_HEADS * MIRAGE_MAX_M * MIRAGE_CROSS_V_DIM;
    const mockCrossKv: MirageOrtSessionRunner = {
      async run(feeds) {
        expect(feeds.E.dims).toEqual([1, 256, MIRAGE_EMBEDDING_DIM]);
        expect(feeds.slot_valid.dims).toEqual([MIRAGE_MAX_SLOTS]);
        return {
          ck: new ort.Tensor('float32', new Float32Array(ckSize), [
            MIRAGE_DECODER_LAYERS,
            MIRAGE_CROSS_KV_HEADS,
            MIRAGE_MAX_M,
            MIRAGE_CROSS_HEAD_DIM,
          ]),
          cv: new ort.Tensor('float32', new Float32Array(cvSize), [
            MIRAGE_DECODER_LAYERS,
            MIRAGE_CROSS_KV_HEADS,
            MIRAGE_MAX_M,
            MIRAGE_CROSS_V_DIM,
          ]),
          cross_mask: new ort.Tensor('float32', new Float32Array(MIRAGE_MAX_M), [MIRAGE_MAX_M]),
        };
      },
    };
    const crossKvOut = await runMirageCrossKv(
      mockCrossKv,
      E,
      D,
      meta,
      objProps,
      txtProps,
      tableReader
    );
    expect(crossKvOut.crossK.length).toBe(ckSize);
    expect(crossKvOut.refs).toEqual(kvInputs.refs);

    // Mock greedy decoder: step 0 feeds bos, step 1 feeds <|en|>.
    // On step 1 output, rank forbidden token `<|fr|>` AND forbidden `<obj1>` HIGHER than `<obj0>`,
    // verifying `banned` blocks them and picks `<obj0>`; then step 2 picks eos (`</s>`).
    const seenEngramHashCheck = engramIndex([tableReader.vocab.bos, tableReader.vocab.langIds.en], 1, 0, 2, 0);
    expect(seenEngramHashCheck).toBeGreaterThanOrEqual(0);

    const fedTokens: number[] = [];
    const frId = tableReader.vocab.langIds.fr;
    const bannedObj1 = tableReader.vocab.refId('object', 1);
    const allowedObj0 = tableReader.vocab.refId('object', 0);

    const mockDecoderStep: MirageOrtSessionRunner = {
      async run(feeds) {
        const logits = new Float32Array(tableReader.vocab.size).fill(-10);
        const posArr = feeds.pos_onehot.data as Float32Array;
        const pos = posArr.indexOf(1);
        fedTokens.push(pos);
        if (pos === 1) {
          logits[frId] = 100; // banned language switch
          logits[bannedObj1] = 99; // banned unreferenced slot
          logits[allowedObj0] = 50; // allowed referenced slot!
        } else if (pos === 2) {
          logits[tableReader.vocab.eos] = 80;
        }
        return {
          logits: new ort.Tensor('float32', logits, [tableReader.vocab.size]),
          new_past_k: new ort.Tensor(
            'float32',
            new Float32Array(
              MIRAGE_DECODER_LAYERS *
                MIRAGE_DECODER_KV_HEADS *
                MIRAGE_MAX_SEQ *
                MIRAGE_CROSS_HEAD_DIM
            ),
            [
              MIRAGE_DECODER_LAYERS,
              MIRAGE_DECODER_KV_HEADS,
              MIRAGE_MAX_SEQ,
              MIRAGE_CROSS_HEAD_DIM,
            ]
          ),
          new_past_v: new ort.Tensor(
            'float32',
            new Float32Array(
              MIRAGE_DECODER_LAYERS *
                MIRAGE_DECODER_KV_HEADS *
                MIRAGE_MAX_SEQ *
                MIRAGE_CROSS_V_DIM
            ),
            [
              MIRAGE_DECODER_LAYERS,
              MIRAGE_DECODER_KV_HEADS,
              MIRAGE_MAX_SEQ,
              MIRAGE_CROSS_V_DIM,
            ]
          ),
          new_raw_q_hist: new ort.Tensor(
            'float32',
            new Float32Array(MIRAGE_DECODER_LAYERS * 2 * 384),
            [MIRAGE_DECODER_LAYERS, 2, 384]
          ),
          new_raw_k_hist: new ort.Tensor(
            'float32',
            new Float32Array(MIRAGE_DECODER_LAYERS * 2 * 96),
            [MIRAGE_DECODER_LAYERS, 2, 96]
          ),
          new_raw_v_hist: new ort.Tensor(
            'float32',
            new Float32Array(MIRAGE_DECODER_LAYERS * 2 * 128),
            [MIRAGE_DECODER_LAYERS, 2, 128]
          ),
          new_eng_v_hist: new ort.Tensor(
            'float32',
            new Float32Array(2 * 9 * MIRAGE_EMBEDDING_DIM),
            [2, 9, MIRAGE_EMBEDDING_DIM]
          ),
        };
      },
    };

    const greedy = await runMirageGreedyDecode(mockDecoderStep, crossKvOut, tableReader, {
      language: 'en',
      maxTokens: 10,
    });
    expect(greedy.language).toBe('en');
    expect(greedy.generatedTokenIds).toEqual([allowedObj0]);
    expect(greedy.decodedText).toBe('<obj0>');
    expect(greedy.stepsExecuted).toBe(3);
  });

  it('parses Mirage grammar lines, extracts mask polygons, and assembles DetectedObjects faithfully', () => {
    const sampleDecoded = [
      '<think>two objects on a desk</think>',
      'S location=office;lighting=daylight',
      'E1 coffee mug<obj0>@center {color=white;material=ceramic}',
      'E2 chairs<obj1><obj2> #2@left {type=office}',
      'E3 wooden desk #1@bottom {surface=flat}',
      'T <txt0>',
      'R 1 sits_on 3',
      'EV working_at_desk',
      'stray unparsed token',
      '</s>',
    ].join('\n');

    const parsed = parseMirageDecodedText(sampleDecoded);
    expect(parsed.reasoning).toBe('two objects on a desk');
    expect(parsed.scene).toEqual({ location: 'office', lighting: 'daylight' });
    expect(parsed.entities).toHaveLength(3);
    expect(parsed.entities[0]).toEqual({
      n: 1,
      name: 'coffee mug',
      position: 'center',
      refs: [{ kind: 'obj', index: 0 }],
      description: { color: 'white', material: 'ceramic' },
    });
    expect(parsed.entities[1].count).toBe(2);
    expect(parsed.entities[1].refs).toEqual([
      { kind: 'obj', index: 1 },
      { kind: 'obj', index: 2 },
    ]);
    expect(parsed.entities[2].refs).toEqual([]);
    expect(parsed.text_refs).toEqual([0]);
    expect(parsed.relations).toEqual([{ subject: 1, predicate: 'sits_on', object: 3 }]);
    expect(parsed.events).toEqual(['working_at_desk']);
    expect(parsed.unparsed).toEqual(['stray unparsed token']);

    // Build synthetic 128x128 mask for obj0 within [0.25, 0.25, 0.75, 0.75]
    const mask0 = new Uint8Array(128 * 128);
    for (let y = 32; y < 96; y++) {
      for (let x = 32; x < 96; x++) {
        mask0[y * 128 + x] = 1;
      }
    }
    const objects: MirageProposalLike[] = [
      {
        kind: 'object_proposal',
        index: 0,
        cell: [32, 32],
        score: 0.9124,
        box: [0.25, 0.25, 0.75, 0.75],
        bbox: [250, 250, 750, 750],
        mask: mask0,
      },
      {
        kind: 'object_proposal',
        index: 1,
        cell: [10, 10],
        score: 0.72,
        box: [0.05, 0.1, 0.2, 0.4],
        bbox: [50, 100, 200, 400],
      },
      {
        kind: 'object_proposal',
        index: 2,
        cell: [12, 12],
        score: 0.68,
        box: [0.22, 0.1, 0.38, 0.4],
        bbox: [220, 100, 380, 400],
      },
    ];
    const texts: MirageProposalLike[] = [
      {
        kind: 'text_proposal',
        index: 0,
        cell: [5, 5],
        score: 0.85,
        box: [0.1, 0.05, 0.3, 0.1],
        bbox: [100, 50, 300, 100],
      },
      {
        kind: 'text_proposal',
        index: 1,
        cell: [6, 6],
        score: 0.81,
        box: [0.1, 0.12, 0.3, 0.18],
        bbox: [100, 120, 300, 180],
      },
    ];

    const poly1000 = extractProposalMaskPolygon1000(objects[0]);
    expect(poly1000).toBeDefined();
    expect(poly1000!.length).toBeGreaterThanOrEqual(6);
    for (let i = 0; i < poly1000!.length; i += 2) {
      expect(poly1000![i]).toBeGreaterThanOrEqual(240);
      expect(poly1000![i]).toBeLessThanOrEqual(760);
      expect(poly1000![i + 1]).toBeGreaterThanOrEqual(240);
      expect(poly1000![i + 1]).toBeLessThanOrEqual(760);
    }

    const assembled = assembleMirageObservation({
      decodedText: sampleDecoded,
      generatedTokenIds: [12, 1],
      language: 'en',
      objects,
      texts,
      ocrStrings: ['CAFE', 'OPEN'],
    });
    expect(assembled.objects).toHaveLength(2);
    expect(assembled.objects[0].name).toBe('coffee mug');
    expect(assembled.objects[0].mask).toBeDefined();
    expect(assembled.objects[1].instances).toHaveLength(2);
    expect(assembled.regions).toHaveLength(1);
    expect(assembled.regions[0].name).toBe('wooden desk');
    expect(assembled.text).toEqual([
      { string: 'CAFE', bbox: [100, 50, 300, 100], score: 0.85, mentioned: true },
      { string: 'OPEN', bbox: [100, 120, 300, 180], score: 0.81, mentioned: false },
    ]);

    const rawObs = toMirageRawObservation(assembled);
    const mapped = mapMirageObservationToDetectedObjects(rawObs, {
      confidenceThreshold: 0.8, // Filters out entity E2 (`chairs` score 0.72) to verify mask matching doesn't shift!
      textConfidenceThreshold: 0.5,
      includeOcrAsDetections: true,
      maxDetections: 20,
      bboxFormat: 'xyxy',
    });

    const renderedIds: number[] = [];
    attachDiagnosticMasksToSemanticObjects(
      mapped.objects,
      assembled,
      sampleDecoded,
      objects,
      (mask) => {
        renderedIds.push(mask.length);
        return { width: 128, height: 128 } as unknown as HTMLCanvasElement;
      }
    );
    expect(renderedIds).toEqual([128 * 128]);
    expect(mapped.objects[0].label).toBe('coffee mug');
    expect(mapped.objects[0].maskCanvas).toBeDefined();
    expect(mapped.objects[0].maskPolygon?.length).toBeGreaterThanOrEqual(3);
  });
});
