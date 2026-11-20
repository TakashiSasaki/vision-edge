// @vitest-environment node
import { describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import {
  decomposeOnnx5dConvNodesForWebGpu,
  inspectStereoOnnxWebGpuCompatibility,
  validateOnnxGraphNodeInputs,
} from './stereoDepthWorker';

const textEncoder = new TextEncoder();

function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = [];
  let v = value >>> 0;
  while (v >= 0x80) {
    bytes.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  bytes.push(v);
  return new Uint8Array(bytes);
}

function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.byteLength;
  }
  return out;
}

function encodeLenField(fn: number, payload: Uint8Array): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 2), encodeVarint(payload.byteLength), payload]);
}

function encodeVarintField(fn: number, val: number): Uint8Array {
  return concatBytes([encodeVarint((fn << 3) | 0), encodeVarint(val)]);
}

function encodeStrField(fn: number, value: string): Uint8Array {
  return encodeLenField(fn, textEncoder.encode(value));
}

function encodeAttrInt(name: string, val: number): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([encodeStrField(1, name), encodeVarintField(3, val), encodeVarintField(20, 2)])
  );
}

function encodeAttrInts(name: string, vals: readonly number[]): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([
      encodeStrField(1, name),
      ...vals.map((v) => encodeVarintField(8, v)),
      encodeVarintField(20, 7),
    ])
  );
}

function encodeAttrString(name: string, val: string): Uint8Array {
  return encodeLenField(
    5,
    concatBytes([encodeStrField(1, name), encodeStrField(4, val), encodeVarintField(20, 3)])
  );
}

function encodeValueInfoFloat(name: string, dims: readonly number[]): Uint8Array {
  const dimFields = dims.map((d) => encodeLenField(1, encodeVarintField(1, d)));
  const shapeProto = encodeLenField(2, concatBytes(dimFields));
  const tensorTypeProto = encodeLenField(1, concatBytes([encodeVarintField(1, 1), shapeProto]));
  return concatBytes([encodeStrField(1, name), encodeLenField(2, tensorTypeProto)]);
}

function encodeFloatTensorProto(
  name: string,
  dims: readonly number[],
  values: Float32Array
): Uint8Array {
  const raw = new Uint8Array(values.length * 4);
  const dv = new DataView(raw.buffer);
  for (let i = 0; i < values.length; i++) {
    dv.setFloat32(i * 4, values[i], true);
  }
  return concatBytes([
    ...dims.map((d) => encodeVarintField(1, d)),
    encodeVarintField(2, 1), // FLOAT
    encodeStrField(8, name),
    encodeLenField(9, raw),
  ]);
}

interface Conv3dFixtureSpec {
  nodeName: string;
  inputDims: [number, number, number, number, number];
  weightDims: [number, number, number, number, number];
  strides: [number, number, number];
  pads: [number, number, number, number, number, number];
  dilations?: [number, number, number];
  group?: number;
  withBias?: boolean;
  duplicateWeightInGraphInput?: boolean;
  opsetVersion?: number;
}

function buildValidConv3dOnnxModel(spec: Conv3dFixtureSpec): {
  modelBytes: Uint8Array;
  inputData: Float32Array;
  expectedOutDims: [number, number, number, number, number];
} {
  const [N, CIn, DIn, HIn, WIn] = spec.inputDims;
  const [COut, CInPerGroup, kD, kH, kW] = spec.weightDims;
  const [sD, sH, sW] = spec.strides;
  const [pD0, pH0, pW0, pD1, pH1, pW1] = spec.pads;
  const [dD, dH, dW] = spec.dilations ?? [1, 1, 1];
  const group = spec.group ?? 1;

  const DOut = Math.floor((DIn + pD0 + pD1 - dD * (kD - 1) - 1) / sD) + 1;
  const HOut = Math.floor((HIn + pH0 + pH1 - dH * (kH - 1) - 1) / sH) + 1;
  const WOut = Math.floor((WIn + pW0 + pW1 - dW * (kW - 1) - 1) / sW) + 1;

  const weightLen = COut * CInPerGroup * kD * kH * kW;
  const weightData = new Float32Array(weightLen);
  for (let i = 0; i < weightLen; i++) {
    weightData[i] = ((i * 7 + 3) % 19 - 9) * 0.08;
  }

  const inputLen = N * CIn * DIn * HIn * WIn;
  const inputData = new Float32Array(inputLen);
  for (let i = 0; i < inputLen; i++) {
    inputData[i] = ((i * 11 + 5) % 23 - 11) * 0.1;
  }

  const biasData = new Float32Array(COut);
  for (let i = 0; i < COut; i++) {
    biasData[i] = (i + 1) * 0.15;
  }

  const convInputs = spec.withBias ? ['x', 'w', 'b'] : ['x', 'w'];
  const attrs: Uint8Array[] = [
    encodeAttrInts('kernel_shape', [kD, kH, kW]),
    encodeAttrInts('strides', [sD, sH, sW]),
    encodeAttrInts('pads', [pD0, pH0, pW0, pD1, pH1, pW1]),
    encodeAttrInts('dilations', [dD, dH, dW]),
  ];
  if (group > 1) {
    attrs.push(encodeAttrInt('group', group));
  }
  const convNode = encodeLenField(
    1,
    concatBytes([
      ...convInputs.map((inp) => encodeStrField(1, inp)),
      encodeStrField(2, 'y'),
      encodeStrField(3, spec.nodeName),
      encodeStrField(4, 'Conv'),
      ...attrs,
    ])
  );

  const graphParts: Uint8Array[] = [
    convNode,
    encodeStrField(2, 'conv3d_test_graph'),
    encodeLenField(5, encodeFloatTensorProto('w', spec.weightDims, weightData)),
  ];
  if (spec.withBias) {
    graphParts.push(encodeLenField(5, encodeFloatTensorProto('b', [COut], biasData)));
  }
  graphParts.push(encodeLenField(11, encodeValueInfoFloat('x', spec.inputDims)));
  if (spec.duplicateWeightInGraphInput) {
    graphParts.push(encodeLenField(11, encodeValueInfoFloat('w', spec.weightDims)));
  }
  graphParts.push(
    encodeLenField(12, encodeValueInfoFloat('y', [N, COut, DOut, HOut, WOut]))
  );

  const opsetImport = encodeLenField(
    8,
    concatBytes([encodeStrField(1, ''), encodeVarintField(2, spec.opsetVersion ?? 13)])
  );

  const modelBytes = concatBytes([
    encodeVarintField(1, 7),
    encodeLenField(7, concatBytes(graphParts)),
    opsetImport,
  ]);

  return {
    modelBytes,
    inputData,
    expectedOutDims: [N, COut, DOut, HOut, WOut],
  };
}

describe('decomposeOnnx5dConvNodesForWebGpu real ORT verification', () => {
  ort.env.wasm.numThreads = 1;

  it('produces ORT-valid 2D Conv graphs with exact numerical parity for kD=3 (padding+bias), kD=3 (strided), grouped+dilated, kD=1 fast path, and legacy duplicated graph inputs', async () => {
    const cases: Conv3dFixtureSpec[] = [
      {
        nodeName: 'Conv_1005',
        inputDims: [1, 2, 4, 5, 6],
        weightDims: [3, 2, 3, 3, 3],
        strides: [1, 1, 1],
        pads: [1, 1, 1, 1, 1, 1],
        withBias: true,
      },
      {
        nodeName: 'Conv_strided_3d',
        inputDims: [1, 2, 6, 5, 6],
        weightDims: [2, 2, 3, 3, 3],
        strides: [2, 1, 2],
        pads: [1, 0, 1, 1, 0, 1],
        withBias: false,
        duplicateWeightInGraphInput: true,
      },
      {
        nodeName: 'Conv_grouped_dilated',
        inputDims: [1, 4, 6, 5, 5],
        weightDims: [4, 2, 2, 3, 3],
        strides: [1, 1, 1],
        pads: [1, 1, 1, 1, 1, 1],
        dilations: [2, 1, 1],
        group: 2,
        withBias: true,
      },
      {
        nodeName: 'Conv_kd1_fast',
        inputDims: [1, 2, 4, 4, 5],
        weightDims: [3, 2, 1, 3, 3],
        strides: [1, 1, 1],
        pads: [0, 1, 1, 0, 1, 1],
        withBias: true,
      },
    ];

    for (const spec of cases) {
      const { modelBytes, inputData, expectedOutDims } = buildValidConv3dOnnxModel(spec);

      expect(validateOnnxGraphNodeInputs(modelBytes)).toMatchObject({
        valid: true,
        missingInputs: [],
      });
      expect(inspectStereoOnnxWebGpuCompatibility(modelBytes).hasConv3d).toBe(true);

      const decomposed = decomposeOnnx5dConvNodesForWebGpu(modelBytes);
      expect(decomposed.decomposedCount).toBe(1);
      expect(decomposed.decomposedNodeNames).toEqual([spec.nodeName]);
      expect(inspectStereoOnnxWebGpuCompatibility(decomposed.bytes)).toMatchObject({
        compatible: true,
        hasConv3d: false,
      });
      expect(validateOnnxGraphNodeInputs(decomposed.bytes)).toMatchObject({
        valid: true,
        missingInputs: [],
      });

      const origSession = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const decSession = await ort.InferenceSession.create(decomposed.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });

      try {
        const xTensor = new ort.Tensor('float32', inputData, spec.inputDims);
        const origOut = (await origSession.run({ x: xTensor })).y;
        const decOut = (await decSession.run({ x: xTensor })).y;

        expect(origOut.dims).toEqual(expectedOutDims);
        expect(decOut.dims).toEqual(expectedOutDims);

        const origArr = origOut.data as Float32Array;
        const decArr = decOut.data as Float32Array;
        expect(decArr.length).toBe(origArr.length);

        let maxAbsDiff = 0;
        for (let i = 0; i < origArr.length; i++) {
          const diff = Math.abs(origArr[i] - decArr[i]);
          if (diff > maxAbsDiff) maxAbsDiff = diff;
        }
        expect(maxAbsDiff).toBeLessThan(1e-4);
      } finally {
        await origSession.release();
        await decSession.release();
      }
    }
  });

  it('decomposes multiple chained 5D Conv3D nodes sharing helper constants with exact numerical parity', async () => {
    const inputDims = [1, 2, 4, 4, 4] as const;
    const w1Data = new Float32Array(3 * 2 * 3 * 3 * 3).map((_, i) => ((i * 5 + 1) % 13 - 6) * 0.07);
    const w2Data = new Float32Array(2 * 3 * 3 * 3 * 3).map((_, i) => ((i * 3 + 2) % 11 - 5) * 0.06);
    const inputData = new Float32Array(1 * 2 * 4 * 4 * 4).map((_, i) => ((i * 9 + 4) % 17 - 8) * 0.1);

    const conv1 = encodeLenField(
      1,
      concatBytes([
        encodeStrField(1, 'x'),
        encodeStrField(1, 'w1'),
        encodeStrField(2, 'mid'),
        encodeStrField(3, 'Conv_1005'),
        encodeStrField(4, 'Conv'),
        encodeAttrInts('kernel_shape', [3, 3, 3]),
        encodeAttrInts('strides', [1, 1, 1]),
        encodeAttrInts('pads', [1, 1, 1, 1, 1, 1]),
      ])
    );
    const conv2 = encodeLenField(
      1,
      concatBytes([
        encodeStrField(1, 'mid'),
        encodeStrField(1, 'w2'),
        encodeStrField(2, 'y'),
        encodeStrField(3, 'Conv_1006'),
        encodeStrField(4, 'Conv'),
        encodeAttrInts('kernel_shape', [3, 3, 3]),
        encodeAttrInts('strides', [2, 1, 1]),
        encodeAttrInts('pads', [1, 1, 1, 1, 1, 1]),
      ])
    );

    const modelBytes = concatBytes([
      encodeVarintField(1, 7),
      encodeLenField(
        7,
        concatBytes([
          conv1,
          conv2,
          encodeStrField(2, 'chained_conv3d_graph'),
          encodeLenField(5, encodeFloatTensorProto('w1', [3, 2, 3, 3, 3], w1Data)),
          encodeLenField(5, encodeFloatTensorProto('w2', [2, 3, 3, 3, 3], w2Data)),
          encodeLenField(11, encodeValueInfoFloat('x', inputDims)),
          encodeLenField(12, encodeValueInfoFloat('y', [1, 2, 2, 4, 4])),
        ])
      ),
      encodeLenField(8, concatBytes([encodeStrField(1, ''), encodeVarintField(2, 13)])),
    ]);

    const decomposed = decomposeOnnx5dConvNodesForWebGpu(modelBytes);
    expect(decomposed.decomposedCount).toBe(2);
    expect(decomposed.decomposedNodeNames).toEqual(['Conv_1005', 'Conv_1006']);
    expect(validateOnnxGraphNodeInputs(decomposed.bytes)).toMatchObject({
      valid: true,
      missingInputs: [],
    });

    const origSession = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'disabled',
      logSeverityLevel: 3,
    });
    const decSession = await ort.InferenceSession.create(decomposed.bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    try {
      const xTensor = new ort.Tensor('float32', inputData, [...inputDims]);
      const origArr = (await origSession.run({ x: xTensor })).y.data as Float32Array;
      const decArr = (await decSession.run({ x: xTensor })).y.data as Float32Array;
      expect(decArr.length).toBe(origArr.length);
      let maxAbsDiff = 0;
      for (let i = 0; i < origArr.length; i++) {
        maxAbsDiff = Math.max(maxAbsDiff, Math.abs(origArr[i] - decArr[i]));
      }
      expect(maxAbsDiff).toBeLessThan(1e-4);
    } finally {
      await origSession.release();
      await decSession.release();
    }
  });

  it('skips 5D Conv decomposition cleanly on ONNX opset < 11', () => {
    const { modelBytes } = buildValidConv3dOnnxModel({
      nodeName: 'Conv_old_opset',
      inputDims: [1, 2, 4, 4, 4],
      weightDims: [2, 2, 3, 3, 3],
      strides: [1, 1, 1],
      pads: [1, 1, 1, 1, 1, 1],
      opsetVersion: 10,
    });
    const res = decomposeOnnx5dConvNodesForWebGpu(modelBytes);
    expect(res.decomposedCount).toBe(0);
    expect(res.bytes).toBe(modelBytes);
  });

  interface ConvTranspose3dFixtureSpec {
    nodeName: string;
    inputDims: [number, number, number, number, number];
    weightDims: [number, number, number, number, number];
    strides?: [number, number, number];
    pads?: [number, number, number, number, number, number];
    dilations?: [number, number, number];
    outputPadding?: [number, number, number];
    outputShape?: [number, number, number, number, number];
    kernelShapeAttr?: number[];
    autoPad?: string;
    group?: number;
    withBias?: boolean;
    shareWeightAcrossTwoNodes?: boolean;
    opsetVersion?: number;
  }

  function buildValidConvTranspose3dOnnxModel(spec: ConvTranspose3dFixtureSpec): {
    modelBytes: Uint8Array;
    inputData: Float32Array;
  } {
    const [N, cIn, d0, d1, d2] = spec.inputDims;
    const [wCin, mPerGroup, k0, k1, k2] = spec.weightDims;
    const group = spec.group ?? 1;
    const cOut = mPerGroup * group;

    const weightLen = wCin * mPerGroup * k0 * k1 * k2;
    const weightData = new Float32Array(weightLen);
    for (let i = 0; i < weightLen; i++) {
      weightData[i] = ((i * 5 + 2) % 17 - 8) * 0.09;
    }

    const inputLen = N * cIn * d0 * d1 * d2;
    const inputData = new Float32Array(inputLen);
    for (let i = 0; i < inputLen; i++) {
      inputData[i] = ((i * 7 + 3) % 19 - 9) * 0.1;
    }

    const biasData = new Float32Array(cOut);
    for (let i = 0; i < cOut; i++) {
      biasData[i] = (i + 1) * 0.12;
    }

    const buildAttrs = (): Uint8Array[] => {
      const attrs: Uint8Array[] = [
        encodeAttrInts('kernel_shape', spec.kernelShapeAttr ?? [k0, k1, k2]),
      ];
      if (spec.strides) attrs.push(encodeAttrInts('strides', spec.strides));
      if (spec.pads) attrs.push(encodeAttrInts('pads', spec.pads));
      if (spec.dilations) attrs.push(encodeAttrInts('dilations', spec.dilations));
      if (spec.outputPadding) attrs.push(encodeAttrInts('output_padding', spec.outputPadding));
      if (spec.outputShape) attrs.push(encodeAttrInts('output_shape', spec.outputShape));
      if (spec.autoPad) attrs.push(encodeAttrString('auto_pad', spec.autoPad));
      if (group > 1) attrs.push(encodeAttrInt('group', group));
      return attrs;
    };

    const ctInputs = spec.withBias ? ['x', 'w', 'b'] : ['x', 'w'];
    const nodes: Uint8Array[] = [];
    if (spec.shareWeightAcrossTwoNodes) {
      nodes.push(
        encodeLenField(
          1,
          concatBytes([
            ...ctInputs.map((inp) => encodeStrField(1, inp)),
            encodeStrField(2, 'y1'),
            encodeStrField(3, `${spec.nodeName}_1`),
            encodeStrField(4, 'ConvTranspose'),
            ...buildAttrs(),
          ])
        ),
        encodeLenField(
          1,
          concatBytes([
            ...ctInputs.map((inp) => encodeStrField(1, inp)),
            encodeStrField(2, 'y2'),
            encodeStrField(3, `${spec.nodeName}_2`),
            encodeStrField(4, 'ConvTranspose'),
            ...buildAttrs(),
          ])
        ),
        encodeLenField(
          1,
          concatBytes([
            encodeStrField(1, 'y1'),
            encodeStrField(1, 'y2'),
            encodeStrField(2, 'y'),
            encodeStrField(3, 'Add_shared'),
            encodeStrField(4, 'Add'),
          ])
        )
      );
    } else {
      nodes.push(
        encodeLenField(
          1,
          concatBytes([
            ...ctInputs.map((inp) => encodeStrField(1, inp)),
            encodeStrField(2, 'y'),
            encodeStrField(3, spec.nodeName),
            encodeStrField(4, 'ConvTranspose'),
            ...buildAttrs(),
          ])
        )
      );
    }

    const graphParts: Uint8Array[] = [
      ...nodes,
      encodeStrField(2, 'conv_transpose3d_test_graph'),
      encodeLenField(5, encodeFloatTensorProto('w', spec.weightDims, weightData)),
    ];
    if (spec.withBias) {
      graphParts.push(encodeLenField(5, encodeFloatTensorProto('b', [cOut], biasData)));
    }
    graphParts.push(encodeLenField(11, encodeValueInfoFloat('x', spec.inputDims)));
    const dynamicOutShape = encodeLenField(
      2,
      concatBytes([0, 1, 2, 3, 4].map(() => encodeLenField(1, new Uint8Array(0))))
    );
    const dynamicOutType = encodeLenField(
      1,
      concatBytes([encodeVarintField(1, 1), dynamicOutShape])
    );
    graphParts.push(
      encodeLenField(
        12,
        concatBytes([encodeStrField(1, 'y'), encodeLenField(2, dynamicOutType)])
      )
    );

    const opsetImport = encodeLenField(
      8,
      concatBytes([encodeStrField(1, ''), encodeVarintField(2, spec.opsetVersion ?? 12)])
    );

    const modelBytes = concatBytes([
      encodeVarintField(1, 7),
      encodeLenField(7, concatBytes(graphParts)),
      opsetImport,
    ]);

    return { modelBytes, inputData };
  }

  it('decomposes trailing-unit-axis 5D ConvTranspose (including HITNet [3,1,4,4,1] shared weights, dynamic D2>1, bias, group, pads, output_padding) with exact ORT parity', async () => {
    const cases: Array<{ spec: ConvTranspose3dFixtureSpec; expectedDecomposedCount: number }> = [
      {
        spec: {
          nodeName: 'get_warp_disparity',
          inputDims: [1, 3, 6, 8, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 1],
          pads: [0, 0, 0, 0, 0, 0],
          dilations: [1, 1, 1],
          autoPad: 'NOTSET',
          shareWeightAcrossTwoNodes: true,
          opsetVersion: 12,
        },
        expectedDecomposedCount: 2,
      },
      {
        spec: {
          nodeName: 'ct_dynamic_d2_bias_pad_outpad',
          inputDims: [2, 3, 4, 5, 3],
          weightDims: [3, 2, 3, 3, 1],
          strides: [2, 2, 1],
          pads: [1, 1, 0, 1, 1, 0],
          dilations: [1, 1, 1],
          outputPadding: [1, 1, 0],
          withBias: true,
          opsetVersion: 12,
        },
        expectedDecomposedCount: 1,
      },
      {
        spec: {
          nodeName: 'ct_grouped_dilated',
          inputDims: [1, 4, 5, 5, 2],
          weightDims: [4, 2, 3, 3, 1],
          strides: [1, 2, 1],
          pads: [1, 0, 0, 1, 0, 0],
          dilations: [2, 1, 1],
          group: 2,
          withBias: true,
          opsetVersion: 12,
        },
        expectedDecomposedCount: 1,
      },
    ];

    for (const { spec, expectedDecomposedCount } of cases) {
      const { modelBytes, inputData } = buildValidConvTranspose3dOnnxModel(spec);
      expect(inspectStereoOnnxWebGpuCompatibility(modelBytes).hasConv3d).toBe(true);

      const decomposed = decomposeOnnx5dConvNodesForWebGpu(modelBytes);
      expect(decomposed.decomposedCount).toBe(expectedDecomposedCount);
      expect(inspectStereoOnnxWebGpuCompatibility(decomposed.bytes)).toMatchObject({
        compatible: true,
        hasConv3d: false,
      });
      expect(validateOnnxGraphNodeInputs(decomposed.bytes)).toMatchObject({
        valid: true,
        missingInputs: [],
      });

      const origSession = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'disabled',
        logSeverityLevel: 3,
      });
      const decSession = await ort.InferenceSession.create(decomposed.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        logSeverityLevel: 3,
      });

      try {
        const xTensor = new ort.Tensor('float32', inputData, spec.inputDims);
        const origOut = (await origSession.run({ x: xTensor })).y;
        const decOut = (await decSession.run({ x: xTensor })).y;

        expect(decOut.dims).toEqual(origOut.dims);
        const origArr = origOut.data as Float32Array;
        const decArr = decOut.data as Float32Array;
        expect(decArr.length).toBe(origArr.length);

        let maxAbsDiff = 0;
        for (let i = 0; i < origArr.length; i++) {
          expect(Number.isFinite(decArr[i])).toBe(true);
          maxAbsDiff = Math.max(maxAbsDiff, Math.abs(origArr[i] - decArr[i]));
        }
        expect(maxAbsDiff).toBeLessThan(1e-4);
      } finally {
        await origSession.release();
        await decSession.release();
      }
    }
  });

  it('leaves 5D ConvTranspose unconverted when attributes or trailing axis violate strict unit-axis constraints', () => {
    const incompatibleCases: Array<{ label: string; spec: ConvTranspose3dFixtureSpec }> = [
      {
        label: 'k2 > 1',
        spec: {
          nodeName: 'ct_k2_2',
          inputDims: [1, 3, 4, 4, 2],
          weightDims: [3, 1, 4, 4, 2],
          strides: [4, 4, 1],
          pads: [0, 0, 0, 0, 0, 0],
        },
      },
      {
        label: 's2 > 1',
        spec: {
          nodeName: 'ct_s2_2',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 2],
          pads: [0, 0, 0, 0, 0, 0],
        },
      },
      {
        label: 'd2 > 1',
        spec: {
          nodeName: 'ct_d2_2',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 1],
          dilations: [1, 1, 2],
          pads: [0, 0, 0, 0, 0, 0],
        },
      },
      {
        label: 'trailing pad != 0',
        spec: {
          nodeName: 'ct_pad2_1',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 1],
          pads: [0, 0, 1, 0, 0, 1],
        },
      },
      {
        label: 'trailing output_padding != 0',
        spec: {
          nodeName: 'ct_op2_1',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [2, 2, 1],
          outputPadding: [0, 0, 1],
        },
      },
      {
        label: 'auto_pad SAME_UPPER',
        spec: {
          nodeName: 'ct_autopad_same',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 1],
          autoPad: 'SAME_UPPER',
        },
      },
      {
        label: 'output_shape present',
        spec: {
          nodeName: 'ct_output_shape',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 1],
          outputShape: [1, 1, 16, 16, 1],
        },
      },
      {
        label: 'kernel_shape mismatch',
        spec: {
          nodeName: 'ct_kshape_mismatch',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          kernelShapeAttr: [4, 3, 1],
        },
      },
      {
        label: 'opset < 11',
        spec: {
          nodeName: 'ct_opset10',
          inputDims: [1, 3, 4, 4, 1],
          weightDims: [3, 1, 4, 4, 1],
          strides: [4, 4, 1],
          opsetVersion: 10,
        },
      },
    ];

    for (const { label, spec } of incompatibleCases) {
      const { modelBytes } = buildValidConvTranspose3dOnnxModel(spec);
      const res = decomposeOnnx5dConvNodesForWebGpu(modelBytes);
      expect(res.decomposedCount, label).toBe(0);
      expect(res.bytes, label).toBe(modelBytes);
      expect(inspectStereoOnnxWebGpuCompatibility(res.bytes), label).toMatchObject({
        compatible: false,
        hasConv3d: true,
      });
    }
  });
});
