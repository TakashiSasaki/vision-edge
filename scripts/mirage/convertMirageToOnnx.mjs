import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { buildMirageWebGpuOnnxBytes } from '../../src/services/mirageOnnxConverter.ts';

const SAFETENSORS_URL =
  'https://huggingface.co/Cactus-Compute/mirage/resolve/main/mirage.safetensors';

async function fetchRangeFloat32(url, startByte, endByteInclusive) {
  const res = await fetch(url, {
    headers: { Range: `bytes=${startByte}-${endByteInclusive}` },
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`HTTP ${res.status} fetching byte range ${startByte}-${endByteInclusive}`);
  }
  const buf = await res.arrayBuffer();
  return new Float32Array(buf);
}

async function fetchUpstreamMirageTensors() {
  try {
    const lenRes = await fetch(SAFETENSORS_URL, {
      headers: { Range: 'bytes=0-7' },
    });
    if (!lenRes.ok && lenRes.status !== 206) {
      return undefined;
    }
    const lenBuf = Buffer.from(await lenRes.arrayBuffer());
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    const hdrRes = await fetch(SAFETENSORS_URL, {
      headers: { Range: `bytes=8-${8 + headerLen - 1}` },
    });
    if (!hdrRes.ok && hdrRes.status !== 206) {
      return undefined;
    }
    const header = JSON.parse(
      Buffer.from(await hdrRes.arrayBuffer()).toString('utf8')
    );
    const dataBase = 8 + headerLen;

    const getTensor = async (name) => {
      const entry = header[name];
      if (!entry || !Array.isArray(entry.data_offsets)) return undefined;
      const [relStart, relEnd] = entry.data_offsets;
      return fetchRangeFloat32(
        SAFETENSORS_URL,
        dataBase + relStart,
        dataBase + relEnd - 1
      );
    };

    const [
      stemW0,
      stemPw1,
      stemPw2,
      stemPw3,
      stemPw4,
      hconvGate,
      denseEdgeKernel,
      denseFineKernel,
      denseMidKernel,
      denseCoarseKernel,
      denseCoefKernel,
      denseOutKernel,
      denseOutBias,
      denseProtosKernel,
    ] = await Promise.all([
      getTensor('stem/w0'),
      getTensor('stem/pw_1/kernel'),
      getTensor('stem/pw_2/kernel'),
      getTensor('stem/pw_3/kernel'),
      getTensor('stem/pw_4/kernel'),
      getTensor('body_0/hconv/gate_conv'),
      getTensor('dense/edge/kernel'),
      getTensor('dense/fine/kernel'),
      getTensor('dense/mid/kernel'),
      getTensor('dense/coarse/kernel'),
      getTensor('dense/coef/kernel'),
      getTensor('dense/out/kernel'),
      getTensor('dense/bias'),
      getTensor('dense/protos/kernel'),
    ]);

    return {
      stemW0,
      stemPw1,
      stemPw2,
      stemPw3,
      stemPw4,
      hconvGate,
      denseEdgeKernel,
      denseFineKernel,
      denseMidKernel,
      denseCoarseKernel,
      denseCoefKernel,
      denseOutKernel,
      denseOutBias,
      denseProtosKernel,
    };
  } catch {
    return undefined;
  }
}

export async function generateMirageWebGpuOnnxFile(
  outputPath = path.join(process.cwd(), '.cache', 'container-models', 'mirage-webgpu.onnx')
) {
  const upstream = await fetchUpstreamMirageTensors();
  const onnxBytes = buildMirageWebGpuOnnxBytes(upstream);
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, onnxBytes);
  return {
    outputPath,
    byteLength: onnxBytes.byteLength,
    usedUpstream: Boolean(upstream?.stemW0),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const result = await generateMirageWebGpuOnnxFile();
  console.log(
    JSON.stringify({
      status: 'ok',
      ...result,
    })
  );
}
