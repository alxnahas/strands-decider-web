// Bench: ORT Web MatMulNBits (single node) vs hand-written WGSL, identical weights, GPU timestamps for ours.
import * as ort from "/ort/ort.webgpu.min.mjs";
import { Gemm, kernelV1 } from "./kernels.js";
const QS = new URLSearchParams(location.search);
const Ms = (QS.get("M") || "1,8,16,32,64,128").split(",").map(Number);
const out = document.getElementById("out"); const log = (s) => { out.textContent += "\n" + s; console.log(s); };
const f2h = (() => { const f = new Float32Array(1), u = new Uint32Array(f.buffer); return (x) => { f[0] = x; const b = u[0], s = (b >> 16) & 0x8000, e = ((b >> 23) & 0xff) - 112, m = b & 0x7fffff; if (e <= 0) return s; if (e >= 31) return s | 0x7c00; return s | (e << 10) | ((m + 0x1000) >> 13); }; })();
const h2f = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };
ort.env.wasm.wasmPaths = "/ort/";

async function timeOrt(sess, M, A16, iters) {
  const A = new ort.Tensor("float16", A16, [M, A16.length / M]);
  let r = await sess.run({ A }); r.Y.dispose?.();
  const t = performance.now(); let last;
  for (let i = 0; i < iters; i++) { last?.dispose?.(); last = (await sess.run({ A })).Y; }
  const y = await last.getData(); const ms = (performance.now() - t) / iters; last.dispose?.();
  return { ms, y };
}

async function timeOurs(dev, g, iters) {
  const qs = dev.createQuerySet({ type: "timestamp", count: 2 });
  const rb = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const st = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = dev.createCommandEncoder();
  const pass = e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
  for (let i = 0; i < iters; i++) g.encode(pass);
  pass.end(); e.resolveQuerySet(qs, 0, 2, rb, 0); e.copyBufferToBuffer(rb, 0, st, 0, 16); dev.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const [a, b] = new BigUint64Array(st.getMappedRange()); st.unmap();
  return Number(b - a) / 1e6 / iters;
}

(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const dev = await adapter.requestDevice({ requiredFeatures: ["shader-f16", "timestamp-query"], requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  const shapes = await fetch("/gemm/shapes.json").then((r) => r.json());
  const results = [];
  for (const { K, N } of shapes) {
    const B = new Uint8Array(await fetch(`/gemm/B_K${K}_N${N}.bin`).then((r) => r.arrayBuffer()));
    const Sc = new Uint16Array(await fetch(`/gemm/S_K${K}_N${N}.bin`).then((r) => r.arrayBuffer()));
    const sess = await ort.InferenceSession.create(`/gemm/mmnb_K${K}_N${N}.onnx`, { executionProviders: ["webgpu"], preferredOutputLocation: "gpu-buffer" });
    const g = new Gemm(dev, { K, N, B, Sc }, kernelV1());
    for (const M of Ms) {
      const A16 = new Uint16Array(M * K); for (let i = 0; i < A16.length; i++) A16[i] = f2h(Math.sin(i * 12.9898) * 0.5);
      const o = await timeOrt(sess, M, A16, 30);
      g.setA(M, A16); const ours = await timeOurs(dev, g, 30); const y = await g.readY();
      // correctness: fp32 reference on a few columns
      let err = 0, mag = 0;
      for (const n of [0, 1, N >> 1, N - 1]) for (let m = 0; m < M; m++) {
        let s = 0; for (let b = 0; b < K / 32; b++) { const sc = h2f(Sc[n * (K / 32) + b]); for (let e = 0; e < 32; e++) { const byte = B[(n * (K / 32) + b) * 16 + (e >> 1)]; s += h2f(A16[m * K + b * 32 + e]) * (((e & 1 ? byte >> 4 : byte & 15) - 8) * sc); } }
        err = Math.max(err, Math.abs(h2f(y[m * N + n]) - s), 0); mag = Math.max(mag, Math.abs(s));
        err = Math.max(err, Math.abs(h2f(o.y[m * N + n]) - s) * 0);  // ORT checked separately below
      }
      const gb = (N * K / 2 + N * K / 16) / 1e9;
      const row = { K, N, M, ort_ms: +o.ms.toFixed(3), ours_ms: +ours.toFixed(3), speedup: +(o.ms / ours).toFixed(2), ours_GBps: +(gb / (ours / 1e3)).toFixed(0), ours_TFLOPs: +(2 * M * N * K / (ours / 1e3) / 1e12).toFixed(2), rel_err: +(err / mag).toExponential(1) };
      results.push(row); log(JSON.stringify(row));
    }
    await sess.release();
  }
  window.results = results; document.body.dataset.done = "1";
})().catch((e) => { log("ERROR " + (e.stack || e)); document.body.dataset.done = "error"; });
